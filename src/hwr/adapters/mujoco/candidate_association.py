"""MuJoCo sidecar and replay bridge for frozen R0001-P68-E1."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping

import mujoco
import numpy as np

from hwr.adapters.mujoco.candidate_acquisition import (
    _bytes_sequence_sha256,
    _canonical_sha256,
    _physical_trace_sha256,
    _runtime_latency_contract,
    _score_sha256,
    _sha256,
    _trace_sha256,
    capture_policy_input,
)
from hwr.adapters.mujoco.entity_contact_graph import (
    p40_conservation_differences,
)
from hwr.adapters.mujoco.formal_household_backend import (
    MujocoFormalHouseholdDualArmBackend,
)
from hwr.adapters.mujoco.target_selection_diagnostic import (
    INVALID_GRAPH_FIELDS,
    MainEventTracker,
    TargetSelectionDiagnostic,
    _AcquisitionState,
    _acquisition_phase,
    _append_history,
    _graph_from_backend,
    _hold,
    _input_failure,
    _step,
    policy_input_bytes,
)
from hwr.eval.target_selection import (
    ACQUISITION_STEPS,
    candidate_scores,
    deserialize_policy_input,
    generate_candidate_set,
    select_candidate_index,
)


class CandidateAssociationBackend(MujocoFormalHouseholdDualArmBackend):
    """Capture segmentation at source observations without changing observations."""

    def __init__(self, task, binding) -> None:
        super().__init__(
            task,
            binding,
            camera_width=256,
            camera_height=192,
            evaluation_profile=True,
        )
        self._segmentation_renderer = mujoco.Renderer(
            self.model, height=192, width=256
        )
        self._segmentation_by_identity: dict[
            tuple[int, int], np.ndarray
        ] = {}
        self._segmentation_order: deque[tuple[int, int]] = deque()

    def _observation(self):
        observation = super()._observation()
        self._segmentation_renderer.enable_segmentation_rendering()
        self._segmentation_renderer.update_scene(
            self.data, camera="head_depth"
        )
        segmentation = np.ascontiguousarray(
            self._segmentation_renderer.render(), dtype=np.int32
        )
        identity = (observation.timestamp_ns, observation.sequence_id)
        self._segmentation_by_identity[identity] = segmentation
        self._segmentation_order.append(identity)
        while len(self._segmentation_order) > 4:
            expired = self._segmentation_order.popleft()
            self._segmentation_by_identity.pop(expired, None)
        return observation

    def segmentation_for(self, identity: tuple[int, int]) -> np.ndarray:
        try:
            return self._segmentation_by_identity[identity].copy()
        except KeyError as error:
            raise RuntimeError(
                "segmentation source identity fell outside latency queue"
            ) from error

    def close(self) -> None:
        self._segmentation_renderer.close()
        super().close()


class CandidateAssociationDiagnostic(TargetSelectionDiagnostic):
    """Replay one frozen acquisition while capturing aligned segmentation."""

    def _backend(self) -> CandidateAssociationBackend:
        return CandidateAssociationBackend(self.task, self.binding)

    def run_episode(
        self, plan: Mapping[str, object]
    ) -> dict[str, object]:
        return _run_association_once(self, plan)


def _run_association_once(
    diagnostic: CandidateAssociationDiagnostic,
    plan: Mapping[str, object],
) -> dict[str, object]:
    backend = diagnostic._backend()
    graph = _graph_from_backend(backend)
    original_substep = backend._after_physics_substep

    def observe_substep() -> None:
        original_substep()
        graph.sample_mujoco_substep(backend.model, backend.data)

    backend._after_physics_substep = observe_substep
    history: list[tuple[float, ...]] = []
    available: list[bool] = []
    keyframes: list[bytes] = []
    captures = []
    segmentations = []
    proposed, applied, observations, payloads, trace = [], [], [], [], []
    failure: str | None = None
    tracker = MainEventTracker()
    previous_identity: tuple[int, int] | None = None
    try:
        backend.contact_ledger.set_enabled(True)
        observation = backend.reset(
            seed=int(plan["environment_seed"]),
            task_id=diagnostic.task.task_id,
        )
        reset_audit = backend.task_audit()
        runtime = _runtime_latency_contract(plan, reset_audit)
        graph.reset()
        acquisition_pose = tuple(observation.proprioception.base_pose)
        state = _AcquisitionState(acquisition_pose)
        for step in range(ACQUISITION_STEPS):
            phase_index, phase, phase_step = _acquisition_phase(step)
            payload = policy_input_bytes(
                observation,
                history,
                available,
                int(plan["policy_rng_seed"]),
                phase_index=phase_index,
                phase_step=phase_step,
            )
            payloads.append(payload)
            failure = failure or _input_failure(
                backend,
                observation,
                payload,
                supported_only=True,
                previous_identity=previous_identity,
            )
            identity = (observation.timestamp_ns, observation.sequence_id)
            previous_identity = identity
            action, capture = state.action(phase, payload)
            if failure is not None:
                action, capture = _hold(deserialize_policy_input(payload)), False
            if capture:
                keyframes.append(payload)
                captures.append(
                    capture_policy_input(
                        payload,
                        capture_ordinal=len(captures),
                        acquisition_phase=phase,
                        final_input=False,
                    )
                )
                segmentations.append(backend.segmentation_for(identity))
            proposed.append(tuple(action))
            observations.append(identity)
            observation, period, row = _step(
                backend, graph, observation, action, step
            )
            tracker.update(
                period, row.pop("_motion_start"), row.pop("_motion_end")
            )
            if tracker.event:
                failure = failure or "main_event_during_acquisition"
            if row["safety_intervened"]:
                failure = failure or "safety_intervention_during_acquisition"
            trace.append(row)
            applied.append(tuple(row["applied_action"]))
            _append_history(history, available, row["applied_action"])
            if row["terminal"]:
                failure = failure or "runtime_terminal_during_acquisition"
                break
        final_payload = policy_input_bytes(
            observation,
            history,
            available,
            int(plan["policy_rng_seed"]),
            phase_index=4,
            phase_step=5,
        )
        payloads.append(final_payload)
        final_identity = (
            observation.timestamp_ns,
            observation.sequence_id,
        )
        captures.append(
            capture_policy_input(
                final_payload,
                capture_ordinal=len(captures),
                acquisition_phase="A4_seal",
                final_input=True,
            )
        )
        segmentations.append(backend.segmentation_for(final_identity))
        candidate_set = (
            None
            if failure is not None
            else generate_candidate_set(
                keyframes,
                acquisition_base_pose=acquisition_pose,
                final_input=final_payload,
            )
        )
        final_value = deserialize_policy_input(final_payload)
        selected = (
            -1
            if candidate_set is None
            else select_candidate_index(
                candidate_set,
                final_value.base_pose,
                acquisition_base_pose=acquisition_pose,
            )
        )
        scores = (
            ()
            if candidate_set is None
            else candidate_scores(
                candidate_set,
                final_value.base_pose,
                acquisition_base_pose=acquisition_pose,
            )
        )
        candidate_bytes = (
            b"" if candidate_set is None else candidate_set.canonical_bytes
        )
        graph_report = graph.report()
        audit = backend.task_audit()
        conservation = p40_conservation_differences(
            graph_report, backend.contact_ledger.report()
        )
        captured_inputs = (*keyframes, final_payload)
        return {
            "environment_seed": int(plan["environment_seed"]),
            "policy_rng_seed": int(plan["policy_rng_seed"]),
            "acquisition_pose": acquisition_pose,
            "captures": captures,
            "segmentations": segmentations,
            "keyframes": keyframes,
            "final_payload": final_payload,
            "candidate_set": candidate_set,
            "candidate_bytes": candidate_bytes,
            "candidate_sha256": _sha256(candidate_bytes),
            "candidate_count": (
                0 if candidate_set is None else len(candidate_set.candidates)
            ),
            "candidate_score_sha256": _score_sha256(scores),
            "selected_index": selected,
            "failure": failure,
            "proposed_action_sha256": _trace_sha256(proposed),
            "applied_action_sha256": _trace_sha256(applied),
            "observation_identity_trace_sha256": _canonical_sha256(observations),
            "policy_input_trace_sha256": _bytes_sequence_sha256(payloads),
            "physical_trace_sha256": _physical_trace_sha256(trace),
            "capture_payload_sha256": _bytes_sequence_sha256(captured_inputs),
            "capture_identity_sequence": [
                list(capture.observation_identity) for capture in captures
            ],
            "trace_step_count": len(trace),
            "runtime_terminal": bool(trace and trace[-1]["terminal"]),
            "runtime_randomization_sha256": _canonical_sha256(
                reset_audit["randomization"]
            ),
            **runtime,
            "action_bounds_valid": all(
                bool(row["action_bounds_valid"]) for row in trace
            ),
            "stale_action_applied_count": sum(
                bool(row["outside_validity_window"])
                and row["applied_action"] != row["hold_action"]
                for row in trace
            ),
            "severe_collision_count": int(audit["severe_collision_count"]),
            "invalid_force_count": sum(
                int(graph_report[name]) for name in INVALID_GRAPH_FIELDS
            ),
            "p40_conservation_maximum_difference": float(
                conservation["maximum_absolute_difference"]
            ),
            "safety_intervention_count": sum(
                bool(row["safety_intervened"]) for row in trace
            ),
            "segmentation_sequence_sha256": _segmentation_sha256(
                segmentations
            ),
        }
    finally:
        backend.close()


def _segmentation_sha256(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value, dtype="<i4")
        digest.update(len(array.tobytes()).to_bytes(8, "little"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def summary_for_identity(run: Mapping[str, object]) -> dict[str, object]:
    """Return fields that must match the historical observer-off run."""
    names = (
        "environment_seed",
        "policy_rng_seed",
        "runtime_observation_latency_steps",
        "runtime_action_latency_steps",
        "latency_override_inactive",
        "runtime_randomization_sha256",
        "physical_trace_sha256",
        "policy_input_trace_sha256",
        "observation_identity_trace_sha256",
        "capture_identity_sequence",
        "capture_payload_sha256",
        "proposed_action_sha256",
        "applied_action_sha256",
        "candidate_sha256",
        "candidate_count",
        "candidate_score_sha256",
        "selected_index",
        "trace_step_count",
        "runtime_terminal",
        "failure",
        "action_bounds_valid",
        "stale_action_applied_count",
        "severe_collision_count",
        "invalid_force_count",
        "p40_conservation_maximum_difference",
        "safety_intervention_count",
    )
    return {name: run[name] for name in names}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
