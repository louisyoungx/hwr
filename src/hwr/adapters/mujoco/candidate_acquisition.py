"""Immutable acquisition evidence for frozen R0001-P50-E1."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence

import mujoco
import numpy as np

from hwr.adapters.mujoco.entity_contact_graph import p40_conservation_differences
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
    CANDIDATE_SCHEMA,
    CandidateSet,
    candidate_scores,
    deserialize_policy_input,
    generate_candidate_set,
    select_candidate_index,
    serialize_policy_input,
)

CAPTURE_SCHEMA = "hwr.p50-acquisition-capture/v1"
CAPSULE_SCHEMA = "hwr.p50-acquisition-capsule/v1"
CANDIDATE_VISIBLE_SCHEMA = "hwr.p50-candidate-visible-input/v1"
EPISODE_SCHEMA = "hwr.p50-acquisition-terminal/v1"
VISIBLE_ARRAYS = (
    "head_rgb_uint8",
    "head_depth_m",
    "head_depth_valid",
    "head_camera_intrinsics",
    "robot_from_head_camera",
)


class AcquisitionContractError(ValueError):
    """Raised when immutable acquisition evidence is incomplete or inconsistent."""

    def __init__(
        self, message: str, *, details: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.details = {} if details is None else dict(details)


@dataclass(frozen=True)
class AcquisitionCapture:
    capture_ordinal: int
    acquisition_phase: str
    final_input: bool
    observation_timestamp_ns: int
    sequence_id: int
    policy_input_bytes: bytes
    policy_input_sha256: str
    policy_input_byte_count: int
    candidate_visible_bytes: bytes
    candidate_visible_sha256: str
    candidate_visible_byte_count: int

    @property
    def observation_identity(self) -> tuple[int, int]:
        return self.observation_timestamp_ns, self.sequence_id

    def __post_init__(self) -> None:
        if self.capture_ordinal < 0:
            raise AcquisitionContractError("capture ordinal must be nonnegative")
        if self.acquisition_phase not in (
            "A1_panorama",
            "A3_panorama",
            "A4_seal",
        ):
            raise AcquisitionContractError("capture phase is outside frozen contract")
        if self.final_input != (self.acquisition_phase == "A4_seal"):
            raise AcquisitionContractError("final-input phase differs")
        try:
            value = deserialize_policy_input(self.policy_input_bytes)
        except ValueError as error:
            raise AcquisitionContractError("policy input round-trip differs") from error
        if serialize_policy_input(value) != self.policy_input_bytes:
            raise AcquisitionContractError("policy input round-trip differs")
        expected_visible = candidate_visible_bytes(value)
        checks = (
            self.observation_identity
            == (value.observation_timestamp_ns, value.sequence_id),
            self.policy_input_byte_count == len(self.policy_input_bytes),
            self.policy_input_sha256 == _sha256(self.policy_input_bytes),
            self.candidate_visible_bytes == expected_visible,
            self.candidate_visible_byte_count == len(expected_visible),
            self.candidate_visible_sha256 == _sha256(expected_visible),
        )
        if not all(checks):
            raise AcquisitionContractError("capture identity differs from payload bytes")


@dataclass(frozen=True)
class AcquisitionCapsule:
    planned_episode_id: str
    task_id: str
    cell_id: str
    cell_ordinal: int
    replicate_ordinal: int
    candidate_ordinal: int
    environment_seed: int
    policy_rng_seed: int
    sampled_observation_latency_steps: int
    sampled_action_latency_steps: int
    runtime_observation_latency_steps: int
    runtime_action_latency_steps: int
    latency_override_inactive: bool
    runtime_randomization_sha256: str
    acquisition_base_pose: tuple[float, float, float]
    captures: tuple[AcquisitionCapture, ...]
    candidate_bytes: bytes
    candidate_sha256: str
    candidate_count: int
    selected_index: int
    candidate_score_sha256: str
    acquisition_failure: str | None
    proposed_action_sha256: str
    applied_action_sha256: str
    observation_identity_trace_sha256: str
    same_seed_validation_replay: bool
    capture_enabled_disabled_identity: bool

    def __post_init__(self) -> None:
        if not _is_sha256(self.planned_episode_id):
            raise AcquisitionContractError("planned Episode identity must be SHA-256")
        if self.replicate_ordinal not in (0, 1):
            raise AcquisitionContractError("replicate ordinal differs")
        if not 0 <= self.candidate_ordinal <= 95:
            raise AcquisitionContractError("candidate ordinal exceeds frozen maximum")
        if (
            self.sampled_observation_latency_steps not in (1, 2)
            or self.sampled_action_latency_steps not in (1, 2)
        ):
            raise AcquisitionContractError("planned Episode is outside support cells")
        if (
            self.runtime_observation_latency_steps
            != self.sampled_observation_latency_steps
            or self.runtime_action_latency_steps
            != self.sampled_action_latency_steps
            or not self.latency_override_inactive
        ):
            raise AcquisitionContractError("runtime latency differs from plan")
        if not _is_sha256(self.runtime_randomization_sha256):
            raise AcquisitionContractError("randomization identity must be SHA-256")
        if len(self.acquisition_base_pose) != 3 or not np.isfinite(
            self.acquisition_base_pose
        ).all():
            raise AcquisitionContractError("acquisition base pose differs")
        ordinals = [capture.capture_ordinal for capture in self.captures]
        if ordinals != list(range(len(self.captures))):
            raise AcquisitionContractError("capture ordinals are incomplete")
        if sum(capture.final_input for capture in self.captures) != 1:
            raise AcquisitionContractError("capsule requires exactly one final input")
        if not self.captures[-1].final_input:
            raise AcquisitionContractError("final input must be the last capture")
        validate_capture_identities(self.captures)
        if self.candidate_count < 0:
            raise AcquisitionContractError("candidate count must be nonnegative")
        if self.candidate_bytes:
            if self.acquisition_failure is not None:
                raise AcquisitionContractError(
                    "failed acquisition cannot publish a formal candidate"
                )
            if self.candidate_sha256 != _sha256(self.candidate_bytes):
                raise AcquisitionContractError("candidate bytes identity differs")
            replay = replay_candidate_set(self)
            if (
                replay.canonical_bytes != self.candidate_bytes
                or replay.candidate_set_sha256 != self.candidate_sha256
                or len(replay.candidates) != self.candidate_count
            ):
                raise AcquisitionContractError("offline candidate replay differs")
            final = next(
                capture for capture in self.captures if capture.final_input
            )
            final_value = deserialize_policy_input(final.policy_input_bytes)
            scores = candidate_scores(
                replay,
                final_value.base_pose,
                acquisition_base_pose=self.acquisition_base_pose,
            )
            if self.candidate_score_sha256 != _score_sha256(scores):
                raise AcquisitionContractError(
                    "candidate score bytes identity differs"
                )
        elif self.candidate_count or self.selected_index != -1:
            raise AcquisitionContractError("missing candidate bytes are inconsistent")
        elif self.acquisition_failure is None:
            raise AcquisitionContractError(
                "successful acquisition must publish formal candidate bytes"
            )
        elif self.candidate_score_sha256 != _score_sha256(()):
            raise AcquisitionContractError("empty candidate score identity differs")
        elif self.candidate_sha256 != _sha256(b""):
            raise AcquisitionContractError("missing candidate bytes identity differs")
        elif self.candidate_sha256 != _sha256(b""):
            raise AcquisitionContractError("missing candidate bytes identity differs")
        if not -1 <= self.selected_index < self.candidate_count:
            raise AcquisitionContractError("selected candidate index is invalid")
        for identity in (
            self.proposed_action_sha256,
            self.applied_action_sha256,
            self.observation_identity_trace_sha256,
        ):
            if not _is_sha256(identity):
                raise AcquisitionContractError("trace identity must be SHA-256")


@dataclass(frozen=True)
class AcquisitionEpisodeResult:
    capsule: AcquisitionCapsule
    trace_step_count: int
    action_bounds_valid: bool
    stale_action_applied_count: int
    severe_collision_count: int
    invalid_force_count: int
    p40_conservation_maximum_difference: float
    safety_intervention_count: int
    runtime_terminal: bool = False
    primary_summary: Mapping[str, object] | None = None
    validation_summary: Mapping[str, object] | None = None
    replay_comparison: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        counts = (
            self.trace_step_count,
            self.stale_action_applied_count,
            self.severe_collision_count,
            self.invalid_force_count,
            self.safety_intervention_count,
        )
        if any(value < 0 for value in counts):
            raise AcquisitionContractError("episode counters must be nonnegative")
        if not np.isfinite(self.p40_conservation_maximum_difference):
            raise AcquisitionContractError("P40 conservation must be finite")


def candidate_visible_bytes(value) -> bytes:
    """Serialize exactly the observation fields read by the formal generator."""
    arrays = []
    for name in VISIBLE_ARRAYS:
        array = np.ascontiguousarray(getattr(value, name))
        arrays.append(array.astype(np.uint8, copy=False).tobytes() if name.endswith("valid")
                      else array.tobytes())
    proprioception = np.asarray(value.proprioception, dtype="<f8")
    candidate_proprioception = np.concatenate(
        (proprioception[:6], proprioception[12:18], proprioception[26:29])
    )
    return b"".join(
        (
            CANDIDATE_VISIBLE_SCHEMA.encode("ascii"),
            b"\0",
            *arrays,
            np.ascontiguousarray(candidate_proprioception, dtype="<f8").tobytes(),
        )
    )


def capture_policy_input(
    payload: bytes,
    *,
    capture_ordinal: int,
    acquisition_phase: str,
    final_input: bool,
) -> AcquisitionCapture:
    value = deserialize_policy_input(payload)
    if serialize_policy_input(value) != payload:
        raise AcquisitionContractError("policy input round-trip differs")
    visible = candidate_visible_bytes(value)
    return AcquisitionCapture(
        capture_ordinal=capture_ordinal,
        acquisition_phase=acquisition_phase,
        final_input=final_input,
        observation_timestamp_ns=value.observation_timestamp_ns,
        sequence_id=value.sequence_id,
        policy_input_bytes=bytes(payload),
        policy_input_sha256=_sha256(payload),
        policy_input_byte_count=len(payload),
        candidate_visible_bytes=visible,
        candidate_visible_sha256=_sha256(visible),
        candidate_visible_byte_count=len(visible),
    )


def validate_capture_identities(
    captures: Sequence[AcquisitionCapture],
) -> dict[tuple[int, int], str]:
    identities: dict[tuple[int, int], str] = {}
    for capture in captures:
        previous = identities.setdefault(
            capture.observation_identity, capture.candidate_visible_sha256
        )
        if previous != capture.candidate_visible_sha256:
            raise AcquisitionContractError("observation identity changed bytes")
    return identities


def replay_candidate_set(capsule: AcquisitionCapsule) -> CandidateSet:
    final = [capture for capture in capsule.captures if capture.final_input]
    if len(final) != 1:
        raise AcquisitionContractError("capsule requires one final input for replay")
    keyframes = tuple(
        capture.policy_input_bytes
        for capture in capsule.captures
        if not capture.final_input
    )
    return generate_candidate_set(
        keyframes,
        acquisition_base_pose=capsule.acquisition_base_pose,
        final_input=final[0].policy_input_bytes,
    )


def compare_episode_replays(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        "randomization": "runtime_randomization_sha256",
        "randomization_content": "runtime_randomization",
        "environment_seed": "environment_seed",
        "policy_rng_seed": "policy_rng_seed",
        "acquisition_base_pose": "acquisition_pose",
        "observation_latency": "runtime_observation_latency_steps",
        "action_latency": "runtime_action_latency_steps",
        "physical_trace": "physical_trace_sha256",
        "policy_input_trace": "policy_input_trace_sha256",
        "observation_trace": "observation_identity_trace_sha256",
        "capture_identity_sequence": "capture_identity_sequence",
        "capture_payload_bytes": "capture_payload_sha256",
        "proposed_action_trace": "proposed_action_sha256",
        "applied_action_trace": "applied_action_sha256",
        "candidate_bytes": "candidate_bytes",
        "candidate_identity": "candidate_sha256",
        "candidate_count": "candidate_count",
        "candidate_scores": "candidate_score_sha256",
        "selected_index": "selected_index",
        "failure": "failure",
        "runtime_terminal": "runtime_terminal",
        "trace_step_count": "trace_step_count",
    }
    checks = {
        name: first[field] == second[field]
        for name, field in fields.items()
    }
    checks["fresh_backend_reset"] = (
        first["_backend_object"] is not second["_backend_object"]
        and first["backend_run_ordinal"] != second["backend_run_ordinal"]
        and first["reset_count"] == second["reset_count"] == 1
    )
    checks["capture_disabled"] = (
        first["capture_persistence_enabled"] is True
        and second["capture_persistence_enabled"] is False
        and not second["captures"]
    )
    checks["latency_override_inactive"] = (
        first["latency_override_inactive"] is True
        and second["latency_override_inactive"] is True
    )
    return {"checks": checks, "passed": all(checks.values())}


class CandidateAcquisitionDiagnostic(TargetSelectionDiagnostic):
    """Runs the unchanged P41 acquisition while recording immutable evidence."""

    def run_episode(self, plan: Mapping[str, object]) -> AcquisitionEpisodeResult:
        primary = _run_acquisition_once(
            self, plan, capture_persistence_enabled=True, backend_run_ordinal=0
        )
        validation = _run_acquisition_once(
            self, plan, capture_persistence_enabled=False, backend_run_ordinal=1
        )
        comparison = compare_episode_replays(primary, validation)
        return _finish_episode(
            plan, self.task.task_id, primary,
            validation=validation,
            comparison=comparison,
        )


def _run_acquisition_once(
    diagnostic: CandidateAcquisitionDiagnostic,
    plan: Mapping[str, object],
    *,
    capture_persistence_enabled: bool,
    backend_run_ordinal: int,
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
    captures: list[AcquisitionCapture] = []
    proposed, applied, observations, payloads, trace = [], [], [], [], []
    failure: str | None = None
    tracker = MainEventTracker()
    previous_identity: tuple[int, int] | None = None
    try:
        backend.contact_ledger.set_enabled(True)
        observation = backend.reset(
            seed=int(plan["environment_seed"]), task_id=diagnostic.task.task_id
        )
        reset_audit = backend.task_audit()
        runtime = _runtime_latency_contract(plan, reset_audit)
        graph.reset()
        acquisition_pose = tuple(observation.proprioception.base_pose)
        state = _AcquisitionState(acquisition_pose)
        for step in range(ACQUISITION_STEPS):
            phase_index, phase, phase_step = _acquisition_phase(step)
            payload = policy_input_bytes(
                observation, history, available, int(plan["policy_rng_seed"]),
                phase_index=phase_index, phase_step=phase_step,
            )
            payloads.append(payload)
            failure = failure or _input_failure(
                backend, observation, payload, supported_only=True,
                previous_identity=previous_identity,
            )
            identity = (observation.timestamp_ns, observation.sequence_id)
            previous_identity = identity
            action, capture = state.action(phase, payload)
            if failure is not None:
                action, capture = _hold(deserialize_policy_input(payload)), False
            if capture:
                keyframes.append(payload)
                if capture_persistence_enabled:
                    captures.append(
                        capture_policy_input(
                            payload,
                            capture_ordinal=len(captures),
                            acquisition_phase=phase,
                            final_input=False,
                        )
                    )
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
            observation, history, available, int(plan["policy_rng_seed"]),
            phase_index=4, phase_step=5,
        )
        payloads.append(final_payload)
        if capture_persistence_enabled:
            captures.append(
                capture_policy_input(
                    final_payload,
                    capture_ordinal=len(captures),
                    acquisition_phase="A4_seal",
                    final_input=True,
                )
            )
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
            -1 if candidate_set is None else select_candidate_index(
                candidate_set, final_value.base_pose,
                acquisition_base_pose=acquisition_pose,
            )
        )
        scores = (
            ()
            if candidate_set is None
            else candidate_scores(
                candidate_set, final_value.base_pose,
                acquisition_base_pose=acquisition_pose,
            )
        )
        candidate_bytes = b"" if candidate_set is None else candidate_set.canonical_bytes
        graph_report = graph.report()
        ledger_report = backend.contact_ledger.report()
        audit = backend.task_audit()
        conservation = p40_conservation_differences(
            graph_report, ledger_report
        )
        captured_inputs = (*keyframes, final_payload)
        return {
            "_backend_object": backend,
            "backend_run_ordinal": backend_run_ordinal,
            "reset_count": 1,
            "capture_persistence_enabled": capture_persistence_enabled,
            "environment_seed": int(plan["environment_seed"]),
            "policy_rng_seed": int(plan["policy_rng_seed"]),
            "acquisition_pose": acquisition_pose,
            "captures": captures,
            "capture_identity_sequence": tuple(
                (
                    deserialize_policy_input(payload).observation_timestamp_ns,
                    deserialize_policy_input(payload).sequence_id,
                )
                for payload in captured_inputs
            ),
            "capture_payload_sha256": _bytes_sequence_sha256(captured_inputs),
            "candidate_bytes": candidate_bytes,
            "candidate_sha256": _sha256(candidate_bytes),
            "candidate_count": 0 if candidate_set is None else len(candidate_set.candidates),
            "selected_index": selected,
            "candidate_score_sha256": _score_sha256(scores),
            "failure": failure,
            "proposed_action_sha256": _trace_sha256(proposed),
            "applied_action_sha256": _trace_sha256(applied),
            "observation_identity_trace_sha256": _canonical_sha256(observations),
            "policy_input_trace_sha256": _bytes_sequence_sha256(payloads),
            "physical_trace_sha256": _physical_trace_sha256(trace),
            "trace_step_count": len(trace),
            "runtime_terminal": bool(trace and trace[-1]["terminal"]),
            "runtime_randomization": json.loads(
                json.dumps(reset_audit["randomization"], sort_keys=True)
            ),
            **runtime,
            "trace": trace,
            "action_bounds_valid": all(
                bool(row.get("action_bounds_valid", False)) for row in trace
            ),
            "stale_action_applied_count": sum(
                bool(row.get("outside_validity_window"))
                and row.get("applied_action") != row.get("hold_action")
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
                bool(row.get("safety_intervened")) for row in trace
            ),
        }
    finally:
        backend.close()


def _finish_episode(
    plan,
    task_id,
    run,
    *,
    validation,
    comparison,
) -> AcquisitionEpisodeResult:
    candidate_bytes = run["candidate_bytes"]
    capsule = AcquisitionCapsule(
        planned_episode_id=str(plan["planned_episode_id"]),
        task_id=task_id,
        cell_id=str(plan["cell_id"]),
        cell_ordinal=int(plan["cell_ordinal"]),
        replicate_ordinal=int(plan["replicate_ordinal"]),
        candidate_ordinal=int(plan["candidate_ordinal"]),
        environment_seed=int(plan["environment_seed"]),
        policy_rng_seed=int(plan["policy_rng_seed"]),
        sampled_observation_latency_steps=int(
            plan["sampled_observation_latency_steps"]
        ),
        sampled_action_latency_steps=int(plan["sampled_action_latency_steps"]),
        runtime_observation_latency_steps=run["runtime_observation_latency_steps"],
        runtime_action_latency_steps=run["runtime_action_latency_steps"],
        latency_override_inactive=run["latency_override_inactive"],
        runtime_randomization_sha256=run["runtime_randomization_sha256"],
        acquisition_base_pose=run["acquisition_pose"],
        captures=tuple(run["captures"]),
        candidate_bytes=candidate_bytes,
        candidate_sha256=_sha256(candidate_bytes),
        candidate_count=run["candidate_count"],
        selected_index=run["selected_index"],
        candidate_score_sha256=run["candidate_score_sha256"],
        acquisition_failure=run["failure"],
        proposed_action_sha256=run["proposed_action_sha256"],
        applied_action_sha256=run["applied_action_sha256"],
        observation_identity_trace_sha256=run["observation_identity_trace_sha256"],
        same_seed_validation_replay=comparison["passed"],
        capture_enabled_disabled_identity=comparison["passed"],
    )
    return AcquisitionEpisodeResult(
        capsule=capsule,
        trace_step_count=run["trace_step_count"],
        action_bounds_valid=run["action_bounds_valid"],
        stale_action_applied_count=run["stale_action_applied_count"],
        severe_collision_count=run["severe_collision_count"],
        invalid_force_count=run["invalid_force_count"],
        p40_conservation_maximum_difference=run[
            "p40_conservation_maximum_difference"
        ],
        safety_intervention_count=run["safety_intervention_count"],
        runtime_terminal=run["runtime_terminal"],
        primary_summary=_run_summary(run),
        validation_summary=_run_summary(validation),
        replay_comparison=comparison,
    )


def _runtime_latency_contract(plan, audit) -> dict[str, object]:
    randomization = audit["randomization"]
    runtime = {
        "runtime_observation_latency_steps": int(
            randomization["observation_latency_steps"]
        ),
        "runtime_action_latency_steps": int(randomization["action_latency_steps"]),
        "latency_override_inactive": all(
            audit[name] is None
            for name in (
                "action_latency_diagnostic",
                "observation_latency_diagnostic",
                "latency_pair_diagnostic",
            )
        ),
        "runtime_randomization_sha256": _canonical_sha256(randomization),
    }
    planned = {
        "planned_observation_latency_steps": int(
            plan["sampled_observation_latency_steps"]
        ),
        "planned_action_latency_steps": int(plan["sampled_action_latency_steps"]),
    }
    if (
        runtime["runtime_observation_latency_steps"]
        != planned["planned_observation_latency_steps"]
        or runtime["runtime_action_latency_steps"]
        != planned["planned_action_latency_steps"]
        or not runtime["latency_override_inactive"]
    ):
        raise AcquisitionContractError(
            "runtime latency or override state differs from plan",
            details={**planned, **runtime},
        )
    return {**planned, **runtime}


def _run_summary(run) -> dict[str, object]:
    return {
        key: run[key]
        for key in (
            "backend_run_ordinal",
            "reset_count",
            "capture_persistence_enabled",
            "environment_seed",
            "policy_rng_seed",
            "runtime_observation_latency_steps",
            "runtime_action_latency_steps",
            "latency_override_inactive",
            "runtime_randomization_sha256",
            "runtime_randomization",
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
    }


def mujoco_runtime_version() -> str:
    return str(mujoco.__version__)


def capture_record(
    capture: AcquisitionCapture,
    *,
    policy_blob: str,
    candidate_visible_blob: str,
) -> dict[str, object]:
    return {
        "schema_version": CAPTURE_SCHEMA,
        "capture_ordinal": capture.capture_ordinal,
        "acquisition_phase": capture.acquisition_phase,
        "final_input": capture.final_input,
        "observation_timestamp_ns": capture.observation_timestamp_ns,
        "sequence_id": capture.sequence_id,
        "policy_input": {
            "path": policy_blob,
            "sha256": capture.policy_input_sha256,
            "bytes": capture.policy_input_byte_count,
        },
        "candidate_visible_input": {
            "path": candidate_visible_blob,
            "sha256": capture.candidate_visible_sha256,
            "bytes": capture.candidate_visible_byte_count,
        },
    }


def _trace_sha256(trace: Sequence[Sequence[float]]) -> str:
    array = np.ascontiguousarray(trace, dtype="<f8")
    return _sha256(array.tobytes())


def _bytes_sequence_sha256(payloads: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _physical_trace_sha256(trace: Sequence[Mapping[str, object]]) -> str:
    return _canonical_sha256(trace)


def _score_sha256(scores: Sequence[float]) -> str:
    return _sha256(np.ascontiguousarray(scores, dtype="<f8").tobytes())


def _action_bytes(action: Sequence[float]) -> bytes:
    return np.ascontiguousarray(action, dtype="<f8").tobytes()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return _sha256(payload)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()
