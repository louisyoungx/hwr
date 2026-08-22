"""Immutable acquisition evidence for frozen R0001-P50-E1."""

from __future__ import annotations

import hashlib
import json
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
    replicate_ordinal: int
    candidate_ordinal: int
    environment_seed: int
    policy_rng_seed: int
    sampled_observation_latency_steps: int
    sampled_action_latency_steps: int
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
    same_seed_lockstep_replay: bool
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
        elif self.candidate_score_sha256 != _score_sha256(()):
            raise AcquisitionContractError("empty candidate score identity differs")
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
    first: AcquisitionEpisodeResult,
    second: AcquisitionEpisodeResult,
) -> dict[str, object]:
    left, right = first.capsule, second.capsule
    checks = {
        "capture_identity_sequence": [
            capture.observation_identity for capture in left.captures
        ] == [capture.observation_identity for capture in right.captures],
        "policy_input_bytes": [
            capture.policy_input_bytes for capture in left.captures
        ] == [capture.policy_input_bytes for capture in right.captures],
        "candidate_visible_bytes": [
            capture.candidate_visible_bytes for capture in left.captures
        ] == [capture.candidate_visible_bytes for capture in right.captures],
        "candidate_bytes": left.candidate_bytes == right.candidate_bytes,
        "proposed_action_trace": (
            left.proposed_action_sha256 == right.proposed_action_sha256
        ),
        "applied_action_trace": left.applied_action_sha256 == right.applied_action_sha256,
        "observation_identity_trace": (
            left.observation_identity_trace_sha256
            == right.observation_identity_trace_sha256
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


class CandidateAcquisitionDiagnostic(TargetSelectionDiagnostic):
    """Runs the unchanged P41 acquisition while recording immutable evidence."""

    def run_episode(self, plan: Mapping[str, object]) -> AcquisitionEpisodeResult:
        primary = _run_acquisition_once(self, plan)
        return _finish_episode(
            plan, self.task.task_id, primary,
            replay_equal=bool(primary["controller_replay_equal"]),
            capture_shadow_equal=bool(primary["capture_shadow_equal"]),
        )


def _run_acquisition_once(
    diagnostic: CandidateAcquisitionDiagnostic,
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
    captures: list[AcquisitionCapture] = []
    proposed, applied, observations, trace = [], [], [], []
    failure: str | None = None
    tracker = MainEventTracker()
    previous_identity: tuple[int, int] | None = None
    replay_state = None
    disabled_state = None
    controller_replay_equal = True
    capture_shadow_equal = True
    try:
        backend.contact_ledger.set_enabled(True)
        observation = backend.reset(
            seed=int(plan["environment_seed"]), task_id=diagnostic.task.task_id
        )
        graph.reset()
        acquisition_pose = tuple(observation.proprioception.base_pose)
        state = _AcquisitionState(acquisition_pose)
        replay_state = _AcquisitionState(acquisition_pose)
        disabled_state = _AcquisitionState(acquisition_pose)
        for step in range(ACQUISITION_STEPS):
            phase_index, phase, phase_step = _acquisition_phase(step)
            payload = policy_input_bytes(
                observation, history, available, int(plan["policy_rng_seed"]),
                phase_index=phase_index, phase_step=phase_step,
            )
            failure = failure or _input_failure(
                backend, observation, payload, supported_only=True,
                previous_identity=previous_identity,
            )
            identity = (observation.timestamp_ns, observation.sequence_id)
            previous_identity = identity
            action, capture = state.action(phase, payload)
            replay_action, replay_capture = replay_state.action(phase, payload)
            disabled_action, disabled_capture = disabled_state.action(phase, payload)
            controller_replay_equal &= (
                _action_bytes(action) == _action_bytes(replay_action)
                and capture == replay_capture
            )
            capture_shadow_equal &= (
                _action_bytes(action) == _action_bytes(disabled_action)
                and capture == disabled_capture
            )
            if failure is not None:
                action, capture = _hold(deserialize_policy_input(payload)), False
            if capture:
                keyframes.append(payload)
                captures.append(
                    capture_policy_input(
                        payload, capture_ordinal=len(captures),
                        acquisition_phase=phase, final_input=False,
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
        captures.append(
            capture_policy_input(
                final_payload, capture_ordinal=len(captures),
                acquisition_phase="A4_seal", final_input=True,
            )
        )
        candidate_set = generate_candidate_set(
            keyframes,
            acquisition_base_pose=acquisition_pose,
            final_input=final_payload,
        )
        final_value = deserialize_policy_input(final_payload)
        selected = (
            -1 if failure is not None else select_candidate_index(
                candidate_set, final_value.base_pose,
                acquisition_base_pose=acquisition_pose,
            )
        )
        scores = candidate_scores(
            candidate_set, final_value.base_pose,
            acquisition_base_pose=acquisition_pose,
        )
        return {
            "acquisition_pose": acquisition_pose,
            "captures": captures,
            "keyframes": tuple(keyframes),
            "final_payload": final_payload,
            "candidate_set": candidate_set,
            "selected_index": selected,
            "candidate_score_sha256": _score_sha256(scores),
            "failure": failure,
            "proposed_action_sha256": _trace_sha256(proposed),
            "applied_action_sha256": _trace_sha256(applied),
            "observation_identity_trace_sha256": _canonical_sha256(observations),
            "trace": trace,
            "audit": backend.task_audit(),
            "graph_report": graph.report(),
            "ledger_report": backend.contact_ledger.report(),
            "controller_replay_equal": controller_replay_equal,
            "capture_shadow_equal": capture_shadow_equal,
        }
    finally:
        backend.close()


def _finish_episode(
    plan,
    task_id,
    run,
    *,
    replay_equal,
    capture_shadow_equal,
) -> AcquisitionEpisodeResult:
    candidate_set = run["candidate_set"]
    candidate_bytes = candidate_set.canonical_bytes
    capsule = AcquisitionCapsule(
        planned_episode_id=str(plan["planned_episode_id"]),
        task_id=task_id,
        cell_id=str(plan["cell_id"]),
        replicate_ordinal=int(plan["replicate_ordinal"]),
        candidate_ordinal=int(plan["candidate_ordinal"]),
        environment_seed=int(plan["environment_seed"]),
        policy_rng_seed=int(plan["policy_rng_seed"]),
        sampled_observation_latency_steps=int(
            plan["sampled_observation_latency_steps"]
        ),
        sampled_action_latency_steps=int(plan["sampled_action_latency_steps"]),
        acquisition_base_pose=run["acquisition_pose"],
        captures=tuple(run["captures"]),
        candidate_bytes=candidate_bytes,
        candidate_sha256=_sha256(candidate_bytes),
        candidate_count=len(candidate_set.candidates),
        selected_index=run["selected_index"],
        candidate_score_sha256=run["candidate_score_sha256"],
        acquisition_failure=run["failure"],
        proposed_action_sha256=run["proposed_action_sha256"],
        applied_action_sha256=run["applied_action_sha256"],
        observation_identity_trace_sha256=run["observation_identity_trace_sha256"],
        same_seed_lockstep_replay=replay_equal,
        capture_enabled_disabled_identity=capture_shadow_equal,
    )
    conservation = p40_conservation_differences(
        run["graph_report"], run["ledger_report"]
    )
    return AcquisitionEpisodeResult(
        capsule=capsule,
        trace_step_count=len(run["trace"]),
        action_bounds_valid=all(
            bool(row.get("action_bounds_valid", False)) for row in run["trace"]
        ),
        stale_action_applied_count=sum(
            bool(row.get("outside_validity_window"))
            and row.get("applied_action") != row.get("hold_action")
            for row in run["trace"]
        ),
        severe_collision_count=int(run["audit"]["severe_collision_count"]),
        invalid_force_count=sum(
            int(run["graph_report"][name]) for name in INVALID_GRAPH_FIELDS
        ),
        p40_conservation_maximum_difference=float(
            conservation["maximum_absolute_difference"]
        ),
        safety_intervention_count=sum(
            bool(row.get("safety_intervened")) for row in run["trace"]
        ),
    )


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


def persist_episode(
    result: AcquisitionEpisodeResult,
) -> tuple[dict[str, object], dict[str, object], dict[str, bytes]]:
    capsule = result.capsule
    prefix = f"blobs/{capsule.planned_episode_id}"
    blobs: dict[str, bytes] = {}
    captures = []
    for capture in capsule.captures:
        suffix = f"{capture.capture_ordinal:02d}"
        policy_path = f"{prefix}/capture-{suffix}-policy.bin"
        visible_path = f"{prefix}/capture-{suffix}-candidate-visible.bin"
        blobs[policy_path] = capture.policy_input_bytes
        blobs[visible_path] = capture.candidate_visible_bytes
        captures.append(capture_record(
            capture, policy_blob=policy_path, candidate_visible_blob=visible_path
        ))
    candidate_path = f"{prefix}/candidate-set.json"
    blobs[candidate_path] = capsule.candidate_bytes
    replay = replay_candidate_set(capsule)
    record = {
        "schema_version": CAPSULE_SCHEMA,
        "planned_episode_id": capsule.planned_episode_id,
        "task_id": capsule.task_id,
        "cell_id": capsule.cell_id,
        "replicate_ordinal": capsule.replicate_ordinal,
        "candidate_ordinal": capsule.candidate_ordinal,
        "environment_seed": capsule.environment_seed,
        "policy_rng_seed": capsule.policy_rng_seed,
        "sampled_observation_latency_steps": capsule.sampled_observation_latency_steps,
        "sampled_action_latency_steps": capsule.sampled_action_latency_steps,
        "acquisition_base_pose": list(capsule.acquisition_base_pose),
        "captures": captures,
        "capture_count": len(captures),
        "candidate_set": {
            "path": candidate_path,
            "sha256": capsule.candidate_sha256,
            "bytes": len(capsule.candidate_bytes),
            "candidate_count": capsule.candidate_count,
            "selected_index": capsule.selected_index,
            "score_bytes_sha256": capsule.candidate_score_sha256,
            "schema_version": CANDIDATE_SCHEMA,
        },
        "acquisition_failure": capsule.acquisition_failure,
        "proposed_action_sha256": capsule.proposed_action_sha256,
        "applied_action_sha256": capsule.applied_action_sha256,
        "observation_identity_trace_sha256": capsule.observation_identity_trace_sha256,
        "same_seed_lockstep_replay": capsule.same_seed_lockstep_replay,
        "capture_enabled_disabled_identity": capsule.capture_enabled_disabled_identity,
        "offline_candidate_replay_bit_identical": (
            replay.canonical_bytes == capsule.candidate_bytes
            and replay.candidate_set_sha256 == capsule.candidate_sha256
        ),
        "anchor_blobs_complete": all(
            blobs[row[kind]["path"]]
            and _sha256(blobs[row[kind]["path"]]) == row[kind]["sha256"]
            for row in captures
            for kind in ("policy_input", "candidate_visible_input")
        ),
    }
    terminal = {
        "schema_version": EPISODE_SCHEMA,
        "planned_episode_id": capsule.planned_episode_id,
        "task_id": capsule.task_id,
        "cell_id": capsule.cell_id,
        "replicate_ordinal": capsule.replicate_ordinal,
        "candidate_ordinal": capsule.candidate_ordinal,
        "replacement": False,
        "resolved": True,
        "trace_step_count": result.trace_step_count,
        "acquisition_failure": capsule.acquisition_failure,
        "candidate_count": capsule.candidate_count,
        "selected_index": capsule.selected_index,
        "action_bounds_valid": result.action_bounds_valid,
        "stale_action_applied_count": result.stale_action_applied_count,
        "severe_collision_count": result.severe_collision_count,
        "invalid_force_count": result.invalid_force_count,
        "p40_conservation_maximum_difference": result.p40_conservation_maximum_difference,
        "safety_intervention_count": result.safety_intervention_count,
    }
    return terminal, record, blobs


def validate_terminal_ledger(plan, terminals) -> dict[str, object]:
    planned = [str(value["planned_episode_id"]) for value in plan["episodes"]]
    published = [str(value.get("planned_episode_id")) for value in terminals]
    missing = sorted(set(planned) - set(published))
    unplanned = sorted(set(published) - set(planned))
    duplicate_count = len(published) - len(set(published))
    replacement_count = sum(bool(value.get("replacement", True)) for value in terminals)
    passed = (
        len(published) == len(planned)
        and not missing and not unplanned
        and duplicate_count == 0 and replacement_count == 0
    )
    return {
        "planned_count": len(planned),
        "published_count": len(published),
        "missing": missing,
        "unplanned": unplanned,
        "duplicate_count": duplicate_count,
        "replacement_count": replacement_count,
        "passed": passed,
    }


def _trace_sha256(trace: Sequence[Sequence[float]]) -> str:
    array = np.ascontiguousarray(trace, dtype="<f8")
    return _sha256(array.tobytes())


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
