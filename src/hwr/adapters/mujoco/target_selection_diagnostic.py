"""MuJoCo bridge for the frozen R0001-P41-E2 interaction diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from hwr.adapters.mujoco.contact_ledger import resolve_allowed_contact_role_ids
from hwr.adapters.mujoco.entity_contact_graph import (
    EntityContactGraph,
    EntityMotionSource,
    p40_conservation_differences,
    resolve_robot_part_by_geom,
)
from hwr.adapters.mujoco.formal_household_backend import MujocoFormalHouseholdDualArmBackend
from hwr.core.embodied import DualArmAction, DualArmActionFrame, DualArmObservation
from hwr.eval.target_selection import (
    ACQUISITION_STEPS,
    PHASES,
    PLANNED_HORIZON,
    Candidate,
    CandidateSet,
    PolicyVisibleInput,
    corridor_obstacle_count,
    deserialize_policy_input,
    generate_candidate_set,
    phase_for_step,
    primitive_action,
    select_candidate_index,
    serialize_policy_input,
)
from hwr.eval.target_selection_safety import evaluate_safety_guards
DIAGNOSTIC_SCHEMA = "hwr.p41-target-selection-diagnostic/v1"
MAIN_EVENT = "same_entity_dual_arm_contact_associated_motion"
SOURCE = "R0001-P41-E2-fixed-primitive"
VALIDITY_NS = 100_000_000
ACQUISITION_PHASES = (
    ("A0_stable", 10), ("A1_panorama", 380), ("A2_forward", 220),
    ("A3_panorama", 380), ("A4_seal", 5),
)
INVALID_GRAPH_FIELDS = (
    "missing_normal_force_count", "nonfinite_normal_force_count",
    "invalid_negative_normal_force_count", "unknown_mapping_count", "invalid_motion_state_count",
)
@dataclass(frozen=True)
class BranchConfiguration:
    environment_seed: int
    policy_rng_seed: int
    forced_index: int | None = None
    supported_domain: bool = True


class MainEventTracker:
    """Evaluator-private same-entity bout tracker."""

    def __init__(self) -> None:
        self._bout_starts: dict[str, object] = {}
        self.event = False
        self.entities: set[str] = set()
        self.maximum_motion: dict[str, float] = {}

    def update(
        self,
        period: Mapping[str, object],
        start_state: Mapping[str, object],
        end_state: Mapping[str, object],
    ) -> None:
        if period["reset_settling_excluded"]:
            self._bout_starts.clear()
            return
        qualified: set[str] = set()
        for substep in period["substeps"]:
            qualified.update(substep["same_object_dual_arm_grasps"])
            qualified.update(
                entity
                for entity in substep["same_entity_dual_arm_contacts"]
                if str(entity).startswith("articulation:")
            )
        for entity in tuple(self._bout_starts):
            if entity not in qualified:
                del self._bout_starts[entity]
        for entity in qualified:
            self._bout_starts.setdefault(entity, start_state[entity])
            motion = _motion_delta(
                self._bout_starts[entity],
                end_state[entity],
            )
            self.maximum_motion[entity] = max(
                self.maximum_motion.get(entity, 0.0), motion
            )
            if motion >= 0.01:
                self.event = True
                self.entities.add(entity)


class TargetSelectionDiagnostic:
    """Runs one branch while keeping outcome measurement evaluator-private."""

    def __init__(self, task, binding) -> None:
        self.task = task
        self.binding = binding

    def sample_latencies(self, seed: int) -> tuple[int, int]:
        rng = random.Random(seed ^ 0x5A17C0DE)
        spec = self.task.evaluation_randomization
        for _ in self.task.objects:
            spec.mass_scale.sample(rng)
            spec.friction_scale.sample(rng)
        spec.light_scale.sample(rng)
        spec.material_tint.sample(rng)
        spec.focal_scale.sample(rng)
        spec.rgb_noise_std.sample(rng)
        spec.depth_dropout.sample(rng)
        spec.depth_noise_std_m.sample(rng)
        spec.actuator_scale.sample(rng)
        action = int(round(spec.action_latency_steps.sample(rng)))
        observation = int(round(spec.observation_latency_steps.sample(rng)))
        return observation, action

    def run_branch(self, configuration: BranchConfiguration) -> dict[str, object]:
        backend = self._backend()
        graph = _graph_from_backend(backend)
        original_substep = backend._after_physics_substep

        def observe_substep() -> None:
            original_substep()
            graph.sample_mujoco_substep(backend.model, backend.data)

        backend._after_physics_substep = observe_substep
        history: list[tuple[float, ...]] = []
        history_available: list[bool] = []
        trace: list[dict[str, object]] = []
        acquisition_tracker = MainEventTracker()
        branch_tracker = MainEventTracker()
        keyframes: list[bytes] = []
        failure: str | None = None
        candidate_set: CandidateSet | None = None
        selected_index = -1
        acquisition_pose: tuple[float, float, float] | None = None
        previous_identity: tuple[int, int] | None = None
        try:
            backend.contact_ledger.set_enabled(True)
            observation = backend.reset(
                seed=configuration.environment_seed,
                task_id=self.task.task_id,
            )
            graph.reset()
            acquisition_pose = observation.proprioception.base_pose
            state = _AcquisitionState(acquisition_pose)
            for step in range(ACQUISITION_STEPS):
                phase_index, phase, phase_step = _acquisition_phase(step)
                payload = policy_input_bytes(
                    observation,
                    history,
                    history_available,
                    configuration.policy_rng_seed,
                    phase_index=phase_index,
                    phase_step=phase_step,
                )
                failure = failure or _input_failure(
                    backend,
                    observation,
                    payload,
                    supported_only=configuration.supported_domain,
                    previous_identity=previous_identity,
                )
                previous_identity = (
                    observation.timestamp_ns, observation.sequence_id
                )
                action, capture = state.action(phase, payload)
                if failure is not None:
                    action, capture = _hold(deserialize_policy_input(payload)), False
                if capture:
                    keyframes.append(payload)
                observation, period, row = _step(
                    backend, graph, observation, action, step
                )
                acquisition_tracker.update(
                    period, row.pop("_motion_start"), row.pop("_motion_end")
                )
                if acquisition_tracker.event:
                    failure = failure or "main_event_during_acquisition"
                if row["safety_intervened"]:
                    failure = failure or "safety_intervention_during_acquisition"
                trace.append(row)
                _append_history(history, history_available, row["applied_action"])
                if row["terminal"]:
                    failure = failure or "runtime_terminal_during_acquisition"
                    break
            final_payload = policy_input_bytes(
                observation,
                history,
                history_available,
                configuration.policy_rng_seed,
                phase_index=4,
                phase_step=5,
            )
            if failure is None:
                candidate_set = generate_candidate_set(
                    keyframes,
                    acquisition_base_pose=acquisition_pose,
                    final_input=final_payload,
                )
                selected_index = (
                    configuration.forced_index
                    if configuration.forced_index is not None
                    else select_candidate_index(
                        candidate_set,
                        deserialize_policy_input(final_payload).base_pose,
                        acquisition_base_pose=acquisition_pose,
                    )
                )
                if not -1 <= selected_index < len(candidate_set.candidates):
                    failure = "selected_index_out_of_range"
                    selected_index = -1
            candidate = (
                None
                if candidate_set is None or selected_index < 0
                else candidate_set.candidates[selected_index]
            )
            safety_latched = failure is not None
            for post_step in range(PLANNED_HORIZON - ACQUISITION_STEPS):
                absolute_step = ACQUISITION_STEPS + post_step
                if trace[-1]["terminal"] if trace else False:
                    break
                phase_name, phase_step = phase_for_step(post_step)
                phase_index = 5 + next(
                    index
                    for index, (name, _) in enumerate(PHASES)
                    if name == phase_name
                )
                payload = policy_input_bytes(
                    observation,
                    history,
                    history_available,
                    configuration.policy_rng_seed,
                    phase_index=phase_index,
                    phase_step=phase_step,
                )
                action = (
                    _hold(deserialize_policy_input(payload))
                    if safety_latched
                    else primitive_action(
                        payload, candidate, acquisition_pose, post_step
                    )
                )
                observation, period, row = _step(
                    backend, graph, observation, action, absolute_step
                )
                branch_tracker.update(
                    period, row.pop("_motion_start"), row.pop("_motion_end")
                )
                trace.append(row)
                _append_history(history, history_available, row["applied_action"])
                safety_latched |= bool(row["safety_intervened"])
            graph_report = graph.report()
            ledger_report = backend.contact_ledger.report()
            audit = backend.task_audit()
            return _branch_report(
                task_id=self.task.task_id,
                configuration=configuration,
                audit=audit,
                failure=failure,
                acquisition_tracker=acquisition_tracker,
                branch_tracker=branch_tracker,
                keyframes=keyframes,
                candidate_set=candidate_set,
                selected_index=selected_index,
                candidate=candidate,
                trace=trace,
                graph_report=graph_report,
                ledger_report=ledger_report,
            )
        finally:
            backend.close()

    def _backend(self) -> MujocoFormalHouseholdDualArmBackend:
        return MujocoFormalHouseholdDualArmBackend(
            self.task,
            self.binding,
            camera_width=256,
            camera_height=192,
            evaluation_profile=True,
        )


class _AcquisitionState:
    def __init__(self, acquisition_pose: tuple[float, float, float]) -> None:
        self.acquisition_pose = acquisition_pose
        self.panorama_origin_yaw = acquisition_pose[2]
        self.previous_yaw = acquisition_pose[2]
        self.unwrapped_yaw = 0.0
        self.next_keyframe = 0
        self.forward_stopped = False
        self.obstacle_streak = 0

    def action(
        self, phase: str, payload: bytes
    ) -> tuple[tuple[float, ...], bool]:
        value = deserialize_policy_input(payload)
        hold = _hold(value)
        capture = False
        if phase in ("A1_panorama", "A3_panorama"):
            if value.phase_step == 0:
                self.panorama_origin_yaw = value.base_pose[2]
                self.previous_yaw = value.base_pose[2]
                self.unwrapped_yaw = 0.0
                self.next_keyframe = 0
            increment = math.atan2(
                math.sin(value.base_pose[2] - self.previous_yaw),
                math.cos(value.base_pose[2] - self.previous_yaw),
            )
            self.unwrapped_yaw += max(0.0, increment)
            self.previous_yaw = value.base_pose[2]
            threshold = self.next_keyframe * math.pi / 12.0
            capture = self.next_keyframe < 24 and self.unwrapped_yaw >= threshold
            self.next_keyframe += int(capture)
            if self.unwrapped_yaw < 2.0 * math.pi:
                return _base_action(0.0, 0.35), capture
        elif phase == "A2_forward":
            if corridor_obstacle_count(payload) >= 25:
                self.obstacle_streak += 1
            else:
                self.obstacle_streak = 0
            self.forward_stopped |= self.obstacle_streak >= 3
            delta_x = value.base_pose[0] - self.acquisition_pose[0]
            delta_y = value.base_pose[1] - self.acquisition_pose[1]
            distance = (
                delta_x * math.cos(self.acquisition_pose[2])
                + delta_y * math.sin(self.acquisition_pose[2])
            )
            if not self.forward_stopped and distance < 1.20:
                return _base_action(0.12, 0.0), False
        return hold, capture


def _branch_report(
    *,
    task_id,
    configuration,
    audit,
    failure,
    acquisition_tracker,
    branch_tracker,
    keyframes,
    candidate_set,
    selected_index,
    candidate,
    trace,
    graph_report,
    ledger_report,
):
    trace_sha256 = _canonical_sha256(trace)
    graph_sha256 = _canonical_sha256(graph_report)
    ledger_sha256 = _canonical_sha256(ledger_report)
    candidate_bytes = b"" if candidate_set is None else candidate_set.canonical_bytes
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "task_id": task_id,
        "environment_seed": configuration.environment_seed,
        "policy_rng_seed": configuration.policy_rng_seed,
        "sampled_observation_latency_steps": int(
            audit["randomization"]["observation_latency_steps"]
        ),
        "sampled_action_latency_steps": int(
            audit["randomization"]["action_latency_steps"]
        ),
        "acquisition_failed": failure is not None,
        "acquisition_failure_reason": failure,
        "acquisition_main_event": acquisition_tracker.event,
        "keyframe_count": len(keyframes),
        "candidate_count": 0 if candidate_set is None else len(candidate_set.candidates),
        "candidate_set_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes_hex": candidate_bytes.hex(),
        "selected_index": selected_index,
        "selected_candidate": _candidate_dict(candidate),
        "main_event_name": MAIN_EVENT,
        "main_event": branch_tracker.event,
        "main_event_entities": sorted(branch_tracker.entities),
        "maximum_bout_motion": dict(sorted(branch_tracker.maximum_motion.items())),
        "trace_sha256": trace_sha256,
        "trace_step_count": len(trace),
        "executed_step_count": sum(row.get("executed", False) for row in trace),
        "unexecuted_step_count": PLANNED_HORIZON - len(trace),
        "action_summary": _action_summary(trace),
        "safety_intervention_count": sum(
            bool(row.get("safety_intervened")) for row in trace
        ),
        "entity_contact_graph": _graph_summary(graph_report),
        "contact_ledger_sha256": ledger_sha256,
        "p40_conservation": p40_conservation_differences(
            graph_report, ledger_report
        ),
        "severe_collision_count": audit["severe_collision_count"],
        "maximum_forbidden_force": audit["maximum_forbidden_force"],
        "stale_action_applied_count": sum(
            bool(row.get("outside_validity_window"))
            and row.get("applied_action") != row.get("hold_action")
            for row in trace
            if row.get("executed", False)
        ),
        "invalid_force_count": sum(
            int(graph_report[name]) for name in INVALID_GRAPH_FIELDS
        ),
        "action_bounds_valid": all(
            row.get("action_bounds_valid", True) for row in trace
        ),
        "resolved": True,
        "_full_trace_sha256": trace_sha256,
        "_full_graph_sha256": graph_sha256,
        "_full_ledger_sha256": ledger_sha256,
        "_candidate_bytes": candidate_bytes,
        "_candidate_set": candidate_set,
    }


def policy_input_bytes(
    observation: DualArmObservation,
    history: list[tuple[float, ...]],
    available: list[bool],
    policy_rng_seed: int,
    *,
    phase_index: int,
    phase_step: int,
) -> bytes:
    frames = {frame.camera_id: frame for frame in observation.cameras}
    calibrations = {
        value.camera_id: value for value in observation.camera_calibrations
    }
    if set(("head_rgb", "head_depth")) - set(frames):
        raise ValueError("P41 requires head RGB-D frames")
    if "head_depth" not in calibrations:
        raise ValueError("P41 requires dynamic head-depth calibration")
    rgb = frames["head_rgb"]
    depth = frames["head_depth"]
    if rgb.payload is None or depth.payload is None:
        raise ValueError("P41 requires inline RGB-D payloads")
    rgb_array = np.frombuffer(rgb.payload, np.uint8).reshape(192, 256, 3).copy()
    depth_array = np.frombuffer(depth.payload, np.dtype("<f4")).reshape(192, 256).copy()
    depth_valid = np.isfinite(depth_array) & (depth_array >= 0.10) & (depth_array <= 5.00)
    padded_history = [(0.0,) * 16] * max(0, 4 - len(history)) + history[-4:]
    padded_available = [False] * max(0, 4 - len(available)) + available[-4:]
    return serialize_policy_input(
        PolicyVisibleInput(
            observation_timestamp_ns=observation.timestamp_ns,
            sequence_id=observation.sequence_id,
            phase_index=phase_index,
            phase_step=phase_step,
            policy_rng_seed=policy_rng_seed,
            safety_state=observation.safety_state.value,
            head_rgb_uint8=np.ascontiguousarray(rgb_array, dtype=np.uint8),
            head_depth_m=np.ascontiguousarray(depth_array, dtype=np.dtype("<f4")),
            head_depth_valid=np.ascontiguousarray(depth_valid),
            head_camera_intrinsics=np.asarray(
                calibrations["head_depth"].intrinsics, dtype=np.dtype("<f8")
            ),
            robot_from_head_camera=np.asarray(
                calibrations["head_depth"].robot_from_camera,
                dtype=np.dtype("<f8"),
            ).reshape(4, 4),
            proprioception=np.asarray(
                observation.proprioception.vector(), dtype=np.dtype("<f8")
            ),
            executed_action_history=np.asarray(
                padded_history, dtype=np.dtype("<f8")
            ),
            history_available=np.asarray(padded_available, dtype=np.bool_),
        )
    )


def _step(backend, graph, observation, vector, step):
    start = graph.capture_motion_state(backend.model, backend.data)
    graph.begin_control_period(start)
    action = DualArmAction.from_vector(vector)
    frame = DualArmActionFrame(
        observation.timestamp_ns,
        observation.timestamp_ns,
        observation.timestamp_ns + VALIDITY_NS,
        SOURCE,
        action,
    )
    outcome = backend.apply(frame)
    end = graph.capture_motion_state(backend.model, backend.data)
    period = graph.end_control_period(end)
    applied = tuple(outcome.info["applied_action"].action.vector())
    events = [
        {
            "timestamp_ns": event.timestamp_ns,
            "event_type": event.event_type,
            "source": event.source,
            "details": dict(event.details),
        }
        for event in outcome.events
    ]
    hold = (
        0.0, 0.0, *(0.0,) * 12,
        observation.proprioception.left_gripper_position,
        observation.proprioception.right_gripper_position,
    )
    row = {
        "step": step,
        "executed": True,
        "proposed_action": list(vector),
        "applied_action": list(applied),
        "hold_action": list(hold),
        "proprioception": list(outcome.observation.proprioception.vector()),
        "events": events,
        "safety_intervened": outcome.info["safety_intervened"],
        "outside_validity_window": any(
            event["details"].get("reason") == "outside_validity_window"
            for event in events
        ),
        "action_bounds_valid": _action_bounds_valid(vector, applied),
        "terminated": outcome.terminated,
        "truncated": outcome.truncated,
        "terminal": outcome.terminated or outcome.truncated,
        "_motion_start": start,
        "_motion_end": end,
    }
    return outcome.observation, period, row


def _graph_from_backend(backend) -> EntityContactGraph:
    model, ids, binding = backend.model, backend.household_ids, backend.binding
    robot_geoms = frozenset(int(value) for value in ids.robot_geoms)
    robot_parts, roots = resolve_robot_part_by_geom(model, robot_geoms)
    _, role_by_geom = resolve_allowed_contact_role_ids(
        model,
        binding.allowed_robot_contact_geoms,
        binding.allowed_robot_contact_roles,
    )
    object_by_geom = {
        int(model.geom(value.collision_geom).id): object_id
        for object_id, value in binding.objects.items()
    }
    entity_by_geom = {}
    for geom in range(model.ngeom):
        if geom in robot_geoms:
            continue
        role = role_by_geom.get(geom, "forbidden")
        if role == "manipulated_object":
            identifier = object_by_geom[geom]
        elif role == "articulation":
            identifier = binding.articulation.articulation_id
        else:
            identifier = model.geom(geom).name or f"geom_{geom}"
        entity_by_geom[geom] = f"{role}:{identifier}"
    motion_sources = {
        f"manipulated_object:{object_id}": EntityMotionSource("translation", geom)
        for geom, object_id in object_by_geom.items()
    }
    if binding.articulation:
        motion_sources[
            f"articulation:{binding.articulation.articulation_id}"
        ] = EntityMotionSource(
            "joint", int(model.joint(binding.articulation.joint).id)
        )
    pads = {
        part: (
            (int(model.geom(f"{side}_gripper_left_pad").id),),
            (int(model.geom(f"{side}_gripper_right_pad").id),),
        )
        for part, side in (("left_arm", "left"), ("right_arm", "right"))
    }
    return EntityContactGraph(
        all_geom_ids=range(model.ngeom),
        robot_part_by_geom=robot_parts,
        entity_by_geom=entity_by_geom,
        timestep=float(model.opt.timestep),
        enabled=True,
        excluded_initial_periods=1,
        motion_source_by_entity=motion_sources,
        gripper_pad_groups=pads,
        geom_name_by_id={
            geom: model.geom(geom).name or f"geom_{geom}"
            for geom in range(model.ngeom)
        },
        robot_body_roots=roots,
    )


def _input_failure(
    backend, observation, payload, *, supported_only, previous_identity=None
):
    try:
        value = deserialize_policy_input(payload)
    except ValueError:
        return "invalid_policy_visible_input"
    age = backend._timestamp_ns() - observation.timestamp_ns
    if supported_only and age > VALIDITY_NS:
        return "supported_source_age_exceeded"
    identity = (observation.timestamp_ns, observation.sequence_id)
    if previous_identity is not None and (
        identity[0] < previous_identity[0]
        or identity[1] < previous_identity[1]
        or ((identity[0] == previous_identity[0]) != (
            identity[1] == previous_identity[1]
        ))
    ):
        return "nonmonotonic_observation"
    if value.safety_state != "ok":
        return "safety_state_not_ok"
    return None


def _graph_summary(report):
    return {
        key: value for key, value in report.items()
        if key not in ("periods", "substeps", "mapping")
    } | {"mapping_sha256": _canonical_sha256(report["mapping"])}


def _action_summary(trace):
    proposed = np.asarray(
        [row["proposed_action"] for row in trace if row.get("executed")],
        dtype=np.float64,
    )
    executed = np.asarray(
        [row["applied_action"] for row in trace if row.get("executed")],
        dtype=np.float64,
    )
    if not len(executed):
        zeros = [0.0] * 16
        active = [False] * 16
    else:
        zeros = np.sqrt(np.mean(np.square(executed), axis=0)).tolist()
        active = np.any(np.abs(executed) > 1.0e-9, axis=0).tolist()
    return {
        "proposed_action_sha256": _canonical_sha256(proposed.tolist()),
        "executed_action_sha256": _canonical_sha256(executed.tolist()),
        "executed_rms_per_dimension": zeros,
        "executed_rms": float(np.sqrt(np.mean(np.square(executed))))
        if len(executed) else 0.0,
        "active_dimension_count": sum(active),
        "active_dimension_fraction": sum(active) / 16.0,
        "acquisition_completed": len(trace) >= ACQUISITION_STEPS,
        "post_selection_executed_steps": max(0, len(trace) - ACQUISITION_STEPS),
        "route_completion_fraction": len(trace) / PLANNED_HORIZON,
    }


def _acquisition_phase(step: int) -> tuple[int, str, int]:
    offset = 0
    for phase_index, (name, length) in enumerate(ACQUISITION_PHASES):
        if step < offset + length:
            return phase_index, name, step - offset
        offset += length
    raise ValueError("acquisition step outside frozen horizon")


def _append_history(history, available, vector) -> None:
    history.append(tuple(float(item) for item in vector))
    available.append(True)
    del history[:-4]
    del available[:-4]


def _hold(value: PolicyVisibleInput) -> tuple[float, ...]:
    return (
        0.0,
        0.0,
        *(0.0,) * 12,
        float(value.proprioception[24]),
        float(value.proprioception[25]),
    )


def _base_action(linear: float, angular: float) -> tuple[float, ...]:
    return (linear, angular, *(0.0,) * 14)


def _action_bounds_valid(proposed, applied) -> bool:
    lower = np.asarray((-0.18, -0.50, *(-0.35,) * 12, 0.0, 0.0))
    upper = np.asarray((0.18, 0.50, *(0.35,) * 12, 1.0, 1.0))
    return all(
        len(value) == 16
        and np.isfinite(value).all()
        and np.all(value >= lower)
        and np.all(value <= upper)
        for value in (np.asarray(proposed), np.asarray(applied))
    )


def _candidate_dict(candidate: Candidate | None) -> dict[str, object] | None:
    if candidate is None:
        return None
    return asdict(candidate) | {"canonical_key": list(candidate.canonical_key())}


def _motion_delta(start: object, end: object) -> float:
    if isinstance(start, tuple):
        return float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    return abs(float(end) - float(start))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
