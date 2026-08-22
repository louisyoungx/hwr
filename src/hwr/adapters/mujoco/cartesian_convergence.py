"""MuJoCo execution bridge for frozen R0001-P51-E1 convergence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from hwr.adapters.mujoco.cartesian_convergence_provenance import (
    bank_prefix_record,
    continuation_identity,
    raw_runtime_step_evidence,
)
from hwr.adapters.mujoco.entity_contact_graph import p40_conservation_differences
from hwr.adapters.mujoco.target_selection_diagnostic import (
    INVALID_GRAPH_FIELDS,
    MainEventTracker,
    TargetSelectionDiagnostic,
    _AcquisitionState,
    _acquisition_phase,
    _append_history,
    _graph_from_backend,
    _input_failure,
    _step,
    policy_input_bytes,
)
from hwr.core.embodied import DualArmObservation
from hwr.eval import target_selection
from hwr.eval.cartesian_convergence import (
    B2_STEPS,
    ORDINARY_TERMINAL_REASONS,
    ROLES,
    CartesianConvergenceContractError,
    action_summary,
    arm_outcome,
    attach_pair_invariants,
    canonical_sha256,
    carry_forward_records,
    first_treatment_guard,
    identity,
    legacy_transform,
    preposition_targets,
    signed_derivatives,
    treatment_guard_passes,
    wrap_angle,
)
from hwr.eval.target_selection import (
    ACQUISITION_STEPS,
    Candidate,
    CandidateSet,
    deserialize_policy_input,
    generate_candidate_set,
    select_candidate_index,
)


PREFIX_STEPS, B2_PHASE_INDEX = ACQUISITION_STEPS + 400, 7
@dataclass
class _PrefixRun:
    backend: object
    graph: object
    observation: DualArmObservation
    history: list[tuple[float, ...]]
    history_available: list[bool]
    environment_seed: int
    policy_rng_seed: int
    acquisition_pose: tuple[float, float, float]
    acquisition_world_origin: tuple[float, float, float]
    candidate_set: CandidateSet | None
    selected_index: int
    trace: list[dict[str, object]]
    acquisition_input_hashes: list[str]
    failure: str | None
    input_failure: str | None
    acquisition_main_event: bool
    continuation_identity: dict[str, object]
    first_actions: dict[str, tuple[float, ...]]
    first_guard: dict[str, object]
    preposition_targets: dict[str, tuple[float, float, float]]
    primitive_target_crosscheck: dict[str, object]

    @property
    def candidate(self) -> Candidate | None:
        if self.candidate_set is None or self.selected_index < 0:
            return None
        return self.candidate_set.candidates[self.selected_index]

    def close(self) -> None:
        self.backend.close()
class CartesianConvergenceMujoco:
    """Build treatment-free prefixes and execute B2 treatment arms."""

    def __init__(self, task, binding) -> None:
        self.task = task
        self.binding = binding
        self._diagnostic = TargetSelectionDiagnostic(task, binding)

    def sample_latencies(self, environment_seed: int) -> tuple[int, int]:
        return self._diagnostic.sample_latencies(environment_seed)
    def inspect_prefix(
        self, environment_seed: int, policy_rng_seed: int
    ) -> dict[str, object]:
        run = self._run_prefix(environment_seed, policy_rng_seed)
        try:
            return bank_prefix_record(run)
        finally:
            run.close()

    def evaluate_pair(self, pair: Mapping[str, object]) -> dict[str, object]:
        arms: dict[str, object] = {}
        identities: dict[str, object] = {}
        expected_identity = pair["continuation_identity"]
        expected_candidate = str(pair["candidate_set_sha256"])
        expected_trace = str(pair["prefix_trace_sha256"])
        expected_index = int(pair["selected_index"])
        prefix_valid = True
        hard_stop = False
        for role_value in pair["role_order"]:
            role = str(role_value)
            run = self._run_prefix(
                int(pair["environment_seed"]), int(pair["policy_rng_seed"])
            )
            try:
                prefix_valid &= (
                    run.continuation_identity == expected_identity
                    and run.candidate_set is not None
                    and run.candidate_set.candidate_set_sha256
                    == expected_candidate
                    and run.selected_index == expected_index
                    and canonical_sha256(run.trace) == expected_trace
                    and run.first_guard == pair["first_treatment_guard"]
                    and {
                        name: list(value)
                        for name, value in run.preposition_targets.items()
                    }
                    == pair["preposition_targets"]
                    and run.primitive_target_crosscheck
                    == pair["primitive_target_crosscheck"]
                    and {
                        value: list(action)
                        for value, action in run.first_actions.items()
                    }
                    == pair["first_treatment_actions"]
                )
                if not prefix_valid:
                    raise CartesianConvergenceContractError(
                        "committed continuation replay differs"
                    )
                identities[role] = run.continuation_identity
                arm = self._run_b2(run, role)
                arms[role] = arm
                if not arm["hard_guard_passed"]:
                    hard_stop = True
                    break
            finally:
                run.close()
        complete = set(arms) == set(ROLES)
        record = {
            "pair_id": pair["pair_id"],
            "planned_episode_id": pair["planned_episode_id"],
            "task_id": pair["task_id"],
            "cell_id": pair["cell_id"],
            "replicate_ordinal": pair["replicate_ordinal"],
            "observation_latency_steps": pair[
                "observation_latency_steps"
            ],
            "action_latency_steps": pair["action_latency_steps"],
            "environment_seed": pair["environment_seed"],
            "policy_rng_seed": pair["policy_rng_seed"],
            "role_order": list(pair["role_order"]),
            "candidate_set_sha256": expected_candidate,
            "selected_index": expected_index,
            "continuation_identity": expected_identity,
            "continuation_replay_identities": identities,
            "continuation_identity_equal": bool(identities)
            and all(value == expected_identity for value in identities.values()),
            "prefix_trace_sha256": expected_trace,
            "first_treatment_guard": pair["first_treatment_guard"],
            "preposition_targets": pair["preposition_targets"],
            "preposition_target_identity": pair["preposition_target_identity"],
            "preposition_target_identities": pair[
                "preposition_target_identities"
            ],
            "bank_pair_sha256": canonical_sha256(pair),
            "pair_identity_valid": prefix_valid,
            "resolved": (complete or hard_stop) and prefix_valid,
            "hard_safety_stop": hard_stop,
            "arms": arms,
        }
        attach_pair_invariants(record, complete=complete)
        record["delta_i"] = (
            float(arms["frame_legacy"]["normalized_auc"])
            - float(arms["frame_fixed"]["normalized_auc"])
            if complete
            else None
        )
        return record
    def _run_prefix(
        self, environment_seed: int, policy_rng_seed: int
    ) -> _PrefixRun:
        backend = self._diagnostic._backend()
        graph = _graph_from_backend(backend)
        original_substep = backend._after_physics_substep

        def observe_substep() -> None:
            original_substep()
            graph.sample_mujoco_substep(backend.model, backend.data)

        backend._after_physics_substep = observe_substep
        history: list[tuple[float, ...]] = []
        available: list[bool] = []
        trace: list[dict[str, object]] = []
        keyframes: list[bytes] = []
        input_hashes: list[str] = []
        failure: str | None = None
        candidate_set: CandidateSet | None = None
        selected_index = -1
        input_failure: str | None = None
        try:
            backend.contact_ledger.set_enabled(True)
            observation = backend.reset(
                seed=environment_seed, task_id=self.task.task_id
            )
            graph.reset()
            acquisition_pose = observation.proprioception.base_pose
            acquisition_world_origin = tuple(
                float(value)
                for value in backend.data.xpos[backend.bundle.ids.base_body]
            )
            state = _AcquisitionState(acquisition_pose)
            acquisition_tracker = MainEventTracker()
            previous_identity: tuple[int, int] | None = None
            for step in range(ACQUISITION_STEPS):
                phase_index, phase, phase_step = _acquisition_phase(step)
                payload = policy_input_bytes(
                    observation,
                    history,
                    available,
                    policy_rng_seed,
                    phase_index=phase_index,
                    phase_step=phase_step,
                )
                input_hashes.append(hashlib.sha256(payload).hexdigest())
                current_input_failure = _input_failure(
                    backend,
                    observation,
                    payload,
                    supported_only=True,
                    previous_identity=previous_identity,
                )
                input_failure = input_failure or current_input_failure
                failure = failure or current_input_failure
                previous_identity = (
                    observation.timestamp_ns,
                    observation.sequence_id,
                )
                action, capture = state.action(phase, payload)
                if failure is not None:
                    break
                if capture:
                    keyframes.append(payload)
                observation, row, period = _advance(
                    backend, graph, observation, action, step
                )
                acquisition_tracker.update(
                    period,
                    row.pop("_motion_start"),
                    row.pop("_motion_end"),
                )
                trace.append(row)
                _append_history(history, available, row["applied_action"])
                if acquisition_tracker.event:
                    failure = failure or "main_event_during_acquisition"
                failure = failure or _prefix_step_failure(row, backend)
                if failure is not None:
                    break
            final_payload = policy_input_bytes(
                observation,
                history,
                available,
                policy_rng_seed,
                phase_index=4,
                phase_step=5,
            )
            input_hashes.append(hashlib.sha256(final_payload).hexdigest())
            if failure is None:
                candidate_set = generate_candidate_set(
                    keyframes,
                    acquisition_base_pose=acquisition_pose,
                    final_input=final_payload,
                )
                selected_index = select_candidate_index(
                    candidate_set,
                    deserialize_policy_input(final_payload).base_pose,
                    acquisition_base_pose=acquisition_pose,
                )
                if not candidate_set.candidates:
                    failure = "candidate_set_empty"
                elif not 0 <= selected_index < len(
                    candidate_set.candidates
                ):
                    failure = "selected_index_out_of_range"
            observation, failure = _run_b0_b1(
                backend,
                graph,
                observation,
                history,
                available,
                candidate_set,
                selected_index,
                acquisition_pose,
                policy_rng_seed,
                trace,
                failure,
            )
            failure = failure or _final_prefix_failure(backend, graph)
            first_actions, first_guard, targets, target_check = _first_actions(
                observation,
                history,
                available,
                policy_rng_seed,
                candidate_set,
                selected_index,
                acquisition_pose,
            )
            relative_yaw = wrap_angle(
                backend._base_state()[0][2] - acquisition_pose[2]
            )
            if failure is None and abs(relative_yaw) < math.pi / 6.0:
                failure = "relative_yaw_below_pi_over_6"
            if failure is None and not target_check["passed"]:
                failure = "primitive_target_crosscheck_failed"
            if failure is None and not treatment_guard_passes(first_guard):
                failure = "first_treatment_action_ineligible"
            identity = continuation_identity(
                backend, observation, history, available, graph
            )
            return _PrefixRun(
                backend,
                graph,
                observation,
                history,
                available,
                environment_seed,
                policy_rng_seed,
                acquisition_pose,
                acquisition_world_origin,
                candidate_set,
                selected_index,
                trace,
                input_hashes,
                failure,
                input_failure,
                acquisition_tracker.event,
                identity,
                first_actions,
                first_guard,
                targets,
                target_check,
            )
        except BaseException:
            backend.close()
            raise
    def _run_b2(
        self, run: _PrefixRun, role: str
    ) -> dict[str, object]:
        if run.failure is not None or run.candidate is None:
            raise CartesianConvergenceContractError(
                "ineligible prefix entered B2"
            )
        first_payload = policy_input_bytes(
            run.observation,
            run.history,
            run.history_available,
            run.policy_rng_seed,
            phase_index=B2_PHASE_INDEX,
            phase_step=0,
        )
        current_pose = deserialize_policy_input(first_payload).base_pose
        targets = preposition_targets(
            run.candidate, run.acquisition_pose, current_pose
        )
        if targets != run.preposition_targets:
            raise CartesianConvergenceContractError(
                "B2 preposition target replay differs"
            )
        distances = [_distance_record(run, targets)]
        proposed: list[list[float]] = []
        applied: list[list[float]] = []
        runtime_trace: list[dict[str, object]] = []
        hard_failure: str | None = None
        for step in range(B2_STEPS):
            payload = policy_input_bytes(
                run.observation,
                run.history,
                run.history_available,
                run.policy_rng_seed,
                phase_index=B2_PHASE_INDEX,
                phase_step=step,
            )
            action = _primitive_action(
                payload,
                run.candidate,
                run.acquisition_pose,
                400 + step,
                role,
            )
            run.observation, row, _ = _advance(
                run.backend,
                run.graph,
                run.observation,
                action,
                PREFIX_STEPS + step,
            )
            if step == 0 and tuple(row["proposed_action"]) != run.first_actions[role]:
                raise CartesianConvergenceContractError(
                    "first treatment action replay differs"
                )
            row.pop("_motion_start")
            row.pop("_motion_end")
            run.trace.append(row)
            _append_history(
                run.history, run.history_available, row["applied_action"]
            )
            proposed.append(row["proposed_action"])
            applied.append(row["applied_action"])
            distances.append(_distance_record(run, targets))
            hard_failure = _b2_hard_failure(row, run.backend, run.graph)
            runtime_trace.append(
                raw_runtime_step_evidence(
                    step + 1,
                    row,
                    run.backend,
                    run.graph,
                    distances[-1],
                    hard_failure,
                )
            )
            if hard_failure is not None:
                break
            if row["terminal"]:
                break
        carried = carry_forward_records(distances)
        outcome = arm_outcome([value["mean_m"] for value in distances])
        graph_report = run.graph.report()
        ledger_report = run.backend.contact_ledger.report()
        conservation = p40_conservation_differences(
            graph_report, ledger_report
        )
        audit = run.backend.task_audit()
        invariants = {
            "safety": {
                "limits": asdict(run.backend.safety.limits),
                "source": "hwr.safety.dual_arm.DualArmSafetySupervisor",
            },
            "cap": {"cartesian_velocity_max_mps": 0.08},
            "gripper": {
                "primitive_b2_target": 0.0,
            },
            "phase": {
                "name": "B2_preposition",
                "post_selection_start": 400,
                "maximum_control_steps": B2_STEPS,
            },
            "target": {
                name: list(value) for name, value in targets.items()
            },
            "fk": {
                "left_site_id": int(run.backend._left_tool_site),
                "right_site_id": int(run.backend._right_tool_site),
            },
            "backend": {
                "class": type(run.backend).__qualname__,
                "model_nq": int(run.backend.model.nq),
                "model_nv": int(run.backend.model.nv),
                "model_nu": int(run.backend.model.nu),
            },
        }
        return {
            **outcome,
            "role": role,
            "b2_control_step_limit": B2_STEPS,
            "executed_b2_steps": len(proposed),
            "ordinary_runtime_terminal": (
                None
                if (
                    not runtime_trace
                    or runtime_trace[-1]["episode_result"] is None
                    or runtime_trace[-1]["episode_result"]["reason"]
                    not in ORDINARY_TERMINAL_REASONS
                )
                else runtime_trace[-1]["episode_result"]["reason"]
            ),
            "raw_runtime_step_trace": runtime_trace,
            "raw_runtime_step_trace_sha256": canonical_sha256(runtime_trace),
            "carried_forward_step_count": B2_STEPS + 1 - len(distances),
            "distance_metric": (
                "mean(left_tool_to_left_preposition,"
                "right_tool_to_right_preposition)"
            ),
            "tool_distances": carried,
            "preposition_targets": {
                name: list(value) for name, value in targets.items()
            },
            "preposition_target_identity": identity(
                {name: list(value) for name, value in targets.items()}
            ),
            "preposition_target_identities": {
                name: identity(list(value)) for name, value in targets.items()
            },
            "first_treatment_action": list(run.first_actions[role]),
            "first_treatment_guard": run.first_guard,
            "proposed_actions": proposed,
            "applied_actions": applied,
            "proposed_action_sha256": canonical_sha256(proposed),
            "applied_action_sha256": canonical_sha256(applied),
            "action_summary": action_summary(proposed, applied),
            "first_10_applied_nonzero_arm_signed_derivatives": (
                signed_derivatives(carried, applied)
            ),
            "action_bounds_valid": all(
                row["action_bounds_valid"] for row in runtime_trace
            ),
            "safety_intervention_count": sum(
                bool(row["safety_intervened"]) for row in runtime_trace
            ),
            "stale_action_applied_count": sum(
                bool(row["outside_validity_window"])
                and row["applied_action"] != row["hold_action"]
                for row in runtime_trace
            ),
            "severe_collision_count": int(
                audit["severe_collision_count"]
            ),
            "invalid_force_count": sum(
                int(graph_report[name]) for name in INVALID_GRAPH_FIELDS
            ),
            "p40_conservation_maximum_absolute_difference": float(
                conservation["maximum_absolute_difference"]
            ),
            "nonfinite_runtime_value_count": sum(
                int(row["nonfinite_value_count"]) for row in runtime_trace
            ),
            "hard_failure_reason": hard_failure,
            "hard_guard_passed": hard_failure is None,
            "invariant_identities": {
                name: identity(value)
                for name, value in invariants.items()
            },
        }
def _run_b0_b1(
    backend,
    graph,
    observation,
    history,
    available,
    candidate_set,
    selected_index,
    acquisition_pose,
    policy_rng_seed,
    trace,
    failure,
):
    for post_step in range(400):
        if failure is not None:
            break
        phase_index = 5 + int(post_step >= 100)
        phase_step = post_step if post_step < 100 else post_step - 100
        payload = policy_input_bytes(
            observation,
            history,
            available,
            policy_rng_seed,
            phase_index=phase_index,
            phase_step=phase_step,
        )
        action = target_selection.primitive_action(
            payload,
            candidate_set.candidates[selected_index],
            acquisition_pose,
            post_step,
        )
        observation, row, _ = _advance(
            backend,
            graph,
            observation,
            action,
            ACQUISITION_STEPS + post_step,
        )
        row.pop("_motion_start")
        row.pop("_motion_end")
        trace.append(row)
        _append_history(history, available, row["applied_action"])
        failure = failure or _prefix_step_failure(row, backend)
    return observation, failure
def _first_actions(
    observation,
    history,
    available,
    policy_rng_seed,
    candidate_set,
    selected_index,
    acquisition_pose,
):
    empty_guard = {
        "finite": False,
        "different_bytes": False,
        "differing_indices": [],
        "only_arm_linear_xy_differs": False,
        "arm_action_noncollapsed": False,
    }
    if candidate_set is None or not 0 <= selected_index < len(
        candidate_set.candidates
    ):
        return {}, empty_guard, {}, {"passed": False}
    payload = policy_input_bytes(
        observation,
        history,
        available,
        policy_rng_seed,
        phase_index=B2_PHASE_INDEX,
        phase_step=0,
    )
    candidate = candidate_set.candidates[selected_index]
    value = deserialize_policy_input(payload)
    targets = preposition_targets(
        candidate, acquisition_pose, value.base_pose
    )
    fixed, target_check = _fixed_action_with_target_crosscheck(
        payload, candidate, acquisition_pose, targets
    )
    actions = {
        "frame_legacy": _primitive_action(
            payload, candidate, acquisition_pose, 400, "frame_legacy"
        ),
        "frame_fixed": fixed,
    }
    guard = first_treatment_guard(
        actions["frame_legacy"], actions["frame_fixed"]
    )
    return actions, guard, targets, target_check


def _fixed_action_with_target_crosscheck(
    payload, candidate, acquisition_pose, targets
):
    calls = []
    original = target_selection.acquisition_error_to_base_velocity

    def traced(error, maximum, **yaws):
        calls.append(
            {
                "error": list(np.asarray(error, np.float64)),
                "velocity_max": float(maximum),
                "yaws": {name: float(value) for name, value in yaws.items()},
            }
        )
        return original(error, maximum, **yaws)

    target_selection.acquisition_error_to_base_velocity = traced
    try:
        action = target_selection.primitive_action(
            payload, candidate, acquisition_pose, 400
        )
    finally:
        target_selection.acquisition_error_to_base_velocity = original
    value = deserialize_policy_input(payload)
    tools = target_selection.tool_positions_in_acquisition(
        value, acquisition_pose
    )
    reconstructed = {
        arm: tuple(tools[index] + np.asarray(calls[index]["error"]))
        for index, arm in enumerate(("left", "right"))
    }
    passed = (
        len(calls) == 2
        and all(call["velocity_max"] == 0.08 for call in calls)
        and all(
            np.allclose(reconstructed[arm], targets[arm], atol=1.0e-12, rtol=0.0)
            for arm in ("left", "right")
        )
    )
    return action, {
        "actual_error_calls": calls,
        "reconstructed_targets": {
            arm: list(value) for arm, value in reconstructed.items()
        },
        "passed": passed,
    }
def _primitive_action(
    payload: bytes,
    candidate: Candidate,
    acquisition_pose: Sequence[float],
    post_step: int,
    role: str,
) -> tuple[float, ...]:
    if role == "frame_fixed":
        return target_selection.primitive_action(
            payload, candidate, acquisition_pose, post_step
        )
    if role != "frame_legacy":
        raise CartesianConvergenceContractError(
            f"unknown treatment role: {role}"
        )
    original = target_selection.acquisition_error_to_base_velocity
    target_selection.acquisition_error_to_base_velocity = legacy_transform
    try:
        return target_selection.primitive_action(
            payload, candidate, acquisition_pose, post_step
        )
    finally:
        target_selection.acquisition_error_to_base_velocity = original
def _advance(backend, graph, observation, action, step):
    next_observation, period, row = _step(
        backend, graph, observation, action, step
    )
    return next_observation, row, period
def _prefix_step_failure(row, backend) -> str | None:
    if not row["action_bounds_valid"]:
        return "action_bounds_violation"
    if (
        row["outside_validity_window"]
        and row["applied_action"] != row["hold_action"]
    ):
        return "stale_action_applied"
    if int(backend.task_audit()["severe_collision_count"]):
        return "severe_collision"
    if row["safety_intervened"]:
        return "safety_intervention_during_prefix"
    if row["terminal"]:
        return "runtime_terminal_during_prefix"
    return None
def _final_prefix_failure(backend, graph) -> str | None:
    graph_report = graph.report()
    if any(int(graph_report[name]) for name in INVALID_GRAPH_FIELDS):
        return "invalid_force"
    conservation = p40_conservation_differences(
        graph_report, backend.contact_ledger.report()
    )
    if float(conservation["maximum_absolute_difference"]) != 0.0:
        return "p40_conservation_violation"
    return None
def _b2_hard_failure(row, backend, graph) -> str | None:
    if not row["action_bounds_valid"]:
        return "action_bounds_violation"
    if (
        row["outside_validity_window"]
        and row["applied_action"] != row["hold_action"]
    ):
        return "stale_action_applied"
    if int(backend.task_audit()["severe_collision_count"]):
        return "severe_collision"
    if row["safety_intervened"]:
        return "safety_intervention"
    return _final_prefix_failure(backend, graph)
def _distance_record(
    run: _PrefixRun,
    targets: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    origin = np.asarray(run.acquisition_world_origin, np.float64)
    yaw = run.acquisition_pose[2]
    rotation = np.asarray(
        (
            (math.cos(yaw), math.sin(yaw), 0.0),
            (-math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        ),
        np.float64,
    )
    positions = {
        "left": rotation
        @ (
            run.backend.data.site_xpos[run.backend._left_tool_site]
            - origin
        ),
        "right": rotation
        @ (
            run.backend.data.site_xpos[run.backend._right_tool_site]
            - origin
        ),
    }
    values = {
        arm: float(
            np.linalg.norm(
                positions[arm] - np.asarray(targets[arm], np.float64)
            )
        )
        for arm in ("left", "right")
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise CartesianConvergenceContractError(
            "nonfinite evaluator-private distance"
        )
    return {"left_m": values["left"], "right_m": values["right"],
            "mean_m": float(np.mean(list(values.values())))}
