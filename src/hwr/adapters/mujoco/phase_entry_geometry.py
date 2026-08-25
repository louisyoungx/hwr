"""MuJoCo prefix bridge for frozen R0001-P60 geometry evidence."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np

from hwr.adapters.mujoco.cartesian_convergence import (
    _advance,
    _final_prefix_failure,
    _prefix_step_failure,
)
from hwr.adapters.mujoco.entity_contact_graph import p40_conservation_differences
from hwr.adapters.mujoco.target_selection_diagnostic import (
    INVALID_GRAPH_FIELDS,
    TargetSelectionDiagnostic,
    _AcquisitionState,
    _acquisition_phase,
    _append_history,
    _graph_from_backend,
    _input_failure,
    policy_input_bytes,
)
from hwr.eval import target_selection
from hwr.eval.phase_entry_geometry import (
    ACQUISITION_STEPS,
    B0_STEPS,
    B1_STEPS,
    FLOAT_TOLERANCE_M,
    PREFIX_STEPS,
    PhaseEntryGeometryContractError,
    canonical_sha256,
    measure_phase_entry_geometry,
)
from hwr.eval.target_selection import (
    CandidateSet,
    deserialize_policy_input,
    generate_candidate_set,
    select_candidate_index,
    tool_positions_in_acquisition,
)


B2_PHASE_INDEX = 7


class PhaseEntryGeometryMujoco:
    """Run acquisition plus B0/B1 and stop before generating a B2 action."""

    def __init__(self, task, binding) -> None:
        self.task = task
        self.binding = binding
        self._diagnostic = TargetSelectionDiagnostic(task, binding)

    def sample_latencies(self, environment_seed: int) -> tuple[int, int]:
        return self._diagnostic.sample_latencies(environment_seed)

    def inspect_prefix(
        self,
        environment_seed: int,
        policy_rng_seed: int,
    ) -> dict[str, object]:
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
        input_hashes: list[str] = []
        keyframes: list[bytes] = []
        candidate_set: CandidateSet | None = None
        selected_index = -1
        failure: str | None = None
        input_failure: str | None = None
        hard_safety_failure = False
        previous_identity: tuple[int, int] | None = None
        try:
            backend.contact_ledger.set_enabled(True)
            observation = backend.reset(
                seed=environment_seed,
                task_id=self.task.task_id,
            )
            graph.reset()
            acquisition_pose = tuple(observation.proprioception.base_pose)
            acquisition_state = _AcquisitionState(acquisition_pose)
            final_payload_hash = None
            entry_payload_hash = None
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
                current_failure = _input_failure(
                    backend,
                    observation,
                    payload,
                    supported_only=True,
                    previous_identity=previous_identity,
                )
                input_failure = input_failure or current_failure
                failure = failure or current_failure
                previous_identity = (
                    observation.timestamp_ns,
                    observation.sequence_id,
                )
                if failure is not None:
                    break
                action, capture = acquisition_state.action(phase, payload)
                if capture:
                    keyframes.append(payload)
                observation, row, _ = _advance(
                    backend,
                    graph,
                    observation,
                    action,
                    step,
                )
                row.pop("_motion_start")
                row.pop("_motion_end")
                trace.append(row)
                _append_history(history, available, row["applied_action"])
                step_failure = _prefix_step_failure(row, backend)
                hard_safety_failure |= _is_hard_safety(step_failure)
                failure = failure or step_failure
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
            final_payload_hash = hashlib.sha256(final_payload).hexdigest()
            final_input_failure = _input_failure(
                backend,
                observation,
                final_payload,
                supported_only=True,
                previous_identity=previous_identity,
            )
            input_failure = input_failure or final_input_failure
            failure = failure or final_input_failure
            previous_identity = (
                observation.timestamp_ns,
                observation.sequence_id,
            )
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
                elif not 0 <= selected_index < len(candidate_set.candidates):
                    failure = "selected_index_out_of_range"
            if failure is None:
                (
                    observation,
                    failure,
                    input_failure,
                    hard_safety_failure,
                    previous_identity,
                ) = _run_b0_b1(
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
                    input_hashes,
                    previous_identity,
                )
            final_failure = _final_prefix_failure(backend, graph)
            hard_safety_failure |= _is_hard_safety(final_failure)
            failure = failure or final_failure
            (
                geometry,
                fk_error,
                entry_payload_hash,
                entry_failure,
            ) = _entry_geometry(
                backend,
                observation,
                history,
                available,
                policy_rng_seed,
                previous_identity,
                candidate_set,
                selected_index,
                acquisition_pose,
                enabled=failure is None,
            )
            input_failure = input_failure or entry_failure
            failure = failure or entry_failure
            if geometry is not None:
                if not geometry["target_formula_crosscheck"]["passed"]:
                    failure = "primitive_target_crosscheck_failed"
                elif fk_error > FLOAT_TOLERANCE_M:
                    failure = "fk_crosscheck_failed"
            if entry_failure == "nonfinite_geometry":
                hard_safety_failure = True
            return _record(
                backend=backend,
                graph=graph,
                observation=observation,
                history=history,
                available=available,
                environment_seed=environment_seed,
                policy_rng_seed=policy_rng_seed,
                acquisition_pose=acquisition_pose,
                candidate_set=candidate_set,
                selected_index=selected_index,
                trace=trace,
                input_hashes=input_hashes,
                final_payload_hash=final_payload_hash,
                entry_payload_hash=entry_payload_hash,
                failure=failure,
                input_failure=input_failure,
                hard_safety_failure=hard_safety_failure,
                geometry=geometry,
                fk_error=fk_error,
            )
        finally:
            backend.close()


def _entry_geometry(
    backend,
    observation,
    history,
    available,
    policy_rng_seed,
    previous_identity,
    candidate_set,
    selected_index,
    acquisition_pose,
    *,
    enabled,
):
    if not enabled:
        return None, None, None, None
    entry_payload = policy_input_bytes(
        observation,
        history,
        available,
        policy_rng_seed,
        phase_index=B2_PHASE_INDEX,
        phase_step=0,
    )
    entry_payload_hash = hashlib.sha256(entry_payload).hexdigest()
    entry_failure = _input_failure(
        backend,
        observation,
        entry_payload,
        supported_only=True,
        previous_identity=previous_identity,
    )
    if entry_failure is not None:
        return None, None, entry_payload_hash, entry_failure
    value = deserialize_policy_input(entry_payload)
    candidate = candidate_set.candidates[selected_index]
    try:
        geometry = measure_phase_entry_geometry(
            candidate,
            acquisition_pose,
            value.base_pose,
            value.left_joint_position,
            value.right_joint_position,
        )
        reused_tools = tool_positions_in_acquisition(
            value,
            acquisition_pose,
        )
        fk_error = max(
            float(
                np.linalg.norm(
                    reused_tools[index]
                    - np.asarray(
                        geometry["arms"][arm]["tool_acquisition_m"],
                        np.float64,
                    )
                )
            )
            for index, arm in enumerate(("left", "right"))
        )
    except ValueError as error:
        failure = (
            "nonfinite_geometry"
            if "nonfinite" in str(error)
            else "geometry_contract_failure"
        )
        return None, None, entry_payload_hash, failure
    return geometry, fk_error, entry_payload_hash, None


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
    input_hashes,
    previous_identity,
):
    failure = None
    input_failure = None
    hard_safety_failure = False
    for post_step in range(B0_STEPS + B1_STEPS):
        phase_index = 5 + int(post_step >= B0_STEPS)
        phase_step = post_step if post_step < B0_STEPS else post_step - B0_STEPS
        payload = policy_input_bytes(
            observation,
            history,
            available,
            policy_rng_seed,
            phase_index=phase_index,
            phase_step=phase_step,
        )
        input_hashes.append(hashlib.sha256(payload).hexdigest())
        current_failure = _input_failure(
            backend,
            observation,
            payload,
            supported_only=True,
            previous_identity=previous_identity,
        )
        input_failure = input_failure or current_failure
        failure = failure or current_failure
        previous_identity = (observation.timestamp_ns, observation.sequence_id)
        if failure is not None:
            break
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
        step_failure = _prefix_step_failure(row, backend)
        hard_safety_failure |= _is_hard_safety(step_failure)
        failure = failure or step_failure
        if failure is not None:
            break
    return (
        observation,
        failure,
        input_failure,
        hard_safety_failure,
        previous_identity,
    )


def _record(
    *,
    backend,
    graph,
    observation,
    history,
    available,
    environment_seed,
    policy_rng_seed,
    acquisition_pose,
    candidate_set,
    selected_index,
    trace,
    input_hashes,
    final_payload_hash,
    entry_payload_hash,
    failure,
    input_failure,
    hard_safety_failure,
    geometry,
    fk_error,
):
    graph_report = graph.report()
    audit = backend.task_audit()
    conservation = p40_conservation_differences(
        graph_report,
        backend.contact_ledger.report(),
    )
    candidate_bytes = b"" if candidate_set is None else candidate_set.canonical_bytes
    candidate = (
        None
        if candidate_set is None or not 0 <= selected_index < len(candidate_set.candidates)
        else candidate_set.candidates[selected_index]
    )
    complete = (
        len(trace) == PREFIX_STEPS
        and not any(row["terminal"] for row in trace)
    )
    hard_safety_failure |= any(
        (
            not all(row["action_bounds_valid"] for row in trace),
            any(
                row["outside_validity_window"]
                and row["applied_action"] != row["hold_action"]
                for row in trace
            ),
            any(row["safety_intervened"] for row in trace),
            int(audit["severe_collision_count"]) != 0,
            any(int(graph_report[name]) for name in INVALID_GRAPH_FIELDS),
            float(conservation["maximum_absolute_difference"]) != 0.0,
        )
    )
    return {
        "environment_seed": environment_seed,
        "policy_rng_seed": policy_rng_seed,
        "eligible": failure is None and complete and geometry is not None,
        "eligibility_reason": "eligible" if failure is None and complete else failure,
        "hard_safety_failure": hard_safety_failure,
        "candidate_count": 0 if candidate_set is None else len(candidate_set.candidates),
        "candidate_set_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes_hex": candidate_bytes.hex(),
        "selected_index": selected_index,
        "selected_record": None if candidate is None else asdict(candidate),
        "runtime_observation_latency_steps": int(
            audit["randomization"]["observation_latency_steps"]
        ),
        "runtime_action_latency_steps": int(
            audit["randomization"]["action_latency_steps"]
        ),
        "latency_override_inactive": all(
            audit[name] is None
            for name in (
                "action_latency_diagnostic",
                "observation_latency_diagnostic",
                "latency_pair_diagnostic",
            )
        ),
        "runtime_randomization_sha256": canonical_sha256(
            audit["randomization"]
        ),
        "input_failure_reason": input_failure,
        "prefix_failure_reason": failure,
        "prefix_step_count": len(trace),
        "prefix_complete": complete,
        "prefix_terminal_observed": any(row["terminal"] for row in trace),
        "prefix_action_bounds_valid": all(
            row["action_bounds_valid"] for row in trace
        ),
        "prefix_stale_action_applied_count": sum(
            bool(row["outside_validity_window"])
            and row["applied_action"] != row["hold_action"]
            for row in trace
        ),
        "prefix_safety_intervention_count": sum(
            bool(row["safety_intervened"]) for row in trace
        ),
        "prefix_severe_collision_count": int(audit["severe_collision_count"]),
        "prefix_invalid_force_count": sum(
            int(graph_report[name]) for name in INVALID_GRAPH_FIELDS
        ),
        "prefix_p40_conservation_maximum_absolute_difference": float(
            conservation["maximum_absolute_difference"]
        ),
        "policy_input_count": len(input_hashes),
        "policy_input_sha256": input_hashes,
        "policy_input_sequence_sha256": canonical_sha256(input_hashes),
        "candidate_final_policy_input_sha256": final_payload_hash,
        "b2_entry_policy_input_sha256": entry_payload_hash,
        "raw_prefix_trace": trace,
        "raw_prefix_trace_sha256": canonical_sha256(trace),
        "acquisition_base_pose": list(acquisition_pose),
        "b2_policy_base_pose": list(observation.proprioception.base_pose),
        "geometry": geometry,
        "fk_crosscheck_max_error_m": fk_error,
        "b2_action_generated": False,
        "b2_action_executed": False,
        "post_prefix_action_count": 0,
    }


def _is_hard_safety(reason: str | None) -> bool:
    return reason in {
        "action_bounds_violation",
        "stale_action_applied",
        "severe_collision",
        "safety_intervention_during_prefix",
        "invalid_force",
        "p40_conservation_violation",
    }
