"""Continuation and bank provenance helpers for R0001-P51-E1."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Sequence

import mujoco
import numpy as np

from hwr.core.embodied import DualArmObservation
from hwr.eval.cartesian_convergence import (
    canonical_sha256,
    identity,
    observation_identity_record,
    runtime_counter_record,
    wrap_angle,
)
from hwr.eval.target_selection import ACQUISITION_STEPS


CONTINUATION_SCHEMA = "hwr.p51-cartesian-continuation-identity/v1"


def continuation_identity(
    backend,
    observation: DualArmObservation,
    history: Sequence[Sequence[float]],
    history_available: Sequence[bool],
    graph,
) -> dict[str, object]:
    specification = mujoco.mjtState(
        int(mujoco.mjtState.mjSTATE_INTEGRATION)
        | int(mujoco.mjtState.mjSTATE_CTRL)
    )
    runtime = np.empty(
        mujoco.mj_stateSize(backend.model, specification), np.float64
    )
    mujoco.mj_getState(backend.model, backend.data, runtime, specification)
    components = {
        "mujoco_model_state": _array_bundle_identity(
            (
                backend.model.body_mass,
                backend.model.body_inertia,
                backend.model.geom_friction,
                backend.model.light_diffuse,
                backend.model.mat_rgba,
                backend.model.cam_pos,
                backend.model.cam_quat,
                backend.model.cam_fovy,
            )
        ),
        "mujoco_data_state": _array_bundle_identity(
            (runtime, backend.data.ctrl)
        ),
        "actuator_servo_targets": identity(
            {
                "ctrl": backend.data.ctrl.tolist(),
                "left_targets": backend._left_targets.tolist(),
                "right_targets": backend._right_targets.tolist(),
            }
        ),
        "action_latency_queue": identity(
            [list(action.vector()) for action in backend._action_queue]
        ),
        "observation_latency_queue": identity(
            [
                observation_identity_record(value)
                for value in backend._observation_queue
            ]
        ),
        "policy_history_availability": identity(
            {
                "history": [list(value) for value in history],
                "available": list(history_available),
            }
        ),
        "current_observation": identity(
            observation_identity_record(observation)
        ),
        "timestamp_sequence_runtime_safety_counters": identity(
            runtime_counter_record(backend, observation, graph)
        ),
    }
    payload = {
        "schema_version": CONTINUATION_SCHEMA,
        "components": components,
    }
    return {**payload, "identity": identity(payload)}


def bank_prefix_record(run) -> dict[str, object]:
    candidate_bytes = (
        b""
        if run.candidate_set is None
        else run.candidate_set.canonical_bytes
    )
    return {
        "eligible": run.failure is None,
        "eligibility_reason": (
            "eligible" if run.failure is None else run.failure
        ),
        "candidate_count": (
            0
            if run.candidate_set is None
            else len(run.candidate_set.candidates)
        ),
        "candidate_set_sha256": hashlib.sha256(
            candidate_bytes
        ).hexdigest(),
        "candidate_bytes_hex": candidate_bytes.hex(),
        "selected_index": run.selected_index,
        "selected_record": (
            None if run.candidate is None else asdict(run.candidate)
        ),
        "acquisition_input_hashes": run.acquisition_input_hashes,
        "acquisition_input_sequence_sha256": canonical_sha256(
            run.acquisition_input_hashes
        ),
        "prefix_trace_sha256": canonical_sha256(run.trace),
        "b0_b1_proposed_action_sha256": canonical_sha256(
            [
                row["proposed_action"]
                for row in run.trace[ACQUISITION_STEPS:]
            ]
        ),
        "b0_b1_applied_action_sha256": canonical_sha256(
            [
                row["applied_action"]
                for row in run.trace[ACQUISITION_STEPS:]
            ]
        ),
        "relative_yaw_at_b2": wrap_angle(
            run.backend._base_state()[0][2] - run.acquisition_pose[2]
        ),
        "acquisition_base_pose": list(run.acquisition_pose),
        "acquisition_world_origin": list(run.acquisition_world_origin),
        "continuation_identity": run.continuation_identity,
        "first_treatment_actions": {
            role: list(action)
            for role, action in run.first_actions.items()
        },
        "first_treatment_guard": run.first_guard,
        "preposition_targets": {
            name: list(value)
            for name, value in run.preposition_targets.items()
        },
        "primitive_target_crosscheck": run.primitive_target_crosscheck,
    }


def _array_bundle_identity(
    values: Sequence[np.ndarray],
) -> dict[str, object]:
    chunks = []
    for value in values:
        array = np.ascontiguousarray(value)
        chunks.extend(
            (
                array.dtype.str.encode("ascii"),
                str(tuple(array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return identity(b"".join(chunks))
