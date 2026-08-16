from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from hwr.adapters.mujoco import (  # noqa: E402
    MujocoFormalHouseholdDualArmBackend,
    load_default_formal_household_catalogs,
)
from hwr.core.embodied import DualArmAction, DualArmActionFrame  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TASKS, BINDINGS = load_default_formal_household_catalogs(ROOT)


def _backend(
    task_id: str, *, evaluation: bool = False
) -> MujocoFormalHouseholdDualArmBackend:
    return MujocoFormalHouseholdDualArmBackend(
        TASKS[task_id],
        BINDINGS[task_id],
        camera_width=32,
        camera_height=24,
        evaluation_profile=evaluation,
    )


def _idle(observation) -> DualArmActionFrame:
    action = DualArmAction(
        0.0,
        0.0,
        (0.0,) * 6,
        (0.0,) * 6,
        observation.proprioception.left_gripper_position,
        observation.proprioception.right_gripper_position,
    )
    return DualArmActionFrame(
        observation.timestamp_ns,
        observation.timestamp_ns,
        observation.timestamp_ns + 250_000_000,
        "test_policy",
        action,
    )


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_formal_household_runtime_uses_canonical_multicamera_contract(task_id) -> None:
    backend = _backend(task_id)
    try:
        observation = backend.reset(seed=17, task_id=task_id)
        outcome = backend.apply(_idle(observation))
        audit = backend.task_audit()
    finally:
        backend.close()

    assert tuple(frame.camera_id for frame in observation.cameras) == (
        "head_rgb",
        "head_depth",
        "left_wrist_rgb",
        "right_wrist_rgb",
    )
    assert observation.instruction.text in TASKS[task_id].training_instructions
    assert len(outcome.info["applied_action"].action.vector()) == 16
    assert not outcome.terminated
    assert not outcome.truncated
    assert audit["instruction_split"] == "train"
    assert audit["randomization"]["profile"] == "train"
    assert audit["severe_collision_count"] == 0


def test_formal_evaluation_holds_out_language_and_physical_domain() -> None:
    task_id = "tidy_living_room_3d/v1"
    training = _backend(task_id)
    evaluation = _backend(task_id, evaluation=True)
    try:
        training_observation = training.reset(seed=23, task_id=task_id)
        evaluation_observation = evaluation.reset(seed=23, task_id=task_id)
        training_audit = training.task_audit()
        evaluation_audit = evaluation.task_audit()
    finally:
        training.close()
        evaluation.close()

    assert training_observation.instruction.text in TASKS[task_id].training_instructions
    assert (
        evaluation_observation.instruction.text
        in TASKS[task_id].evaluation_instructions
    )
    assert evaluation_audit["randomization"]["profile"] == "evaluation"
    assert training_audit["randomization"] != evaluation_audit["randomization"]
    assert (
        training_observation.camera_calibrations[0].intrinsics
        != evaluation_observation.camera_calibrations[0].intrinsics
    )


def test_formal_randomization_is_reproducible_for_the_same_profile_and_seed() -> None:
    task_id = "clear_dining_table_3d/v1"
    first = _backend(task_id)
    second = _backend(task_id)
    try:
        first.reset(seed=41, task_id=task_id)
        second.reset(seed=41, task_id=task_id)
        first_audit = first.task_audit()
        second_audit = second.task_audit()
    finally:
        first.close()
        second.close()

    assert first_audit["randomization"] == second_audit["randomization"]


def test_formal_diagnostic_override_changes_only_observation_latency() -> None:
    task_id = "clear_dining_table_3d/v1"
    backend = _backend(task_id)
    try:
        backend.reset_for_observation_latency_diagnostic(
            seed=41, task_id=task_id, observation_latency_steps=0
        )
        lag_zero = backend.task_audit()
        backend.reset_for_observation_latency_diagnostic(
            seed=41, task_id=task_id, observation_latency_steps=1
        )
        lag_one = backend.task_audit()
    finally:
        backend.close()

    zero_randomization = dict(lag_zero["randomization"])
    one_randomization = dict(lag_one["randomization"])
    assert zero_randomization.pop("observation_latency_steps") == 0
    assert one_randomization.pop("observation_latency_steps") == 1
    assert zero_randomization == one_randomization
    zero_provenance = lag_zero["observation_latency_diagnostic"]
    one_provenance = lag_one["observation_latency_diagnostic"]
    assert zero_provenance["sampled_randomization_sha256"] == (
        one_provenance["sampled_randomization_sha256"]
    )
    assert zero_provenance["other_randomization_sha256"] == (
        one_provenance["other_randomization_sha256"]
    )
    assert zero_provenance["verified_only_observation_latency_changed"] is True
    assert one_provenance["verified_only_observation_latency_changed"] is True


def test_formal_runtime_can_defer_and_resume_camera_rendering() -> None:
    task_id = "clear_dining_table_3d/v1"
    backend = _backend(task_id)
    try:
        observation = backend.reset(seed=42, task_id=task_id)
        initial_payload = observation.cameras[0].payload
        backend.set_camera_rendering(False)
        deferred = backend.apply(_idle(observation)).observation
        backend.set_camera_rendering(True)
        resumed = backend.observe()
    finally:
        backend.close()

    assert deferred.cameras[0].payload == initial_payload
    assert deferred.sequence_id in (
        observation.sequence_id,
        observation.sequence_id + 1,
    )
    assert resumed.sequence_id == observation.sequence_id + 1
    assert resumed.cameras[0].timestamp_ns == resumed.timestamp_ns


def test_formal_runtime_rejects_predicted_severe_collision_before_commit() -> None:
    task_id = "clear_dining_table_3d/v1"
    backend = _backend(task_id)
    try:
        observation = backend.reset(seed=43, task_id=task_id)
        backend._predictive_safety_violation = lambda: True  # type: ignore[method-assign]  # noqa: SLF001
        action = DualArmAction(
            0.18,
            0.5,
            (0.35,) * 6,
            (-0.35,) * 6,
            0.0,
            1.0,
        )
        outcome = backend.apply(
            DualArmActionFrame(
                observation.timestamp_ns,
                observation.timestamp_ns,
                observation.timestamp_ns + 250_000_000,
                "test_policy",
                action,
            )
        )
    finally:
        backend.close()

    applied = outcome.info["applied_action"]
    assert applied.source == "safety"
    assert outcome.info["safety_intervened"] is True
    assert outcome.info["physics_advanced"] is False
    assert outcome.events[-1].event_type == "action_rejected"
    assert outcome.events[-1].details["reason"] == "predicted_severe_collision"


def test_formal_runtime_terminates_after_actual_severe_collision() -> None:
    task_id = "clear_dining_table_3d/v1"
    backend = _backend(task_id)
    try:
        observation = backend.reset(seed=47, task_id=task_id)
        backend._severe_collision_count = 1
        result = backend._task_result_after_step()
    finally:
        backend.close()

    assert result is not None
    assert result.success is False
    assert result.reason == "severe_collision"


def test_formal_success_requires_both_arms_and_concurrent_contact() -> None:
    task_id = "tidy_living_room_3d/v1"
    backend = _backend(task_id)
    try:
        backend.reset(seed=5, task_id=task_id)
        required = round(TASKS[task_id].control_hz * 0.5)
        backend._left_contact_steps = required
        backend._right_contact_steps = required
        backend._maximum_concurrent_steps = 0
        for object_id, joint_id in backend.household_ids.object_joints.items():
            site_id = backend.household_ids.target_sites[object_id]
            address = int(backend.model.jnt_qposadr[joint_id])
            backend.data.qpos[address : address + 3] = backend.data.site_xpos[site_id]
            dof = int(backend.model.jnt_dofadr[joint_id])
            backend.data.qvel[dof : dof + 6] = 0.0
        import mujoco

        mujoco.mj_forward(backend.model, backend.data)
        result = None
        for _ in range(round(TASKS[task_id].control_hz * 2.0)):
            result = backend._task_result_after_step()
        assert result is None
        backend._maximum_concurrent_steps = required
        result = backend._task_result_after_step()
    finally:
        backend.close()

    assert result is not None
    assert result.success
