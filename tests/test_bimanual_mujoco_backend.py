from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hwr.adapters.mujoco import (
    MujocoBimanualTaskBackend,
    load_bimanual_mujoco_bindings,
)
from hwr.core.embodied import DualArmAction, DualArmActionFrame
from hwr.tasks import load_bimanual_task_specs


ROOT = Path(__file__).resolve().parents[1]
TASKS = load_bimanual_task_specs(ROOT / "configs/tasks/bimanual_household_v1.json")
BINDINGS = load_bimanual_mujoco_bindings(
    ROOT / "configs/adapters/mujoco/bimanual_household_v1.json",
    root=ROOT,
)


def _backend(task_id: str) -> MujocoBimanualTaskBackend:
    return MujocoBimanualTaskBackend(
        TASKS[task_id], BINDINGS[task_id], camera_width=32, camera_height=24
    )


def _idle(timestamp_ns: int) -> DualArmActionFrame:
    return DualArmActionFrame(
        timestamp_ns,
        timestamp_ns,
        timestamp_ns + 100_000_000,
        "random_actor",
        DualArmAction(0.0, 0.0, (0.0,) * 6, (0.0,) * 6, 0.0, 0.0),
    )


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_bimanual_scene_compiles_without_welds_and_exposes_four_actor_cameras(
    task_id: str,
) -> None:
    backend = _backend(task_id)
    try:
        observation = backend.reset(seed=11, task_id=task_id)
        camera_ids = tuple(camera.camera_id for camera in observation.cameras)
        state = backend.privileged_training_state()
    finally:
        backend.close()

    assert backend.model.neq == 0
    assert backend.model.nu == 20
    assert camera_ids == (
        "head_rgb",
        "head_depth",
        "left_wrist_rgb",
        "right_wrist_rgb",
    )
    assert observation.instruction.text == TASKS[task_id].instruction
    assert len(state.critic_state) == 62
    assert len(state.achieved_goal) == len(state.desired_goal) == 12
    assert not hasattr(observation, "achieved_goal")
    assert not hasattr(observation, "critic_state")


def test_kitchen_drawer_is_passively_sprung_and_unactuated() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(BINDINGS["hold_drawer_place_item/v1"].model_path)
    )
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    actuated = {int(value) for value in model.actuator_trnid[:, 0]}

    assert model.jnt_stiffness[joint] > 0.0
    assert joint not in actuated
    assert model.neq == 0


@pytest.mark.parametrize(
    ("task_id", "left_posture", "right_posture", "handle_prefix"),
    (
        (
            "carry_living_room_basket/v1",
            (0.0, 1.037, -0.453, 0.0, -0.524, 0.0),
            (0.0, 1.037, -0.453, 0.0, -0.524, 0.0),
            "basket",
        ),
        (
            "carry_dining_tray/v1",
            (0.153, 1.011, -0.596, -0.019, -0.647, 0.008),
            (-0.153, 1.011, -0.596, 0.019, -0.647, -0.008),
            "tray",
        ),
    ),
)
def test_closed_physical_pincers_can_contact_both_sides_of_both_handles(
    task_id: str,
    left_posture: tuple[float, ...],
    right_posture: tuple[float, ...],
    handle_prefix: str,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(BINDINGS[task_id].model_path))
    data = mujoco.MjData(model)
    for side, posture in (("left", left_posture), ("right", right_posture)):
        for index, value in enumerate(posture, 1):
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_joint_{index}"
            )
            data.qpos[model.jnt_qposadr[joint]] = value
        for finger in ("left", "right"):
            joint = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_gripper_{finger}_finger_joint",
            )
            data.qpos[model.jnt_qposadr[joint]] = 0.095
    mujoco.mj_forward(model, data)
    pairs = {
        frozenset(
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            )
        )
        for contact in data.contact[: data.ncon]
    }

    for side in ("left", "right"):
        handle = f"{handle_prefix}_{side}_handle"
        for finger in ("left", "right"):
            pad = f"{side}_gripper_{finger}_pad"
            assert frozenset((handle, pad)) in pairs


def test_procedural_reset_is_seeded_and_changes_continuous_physics() -> None:
    task_id = "carry_dining_tray/v1"
    backend = _backend(task_id)
    try:
        backend.reset(seed=101, task_id=task_id)
        first = backend.privileged_training_state().critic_state
        first_randomization = backend.task_audit()["randomization"]
        backend.reset(seed=101, task_id=task_id)
        repeated = backend.privileged_training_state().critic_state
        repeated_randomization = backend.task_audit()["randomization"]
        backend.reset(seed=102, task_id=task_id)
        changed = backend.privileged_training_state().critic_state
        changed_randomization = backend.task_audit()["randomization"]
    finally:
        backend.close()

    assert repeated == pytest.approx(first)
    assert repeated_randomization == first_randomization
    assert not np.allclose(changed, first)
    assert changed_randomization != first_randomization


def test_adapter_restores_self_discovered_position_as_a_fresh_episode() -> None:
    task_id = "carry_dining_tray/v1"
    backend = _backend(task_id)
    try:
        observation = backend.reset(seed=303, task_id=task_id)
        snapshot = backend.capture_state_snapshot()
        expected = np.asarray(snapshot.generalized_positions)
        assert len(snapshot.generalized_velocities) == backend.model.nv
        assert len(snapshot.generalized_accelerations) == backend.model.nv
        assert len(snapshot.actuator_controls) == backend.model.nu
        assert len(snapshot.solver_state) == backend.model.nv
        moving = DualArmActionFrame(
            observation.timestamp_ns,
            observation.timestamp_ns,
            observation.timestamp_ns + 100_000_000,
            "random_actor",
            DualArmAction(0.12, 0.0, (0.1,) * 6, (-0.1,) * 6, 0.7, 0.3),
        )
        for _ in range(3):
            outcome = backend.apply(moving)
            moving = DualArmActionFrame(
                outcome.observation.timestamp_ns,
                outcome.observation.timestamp_ns,
                outcome.observation.timestamp_ns + 100_000_000,
                "random_actor",
                moving.action,
            )
        restored = backend.reset(
            seed=304, task_id=task_id, initial_state=snapshot
        )
        audit = backend.task_audit()
    finally:
        backend.close()

    assert np.asarray(backend.data.qpos) == pytest.approx(expected)
    assert np.asarray(backend.data.qvel) == pytest.approx(0.0)
    assert restored.sequence_id == 0
    assert restored.timestamp_ns == 0
    assert audit["left_contact_steps"] == 0
    assert audit["right_contact_steps"] == 0
    assert not hasattr(backend, "restore_state_snapshot")


def test_snapshot_restores_dynamical_continuation_and_controller_load() -> None:
    task_id = "carry_dining_tray/v1"
    backend = _backend(task_id)
    action = DualArmAction(
        0.04,
        -0.02,
        (0.02, -0.01, 0.01, 0.0, 0.0, 0.0),
        (0.02, 0.01, 0.01, 0.0, 0.0, 0.0),
        1.0,
        1.0,
    )

    def advance(observation, steps: int) -> None:
        for _ in range(steps):
            outcome = backend.apply(
                DualArmActionFrame(
                    observation.timestamp_ns,
                    observation.timestamp_ns,
                    observation.timestamp_ns + 100_000_000,
                    "random_actor",
                    action,
                )
            )
            observation = outcome.observation

    try:
        observation = backend.reset(seed=811, task_id=task_id)
        advance(observation, 5)
        snapshot = backend.capture_state_snapshot()
        snapshot_left_targets = backend._left_targets.copy()
        snapshot_right_targets = backend._right_targets.copy()
        observation = backend.observe()
        advance(observation, 8)
        expected_positions = backend.data.qpos.copy()
        expected_velocities = backend.data.qvel.copy()
        expected_controls = backend.data.ctrl.copy()

        restored = backend.reset(
            seed=811, task_id=task_id, initial_state=snapshot
        )
        assert backend.data.qpos == pytest.approx(snapshot.generalized_positions)
        assert backend.data.qvel == pytest.approx(snapshot.generalized_velocities)
        assert backend.data.ctrl == pytest.approx(snapshot.actuator_controls)
        assert backend.data.qacc_warmstart == pytest.approx(snapshot.solver_state)
        assert backend._left_targets == pytest.approx(snapshot_left_targets)
        assert backend._right_targets == pytest.approx(snapshot_right_targets)
        advance(restored, 8)
    finally:
        backend.close()

    assert backend.data.qpos == pytest.approx(expected_positions, abs=1.0e-10)
    assert backend.data.qvel == pytest.approx(expected_velocities, abs=1.0e-10)
    assert backend.data.ctrl == pytest.approx(expected_controls, abs=1.0e-10)


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_idle_actor_runs_physics_without_false_success_or_truth_leak(task_id: str) -> None:
    backend = _backend(task_id)
    try:
        observation = backend.reset(seed=17, task_id=task_id)
        for _ in range(20):
            outcome = backend.apply(_idle(observation.timestamp_ns))
            observation = outcome.observation
            assert math.isfinite(outcome.reward)
            assert not outcome.terminated
        audit = backend.task_audit()
    finally:
        backend.close()

    assert audit["severe_collision_count"] == 0
    assert audit["maximum_concurrent_steps"] == 0
    assert not {
        "achieved_goal",
        "desired_goal",
        "critic_state",
        "object_truth",
    }.intersection(outcome.info)
