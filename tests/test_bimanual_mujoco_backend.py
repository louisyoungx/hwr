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
    assert len(state.critic_state) == 60
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
