from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from hwr.adapters.mujoco import (  # noqa: E402
    MujocoHouseholdBackend,
    load_mujoco_task_bindings,
)
from hwr.core.types import ActionFrame  # noqa: E402
from hwr.scenarios.formal3d import load_formal_3d_tasks  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TASKS = load_formal_3d_tasks(ROOT / "configs/tasks/formal_3d_v1.json")
BINDINGS = load_mujoco_task_bindings(
    ROOT / "configs/adapters/mujoco/formal_3d_v1.json", root=ROOT
)


def _backend(task_id: str) -> MujocoHouseholdBackend:
    return MujocoHouseholdBackend(
        TASKS[task_id], BINDINGS[task_id], camera_width=32, camera_height=24
    )


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_formal_backend_reset_is_seeded_visual_and_non_privileged(task_id: str) -> None:
    backend = _backend(task_id)
    try:
        first = backend.reset(seed=301, task_id=task_id)
        first_audit = backend.audit_snapshot()
        repeated = backend.reset(seed=301, task_id=task_id)
        repeated_audit = backend.audit_snapshot()
        observed_again = backend.observe()
    finally:
        backend.close()

    assert first.base_pose == repeated.base_pose
    assert first_audit["randomization"] == repeated_audit["randomization"]
    assert first_audit["objects"] == repeated_audit["objects"]
    assert repeated.features == {}
    assert repeated.task_stage == "instruction_following"
    assert tuple(frame.camera_id for frame in repeated.cameras) == (
        "head_rgb",
        "head_depth",
        "wrist_rgb",
    )
    assert [frame.payload for frame in repeated.cameras] == [
        frame.payload for frame in observed_again.cameras
    ]
    assert all(
        backend.model.body_mass[body_id] < 1.0
        for body_id in backend.household_ids.object_bodies.values()
    )


def test_formal_backend_randomizes_physics_and_pose_across_seeds() -> None:
    task_id = "tidy_living_room_3d/v1"
    backend = _backend(task_id)
    try:
        first = backend.reset(seed=301, task_id=task_id)
        first_audit = backend.audit_snapshot()
        second = backend.reset(seed=302, task_id=task_id)
        second_audit = backend.audit_snapshot()
    finally:
        backend.close()

    assert first.base_pose != second.base_pose
    assert first_audit["randomization"] != second_audit["randomization"]
    assert first_audit["objects"]["duck"]["position"] != second_audit["objects"]["duck"]["position"]


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_idle_formal_episode_is_physical_and_not_a_false_success(task_id: str) -> None:
    backend = _backend(task_id)
    try:
        observation = backend.reset(seed=401, task_id=task_id)
        for _ in range(45):
            action = ActionFrame(
                observation.timestamp_ns,
                observation.timestamp_ns,
                observation.timestamp_ns + 100_000_000,
                "loaded_policy/test",
                arm_command=(0.0,) * 6,
            )
            outcome = backend.apply(action)
            observation = outcome.observation
        audit = backend.audit_snapshot()
    finally:
        backend.close()

    assert outcome.terminated is False
    assert outcome.truncated is False
    assert backend.result() is None
    assert audit["stable_steps"] == 0
    assert audit["severe_collision_count"] == 0
    assert all(not value["inside_target"] for value in audit["objects"].values())
    assert outcome.info["applied_action"].source == "loaded_policy/test"
