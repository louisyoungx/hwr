from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from hwr.adapters.mujoco import (  # noqa: E402
    MujocoHouseholdBackend,
    PrivilegedHouseholdExpert,
    load_mujoco_task_bindings,
)
from hwr.scenarios.formal3d import load_formal_3d_tasks  # noqa: E402
from hwr.adapters.mujoco.formal_routes import (  # noqa: E402
    top_down_gripper_rotation,
    top_down_site_compensation,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = load_formal_3d_tasks(ROOT / "configs/tasks/formal_3d_v1.json")
BINDINGS = load_mujoco_task_bindings(
    ROOT / "configs/adapters/mujoco/formal_3d_v1.json", root=ROOT
)


def test_floor_grasp_rotation_points_finger_length_downward() -> None:
    rotation = top_down_gripper_rotation(math.pi / 2)

    assert tuple(row[0] for row in rotation) == pytest.approx((0.0, 0.0, -1.0))
    assert top_down_site_compensation(0.0) == pytest.approx((0.04, 0.0, 0.0))
    assert top_down_site_compensation(math.pi) == pytest.approx((-0.04, 0.0, 0.0))


def test_dining_plate_navigation_aligns_before_entering_table_clearance() -> None:
    task_id = "clear_dining_table_3d/v1"
    backend = MujocoHouseholdBackend(
        TASKS[task_id], BINDINGS[task_id], camera_width=2, camera_height=2
    )
    try:
        backend.reset(seed=301, task_id=task_id)
        expert = PrivilegedHouseholdExpert(backend)
        expert.stage_index = next(
            index
            for index, stage in enumerate(expert.stages)
            if stage.kind == "nav_object" and stage.object_id == "plate"
        )
        expert._enter_stage()  # noqa: SLF001 - assert privileged route contract
        prealign, goal = expert.nav_targets[-2:]
        expert.stage_index = next(
            index
            for index, stage in enumerate(expert.stages)
            if stage.kind == "nav_target" and stage.object_id == "plate"
        )
        base_position = backend.data.xpos[backend.bundle.ids.base_body].copy()
        expert._enter_stage()  # noqa: SLF001 - assert privileged route contract
        retreat = expert.nav_targets[0]
        corridor = expert.nav_targets[1:3]
        target_prealign, target_goal = expert.nav_targets[-2:]
    finally:
        backend.close()

    assert prealign[2] == goal[2] == math.pi / 2
    assert math.dist(prealign[:2], goal[:2]) == pytest.approx(0.30)
    assert retreat[2] == math.pi / 2
    assert retreat[0] == pytest.approx(base_position[0])
    assert retreat[1] == pytest.approx(base_position[1] - 0.30)
    assert all(target[1] == pytest.approx(-0.65) for target in corridor)
    assert target_prealign[2] == target_goal[2] == math.pi / 2
    assert math.dist(target_prealign[:2], target_goal[:2]) == pytest.approx(0.35)


def test_living_floor_pickup_stops_inside_arm_workspace() -> None:
    task_id = "tidy_living_room_3d/v1"
    backend = MujocoHouseholdBackend(
        TASKS[task_id], BINDINGS[task_id], camera_width=2, camera_height=2
    )
    try:
        backend.reset(seed=301, task_id=task_id)
        expert = PrivilegedHouseholdExpert(backend)
        duck_stages = [
            stage.kind for stage in expert.stages if stage.object_id == "duck"
        ]
        expert.stage_index = next(
            index
            for index, stage in enumerate(expert.stages)
            if stage.kind == "nav_object" and stage.object_id == "duck"
        )
        expert._enter_stage()  # noqa: SLF001 - assert privileged route contract
        goal = expert.nav_targets[-1]
        expert.stage_index = next(
            index
            for index, stage in enumerate(expert.stages)
            if stage.kind == "nav_target" and stage.object_id == "duck"
        )
        expert._enter_stage()  # noqa: SLF001 - assert privileged route contract
        target_route = expert.nav_targets
        football_grip = expert._grip_fraction("football")  # noqa: SLF001
        object_xy = expert._object_position("duck")[:2]  # noqa: SLF001
        shoulder = backend.model.body_pos[
            backend.model.body("right_shoulder_pan_link").id, :2
        ]
        yaw = goal[2]
        shoulder_xy = (
            goal[0] + shoulder[0] * math.cos(yaw) - shoulder[1] * math.sin(yaw),
            goal[1] + shoulder[0] * math.sin(yaw) + shoulder[1] * math.cos(yaw),
        )
    finally:
        backend.close()

    assert math.dist(shoulder_xy, object_xy) == pytest.approx(0.58)
    assert target_route[:2] == [(1.05, 0.30, None), (1.28, 0.92, None)]
    assert football_grip == pytest.approx(0.60)
    assert duck_stages.index("arm_object_clearance") + 1 == duck_stages.index(
        "arm_object_above"
    )


def test_kitchen_drawer_navigation_aligns_east_of_island() -> None:
    task_id = "store_kitchen_items_3d/v1"
    backend = MujocoHouseholdBackend(
        TASKS[task_id], BINDINGS[task_id], camera_width=2, camera_height=2
    )
    try:
        backend.reset(seed=301, task_id=task_id)
        expert = PrivilegedHouseholdExpert(backend)
        expert.stage_index = next(
            index for index, stage in enumerate(expert.stages) if stage.kind == "nav_drawer"
        )
        expert._enter_stage()  # noqa: SLF001 - assert privileged route contract
        prealign, goal = expert.nav_targets[-2:]
        handle = expert._drawer_handle_position()  # noqa: SLF001
        grasp = expert._drawer_grasp_target()  # noqa: SLF001
        expert.stage_index = next(
            index
            for index, stage in enumerate(expert.stages)
            if stage.kind == "nav_target" and stage.object_id == "cleaner_yellow"
        )
        expert._enter_stage()  # noqa: SLF001 - assert privileged route contract
        target_prealign, target_goal = expert.nav_targets[-2:]
    finally:
        backend.close()

    assert prealign[0] == pytest.approx(goal[0])
    assert prealign[1:] == pytest.approx((0.65, math.pi / 2))
    assert goal[0] >= 1.20
    assert handle[0] == pytest.approx(1.55)
    assert grasp == pytest.approx((handle[0], handle[1] - 0.024, handle[2] + 0.03))
    assert target_prealign == pytest.approx(
        (target_goal[0], 0.65, math.pi / 2)
    )


@pytest.mark.skip(
    reason="legacy privileged-expert acceptance is outside the foundation RL lineage"
)
@pytest.mark.parametrize(
    ("task_id", "seed"),
    (
        ("tidy_living_room_3d/v1", 301),
        ("tidy_living_room_3d/v1", 1000),
        ("tidy_living_room_3d/v1", 2001),
        ("tidy_living_room_3d/v1", 2002),
        ("clear_dining_table_3d/v1", 301),
        ("clear_dining_table_3d/v1", 1000),
        ("clear_dining_table_3d/v1", 1001),
        ("store_kitchen_items_3d/v1", 301),
        ("store_kitchen_items_3d/v1", 3000),
        ("store_kitchen_items_3d/v1", 3001),
        ("store_kitchen_items_3d/v1", 3002),
    ),
)
def test_household_expert_completes_contact_only_household_task(
    task_id: str, seed: int,
) -> None:
    backend = MujocoHouseholdBackend(
        TASKS[task_id], BINDINGS[task_id], camera_width=2, camera_height=2
    )
    try:
        observation = backend.reset(seed=seed, task_id=task_id)
        expert = PrivilegedHouseholdExpert(backend)
        if task_id.startswith("store_kitchen"):
            assert expert.phase_names[0] == "stow_for_drawer"
        else:
            assert expert.phase_names[0].startswith("stow_for_nav_object_")
        for _ in range(TASKS[task_id].max_steps):
            output = expert.action(observation)
            outcome = backend.apply(output.action)
            observation = outcome.observation
            if outcome.terminated or outcome.truncated:
                break
        result = backend.result()
        audit = backend.audit_snapshot()
    finally:
        backend.close()

    assert result is not None and result.success, audit
    assert expert.failed is False
    assert audit["stable_steps"] >= 40
    assert audit["severe_collision_count"] == 0
    assert all(value["inside_target"] for value in audit["objects"].values())
    assert all(value["bilateral_contact_steps"] > 0 for value in audit["objects"].values())
    if task_id.startswith("store_kitchen"):
        assert audit["articulation_satisfied"] is True
        assert audit["drawer_bilateral_contact_steps"] > 0
    assert backend.model.neq == 0
