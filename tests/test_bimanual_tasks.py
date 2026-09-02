from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hwr.tasks import BimanualTaskSample, BimanualTaskTracker, load_bimanual_task_specs


ROOT = Path(__file__).resolve().parents[1]
SPECS = load_bimanual_task_specs(ROOT / "configs/tasks/bimanual_household_v1.json")


def _sample(task_id: str, **overrides) -> BimanualTaskSample:
    spec = SPECS[task_id]
    values = {
        "payload_position": (0.7, 0.0, 0.7),
        "target_position": spec.target_position,
        "payload_tilt_radians": 0.0,
        "payload_linear_speed": 0.0,
        "payload_angular_speed": 0.0,
        "left_reach_distance": 0.2,
        "right_reach_distance": 0.2,
        "left_contact": False,
        "right_contact": False,
        "support_contact": False,
        "inside_target": False,
    }
    values.update(overrides)
    return BimanualTaskSample(**values)


def _at_target(task_id: str, **overrides) -> BimanualTaskSample:
    spec = SPECS[task_id]
    values = {
        "payload_position": spec.target_position,
        "left_reach_distance": 0.0,
        "right_reach_distance": 0.0,
        "support_contact": True,
    }
    values.update(overrides)
    return _sample(task_id, **values)


def test_catalog_defines_exactly_three_raw_language_bimanual_tasks() -> None:
    assert set(SPECS) == {
        "carry_living_room_basket/v1",
        "carry_dining_tray/v1",
        "hold_drawer_place_item/v1",
    }
    assert {spec.objective for spec in SPECS.values()} == {
        "carry_payload",
        "hold_drawer_place",
    }
    assert all(spec.hold_steps == 40 for spec in SPECS.values())
    assert all(spec.concurrent_steps == 10 for spec in SPECS.values())
    assert all("both hands" in spec.instruction or "left hand" in spec.instruction for spec in SPECS.values())


@pytest.mark.parametrize(
    "task_id",
    ("carry_living_room_basket/v1", "carry_dining_tray/v1"),
)
def test_carry_success_requires_prior_bimanual_window_and_two_second_stability(
    task_id: str,
) -> None:
    spec = SPECS[task_id]
    tracker = BimanualTaskTracker(spec)
    tracker.reset(_sample(task_id))

    contacted = _sample(task_id, left_contact=True, right_contact=True)
    tracker.update(contacted)
    transported = replace(contacted, payload_position=spec.target_position)
    update = tracker.update(transported)
    for _ in range(spec.concurrent_steps - 2):
        update = tracker.update(transported)
    assert update.metrics["controlled_target_complete"] == 1.0
    assert update.metrics["maximum_controlled_target_progress"] >= (
        update.metrics["required_controlled_target_progress"]
    )
    assert update.maximum_concurrent_steps == spec.concurrent_steps
    assert tracker.left_contact_steps == spec.concurrent_steps
    assert tracker.right_contact_steps == spec.concurrent_steps
    assert tracker.simultaneous_contact_steps == spec.concurrent_steps

    for _ in range(spec.hold_steps - 1):
        update = tracker.update(_at_target(task_id))
        assert not update.success
    update = tracker.update(_at_target(task_id))

    assert update.success and update.terminated
    assert update.stable_steps == 40
    assert update.metrics["severe_collisions"] == 0.0
    assert len(update.achieved_goal) == len(update.desired_goal) == 12


def test_one_arm_contact_can_never_satisfy_bimanual_carry() -> None:
    task_id = "carry_living_room_basket/v1"
    spec = SPECS[task_id]
    tracker = BimanualTaskTracker(spec)
    tracker.reset(_sample(task_id))

    for _ in range(spec.hold_steps + spec.concurrent_steps + 5):
        update = tracker.update(
            _at_target(task_id, left_contact=True, right_contact=False)
        )

    assert not update.success
    assert update.maximum_concurrent_steps == 0


def test_drawer_requires_left_hold_open_while_right_side_places_payload() -> None:
    task_id = "hold_drawer_place_item/v1"
    spec = SPECS[task_id]
    tracker = BimanualTaskTracker(spec)
    tracker.reset(_sample(task_id))

    contacted = _sample(task_id, left_contact=True, right_contact=True)
    tracker.update(contacted)
    transported = replace(
        contacted,
        payload_position=spec.target_position,
        articulation_position=0.30,
    )
    tracker.update(transported)
    for _ in range(spec.concurrent_steps - 2):
        tracker.update(transported)
    for _ in range(spec.hold_steps):
        update = tracker.update(
            _at_target(
                task_id,
                left_contact=True,
                right_contact=False,
                inside_target=True,
                articulation_position=0.30,
                articulation_speed=0.0,
            )
        )

    assert update.success
    closed = replace(
        _at_target(
            task_id,
            left_contact=True,
            inside_target=True,
            articulation_position=0.30,
        ),
        articulation_position=0.10,
    )
    second = BimanualTaskTracker(spec)
    second.reset(_sample(task_id))
    for _ in range(spec.concurrent_steps + spec.hold_steps):
        failed = second.update(closed)
    assert not failed.success


def test_severe_collision_terminates_without_false_success() -> None:
    task_id = "carry_dining_tray/v1"
    tracker = BimanualTaskTracker(SPECS[task_id])
    tracker.reset(_sample(task_id))

    update = tracker.update(
        _at_target(
            task_id,
            left_contact=True,
            right_contact=True,
            severe_collision_count=1,
        )
    )

    assert update.terminated and not update.success
    assert update.reward < 0.0


def test_reward_encourages_closing_both_grippers_only_near_handles() -> None:
    task_id = "carry_living_room_basket/v1"
    spec = SPECS[task_id]
    near = _sample(
        task_id,
        left_reach_distance=0.05,
        right_reach_distance=0.05,
        left_gripper_position=0.0,
        right_gripper_position=0.0,
    )
    tracker = BimanualTaskTracker(spec)
    tracker.reset(near)

    closed = replace(
        near,
        left_gripper_position=1.0,
        right_gripper_position=1.0,
    )
    update = tracker.update(closed)

    assert update.reward > 1.0


def test_reward_prefers_balanced_bimanual_progress_over_one_sided_progress() -> None:
    task_id = "carry_dining_tray/v1"
    spec = SPECS[task_id]
    initial = _sample(
        task_id,
        left_reach_distance=0.20,
        right_reach_distance=0.20,
    )
    one_sided_tracker = BimanualTaskTracker(spec)
    one_sided_tracker.reset(initial)
    one_sided = one_sided_tracker.update(
        replace(initial, left_reach_distance=0.05)
    )
    balanced_tracker = BimanualTaskTracker(spec)
    balanced_tracker.reset(initial)
    balanced = balanced_tracker.update(
        replace(
            initial,
            left_reach_distance=0.125,
            right_reach_distance=0.125,
        )
    )

    assert balanced.reward > one_sided.reward


def test_reward_pays_for_staying_jointly_near_but_not_staying_far() -> None:
    task_id = "carry_dining_tray/v1"
    spec = SPECS[task_id]
    near = _sample(
        task_id,
        left_reach_distance=0.08,
        right_reach_distance=0.08,
    )
    far = replace(
        near,
        left_reach_distance=0.20,
        right_reach_distance=0.20,
    )
    near_tracker = BimanualTaskTracker(spec)
    near_tracker.reset(near)
    far_tracker = BimanualTaskTracker(spec)
    far_tracker.reset(far)

    near_update = near_tracker.update(near)
    far_update = far_tracker.update(far)

    assert near_update.reward > 0.0
    assert far_update.reward < 0.0
    assert near_update.metrics["bilateral_reach_occupancy"] > 0.0


def test_reward_contains_bilateral_contact_synergy() -> None:
    task_id = "carry_living_room_basket/v1"
    spec = SPECS[task_id]
    initial = _sample(task_id)
    single_tracker = BimanualTaskTracker(spec)
    single_tracker.reset(initial)
    single = single_tracker.update(replace(initial, left_contact=True))
    both_tracker = BimanualTaskTracker(spec)
    both_tracker.reset(initial)
    both = both_tracker.update(
        replace(initial, left_contact=True, right_contact=True)
    )

    assert both.reward - single.reward > spec.reward.contact


@pytest.mark.parametrize(
    "task_id",
    ("carry_living_room_basket/v1", "carry_dining_tray/v1"),
)
def test_carry_progress_requires_sustained_bilateral_contact(task_id: str) -> None:
    spec = SPECS[task_id]
    initial = _sample(task_id, payload_position=(0.7, 0.0, 0.7))
    closer = replace(initial, payload_position=spec.target_position)

    pushed_tracker = BimanualTaskTracker(spec)
    pushed_tracker.reset(initial)
    pushed = pushed_tracker.update(closer)

    carried_initial = replace(initial, left_contact=True, right_contact=True)
    carried_tracker = BimanualTaskTracker(spec)
    carried_tracker.reset(carried_initial)
    carried = carried_tracker.update(
        replace(closer, left_contact=True, right_contact=True)
    )

    assert pushed.metrics["controlled_target_progress"] == 0.0
    assert carried.metrics["controlled_target_progress"] > 0.0
    assert carried.metrics["initial_target_distance"] == initial.target_distance
    assert carried.reward > pushed.reward


def test_releasing_payload_withdraws_live_reward_but_keeps_completion_evidence() -> None:
    task_id = "carry_dining_tray/v1"
    spec = SPECS[task_id]
    initial = _sample(
        task_id,
        payload_position=(0.7, 0.0, 0.7),
        left_contact=True,
        right_contact=True,
    )
    closer = replace(initial, payload_position=spec.target_position)
    tracker = BimanualTaskTracker(spec)
    tracker.reset(initial)

    carried = tracker.update(closer)
    released = tracker.update(
        replace(closer, left_contact=False, right_contact=False)
    )

    assert carried.metrics["controlled_target_progress"] > 0.0
    assert released.metrics["controlled_target_progress"] == 0.0
    assert released.metrics["maximum_controlled_target_progress"] > 0.0
    assert released.metrics["controlled_target_complete"] == 1.0
    assert released.reward < 0.0


def test_prior_bimanual_contact_does_not_authorize_an_uncontrolled_push() -> None:
    task_id = "carry_living_room_basket/v1"
    spec = SPECS[task_id]
    initial = _sample(task_id, payload_position=(0.7, 0.0, 0.7))
    tracker = BimanualTaskTracker(spec)
    tracker.reset(initial)

    for _ in range(spec.concurrent_steps):
        tracker.update(replace(initial, left_contact=True, right_contact=True))
    for _ in range(spec.hold_steps):
        update = tracker.update(_at_target(task_id))

    assert not update.success
    assert update.stable_steps == spec.hold_steps
    assert update.maximum_concurrent_steps == spec.concurrent_steps
    assert update.metrics["maximum_controlled_target_progress"] == 0.0
    assert update.metrics["controlled_target_complete"] == 0.0


def test_leaving_target_revokes_causal_completion_until_controlled_return() -> None:
    task_id = "carry_dining_tray/v1"
    spec = SPECS[task_id]
    initial = _sample(task_id, payload_position=(0.7, 0.0, 0.7))
    contacted = replace(initial, left_contact=True, right_contact=True)
    tracker = BimanualTaskTracker(spec)
    tracker.reset(initial)
    tracker.update(contacted)
    tracker.update(replace(contacted, payload_position=spec.target_position))
    assert tracker.update(
        replace(contacted, payload_position=spec.target_position)
    ).metrics["controlled_target_complete"] == 1.0

    tracker.update(initial)
    for _ in range(spec.hold_steps):
        update = tracker.update(_at_target(task_id))

    assert not update.success
    assert update.metrics["maximum_controlled_target_progress"] > 0.0
    assert update.metrics["controlled_target_complete"] == 0.0


def test_drawer_opening_progress_requires_sustained_left_contact() -> None:
    task_id = "hold_drawer_place_item/v1"
    spec = SPECS[task_id]
    initial = _sample(task_id, articulation_position=0.0)
    opened = replace(initial, articulation_position=0.3)

    pushed_tracker = BimanualTaskTracker(spec)
    pushed_tracker.reset(initial)
    pushed = pushed_tracker.update(opened)

    held_initial = replace(initial, left_contact=True)
    held_tracker = BimanualTaskTracker(spec)
    held_tracker.reset(held_initial)
    held = held_tracker.update(replace(opened, left_contact=True))

    assert pushed.metrics["controlled_articulation_progress"] == 0.0
    assert held.metrics["controlled_articulation_progress"] == pytest.approx(
        spec.criteria.minimum_articulation_position
    )
    assert held.reward > pushed.reward


def test_drawer_success_rejects_opening_before_left_contact() -> None:
    task_id = "hold_drawer_place_item/v1"
    spec = SPECS[task_id]
    initial = _sample(task_id, articulation_position=0.0)
    contacted = replace(initial, left_contact=True, right_contact=True)
    tracker = BimanualTaskTracker(spec)
    tracker.reset(initial)
    tracker.update(contacted)
    transported = replace(contacted, payload_position=spec.target_position)
    tracker.update(transported)
    for _ in range(spec.concurrent_steps - 2):
        tracker.update(transported)

    tracker.update(
        _at_target(
            task_id,
            inside_target=True,
            articulation_position=0.30,
        )
    )
    for _ in range(spec.hold_steps):
        update = tracker.update(
            _at_target(
                task_id,
                left_contact=True,
                inside_target=True,
                articulation_position=0.30,
            )
        )

    assert not update.success
    assert update.metrics["controlled_target_complete"] == 1.0
    assert update.metrics["controlled_articulation_complete"] == 0.0
    assert update.metrics["maximum_controlled_articulation_progress"] == 0.0
