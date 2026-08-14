from __future__ import annotations

from pathlib import Path

from hwr.scenarios.formal3d import load_formal_3d_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_formal_task_specs_are_engine_independent_and_multi_object() -> None:
    path = ROOT / "configs/tasks/formal_3d_v1.json"
    tasks = load_formal_3d_tasks(path)

    assert set(tasks) == {
        "tidy_living_room_3d/v1",
        "clear_dining_table_3d/v1",
        "store_kitchen_items_3d/v1",
    }
    assert all(len(task.objects) == 2 for task in tasks.values())
    assert all(task.hold_seconds == 2.0 for task in tasks.values())
    assert all(task.minimum_each_arm_contact_seconds == 0.5 for task in tasks.values())
    assert tasks["store_kitchen_items_3d/v1"].articulation is not None
    assert "mujoco" not in path.read_text(encoding="utf-8").lower()


def test_formal_tasks_have_disjoint_language_and_broader_evaluation_domains() -> None:
    tasks = load_formal_3d_tasks(ROOT / "configs/tasks/formal_3d_v1.json")

    for task in tasks.values():
        assert len(task.training_instructions) >= 3
        assert len(task.evaluation_instructions) >= 3
        assert not set(task.training_instructions) & set(task.evaluation_instructions)
        assert task.instruction_for_seed(0) in task.training_instructions
        assert task.instruction_for_seed(0, evaluation=True) in (
            task.evaluation_instructions
        )
        for name, training_range in task.randomization.ranges().items():
            evaluation_range = task.evaluation_randomization.ranges()[name]
            assert evaluation_range != training_range
            assert (
                evaluation_range.maximum - evaluation_range.minimum
                > training_range.maximum - training_range.minimum
            )
