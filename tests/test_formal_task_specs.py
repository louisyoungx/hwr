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
    assert tasks["store_kitchen_items_3d/v1"].articulation is not None
    assert "mujoco" not in path.read_text(encoding="utf-8").lower()
