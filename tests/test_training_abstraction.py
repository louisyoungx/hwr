from __future__ import annotations

import inspect
from pathlib import Path

from hwr.adapters.mujoco import load_default_bimanual_training_catalogs
from hwr.train.bimanual_training import BimanualTrainingRunner


ROOT = Path(__file__).resolve().parents[1]


def test_training_modules_contain_no_household_scene_branches() -> None:
    forbidden = (
        "carry_payload",
        "hold_drawer_place",
        "carry_dining_tray",
        "carry_living_room_basket",
        "hold_drawer_place_item",
    )
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src/hwr/train").glob("*.py"))
    )

    assert all(token not in sources for token in forbidden)


def test_action_selection_has_no_task_or_environment_argument() -> None:
    parameters = inspect.signature(
        BimanualTrainingRunner._select_action
    ).parameters

    assert "task_id" not in parameters
    assert "environment" not in parameters
    assert set(parameters) == {
        "self",
        "actor_input",
        "previous",
        "random_phase",
        "refresh_random",
    }


def test_scenes_declare_legal_transforms_as_data() -> None:
    tasks, _ = load_default_bimanual_training_catalogs(ROOT)

    assert any(task.legal_transforms for task in tasks.values())
    assert any(not task.legal_transforms for task in tasks.values())
    assert {
        transform
        for task in tasks.values()
        for transform in task.legal_transforms
    } == {"lateral_reflection"}
