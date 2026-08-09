from __future__ import annotations

from pathlib import Path

from scripts.check_architecture import find_mujoco_import_violations


def test_mujoco_dependency_is_confined_to_adapter() -> None:
    root = Path(__file__).resolve().parents[1]

    assert find_mujoco_import_violations(root) == ()
