from __future__ import annotations

from pathlib import Path

from scripts.check_architecture import find_mujoco_import_violations
from scripts.verify_physics_integrity import find_engine_state_write_violations


def test_mujoco_dependency_is_confined_to_adapter() -> None:
    root = Path(__file__).resolve().parents[1]

    assert find_mujoco_import_violations(root) == ()


def test_mujoco_runtime_does_not_teleport_engine_state() -> None:
    root = Path(__file__).resolve().parents[1]

    assert find_engine_state_write_violations(root) == ()
