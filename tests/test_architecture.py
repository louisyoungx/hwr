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


def test_state_write_scanner_cannot_be_bypassed_by_reset_like_name(tmp_path) -> None:
    adapter = tmp_path / "src/hwr/adapters/mujoco"
    adapter.mkdir(parents=True)
    (adapter / "cheat.py").write_text(
        "def _reset_and_teleport(data):\n    data.qpos[0] = 9.0\n",
        encoding="utf-8",
    )

    violations = find_engine_state_write_violations(tmp_path)

    assert len(violations) == 1
    assert violations[0].function == "_reset_and_teleport"
