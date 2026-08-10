from __future__ import annotations

from pathlib import Path

from scripts.check_architecture import (
    find_core_dependency_violations,
    find_mujoco_import_violations,
)
from scripts.verify_physics_integrity import find_engine_state_write_violations


def test_mujoco_dependency_is_confined_to_adapter() -> None:
    root = Path(__file__).resolve().parents[1]

    assert find_mujoco_import_violations(root) == ()


def test_core_schemas_do_not_depend_on_outward_layers() -> None:
    root = Path(__file__).resolve().parents[1]

    assert find_core_dependency_violations(root) == ()


def test_core_dependency_scanner_detects_outward_import(tmp_path) -> None:
    core = tmp_path / "src/hwr/core"
    core.mkdir(parents=True)
    (core / "leaky.py").write_text("from hwr.train.trainer import TrainingResult\n")

    assert find_core_dependency_violations(tmp_path) == (
        (Path("src/hwr/core/leaky.py"), "hwr.train.trainer"),
    )


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
