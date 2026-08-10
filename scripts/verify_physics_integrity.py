"""Reject runtime state teleportation and object weld constraints."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from hwr.adapters.mujoco.model import MujocoModelBundle


RESET_STATE_WRITE_ALLOWLIST = {
    ("src/hwr/adapters/mujoco/backend.py", "_reset_base", "qpos"),
    ("src/hwr/adapters/mujoco/backend.py", "_reset_base", "qvel"),
    ("src/hwr/adapters/mujoco/backend.py", "_reset_arm", "qpos"),
    ("src/hwr/adapters/mujoco/backend.py", "_reset_object", "qpos"),
    ("src/hwr/adapters/mujoco/backend.py", "_reset_object", "qvel"),
    ("src/hwr/adapters/mujoco/scene_preview.py", "_reset_preview_robot", "qpos"),
    ("src/hwr/adapters/mujoco/household_backend.py", "_reset_base", "qpos"),
    ("src/hwr/adapters/mujoco/household_backend.py", "_reset_base", "qvel"),
    ("src/hwr/adapters/mujoco/household_backend.py", "_reset_object", "qpos"),
    ("src/hwr/adapters/mujoco/household_backend.py", "_reset_object", "qvel"),
    ("src/hwr/adapters/mujoco/household_backend.py", "_reset_articulation", "qpos"),
    ("src/hwr/adapters/mujoco/household_backend.py", "_reset_articulation", "qvel"),
    ("src/hwr/adapters/mujoco/dual_arm_backend.py", "_reset_base", "qpos"),
    ("src/hwr/adapters/mujoco/dual_arm_backend.py", "_reset_base", "qvel"),
    ("src/hwr/adapters/mujoco/dual_arm_backend.py", "_reset_arms", "qpos"),
    ("src/hwr/adapters/mujoco/dual_arm_backend.py", "_reset_object", "qpos"),
    ("src/hwr/adapters/mujoco/dual_arm_backend.py", "_reset_object", "qvel"),
    ("src/hwr/adapters/mujoco/dual_arm_backend.py", "_reset_state_snapshot", "qpos"),
    ("src/hwr/adapters/mujoco/dual_arm_backend.py", "_reset_state_snapshot", "qvel"),
    ("src/hwr/adapters/mujoco/bimanual_backend.py", "_reset_base", "qpos"),
    ("src/hwr/adapters/mujoco/bimanual_backend.py", "_reset_base", "qvel"),
    ("src/hwr/adapters/mujoco/bimanual_backend.py", "_reset_object", "qpos"),
    ("src/hwr/adapters/mujoco/bimanual_backend.py", "_reset_object", "qvel"),
}


@dataclass(frozen=True)
class StateWriteViolation:
    path: str
    line: int
    function: str
    field: str


class _StateWriteVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.functions: list[str] = []
        self.violations: list[StateWriteViolation] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def _check_target(self, target: ast.expr) -> None:
        field = _engine_state_field(target)
        if field is None:
            return
        function = self.functions[-1] if self.functions else "<module>"
        key = (self.path.as_posix(), function, field)
        if key in RESET_STATE_WRITE_ALLOWLIST:
            return
        self.violations.append(
            StateWriteViolation(str(self.path), target.lineno, function, field)
        )


def _engine_state_field(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, ast.Subscript):
        current = current.value
    if not isinstance(current, ast.Attribute):
        return None
    if current.attr not in {"qpos", "qvel", "xpos", "xquat"}:
        return None
    return current.attr


def find_engine_state_write_violations(root: Path) -> tuple[StateWriteViolation, ...]:
    adapter_root = root / "src" / "hwr" / "adapters" / "mujoco"
    violations: list[StateWriteViolation] = []
    for path in sorted(adapter_root.rglob("*.py")):
        relative = path.relative_to(root)
        visitor = _StateWriteVisitor(relative)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        violations.extend(visitor.violations)
    return tuple(violations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("assets/mujoco/mobile_manipulator_smoke.xml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    model_path = arguments.model if arguments.model.is_absolute() else root / arguments.model
    bundle = MujocoModelBundle.load(model_path)
    violations = find_engine_state_write_violations(root)
    value: dict[str, object] = {
        "schema_version": "hwr.physics-integrity-report/v1",
        "model_path": str(bundle.model_path),
        "equality_constraint_count": bundle.model.neq,
        "runtime_state_write_violations": [asdict(item) for item in violations],
        "valid": bundle.model.neq == 0 and not violations,
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
