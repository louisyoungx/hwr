"""Enforce third-party engine dependency boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


ALLOWED_MUJOCO_PREFIX = Path("src/hwr/adapters/mujoco")


def _imports_mujoco(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "mujoco" or alias.name.startswith("mujoco.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "mujoco" or (node.module or "").startswith("mujoco."):
                return True
    return False


def find_mujoco_import_violations(root: Path) -> tuple[Path, ...]:
    source_root = root / "src" / "hwr"
    violations: list[Path] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.is_relative_to(ALLOWED_MUJOCO_PREFIX):
            continue
        if _imports_mujoco(path):
            violations.append(relative)
    return tuple(violations)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = find_mujoco_import_violations(root)
    if violations:
        print("MuJoCo imports escaped the adapter boundary:")
        for path in violations:
            print(f"- {path}")
        return 1
    print("Architecture check passed: MuJoCo imports are confined to hwr.adapters.mujoco")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
