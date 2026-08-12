"""Enforce third-party engine dependency boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


ALLOWED_MUJOCO_PREFIX = Path("src/hwr/adapters/mujoco")
ALLOWED_FOUNDATION_PREFIX = Path("src/hwr/adapters/foundation")
FOUNDATION_MODULES = ("transformers", "huggingface_hub", "timm", "mlx")
CORE_ROOT = Path("src/hwr/core")
FORBIDDEN_CORE_PREFIXES = (
    "hwr.adapters",
    "hwr.apps",
    "hwr.data",
    "hwr.eval",
    "hwr.perception",
    "hwr.policy",
    "hwr.render",
    "hwr.safety",
    "hwr.scenarios",
    "hwr.sim",
    "hwr.train",
)


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


def find_foundation_import_violations(root: Path) -> tuple[tuple[Path, str], ...]:
    """Keep third-party model runtimes behind foundation adapters."""
    source_root = root / "src" / "hwr"
    violations: list[tuple[Path, str]] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.is_relative_to(ALLOWED_FOUNDATION_PREFIX):
            continue
        for module in _imported_modules(path):
            if any(module == name or module.startswith(name + ".") for name in FOUNDATION_MODULES):
                violations.append((relative, module))
    return tuple(violations)


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


def find_core_dependency_violations(root: Path) -> tuple[tuple[Path, str], ...]:
    violations: list[tuple[Path, str]] = []
    for path in sorted((root / CORE_ROOT).rglob("*.py")):
        for module in _imported_modules(path):
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in FORBIDDEN_CORE_PREFIXES
            ):
                violations.append((path.relative_to(root), module))
    return tuple(violations)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    engine_violations = find_mujoco_import_violations(root)
    if engine_violations:
        print("MuJoCo imports escaped the adapter boundary:")
        for path in engine_violations:
            print(f"- {path}")
        return 1
    core_violations = find_core_dependency_violations(root)
    if core_violations:
        print("Core schemas import an outward platform layer:")
        for path, module in core_violations:
            print(f"- {path}: {module}")
        return 1
    foundation_violations = find_foundation_import_violations(root)
    if foundation_violations:
        print("Foundation runtime imports escaped the adapter boundary:")
        for path, module in foundation_violations:
            print(f"- {path}: {module}")
        return 1
    print("Architecture check passed: engine, foundation, and core boundaries are intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
