#!/usr/bin/env python3
"""Run the one hard development gate and atomically unlock formal training."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hwr.adapters.foundation import load_foundation_model_locks
from hwr.tasks import load_bimanual_task_specs
from hwr.train.development_gate import (
    DEVELOPMENT_READY_SCHEMA,
    PROTECTED_PATHS,
    current_commit,
    foundation_config_hashes,
    protected_tree_hashes,
)
from hwr.train.foundation_setup import build_foundation_learning_stack


FORBIDDEN_FOUNDATION_IMPORTS = (
    "hwr.adapters.mujoco.expert",
    "hwr.adapters.mujoco.formal_expert",
    "hwr.scenarios.expert",
    "hwr.policy.vla_model",
    "hwr.policy.visual_knn",
)
FORBIDDEN_TASK_LITERALS = (
    "carry_living_room_basket",
    "carry_dining_tray",
    "hold_drawer_place_item",
    "basket",
    "tray",
    "drawer",
    "收纳篮",
    "托盘",
    "抽屉",
)
FOUNDATION_ALGORITHM_PATHS = (
    "src/hwr/train/foundation_augmentation.py",
    "src/hwr/train/foundation_batch.py",
    "src/hwr/train/foundation_collection.py",
    "src/hwr/train/foundation_online.py",
    "src/hwr/train/foundation_setup.py",
    "src/hwr/train/foundation_trainer.py",
    "src/hwr/train/imagination.py",
    "src/hwr/train/imagination_rl.py",
    "src/hwr/policy/foundation_runtime.py",
)


@contextmanager
def _committed_snapshot(root: Path):
    """Expose HEAD as an isolated worktree for reproducible repository checks."""
    with tempfile.TemporaryDirectory(prefix="hwr-development-ready-") as temporary:
        snapshot = Path(temporary) / "source"
        subprocess.run(
            (
                "git",
                "worktree",
                "add",
                "--detach",
                "--quiet",
                str(snapshot),
                "HEAD",
            ),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            yield snapshot
        finally:
            subprocess.run(
                ("git", "worktree", "remove", "--force", str(snapshot)),
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )


def _command(
    root: Path, command: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, **(environment or {})},
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode:
        raise RuntimeError(
            f"development check failed ({' '.join(command)}):\n{output[-8000:]}"
        )
    return {
        "passed": True,
        "command": list(command),
        "output_tail": output[-2000:],
    }


def _protected_tree_clean(root: Path) -> dict[str, Any]:
    tracked = subprocess.run(
        ("git", "diff", "--name-only", "HEAD", "--", *PROTECTED_PATHS),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ("git", "ls-files", "--others", "--exclude-standard", "--", *PROTECTED_PATHS),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    changed = sorted({*tracked, *untracked})
    if changed:
        raise RuntimeError(
            "development gate requires committed protected files: " + ", ".join(changed)
        )
    return {"passed": True, "protected_path_count": len(PROTECTED_PATHS)}


def _algorithm_audit(root: Path) -> dict[str, Any]:
    violations: list[str] = []
    for relative in FOUNDATION_ALGORITHM_PATHS:
        path = root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []
            for module in modules:
                if any(
                    module == forbidden or module.startswith(forbidden + ".")
                    for forbidden in FORBIDDEN_FOUNDATION_IMPORTS
                ):
                    violations.append(f"{relative}:{node.lineno}:import:{module}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                normalized = node.value.lower()
                for literal in FORBIDDEN_TASK_LITERALS:
                    if literal.lower() in normalized:
                        violations.append(
                            f"{relative}:{node.lineno}:task-literal:{literal}"
                        )
    if violations:
        raise RuntimeError("foundation algorithm audit failed: " + "; ".join(violations))
    return {
        "passed": True,
        "files": list(FOUNDATION_ALGORITHM_PATHS),
        "task_literals": False,
        "expert_imports": False,
        "scene_training_branches": False,
    }


def _configuration_audit(root: Path) -> dict[str, Any]:
    tasks = load_bimanual_task_specs(
        root / "configs/tasks/bimanual_household_v1.json"
    )
    if len(tasks) != 3:
        raise RuntimeError("formal configuration must expose exactly three tasks")
    stack = build_foundation_learning_stack(root / "configs/foundation")
    trainer = stack.trainer
    if trainer.world_model.config.action_dimension != 16:
        raise RuntimeError("formal world model action dimension is not canonical")
    if trainer.actor.config.latent_dimension != trainer.world_model.config.feature_dimension:
        raise RuntimeError("formal Actor and world-model latent dimensions differ")
    if trainer.imagination.action_scaling != stack.action_scaling:
        raise RuntimeError("imagined and runtime action units differ")
    deployment_names = {
        name
        for name, _ in __import__(
            "hwr.world_model.deploy", fromlist=["DeployableWorldModelStateFilter"]
        ).DeployableWorldModelStateFilter.from_world_model(
            trainer.world_model
        ).named_modules()
    }
    if any("reward" in name or "critic" in name for name in deployment_names):
        raise RuntimeError("deployment state filter contains training prediction heads")
    return {
        "passed": True,
        "task_count": len(tasks),
        "visual_student_parameters": sum(
            parameter.numel() for parameter in trainer.visual_student.parameters()
        ),
        "world_model_parameters": sum(
            parameter.numel() for parameter in trainer.world_model.parameters()
        ),
        "actor_parameters": sum(parameter.numel() for parameter in trainer.actor.parameters()),
        "action_dimension": trainer.world_model.config.action_dimension,
        "deployment_training_heads": False,
    }


def _weight_audit(root: Path, model_root: Path) -> dict[str, Any]:
    locks = load_foundation_model_locks(
        root / "configs/foundation/model-locks.json", model_root
    )
    errors = {}
    for name, locked in locks.items():
        failures = locked.model_lock.verify_local_files(model_root)
        if failures:
            errors[name] = list(failures)
    if errors:
        raise RuntimeError(f"foundation weight audit failed: {errors}")
    return {
        "passed": True,
        "models": {
            name: {
                "model_id": locked.model_lock.model_id,
                "revision": locked.model_lock.revision,
                "license_id": locked.model_lock.license_id,
                "lock_sha256": locked.model_lock.lock_sha256,
            }
            for name, locked in locks.items()
        },
    }


def _foundation_inference_checks(
    root: Path, model_root: Path, device: str
) -> dict[str, Any]:
    reports = {}
    output_root = root / "artifacts/foundation/development-gate"
    for name in (
        "dinov2-small",
        "siglip2-base-patch16-224",
        "qwen3-embedding-0.6b",
    ):
        output = output_root / f"{name}.json"
        reports[name] = _command(
            root,
            (
                sys.executable,
                "scripts/verify_foundation_models.py",
                name,
                "--device",
                device,
                "--model-root",
                str(model_root),
                "--output",
                str(output),
            ),
        )
    return {"passed": True, "providers": reports}


def verify(
    root: Path, model_root: Path, *, foundation_device: str
) -> dict[str, Any]:
    checks = {
        "protected_tree": _protected_tree_clean(root),
        "algorithm_audit": _algorithm_audit(root),
        "configuration": _configuration_audit(root),
        "weights": _weight_audit(root, model_root),
    }
    with _committed_snapshot(root) as snapshot:
        snapshot_environment = {
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(snapshot / "src"), os.environ.get("PYTHONPATH")))
            )
        }
        checks["architecture"] = _command(
            snapshot, (sys.executable, "scripts/check_architecture.py")
        )
        checks["python_size"] = _command(
            snapshot, (sys.executable, "scripts/check_python_size.py")
        )
        checks["tests"] = _command(
            snapshot,
            (sys.executable, "-m", "pytest", "-q"),
            environment={
                **snapshot_environment,
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
        )
        for name in ("architecture", "python_size", "tests"):
            checks[name]["source_commit"] = current_commit(snapshot)
    checks["foundation_inference"] = _foundation_inference_checks(
        root, model_root, foundation_device
    )
    return {
        "schema_version": DEVELOPMENT_READY_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": current_commit(root),
        "foundation_config_sha256": foundation_config_hashes(root),
        "protected_tree_sha256": protected_tree_hashes(root),
        "checks": checks,
        "training_unlocked": True,
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts/development-ready.json"
    )
    parser.add_argument(
        "--model-root", type=Path, default=root / "models/foundation"
    )
    parser.add_argument("--foundation-device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    report = verify(
        root, arguments.model_root.resolve(), foundation_device=arguments.foundation_device
    )
    _write_atomic(arguments.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
