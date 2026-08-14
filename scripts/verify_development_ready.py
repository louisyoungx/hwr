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
    COMMITTED_SNAPSHOT_CHECKS,
    DEVELOPMENT_READY_SCHEMA,
    PROTECTED_PATHS,
    REQUIRED_DEVELOPMENT_CHECKS,
    current_commit,
    foundation_config_hashes,
    protected_tree_hashes,
)
from hwr.train.foundation_setup import build_foundation_learning_stack
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_resource_budget import foundation_storage_estimate


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
FOUNDATION_ALGORITHM_PATTERNS = (
    "src/hwr/adapters/foundation/*.py",
    "src/hwr/apps/*foundation*.py",
    "src/hwr/data/autonomous_trajectory.py",
    "src/hwr/data/foundation_*.py",
    "src/hwr/data/trajectory_windows.py",
    "src/hwr/perception/foundation.py",
    "src/hwr/perception/geometric_correspondence.py",
    "src/hwr/perception/high_resolution.py",
    "src/hwr/perception/language_cache.py",
    "src/hwr/perception/student*.py",
    "src/hwr/policy/foundation_runtime.py",
    "src/hwr/policy/latent_*.py",
    "src/hwr/safety/*.py",
    "src/hwr/train/foundation_*.py",
    "src/hwr/train/imagination*.py",
    "src/hwr/train/learning_signals.py",
    "src/hwr/train/task_sampling.py",
    "src/hwr/world_model/*.py",
)
EXPECTED_TASK_IDS = frozenset(
    {
        "carry_living_room_basket/v1",
        "carry_dining_tray/v1",
        "hold_drawer_place_item/v1",
    }
)
FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "expert",
        "demonstration",
        "behavior_clone",
        "teacher_action",
        "action_label",
        "waypoint",
        "skill",
        "task_stage",
        "object_token",
        "target_token",
        "action_search",
        "legacy_checkpoint",
    }
)
FORBIDDEN_DEPLOYMENT_NAMES = frozenset(
    {
        "reward",
        "continue",
        "safety",
        "critic",
        "value",
        "teacher",
        "objective",
        "optimizer",
        "augmentation",
        "action_execution",
    }
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
    paths = _foundation_algorithm_paths(root)
    for relative in paths:
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
                if _forbidden_foundation_import(module):
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
        "files": list(paths),
        "file_count": len(paths),
        "task_literals": False,
        "expert_imports": False,
        "scene_training_branches": False,
    }


def _foundation_algorithm_paths(root: Path) -> tuple[str, ...]:
    paths: set[Path] = set()
    for pattern in FOUNDATION_ALGORITHM_PATTERNS:
        matches = tuple(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise RuntimeError(f"foundation algorithm pattern is empty: {pattern}")
        paths.update(matches)
    return tuple(str(path.relative_to(root)) for path in sorted(paths))


def _forbidden_foundation_import(module: str) -> bool:
    if any(
        module == forbidden or module.startswith(forbidden + ".")
        for forbidden in FORBIDDEN_FOUNDATION_IMPORTS
    ):
        return True
    return any(
        part == "expert"
        or part.startswith("expert_")
        or part.endswith("_expert")
        or "_expert_" in part
        for part in module.lower().split(".")
    )


def _forbidden_configuration_keys(root: Path) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted((root / "configs/foundation").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        _scan_configuration_keys(value, path.name, violations)
    return tuple(violations)


def _scan_configuration_keys(
    value: object, location: str, violations: list[str]
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(
                normalized == forbidden
                or normalized.startswith(forbidden + "_")
                or normalized.endswith("_" + forbidden)
                for forbidden in FORBIDDEN_CONFIG_KEYS
            ):
                violations.append(f"{location}:{key}")
            _scan_configuration_keys(item, f"{location}.{key}", violations)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_configuration_keys(item, f"{location}[{index}]", violations)


def _configuration_audit(root: Path) -> dict[str, Any]:
    tasks = load_bimanual_task_specs(
        root / "configs/tasks/bimanual_household_v1.json"
    )
    if set(tasks) != EXPECTED_TASK_IDS:
        raise RuntimeError("formal configuration does not expose the three required tasks")
    lineage_violations = _forbidden_configuration_keys(root)
    if lineage_violations:
        raise RuntimeError(
            "foundation configuration contains forbidden lineage keys: "
            + ", ".join(lineage_violations)
        )
    stack = build_foundation_learning_stack(
        root / "configs/foundation", seed=0
    )
    online = json.loads(
        (root / "configs/foundation/online-training-v1.json").read_text(
            encoding="utf-8"
        )
    )
    online_config = FoundationOnlineTrainingConfig(
        **{name: value for name, value in online.items() if name != "schema_version"}
    )
    storage = foundation_storage_estimate(online_config, task_count=len(tasks))
    if storage["within_configured_budget"] is not True:
        raise RuntimeError("formal foundation storage estimate exceeds its budget")
    if float(online["minimum_action_causality_ratio"]) <= 1.0:
        raise RuntimeError("formal action causality ratio is not a degradation gate")
    if float(online["minimum_action_causality_horizon_fraction"]) < 0.5:
        raise RuntimeError("formal action causality horizon gate is too weak")
    if float(online["minimum_collision_validation_action_sensitivity_ratio"]) <= 1.0:
        raise RuntimeError("formal collision validation is not action-sensitive")
    holdout_episodes = int(online["causality_holdout_episodes_per_task"])
    audit_windows = int(online["causality_audit_windows_per_task"])
    audit_batch = int(online["causality_audit_batch_size"])
    if holdout_episodes < 2:
        raise RuntimeError("formal causality audit needs two holdout Episodes per task")
    if audit_windows < 8:
        raise RuntimeError("formal causality audit has too few windows per task")
    if audit_batch <= 0 or audit_windows % audit_batch:
        raise RuntimeError("formal causality audit batch does not partition task windows")
    if int(online["batch_size"]) > 2:
        raise RuntimeError("formal visual batch exceeds the verified 48 GB envelope")
    motion_correlation = float(
        online["random_exploration_motion_correlation"]
    )
    gripper_flip_probability = float(
        online["random_exploration_gripper_flip_probability"]
    )
    if motion_correlation < 0.90:
        raise RuntimeError("formal random RL motion lacks temporal persistence")
    if not 0.0 < gripper_flip_probability <= 0.10:
        raise RuntimeError("formal random RL gripper dwell is invalid")
    learning_signal_windows = int(online["learning_signal_windows_per_episode"])
    if learning_signal_windows < 2:
        raise RuntimeError("formal Episode learning signals have too few windows")
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
    if any(
        forbidden in name.lower()
        for name in deployment_names
        for forbidden in FORBIDDEN_DEPLOYMENT_NAMES
    ):
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
        "forbidden_lineage_keys": False,
        "action_causality_ratio": online["minimum_action_causality_ratio"],
        "action_causality_horizon_fraction": online[
            "minimum_action_causality_horizon_fraction"
        ],
        "causality_holdout_episodes_per_task": holdout_episodes,
        "causality_audit_windows_per_task": audit_windows,
        "causality_audit_batch_size": audit_batch,
        "formal_batch_size": online["batch_size"],
        "visual_microbatch_observations": trainer.config.visual_microbatch_observations,
        "random_exploration_motion_correlation": motion_correlation,
        "random_exploration_gripper_flip_probability": (
            gripper_flip_probability
        ),
        "learning_signal_windows_per_episode": learning_signal_windows,
        "replay_windows_per_episode": online_config.replay_windows_per_episode,
        "estimated_run_storage_gib": storage["estimated_gib"],
        "holdout_teacher_visual_features": storage["holdout"][
            "teacher_visual_features"
        ],
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


def _model_selection_audit(root: Path, model_root: Path) -> dict[str, Any]:
    """Require requested providers, committed locks, and runtime names to agree."""
    locks = load_foundation_model_locks(
        root / "configs/foundation/model-locks.json", model_root
    )
    sources = json.loads(
        (root / "configs/foundation/model-sources.json").read_text(encoding="utf-8")
    )["models"]
    runtime = json.loads(
        (root / "configs/foundation/runtime-v1.json").read_text(encoding="utf-8")
    )
    requested = {str(value["local_name"]): value for value in sources}
    configured = {
        str(runtime["dense_vision_model"]),
        str(runtime["vision_language_model"]),
        str(runtime["language_model"]),
    }
    if set(requested) != set(locks) or configured != set(locks):
        raise RuntimeError("foundation source, lock, and runtime model sets differ")
    fields = (
        "adapter",
        "model_id",
        "revision",
        "role",
        "license_id",
        "output_dimension",
        "representation_id",
    )
    for name, source in requested.items():
        locked = locks[name]
        actual = {
            "adapter": locked.adapter,
            **{
                field: getattr(locked.model_lock, field)
                for field in fields
                if field != "adapter"
            },
        }
        expected = {field: source[field] for field in fields}
        artifacts = {
            value.relative_path for value in locked.model_lock.artifacts
        }
        expected_artifacts = {
            str(Path(name) / str(value)) for value in source["required_files"]
        }
        if actual != expected or artifacts != expected_artifacts:
            raise RuntimeError(f"foundation source and lock differ for {name}")
    return {"passed": True, "models": sorted(locks)}


def _foundation_dependency_audit() -> dict[str, Any]:
    """Fail before weight access when the DINOv3 processor runtime is unusable."""
    import torch
    import torchvision
    import transformers
    from transformers.models.dinov3_vit.image_processing_dinov3_vit_fast import (
        DINOv3ViTImageProcessorFast,
    )

    torch_version = tuple(int(value) for value in torch.__version__.split("+")[0].split(".")[:2])
    vision_version = tuple(
        int(value) for value in torchvision.__version__.split("+")[0].split(".")[:2]
    )
    transformers_version = tuple(
        int(value) for value in transformers.__version__.split("+")[0].split(".")[:2]
    )
    if torch_version != (2, 13) or vision_version != (0, 28):
        raise RuntimeError("DINOv3 requires Torch 2.13.x with torchvision 0.28.x")
    if transformers_version != (4, 57):
        raise RuntimeError("DINOv3 requires Transformers 4.57.x")
    processor = DINOv3ViTImageProcessorFast()
    if processor.__class__.__name__ != "DINOv3ViTImageProcessorFast":
        raise RuntimeError("DINOv3 fast image processor is unavailable")
    return {
        "passed": True,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "dinov3_image_processor": processor.__class__.__name__,
    }


def _foundation_inference_checks(
    root: Path, model_root: Path, device: str
) -> dict[str, Any]:
    reports = {}
    output_root = root / "artifacts/foundation/development-gate"
    for name in (
        "dinov3-vits16-pretrain-lvd1689m",
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
        "model_selection": _model_selection_audit(root, model_root),
        "foundation_dependencies": _foundation_dependency_audit(),
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
        for name in COMMITTED_SNAPSHOT_CHECKS:
            checks[name]["source_commit"] = current_commit(snapshot)
    checks["foundation_inference"] = _foundation_inference_checks(
        root, model_root, foundation_device
    )
    if set(checks) != REQUIRED_DEVELOPMENT_CHECKS:
        raise RuntimeError("development verifier omitted a mandatory check")
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
