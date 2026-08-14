"""Validate the immutable development-ready report before formal training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DEVELOPMENT_READY_SCHEMA = "hwr.foundation-development-ready/v3"
REQUIRED_DEVELOPMENT_CHECKS = frozenset(
    {
        "protected_tree",
        "algorithm_audit",
        "configuration",
        "model_selection",
        "foundation_dependencies",
        "weights",
        "architecture",
        "python_size",
        "tests",
        "foundation_inference",
        "training_semantics",
    }
)
COMMITTED_SNAPSHOT_CHECKS = frozenset(
    {"architecture", "python_size", "tests", "training_semantics"}
)
FOUNDATION_CONFIG_FILES = (
    "imagination-rl-v1.json",
    "intrinsic-exploration-v1.json",
    "latent-actor-v1.json",
    "model-locks.json",
    "online-training-v1.json",
    "runtime-v1.json",
    "unified-trainer-v1.json",
    "visual-objective-v1.json",
    "visual-student-v1.json",
    "world-model-v1.json",
    "world-objective-v1.json",
)
PROTECTED_PATHS = (
    "assets/mujoco/formal",
    "assets/mujoco/bimanual",
    "configs/adapters/mujoco/formal_3d_v1.json",
    "configs/adapters/mujoco/bimanual_household_v1.json",
    "configs/foundation",
    "configs/tasks/bimanual_household_v1.json",
    "configs/tasks/formal_3d_v1.json",
    "pyproject.toml",
    "scripts/check_architecture.py",
    "scripts/check_python_size.py",
    "scripts/fetch_foundation_models.py",
    "scripts/run_training_with_lark_notify.sh",
    "scripts/send_lark_agent_message.sh",
    "scripts/start_foundation_training_tmux.sh",
    "scripts/verify_development_ready.py",
    "scripts/verify_foundation_models.py",
    "scripts/verify_training_semantics.py",
    "src/hwr/adapters/foundation",
    "src/hwr/adapters/mujoco/bimanual_backend.py",
    "src/hwr/adapters/mujoco/bimanual_bindings.py",
    "src/hwr/adapters/mujoco/dual_arm_backend.py",
    "src/hwr/adapters/mujoco/formal_household_backend.py",
    "src/hwr/adapters/mujoco/training_catalog.py",
    "src/hwr/apps/evaluate_foundation_world_model.py",
    "src/hwr/apps/serve_foundation_dashboard.py",
    "src/hwr/apps/train_foundation_world_model.py",
    "src/hwr/core/embodied.py",
    "src/hwr/core/runtime.py",
    "src/hwr/data/autonomous_trajectory.py",
    "src/hwr/data/foundation_cache.py",
    "src/hwr/data/foundation_features.py",
    "src/hwr/data/foundation_loading.py",
    "src/hwr/data/trajectory_windows.py",
    "src/hwr/eval/bimanual.py",
    "src/hwr/eval/foundation_causality.py",
    "src/hwr/perception/foundation.py",
    "src/hwr/perception/geometric_correspondence.py",
    "src/hwr/perception/high_resolution.py",
    "src/hwr/perception/language_cache.py",
    "src/hwr/perception/student.py",
    "src/hwr/perception/student_input.py",
    "src/hwr/perception/student_objectives.py",
    "src/hwr/policy/foundation_runtime.py",
    "src/hwr/policy/latent_actions.py",
    "src/hwr/policy/latent_actor.py",
    "src/hwr/policy/latent_value.py",
    "src/hwr/safety",
    "src/hwr/tasks/bimanual.py",
    "src/hwr/train/accelerator_memory.py",
    "src/hwr/train/development_gate.py",
    "src/hwr/train/development_semantics.py",
    "src/hwr/train/foundation_augmentation.py",
    "src/hwr/train/foundation_action_probe.py",
    "src/hwr/train/foundation_actor_readiness.py",
    "src/hwr/train/foundation_batch.py",
    "src/hwr/train/foundation_collection.py",
    "src/hwr/train/foundation_diagnostics.py",
    "src/hwr/train/foundation_exploration.py",
    "src/hwr/train/foundation_holdout.py",
    "src/hwr/train/foundation_learning_signals.py",
    "src/hwr/train/foundation_materialization.py",
    "src/hwr/train/foundation_metrics.py",
    "src/hwr/train/foundation_online.py",
    "src/hwr/train/foundation_online_config.py",
    "src/hwr/train/foundation_online_types.py",
    "src/hwr/train/foundation_recovery.py",
    "src/hwr/train/foundation_replay_features.py",
    "src/hwr/train/foundation_run_manifest.py",
    "src/hwr/train/foundation_registry.py",
    "src/hwr/train/foundation_setup.py",
    "src/hwr/train/foundation_trainer.py",
    "src/hwr/train/foundation_update_cycle.py",
    "src/hwr/train/foundation_visual_update.py",
    "src/hwr/train/imagination.py",
    "src/hwr/train/imagination_rl.py",
    "src/hwr/train/intrinsic_exploration.py",
    "src/hwr/train/task_sampling.py",
    "src/hwr/world_model",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def foundation_config_hashes(root: Path) -> dict[str, str]:
    config_root = root / "configs/foundation"
    return {name: sha256(config_root / name) for name in FOUNDATION_CONFIG_FILES}


def protected_tree_hashes(root: Path) -> dict[str, str]:
    files: set[Path] = set()
    for relative in PROTECTED_PATHS:
        path = root / relative
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
        else:
            raise FileNotFoundError(f"protected development path is missing: {relative}")
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(files)
        if "__pycache__" not in path.parts
    }


def require_development_ready(root: Path, report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise RuntimeError(
            "formal training is locked: development-ready report does not exist"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != DEVELOPMENT_READY_SCHEMA:
        raise RuntimeError("formal training is locked: development-ready schema differs")
    if report.get("source_commit") != current_commit(root):
        raise RuntimeError("formal training is locked: source commit changed after verification")
    if report.get("foundation_config_sha256") != foundation_config_hashes(root):
        raise RuntimeError("formal training is locked: foundation configuration changed")
    if report.get("protected_tree_sha256") != protected_tree_hashes(root):
        raise RuntimeError("formal training is locked: verified source tree changed")
    checks = report.get("checks", {})
    if (
        report.get("training_unlocked") is not True
        or not isinstance(checks, dict)
        or set(checks) != REQUIRED_DEVELOPMENT_CHECKS
        or not all(
            isinstance(value, dict) and value.get("passed") is True
            for value in checks.values()
        )
    ):
        raise RuntimeError("formal training is locked: development checks are incomplete")
    if any(
        checks[name].get("source_commit") != report["source_commit"]
        for name in COMMITTED_SNAPSHOT_CHECKS
    ):
        raise RuntimeError(
            "formal training is locked: committed-snapshot evidence differs"
        )
    return report
