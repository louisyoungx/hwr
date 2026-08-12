"""Validate the immutable development-ready report before formal training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DEVELOPMENT_READY_SCHEMA = "hwr.foundation-development-ready/v1"
FOUNDATION_CONFIG_FILES = (
    "imagination-rl-v1.json",
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
    checks = report.get("checks", {})
    if not checks or not all(value.get("passed") is True for value in checks.values()):
        raise RuntimeError("formal training is locked: development checks are incomplete")
    return report
