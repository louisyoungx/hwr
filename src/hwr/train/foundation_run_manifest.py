"""Immutable provenance manifest for one formal foundation training run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from hwr.train.development_gate import DEVELOPMENT_READY_SCHEMA
from hwr.train.foundation_registry import foundation_lineage


FOUNDATION_RUN_SCHEMA = "hwr.foundation-online-run/v3"


def write_or_verify_foundation_run_manifest(
    run_path: Path,
    *,
    source_commit: str,
    development_ready_sha256: str,
    training_config: Mapping[str, object],
    tasks: Sequence[Mapping[str, object]],
    preprocessing: Mapping[str, object],
) -> dict[str, object]:
    """Create one manifest, or require an exact match before resuming."""
    manifest: dict[str, object] = {
        "schema_version": FOUNDATION_RUN_SCHEMA,
        "source_commit": source_commit,
        "development_ready": {
            "schema_version": DEVELOPMENT_READY_SCHEMA,
            "sha256": development_ready_sha256,
            "path": "development-ready.json",
        },
        "training_config": dict(training_config),
        "tasks": [dict(task) for task in tasks],
        "preprocessing": dict(preprocessing),
        "lineage": foundation_lineage(source_commit),
    }
    path = run_path / "run-manifest.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("foundation run manifest differs on resume")
        return manifest
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return manifest
