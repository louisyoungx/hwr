#!/usr/bin/env python3
"""Fetch pinned foundation models and generate an auditable local lock file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SOURCE_SCHEMA = "hwr.foundation-model-sources/v1"
LOCK_SCHEMA = "hwr.foundation-model-lock-set/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sources(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("foundation model source schema mismatch")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("foundation model sources must be non-empty")
    names = [str(value.get("local_name", "")) for value in models]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("foundation model local names must be unique")
    return models


def _download(model: Mapping[str, Any], root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("install the foundation optional dependencies first") from error
    local = root / str(model["local_name"])
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(model["model_id"]),
        revision=str(model["revision"]),
        local_dir=local,
        allow_patterns=[str(value) for value in model["required_files"]],
    )
    return local


def _lock_model(model: Mapping[str, Any], local: Path, root: Path) -> dict[str, Any]:
    artifacts = []
    for filename in model["required_files"]:
        path = local / str(filename)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required foundation file is missing: {path}")
        artifacts.append(
            {
                "relative_path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "adapter": model["adapter"],
        "local_name": model["local_name"],
        "model_id": model["model_id"],
        "revision": model["revision"],
        "role": model["role"],
        "license_id": model["license_id"],
        "output_dimension": int(model["output_dimension"]),
        "representation_id": model["representation_id"],
        "artifacts": artifacts,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=root / "configs/foundation/model-sources.json"
    )
    parser.add_argument("--model-root", type=Path, default=root / "models/foundation")
    parser.add_argument(
        "--lock-output", type=Path, default=root / "configs/foundation/model-locks.json"
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    sources = _load_sources(arguments.sources)
    locks = []
    for model in sources:
        local = arguments.model_root / str(model["local_name"])
        if not arguments.verify_only:
            local = _download(model, arguments.model_root)
        locks.append(_lock_model(model, local, arguments.model_root))
        print(f"locked {model['model_id']} at {model['revision']}")
    payload = {"schema_version": LOCK_SCHEMA, "models": locks}
    _write_json_atomic(arguments.lock_output, payload)
    print(arguments.lock_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
