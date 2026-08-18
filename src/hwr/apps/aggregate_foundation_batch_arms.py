"""Aggregate and verify the nine frozen R0001-P05 formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.eval.foundation_batch_arm_aggregate import (
    aggregate_foundation_batch_arms,
    load_batch_arm_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=9, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    run_paths = tuple(_resolve(root, value) for value in arguments.runs)
    reports = []
    manifests = {}
    for run_path in run_paths:
        manifests[str(run_path)] = _verify_manifest(run_path)
        reports.append(load_batch_arm_report(run_path / "report.json"))
    aggregate = aggregate_foundation_batch_arms(reports)
    aggregate["runs"] = manifests
    output = _resolve(root, arguments.output)
    if output.exists():
        raise FileExistsError(output)
    _write_json(output, aggregate)
    return {
        "output": str(output),
        "decision": aggregate["decision"],
        "report_sha256": _sha256(output),
    }


def _verify_manifest(run_path: Path) -> dict[str, object]:
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "hwr.foundation-batch-arm-artifacts/v1"
        or manifest.get("mode") != "formal"
    ):
        raise ValueError(f"batch-arm artifact manifest differs: {run_path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or "report.json" not in artifacts:
        raise ValueError(f"batch-arm artifact manifest is incomplete: {run_path}")
    for relative, identity in artifacts.items():
        path = run_path / str(relative)
        if (
            not isinstance(identity, Mapping)
            or not path.is_file()
            or path.stat().st_size != int(identity.get("bytes", -1))
            or _sha256(path) != identity.get("sha256")
        ):
            raise ValueError(f"batch-arm artifact identity differs: {path}")
    return {
        "manifest_sha256": _sha256(manifest_path),
        "report_sha256": artifacts["report.json"]["sha256"],
        "artifact_count": len(artifacts),
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
