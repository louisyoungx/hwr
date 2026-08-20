"""Run the frozen R0001-P29 stale-observation validity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.eval.stale_observation_validity import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_REPORT_SHA256,
    evaluate_stale_observation_validity,
)


DEFAULT_P11_RUN = Path(
    "runs/research-loop/0003/r0003-p11-causal-plant-s20261101"
)
DEFAULT_OUTPUT = Path(
    "runs/research-loop/0004/r0004-p29-stale-validity-s20262901"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p11-run", type=Path, default=DEFAULT_P11_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    _require_clean_source(root)
    source_commit = _source_commit(root)
    p11_run = _resolve(root, Path(arguments.p11_run))
    output = _resolve(root, Path(arguments.output))
    _require_frozen_invocation(root, p11_run, output)
    if output.exists():
        raise FileExistsError(output)
    report = evaluate_stale_observation_validity(p11_run)
    report.update(
        {
            "source_commit": source_commit,
            "input": {
                "p11_run": str(p11_run),
                "p11_report_sha256": EXPECTED_REPORT_SHA256,
                "p11_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            },
            "invocation": {
                "module": "hwr.apps.evaluate_stale_observation_validity",
                "p11_run": str(p11_run),
                "output": str(output),
            },
        }
    )
    output.mkdir(parents=True)
    _write_json(output / "report.json", report)
    _write_manifest(output, source_commit)
    return {
        "output": str(output),
        "decision": report["assessment"]["decision"],
        "report_sha256": _sha256(output / "report.json"),
        "manifest_sha256": _sha256(output / "manifest.json"),
    }


def _require_frozen_invocation(
    root: Path, p11_run: Path, output: Path
) -> None:
    if (p11_run, output) != (root / DEFAULT_P11_RUN, root / DEFAULT_OUTPUT):
        raise ValueError("P29 invocation differs from frozen experiment")


def _write_manifest(output: Path, source_commit: str) -> None:
    report = output / "report.json"
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "hwr.stale-observation-validity-artifacts/v1",
            "proposal_id": "R0001-P29",
            "source_commit": source_commit,
            "artifacts": {
                "report.json": {
                    "sha256": _sha256(report),
                    "bytes": report.stat().st_size,
                }
            },
        },
    )


def _source_commit(root: Path) -> str:
    value = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise RuntimeError("P29 diagnostic requires a Git source commit")
    return value


def _require_clean_source(root: Path) -> None:
    value = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if value:
        raise RuntimeError("P29 diagnostic requires clean committed source")


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
