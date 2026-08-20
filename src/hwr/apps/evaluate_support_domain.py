"""Create the frozen R0001-P36-E1 support-domain evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.eval.support_domain import (
    EXPERIMENT_ID,
    P11_MANIFEST_IDENTITY,
    P11_REPORT_IDENTITY,
    P29_MANIFEST_IDENTITY,
    P29_REPORT_IDENTITY,
    evaluate_support_domain,
)


MANIFEST_SCHEMA = "hwr.support-domain-artifacts/v1"
MODULE_NAME = "hwr.apps.evaluate_support_domain"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p11-run", type=Path, required=True)
    parser.add_argument("--p29-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    p11_run = _resolve(root, Path(arguments.p11_run))
    p29_run = _resolve(root, Path(arguments.p29_run))
    output = _resolve(root, Path(arguments.output))
    _validate_paths(p11_run, p29_run, output)
    source_commit = _source_commit(root)
    report = evaluate_support_domain(p11_run, p29_run)
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--p11-run",
        str(p11_run),
        "--p29-run",
        str(p29_run),
        "--output",
        str(output),
    ]
    report["source_commit"] = source_commit
    report["invocation"] = {
        "module": MODULE_NAME,
        "command": command,
        "p11_run": str(p11_run),
        "p29_run": str(p29_run),
        "output": str(output),
    }
    report_bytes = _json_bytes(report)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "proposal_id": "R0001-P36",
        "source_commit": source_commit,
        "command": command,
        "inputs": {
            "p11_report": dict(P11_REPORT_IDENTITY),
            "p11_manifest": dict(P11_MANIFEST_IDENTITY),
            "p29_report": dict(P29_REPORT_IDENTITY),
            "p29_manifest": dict(P29_MANIFEST_IDENTITY),
        },
        "artifacts": {
            "report.json": {
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
                "bytes": len(report_bytes),
            }
        },
    }
    manifest_bytes = _json_bytes(manifest)
    _create_output(output, report_bytes, manifest_bytes)
    return {
        "output": str(output),
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "report_bytes": manifest["artifacts"]["report.json"]["bytes"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_bytes": len(manifest_bytes),
    }


def _validate_paths(p11_run: Path, p29_run: Path, output: Path) -> None:
    if not p11_run.is_dir() or not p29_run.is_dir():
        raise ValueError("P36 input run path must be an existing directory")
    if p11_run == p29_run:
        raise ValueError("P36 input run paths must be distinct")
    if output.exists():
        raise FileExistsError(output)
    for run in (p11_run, p29_run):
        if output == run or output.is_relative_to(run) or run.is_relative_to(output):
            raise ValueError("P36 output must not overlap an input run")


def _create_output(
    output: Path, report_bytes: bytes, manifest_bytes: bytes
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        _atomic_write(output / "report.json", report_bytes)
        _atomic_write(output / "manifest.json", manifest_bytes)
    except BaseException:
        for name in ("report.json.tmp", "manifest.json.tmp", "report.json", "manifest.json"):
            path = output / name
            if path.exists():
                path.unlink()
        output.rmdir()
        raise


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_commit(root: Path) -> str:
    git_path = root / ".git"
    if git_path.is_file():
        marker = git_path.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise RuntimeError("P36 source Git metadata is invalid")
        git_path = (root / marker.removeprefix("gitdir: ")).resolve()
    head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = head.removeprefix("ref: ")
        loose = git_path / reference
        head = (
            loose.read_text(encoding="utf-8").strip()
            if loose.is_file()
            else _packed_reference(git_path, reference)
        )
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("P36 requires a full Git source commit")
    return head


def _packed_reference(git_path: Path, reference: str) -> str:
    for line in (git_path / "packed-refs").read_text(encoding="utf-8").splitlines():
        fields = line.split(" ", 1)
        if len(fields) == 2 and fields[1] == reference:
            return fields[0]
    raise RuntimeError("P36 source Git reference is unresolved")


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
