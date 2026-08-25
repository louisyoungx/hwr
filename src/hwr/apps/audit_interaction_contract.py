"""Audit the frozen R0001-P61 interaction and information-boundary contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.eval.interaction_contract import (
    PROPOSAL_ID,
    REPORT_SCHEMA,
    audit_interaction_contract,
)

MODULE_NAME = "hwr.apps.audit_interaction_contract"
MANIFEST_SCHEMA = "hwr.p61-interaction-contract-artifacts/v1"
FROZEN_DOCUMENT_COMMIT = "a95dbbeacc80a974d7a234f2dc79442249eaf07b"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0012/03-experiment.md")
FROZEN_CONTEXT_PATH = Path("docs/research-loop/0012/00-context.md")
FORMAL_CONTRACT = Path("configs/eval/interaction_contract_v1.json")
FORMAL_OUTPUT = Path(
    "runs/research-loop/0012/r0012-p61-interaction-contract-s20266101"
)
TASK_CONFIGURATION = Path("configs/tasks/formal_3d_v1.json")
BINDING_CONFIGURATION = Path("configs/adapters/mujoco/formal_3d_v1.json")
DIRECT_SOURCE_PATHS = (
    Path("src/hwr/eval/interaction_contract.py"),
    Path("src/hwr/apps/audit_interaction_contract.py"),
    Path("src/hwr/eval/target_selection.py"),
    Path("src/hwr/eval/stability.py"),
    Path("src/hwr/adapters/mujoco/formal_household_backend.py"),
    Path("src/hwr/adapters/mujoco/target_selection_diagnostic.py"),
    Path("src/hwr/adapters/mujoco/cartesian_convergence.py"),
    Path("src/hwr/scenarios/formal3d.py"),
    Path("src/hwr/adapters/mujoco/bindings.py"),
)
HISTORICAL_TREES = {
    "docs/research-loop/0001": "416912b7dc1c19611bcfc4375028180014a1989b",
    "docs/research-loop/0002": "6fb603dbd52451fe1749157daf05aa482ca7222f",
    "docs/research-loop/0003": "f56011eda321ea803bc24051db001e632c1549fb",
    "docs/research-loop/0004": "611c420e539a53a8c7578cd66aa8bdfe46fe82b7",
    "docs/research-loop/0005": "0352d379d5754adb03e9158c0fa72393ab322d58",
    "docs/research-loop/0006": "ee3a6f5b25887f67f812750d2a75424df12823d4",
    "docs/research-loop/0007": "0a696caa153abc9c13403fbc9bd3c081ce71c327",
    "docs/research-loop/0008": "65e626cddbcb0ec9c2e17cca5184b7d40950e1c6",
    "docs/research-loop/0009": "316db8b9ad9739ef491778f641603dbca25e75c9",
    "docs/research-loop/0010": "8a193a24788027d715750c3cd89c2509e71fdbda",
    "docs/research-loop/0011": "85bb445726ecb8e35ff4d8e90606874e2ee36fe4",
}
CLAIM_FLAGS = {
    "training_executed": False,
    "policy_inference_executed": False,
    "capability_claim_allowed": False,
    "task_success_claim_allowed": False,
    "generalization_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
    "entity_coverage_claim_allowed": False,
}
UNCHANGED_FLAGS = {
    "runtime_changed": False,
    "task_or_binding_changed": False,
    "candidate_changed": False,
    "selector_changed": False,
    "primitive_changed": False,
    "action_changed": False,
    "safety_changed": False,
    "reward_changed": False,
    "termination_changed": False,
}
WALL_TIME_LIMIT_SECONDS = 60.0
RSS_LIMIT_BYTES = 1024**3
ARTIFACT_LIMIT_BYTES = 10 * 1024**2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    contract_path = _resolve(root, arguments.contract)
    output = _resolve(root, arguments.output)
    if contract_path != (root / FORMAL_CONTRACT).resolve():
        raise ValueError("contract path differs from frozen formal contract")
    if output != (root / FORMAL_OUTPUT).resolve():
        raise ValueError("output path differs from frozen formal output")
    _require_unused_output(output)
    started_wall = time.time()
    started = time.perf_counter()
    tracemalloc.start()
    source_commit = _source_commit(root)
    provenance = _provenance(root, source_commit)
    _require_provenance(provenance)
    contract = _read_json(contract_path)
    tasks = _read_json(root / TASK_CONFIGURATION)
    bindings = _read_json(root / BINDING_CONFIGURATION)
    sources = _source_documents(root)
    first = audit_interaction_contract(contract, tasks, bindings, sources)
    second = audit_interaction_contract(contract, tasks, bindings, sources)
    deterministic = _canonical_bytes(first) == _canonical_bytes(second)
    if not deterministic:
        raise RuntimeError("P61 deterministic replay differs")
    transitions = first["transitions_document"]
    report = _report(
        source_commit,
        _command(arguments),
        first,
        deterministic=deterministic,
    )
    artifacts = {
        "transitions.json": _json_bytes(transitions),
        "report.json": _json_bytes(report),
    }
    elapsed = time.perf_counter() - started
    _, traced_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss_bytes = _peak_rss_bytes()
    _require_budget(
        elapsed,
        max(traced_peak_bytes, peak_rss_bytes),
        sum(len(content) for content in artifacts.values()),
    )
    manifest = _manifest(
        source_commit,
        _command(arguments),
        provenance,
        artifacts,
        report,
        started_wall=started_wall,
        elapsed_seconds=elapsed,
        traced_peak_bytes=traced_peak_bytes,
        peak_rss_bytes=peak_rss_bytes,
    )
    artifacts["manifest.json"] = _json_bytes(manifest)
    _require_budget(
        elapsed,
        max(traced_peak_bytes, peak_rss_bytes),
        sum(len(content) for content in artifacts.values()),
    )
    _create_output(output, artifacts)
    return {
        "output": str(output),
        "decision": report["decision"],
        "transition_count": transitions["transition_count"],
        "transitions_sha256": manifest["artifacts"]["transitions.json"]["sha256"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
    }


def _report(
    source_commit: str,
    command: Sequence[str],
    audit: Mapping[str, object],
    *,
    deterministic: bool,
) -> dict[str, object]:
    checks = dict(audit["checks"])
    checks["deterministic_reconstruction_bit_identical"] = deterministic
    checks["passed"] = bool(checks.get("passed")) and deterministic
    decision = str(audit["decision"]) if checks["passed"] else "invalid"
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "decision": decision,
        "validation_error": audit["validation_error"],
        "evidence_scope": (
            "static evaluator and primitive information contract only"
        ),
        "allowed_claim": (
            "whether the evaluator and generic primitive information contract "
            "contains a gap"
        ),
        "checks": checks,
        "full_task_contract": audit["full_task_contract"],
        "initial_microinteraction_contract": audit[
            "initial_microinteraction_contract"
        ],
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
    }


def _manifest(
    source_commit: str,
    command: Sequence[str],
    provenance: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    report: Mapping[str, object],
    *,
    started_wall: float,
    elapsed_seconds: float,
    traced_peak_bytes: int,
    peak_rss_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": "complete",
        "source_commit": source_commit,
        "command": list(command),
        "decision": report["decision"],
        "created_unix_seconds": started_wall,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": _package_version("mujoco"),
            "platform": platform.platform(),
        },
        "budgets": {
            "wall_time_limit_seconds": WALL_TIME_LIMIT_SECONDS,
            "elapsed_seconds": elapsed_seconds,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "tracemalloc_peak_bytes": traced_peak_bytes,
            "process_peak_rss_bytes": peak_rss_bytes,
            "artifact_limit_bytes": ARTIFACT_LIMIT_BYTES,
            "artifact_bytes_before_manifest": sum(
                len(content) for content in artifacts.values()
            ),
        },
        "provenance": dict(provenance),
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
        "artifacts": {
            name: _bytes_identity(content)
            for name, content in sorted(artifacts.items())
        },
    }


def _provenance(root: Path, source_commit: str) -> dict[str, object]:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
        cwd=root,
        check=False,
    ).returncode == 0
    frozen_document = _frozen_file_status(
        root, FROZEN_DOCUMENT_COMMIT, FROZEN_DOCUMENT_PATH
    )
    frozen_context = _frozen_file_status(
        root, FROZEN_DOCUMENT_COMMIT, FROZEN_CONTEXT_PATH
    )
    historical = {
        path: {
            "expected_tree": expected,
            "current_tree": _git_output(root, ("rev-parse", f"HEAD:{path}")),
        }
        for path, expected in HISTORICAL_TREES.items()
    }
    tracked = {
        path.as_posix(): _tracked_identity(root, path)
        for path in (
            *DIRECT_SOURCE_PATHS,
            FORMAL_CONTRACT,
            TASK_CONFIGURATION,
            BINDING_CONFIGURATION,
        )
    }
    source_tree = {
        "path": "src/hwr",
        "tree": _git_output(root, ("rev-parse", "HEAD:src/hwr")),
        "python_file_count": len(tuple((root / "src/hwr").rglob("*.py"))),
    }
    checks = {
        "workspace_clean": not status.strip(),
        "source_commit_matches_head": (
            source_commit == _git_output(root, ("rev-parse", "HEAD"))
        ),
        "frozen_document_commit_is_ancestor": ancestor,
        "frozen_document_content_matches": frozen_document["content_matches"],
        "frozen_document_blob_matches": frozen_document["blob_matches"],
        "frozen_context_content_matches": frozen_context["content_matches"],
        "frozen_context_blob_matches": frozen_context["blob_matches"],
        "historical_trees_match": all(
            value["current_tree"] == value["expected_tree"]
            for value in historical.values()
        ),
        "all_inputs_tracked": all(value["tracked"] for value in tracked.values()),
        "all_inputs_match_head": all(
            value["blob_matches_head"] for value in tracked.values()
        ),
    }
    return {
        "checks": {**checks, "passed": all(checks.values())},
        "frozen_document": frozen_document,
        "frozen_context": frozen_context,
        "historical_trees": historical,
        "source_tree": source_tree,
        "inputs": tracked,
    }


def _require_provenance(provenance: Mapping[str, object]) -> None:
    checks = provenance.get("checks")
    if not isinstance(checks, Mapping) or checks.get("passed") is not True:
        failed = (
            sorted(name for name, passed in checks.items() if not passed)
            if isinstance(checks, Mapping)
            else ["missing_checks"]
        )
        raise RuntimeError(f"P61 provenance gate failed: {', '.join(failed)}")


def _frozen_file_status(
    root: Path, commit: str, path: Path
) -> dict[str, object]:
    expected = subprocess.run(
        ("git", "show", f"{commit}:{path.as_posix()}"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    actual = (root / path).read_bytes()
    frozen_blob = _git_output(
        root, ("rev-parse", f"{commit}:{path.as_posix()}")
    )
    current_blob = _git_output(
        root, ("rev-parse", f"HEAD:{path.as_posix()}")
    )
    return {
        "path": path.as_posix(),
        "commit": commit,
        "content_matches": actual == expected,
        "blob_matches": current_blob == frozen_blob,
        "current": _bytes_identity(actual),
        "frozen": _bytes_identity(expected),
        "current_blob": current_blob,
        "frozen_blob": frozen_blob,
    }


def _tracked_identity(root: Path, path: Path) -> dict[str, object]:
    tracked = subprocess.run(
        ("git", "ls-files", "--error-unmatch", path.as_posix()),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    content = (root / path).read_bytes()
    working_blob = _git_output(
        root, ("hash-object", "--", path.as_posix())
    )
    head_blob = (
        _git_output(root, ("rev-parse", f"HEAD:{path.as_posix()}"))
        if tracked
        else None
    )
    return {
        "path": path.as_posix(),
        **_bytes_identity(content),
        "tracked": tracked,
        "head_blob": head_blob,
        "working_blob": working_blob,
        "blob_matches_head": tracked and working_blob == head_blob,
    }


def _source_documents(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((root / "src/hwr").rglob("*.py"))
    }


def _source_commit(root: Path) -> str:
    commit = _git_output(root, ("rev-parse", "HEAD"))
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RuntimeError("P61 requires a full Git source commit")
    return commit


def _command(arguments: argparse.Namespace) -> list[str]:
    return [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--contract",
        str(arguments.contract),
        "--output",
        str(arguments.output),
    ]


def _require_budget(
    elapsed_seconds: float, peak_rss_bytes: int, artifact_bytes: int
) -> None:
    if elapsed_seconds >= WALL_TIME_LIMIT_SECONDS:
        raise RuntimeError("P61 wall-time budget exceeded")
    if peak_rss_bytes >= RSS_LIMIT_BYTES:
        raise RuntimeError("P61 RSS budget exceeded")
    if artifact_bytes >= ARTIFACT_LIMIT_BYTES:
        raise RuntimeError("P61 artifact budget exceeded")


def _require_unused_output(output: Path) -> None:
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    _require_unused_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir()
    try:
        for name, content in artifacts.items():
            temporary = staging / f"{name}.tmp"
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, staging / name)
        os.replace(staging, output)
    except BaseException:
        for path in staging.glob("*"):
            path.unlink()
        if staging.exists():
            staging.rmdir()
        raise


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _git_output(
    root: Path, arguments: Sequence[str]
) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bytes_identity(content: bytes) -> dict[str, object]:
    return {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return (
        0
        if result["decision"]
        == "accepted as interaction-contract gap evidence"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
