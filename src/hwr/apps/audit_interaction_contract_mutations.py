"""Run the frozen R0001-P72-E1 mutation audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from hwr.apps.audit_interaction_contract import build_source_audit
from hwr.eval import interaction_contract as p61_contract
from hwr.eval.interaction_contract_mutation import (
    PROPOSAL_ID,
    audit_interaction_contract_mutations,
    canonical_bytes,
)

MODULE_NAME = "hwr.apps.audit_interaction_contract_mutations"
MANIFEST_SCHEMA = "hwr.p72-interaction-contract-mutation-artifacts/v1"
FROZEN_DOCUMENT_COMMIT = "b1db1368a2321e79f16f673ef140232860be3001"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0013/03-experiment.md")
P61_PRODUCER_COMMIT = "6bf0400f51a25bfb6f45e951299c410efd5c2c7a"
FORMAL_CONTRACT = Path("configs/eval/interaction_contract_v1.json")
FORMAL_INPUT = Path(
    "runs/research-loop/0012/r0012-p61-interaction-contract-s20266101"
)
FORMAL_OUTPUT = Path(
    "runs/research-loop/0013/r0013-p72-p61-mutation-s20267201"
)
TASK_CONFIGURATION = Path("configs/tasks/formal_3d_v1.json")
BINDING_CONFIGURATION = Path("configs/adapters/mujoco/formal_3d_v1.json")
EXPECTED_INPUTS = {
    "transitions.json":
        "1cc139e7f8b02a6325d16282f9b7882e9736c40d03f56644c1739d79ee7bcc0a",
    "report.json":
        "d9a760eaa30198eda95d20e90a4ebf4c9d9f5bcd2e118b6e139446866545719a",
    "manifest.json":
        "6a019a7591a2614c6082dea102c29f9cb24e101f78da8ce21ce3725f60df221d",
}
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
    "docs/research-loop/0012": "db73bb9a6c6155d0366d7d92718aec614e044a5f",
}
MAX_WALL_SECONDS = 60.0
MAX_RSS_BYTES = 1024**3
MAX_ARTIFACT_BYTES = 10 * 1024**2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--p61-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    contract_path = _resolve(root, arguments.contract)
    input_path = _resolve(root, arguments.p61_input)
    output = _resolve(root, arguments.output)
    if contract_path != (root / FORMAL_CONTRACT).resolve():
        raise ValueError("contract path differs from frozen formal contract")
    if input_path != (root / FORMAL_INPUT).resolve():
        raise ValueError("P61 input path differs from frozen input")
    if output != (root / FORMAL_OUTPUT).resolve():
        raise ValueError("output path differs from frozen formal output")
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    source_commit = _git(root, "rev-parse", "HEAD")
    provenance = _provenance(root, input_path, source_commit)
    _require_provenance(provenance)
    contract = _git_json(root, P61_PRODUCER_COMMIT, FORMAL_CONTRACT)
    tasks = _git_json(root, P61_PRODUCER_COMMIT, TASK_CONFIGURATION)
    bindings = _git_json(root, P61_PRODUCER_COMMIT, BINDING_CONFIGURATION)
    sources = _producer_sources(root)
    p61_transitions = _read_json(input_path / "transitions.json")
    result = audit_interaction_contract_mutations(
        contract,
        tasks,
        bindings,
        sources,
        p61_transitions,
        build_source_audit=build_source_audit,
        audit_contract=p61_contract.audit_interaction_contract,
        requirement_fields=p61_contract.source_requirement_fields,
    )
    artifacts = {
        "mutations.json": _json_bytes(result["mutations"]),
        "report.json": _json_bytes(
            {
                **result["report"],
                "source_commit": source_commit,
                "p61_producer_commit": P61_PRODUCER_COMMIT,
                "command": list(_command(arguments)),
            }
        ),
    }
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": "complete",
        "source_commit": source_commit,
        "p61_producer_commit": P61_PRODUCER_COMMIT,
        "command": list(_command(arguments)),
        "decision": result["report"]["decision"],
        "provenance": provenance,
        "runtime": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "wall_seconds": elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "artifacts": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in sorted(artifacts.items())
        },
        **_claim_flags(),
    }
    artifacts["manifest.json"] = _json_bytes(manifest)
    _require_budget(elapsed, manifest["runtime"]["peak_rss_bytes"], artifacts)
    _write_atomic(output, artifacts)
    return {
        "output": str(output),
        "decision": result["report"]["decision"],
        "p68_dependency_gate_passed":
            result["report"]["p68_dependency_gate_passed"],
        "mutations_sha256": hashlib.sha256(
            canonical_bytes(result["mutations"])
        ).hexdigest(),
        "manifest_sha256": _sha256(artifacts["manifest.json"]),
    }


def _producer_sources(root: Path) -> dict[str, str]:
    paths = _git(
        root, "ls-tree", "-r", "--name-only", P61_PRODUCER_COMMIT, "src/hwr"
    ).splitlines()
    return {
        path: subprocess.run(
            ("git", "show", f"{P61_PRODUCER_COMMIT}:{path}"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for path in paths
        if path.endswith(".py")
    }


def _provenance(
    root: Path, input_path: Path, source_commit: str
) -> dict[str, object]:
    identities = {
        name: _file_identity(input_path / name) for name in EXPECTED_INPUTS
    }
    frozen_expected = subprocess.run(
        ("git", "show", f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    frozen_actual = (root / FROZEN_DOCUMENT_PATH).read_bytes()
    p61_manifest = _read_json(input_path / "manifest.json")
    checks = {
        "workspace_clean":
            not _git(root, "status", "--porcelain", "--untracked-files=all"),
        "source_commit_matches_head":
            source_commit == _git(root, "rev-parse", "HEAD"),
        "frozen_document_commit_is_ancestor": subprocess.run(
            ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
            cwd=root,
            check=False,
        ).returncode == 0,
        "frozen_document_content_matches": frozen_actual == frozen_expected,
        "frozen_document_blob_matches": _git(
            root, "rev-parse", f"HEAD:{FROZEN_DOCUMENT_PATH}"
        ) == _git(
            root, "rev-parse", f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}"
        ),
        "input_hashes_match": all(
            identities[name]["sha256"] == expected
            for name, expected in EXPECTED_INPUTS.items()
        ),
        "producer_commit_matches":
            p61_manifest["source_commit"] == P61_PRODUCER_COMMIT,
        "producer_is_ancestor": subprocess.run(
            ("git", "merge-base", "--is-ancestor", P61_PRODUCER_COMMIT, "HEAD"),
            cwd=root,
            check=False,
        ).returncode == 0,
        "historical_trees_match": all(
            _git(root, "rev-parse", f"HEAD:{path}") == expected
            for path, expected in HISTORICAL_TREES.items()
        ),
    }
    return {
        "checks": {**checks, "passed": all(checks.values())},
        "inputs": identities,
        "frozen_document": _file_identity(root / FROZEN_DOCUMENT_PATH),
        "historical_trees": dict(HISTORICAL_TREES),
    }


def _require_provenance(value: Mapping[str, object]) -> None:
    checks = value["checks"]
    if checks["passed"] is not True:
        failed = sorted(
            name for name, passed in checks.items()
            if name != "passed" and not passed
        )
        raise RuntimeError(f"P72 provenance gate failed: {', '.join(failed)}")


def _require_budget(
    elapsed: float, peak_rss: int, artifacts: Mapping[str, bytes]
) -> None:
    if elapsed > MAX_WALL_SECONDS:
        raise RuntimeError("P72 wall-time budget exceeded")
    if peak_rss > MAX_RSS_BYTES:
        raise RuntimeError("P72 RSS budget exceeded")
    if sum(len(value) for value in artifacts.values()) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("P72 artifact budget exceeded")


def _write_atomic(output: Path, artifacts: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for name, content in artifacts.items():
            (staging / name).write_bytes(content)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _claim_flags() -> dict[str, bool]:
    return {
        "training_executed": False,
        "policy_inference_executed": False,
        "capability_claim_allowed": False,
        "task_success_claim_allowed": False,
        "generalization_claim_allowed": False,
        "hardware_safety_claim_allowed": False,
    }


def _file_identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": _sha256(content)}


def _git_json(root: Path, commit: str, path: Path) -> object:
    return json.loads(
        subprocess.run(
            ("git", "show", f"{commit}:{path.as_posix()}"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _command(arguments: argparse.Namespace) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        MODULE_NAME,
        "--contract",
        arguments.contract.as_posix(),
        "--p61-input",
        arguments.p61_input.as_posix(),
        "--output",
        arguments.output.as_posix(),
    )


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
