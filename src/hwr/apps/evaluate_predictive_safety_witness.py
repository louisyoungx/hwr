"""Run the frozen R0001-P66-E1 predictive-safety witness audit."""

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

from hwr.adapters.mujoco.predictive_safety_diagnostic import (
    PredictiveSafetyAnchorReplay,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.eval.predictive_safety_witness import (
    ANCHOR_ID,
    PROPOSAL_ID,
    analyze_predictive_witness,
    canonical_bytes,
    canonical_sha256,
)

MODULE_NAME = "hwr.apps.evaluate_predictive_safety_witness"
MANIFEST_SCHEMA = "hwr.p66-predictive-safety-witness-artifacts/v1"
FROZEN_DOCUMENT_COMMIT = "b1db1368a2321e79f16f673ef140232860be3001"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0013/03-experiment.md")
FORMAL_INPUT = Path(
    "runs/research-loop/0012/r0012-p60-phase-entry-s20266001"
)
FORMAL_OUTPUT = Path(
    "runs/research-loop/0013/r0013-p66-predictive-witness-s20266601"
)
EXPECTED_INPUTS = {
    "episodes.json":
        "681b2ac5f49d8af7fa21108e3adb96f1fcc0bbc894d2a3f6b2d544cb28f64c4e",
    "plan.json":
        "df69da8606f78f94fdaaecef0021a64eb33c8cc5e2b62720bdbdd1f5e2255e5a",
    "seed-audit.json":
        "5bc6fb284f14c689da0167a40e76775c83c2116fb9541d96d45371307dde71f3",
    "report.json":
        "1674278396eb51f647c261d6514b2411cb09ba786deda659ed189c6124841737",
    "manifest.json":
        "ee21c04f009d0ab89bc83f3f00516a36a91955f9369ab50d66bea0f04f9c75df",
}
SOURCE_PATHS = (
    Path("src/hwr/adapters/mujoco/dual_arm_backend.py"),
    Path("src/hwr/adapters/mujoco/formal_household_backend.py"),
    Path("src/hwr/adapters/mujoco/predictive_safety_diagnostic.py"),
    Path("src/hwr/adapters/mujoco/phase_entry_geometry.py"),
    Path("src/hwr/eval/predictive_safety_witness.py"),
    Path("src/hwr/apps/evaluate_predictive_safety_witness.py"),
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
    "docs/research-loop/0012": "db73bb9a6c6155d0366d7d92718aec614e044a5f",
}
MAX_WALL_SECONDS = 20 * 60
MAX_RSS_BYTES = 4 * 1024**3
MAX_ARTIFACT_BYTES = 512 * 1024**2
MIN_DISK_FREE_BYTES = 20 * 1024**3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p60-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    input_path = _resolve(root, arguments.p60_input)
    output = _resolve(root, arguments.output)
    if input_path != (root / FORMAL_INPUT).resolve():
        raise ValueError("P60 input path differs from frozen input")
    if output != (root / FORMAL_OUTPUT).resolve():
        raise ValueError("output path differs from frozen formal output")
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    source_commit = _git(root, "rev-parse", "HEAD")
    provenance = _provenance(root, input_path, source_commit)
    _require_provenance(provenance)
    _require_disk(output)
    historical = _read_json(input_path / "episodes.json")
    anchor = next(
        row for row in historical["records"]
        if row["planned_episode_id"] == ANCHOR_ID
    )
    tasks, bindings = load_default_formal_household_catalogs(root)
    task_id = str(anchor["task_id"])
    replay = PredictiveSafetyAnchorReplay(tasks[task_id], bindings[task_id])
    disabled = replay.run(
        environment_seed=int(anchor["environment_seed"]),
        policy_rng_seed=int(anchor["policy_rng_seed"]),
        observer_enabled=False,
    )
    enabled = replay.run(
        environment_seed=int(anchor["environment_seed"]),
        policy_rng_seed=int(anchor["policy_rng_seed"]),
        observer_enabled=True,
    )
    disabled["prefix"]["planned_episode_id"] = ANCHOR_ID
    enabled["prefix"]["planned_episode_id"] = ANCHOR_ID
    report = analyze_predictive_witness(disabled, enabled)
    artifacts = {
        "disabled.json": _json_bytes(disabled),
        "witness.json": _json_bytes(enabled),
        "report.json": _json_bytes(
            {
                **report,
                "source_commit": source_commit,
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
        "command": list(_command(arguments)),
        "decision": report["decision"],
        "provenance": provenance,
        "runtime": {
            "python": platform.python_version(),
            "mujoco": importlib.metadata.version("mujoco"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "wall_seconds": elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
            "disk_free_bytes": shutil.disk_usage(root).free,
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
        "decision": report["decision"],
        "witness_sha256": canonical_sha256(enabled),
        "manifest_sha256": _sha256(artifacts["manifest.json"]),
    }


def _provenance(
    root: Path, input_path: Path, source_commit: str
) -> dict[str, object]:
    input_identities = {
        name: _file_identity(input_path / name)
        for name in EXPECTED_INPUTS
    }
    frozen = _frozen_status(root)
    source = {
        path.as_posix(): {
            **_file_identity(root / path),
            "head_blob": _git(root, "rev-parse", f"HEAD:{path.as_posix()}"),
            "working_blob": _git(root, "hash-object", "--", path.as_posix()),
        }
        for path in SOURCE_PATHS
    }
    checks = {
        "workspace_clean": not _git(root, "status", "--porcelain", "--untracked-files=all"),
        "source_commit_matches_head": source_commit == _git(root, "rev-parse", "HEAD"),
        "frozen_document_commit_is_ancestor": subprocess.run(
            ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
            cwd=root,
            check=False,
        ).returncode == 0,
        "frozen_document_content_and_blob_match": all(frozen.values()),
        "input_hashes_match": all(
            input_identities[name]["sha256"] == expected
            for name, expected in EXPECTED_INPUTS.items()
        ),
        "source_files_match_head": all(
            item["head_blob"] == item["working_blob"] for item in source.values()
        ),
        "historical_trees_match": all(
            _git(root, "rev-parse", f"HEAD:{path}") == expected
            for path, expected in HISTORICAL_TREES.items()
        ),
    }
    return {
        "checks": {**checks, "passed": all(checks.values())},
        "frozen_document": frozen,
        "inputs": input_identities,
        "sources": source,
        "historical_trees": dict(HISTORICAL_TREES),
    }


def _frozen_status(root: Path) -> dict[str, bool]:
    expected = subprocess.run(
        ("git", "show", f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    actual = (root / FROZEN_DOCUMENT_PATH).read_bytes()
    return {
        "content_matches": actual == expected,
        "blob_matches": _git(
            root, "rev-parse", f"HEAD:{FROZEN_DOCUMENT_PATH}"
        ) == _git(
            root,
            "rev-parse",
            f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}",
        ),
    }


def _require_provenance(value: Mapping[str, object]) -> None:
    checks = value["checks"]
    if checks["passed"] is not True:
        failed = sorted(
            name for name, passed in checks.items()
            if name != "passed" and not passed
        )
        raise RuntimeError(f"P66 provenance gate failed: {', '.join(failed)}")


def _require_budget(
    elapsed: float, peak_rss: int, artifacts: Mapping[str, bytes]
) -> None:
    if elapsed > MAX_WALL_SECONDS:
        raise RuntimeError("P66 wall-time budget exceeded")
    if peak_rss > MAX_RSS_BYTES:
        raise RuntimeError("P66 RSS budget exceeded")
    if sum(len(value) for value in artifacts.values()) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("P66 artifact budget exceeded")


def _require_disk(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("P66 disk-free guard failed")


def _write_atomic(output: Path, artifacts: Mapping[str, bytes]) -> None:
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
        "actual_collision_claim_allowed": False,
    }


def _file_identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": _sha256(content)}


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
        "--p60-input",
        arguments.p60_input.as_posix(),
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
