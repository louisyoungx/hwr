"""Evaluate frozen R0001-P57 bilateral pre-contact reachability evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.eval.precontact_reachability import (
    ACTION_SCALE_M_PER_S,
    B3_STEPS,
    B3_VELOCITY_M_PER_S,
    B4_STEPS,
    B4_VELOCITY_M_PER_S,
    CONTACT_TRANSITION_NOMINAL_M,
    CONTROL_HZ,
    PAIR_SCHEMA,
    PROPOSAL_ID,
    READY_DISTANCE_M,
    analyze_precontact_reachability,
)

MODULE_NAME = "hwr.apps.evaluate_precontact_reachability"
REPORT_SCHEMA = "hwr.p57-precontact-reachability-report/v1"
MANIFEST_SCHEMA = "hwr.p57-precontact-reachability-artifacts/v1"
FROZEN_DOCUMENT_COMMIT = "88992a773ee2b0f214dba7975cdddf25f282d679"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0011/03-experiment.md")
FORMAL_OUTPUT = Path(
    "runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701"
)
INPUT_SPECS = {
    "bank": {
        "path": Path(
            "runs/research-loop/0010/"
            "r0010-p51-e1-bank-s20265101/bank.json"
        ),
        "bytes": 7_160_016,
        "sha256": "09d2fe4e05f2bd8d23ebfe6886fe260d1b34b41771da42992f0f432a8a04f3d3",
    },
    "bank_manifest": {
        "path": Path(
            "runs/research-loop/0010/"
            "r0010-p51-e1-bank-s20265101/manifest.json"
        ),
        "bytes": 53_605,
        "sha256": "7e0d5f9c7757b59ceb8d4dfe3ddcba38cc1d1037c43c358e7168d700310d5e45",
    },
    "terminals": {
        "path": Path(
            "runs/research-loop/0010/"
            "r0010-p51-e1-convergence-s20265101/terminals.json"
        ),
        "bytes": 26_314_706,
        "sha256": "1c54f93a95bfbf4e08076b3c633b22dce295990a6808a48f0f10de18a2b3c2c7",
    },
    "terminal_report": {
        "path": Path(
            "runs/research-loop/0010/"
            "r0010-p51-e1-convergence-s20265101/report.json"
        ),
        "bytes": 4_643,
        "sha256": "3fcac95c2362923d9eb94ef4d7121d5bcb31ea859a308ed352321dfa93771cc9",
    },
    "terminal_manifest": {
        "path": Path(
            "runs/research-loop/0010/"
            "r0010-p51-e1-convergence-s20265101/manifest.json"
        ),
        "bytes": 53_753,
        "sha256": "821f3cf6fea922a86b4096ee5d0ba9c64b9d8f444eacc98dcfc1f164da1328d2",
    },
}
TRACKED_INPUTS = frozenset(("bank", "bank_manifest"))
MANIFEST_BOUND_INPUTS = frozenset(
    ("terminals", "terminal_report", "terminal_manifest")
)
SOURCE_PATHS = (
    Path("src/hwr/eval/precontact_reachability.py"),
    Path("src/hwr/apps/evaluate_precontact_reachability.py"),
    Path("src/hwr/eval/cartesian_convergence.py"),
    Path("src/hwr/eval/cartesian_convergence_validation.py"),
    Path("src/hwr/eval/target_selection.py"),
    FROZEN_DOCUMENT_PATH,
)
CLAIM_FLAGS = {
    "measurement_only": True,
    "mujoco_executed": False,
    "training_executed": False,
    "policy_inference_executed": False,
    "closed_loop_capability_episode_executed": False,
    "candidate_modified": False,
    "primitive_modified": False,
    "phase_modified": False,
    "velocity_modified": False,
    "gripper_modified": False,
    "contact_or_grasp_claim_allowed": False,
    "task_capability_claim_allowed": False,
    "generalization_claim_allowed": False,
    "safety_improvement_claim_allowed": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--terminals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    if output != (root / FORMAL_OUTPUT).resolve():
        raise ValueError("output path differs from frozen formal output")
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    tracemalloc.start()
    source_commit = _source_commit(root)
    source_identities = _source_identities(root)
    _require_clean_source(root, source_identities)
    inputs = _input_identities(root, arguments)
    bank = _read_json(inputs["bank"]["absolute_path"])
    terminals = _read_json(inputs["terminals"]["absolute_path"])
    _validate_input_provenance(root, bank, terminals, inputs)
    first = analyze_precontact_reachability(bank, terminals)
    second = analyze_precontact_reachability(bank, terminals)
    if first["decision"] == "invalid":
        raise RuntimeError(first["validation_error"])
    first_pairs = _pairs_document(first["pairs"])
    second_pairs = _pairs_document(second["pairs"])
    pairs_equal = _canonical_bytes(first_pairs) == _canonical_bytes(second_pairs)
    first_report = _report(
        source_commit,
        first,
        inputs,
        pairs_equal=pairs_equal,
        report_equal=True,
    )
    second_report = _report(
        source_commit,
        second,
        inputs,
        pairs_equal=pairs_equal,
        report_equal=True,
    )
    report_equal = _canonical_bytes(first_report) == _canonical_bytes(second_report)
    report = _report(
        source_commit,
        first,
        inputs,
        pairs_equal=pairs_equal,
        report_equal=report_equal,
    )
    replay_report = _report(
        source_commit,
        second,
        inputs,
        pairs_equal=pairs_equal,
        report_equal=report_equal,
    )
    if (
        _canonical_bytes(report) != _canonical_bytes(replay_report)
        or not pairs_equal
        or not report_equal
    ):
        raise RuntimeError("P57 deterministic replay differs")
    artifacts = {
        "pairs.json": _json_bytes(first_pairs),
        "report.json": _json_bytes(report),
    }
    elapsed = time.perf_counter() - started
    _, traced_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss_bytes = _peak_rss_bytes()
    if elapsed > 300.0:
        raise RuntimeError("P57 wall-time budget exceeded")
    if max(traced_peak_bytes, peak_rss_bytes) > 2 * 1024**3:
        raise RuntimeError("P57 RSS budget exceeded")
    if sum(len(content) for content in artifacts.values()) > 100 * 1024**2:
        raise RuntimeError("P57 artifact budget exceeded")
    manifest = _manifest(
        source_commit,
        _command(arguments),
        source_identities,
        inputs,
        artifacts,
        started,
        traced_peak_bytes=traced_peak_bytes,
        peak_rss_bytes=peak_rss_bytes,
    )
    artifacts["manifest.json"] = _json_bytes(manifest)
    _create_output(output, artifacts)
    return {
        "output": str(output),
        "decision": report["decision"],
        "diagnostic": report["diagnostic"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "pairs_sha256": manifest["artifacts"]["pairs.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
    }


def _report(
    source_commit,
    analysis,
    inputs,
    *,
    pairs_equal: bool,
    report_equal: bool,
) -> dict[str, object]:
    checks = {
        **analysis["checks"],
        "input_provenance_passed": True,
        "deterministic_pairs_bit_identical": pairs_equal,
        "deterministic_report_bit_identical": report_equal,
    }
    checks["passed"] = all(value for name, value in checks.items() if name != "passed")
    decision = (
        analysis["decision"]
        if checks["passed"]
        else "invalid"
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "decision": decision,
        "diagnostic": analysis["diagnostic"] if checks["passed"] else None,
        "sample_unit": "pair",
        "estimand": (
            "bilateral pre-contact readiness and applied-command support "
            "within the fixed P51 frame_fixed cohort"
        ),
        "input_identities": {
            name: {
                "path": value["path"],
                "bytes": value["bytes"],
                "sha256": value["sha256"],
                "commit": value["commit"],
                "provenance_kind": value["provenance_kind"],
            }
            for name, value in inputs.items()
        },
        "checks": checks,
        "summary": analysis["summary"],
        "frozen_thresholds": _frozen_design(),
        **CLAIM_FLAGS,
    }


def _pairs_document(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": PAIR_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "sample_unit": "pair",
        "pair_count": len(rows),
        "arm_count": sum(len(row["arms"]) for row in rows),
        "records": list(rows),
    }


def _manifest(
    source_commit,
    command,
    source_identities,
    inputs,
    artifacts,
    started,
    *,
    traced_peak_bytes,
    peak_rss_bytes,
) -> dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": "complete",
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "command": list(command),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "device": "cpu",
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_platform_units": usage.ru_maxrss,
            "peak_rss_bytes": peak_rss_bytes,
            "tracemalloc_peak_bytes": traced_peak_bytes,
            "disk_free_bytes": shutil.disk_usage(
                Path(__file__).resolve().parents[3]
            ).free,
        },
        "frozen_design": _frozen_design(),
        "source_identities": source_identities,
        "input_identities": {
            name: {
                "path": value["path"],
                "bytes": value["bytes"],
                "sha256": value["sha256"],
                "commit": value["commit"],
                "provenance_kind": value["provenance_kind"],
            }
            for name, value in inputs.items()
        },
        "artifacts": {
            name: _bytes_identity(content)
            for name, content in artifacts.items()
        },
        **CLAIM_FLAGS,
    }


def _frozen_design() -> dict[str, object]:
    return {
        "pair_count": 36,
        "arm_count": 72,
        "distance_steps_per_arm": 101,
        "applied_action_steps_per_pair": 100,
        "ready_distance_m": READY_DISTANCE_M,
        "action_scale_m_per_s": ACTION_SCALE_M_PER_S,
        "control_hz": CONTROL_HZ,
        "b3_steps": B3_STEPS,
        "b3_velocity_m_per_s": B3_VELOCITY_M_PER_S,
        "b4_steps": B4_STEPS,
        "b4_velocity_m_per_s": B4_VELOCITY_M_PER_S,
        "contact_transition_nominal_m": CONTACT_TRANSITION_NOMINAL_M,
        "support_deficit_max_ready_pairs": 6,
        "support_deficit_min_both_negative_margin_pairs": 30,
        "support_deficit_max_ready_pairs_per_task": 4,
        "deficit_rejected_min_ready_pairs": 24,
        "deficit_rejected_min_ready_pairs_per_task": 6,
    }


def _input_identities(root: Path, arguments) -> dict[str, dict[str, object]]:
    provided = {
        "bank": arguments.bank,
        "bank_manifest": arguments.bank.parent / "manifest.json",
        "terminals": arguments.terminals,
        "terminal_report": arguments.terminals.parent / "report.json",
        "terminal_manifest": arguments.terminals.parent / "manifest.json",
    }
    result = {}
    for name, supplied in provided.items():
        path = _resolve(root, supplied)
        expected = (root / INPUT_SPECS[name]["path"]).resolve()
        if path != expected:
            raise ValueError(f"{name} path differs from frozen input")
        content = path.read_bytes()
        identity = _bytes_identity(content)
        if any(identity[field] != INPUT_SPECS[name][field] for field in ("bytes", "sha256")):
            raise ValueError(f"{name} bytes differ from frozen input")
        relative = path.relative_to(root).as_posix()
        result[name] = {
            "absolute_path": path,
            "path": relative,
            **identity,
        }
    producer = _read_json(result["terminal_manifest"]["absolute_path"]).get(
        "source_commit"
    )
    if not _is_commit(producer):
        raise RuntimeError("terminal producer source commit is invalid")
    for name in TRACKED_INPUTS:
        result[name].update(
            {
                "commit": _tracked_input_commit(
                    root, Path(result[name]["absolute_path"]), name
                ),
                "provenance_kind": "tracked_committed_artifact",
            }
        )
    for name in MANIFEST_BOUND_INPUTS:
        result[name].update(
            {
                "commit": producer,
                "provenance_kind": "manifest_bound_ignored_artifact",
            }
        )
    return result


def _validate_input_provenance(root, bank, terminals, inputs) -> None:
    bank_manifest = _read_json(inputs["bank_manifest"]["absolute_path"])
    terminal_report = _read_json(inputs["terminal_report"]["absolute_path"])
    terminal_manifest = _read_json(inputs["terminal_manifest"]["absolute_path"])
    producer = terminal_manifest.get("source_commit")
    if any(
        inputs[name].get("provenance_kind")
        != (
            "tracked_committed_artifact"
            if name in TRACKED_INPUTS
            else "manifest_bound_ignored_artifact"
        )
        for name in INPUT_SPECS
    ):
        raise RuntimeError("input provenance classification differs")
    if any(
        inputs[name].get("commit") != producer for name in MANIFEST_BOUND_INPUTS
    ):
        raise RuntimeError("ignored artifact producer commit differs")
    if bank_manifest.get("artifacts", {}).get("bank.json") != _public_identity(
        inputs["bank"]
    ):
        raise RuntimeError("bank manifest does not bind bank")
    if terminal_manifest.get("artifacts", {}).get("terminals.json") != _public_identity(
        inputs["terminals"]
    ):
        raise RuntimeError("terminal manifest does not bind terminals")
    if terminal_manifest.get("artifacts", {}).get("report.json") != _public_identity(
        inputs["terminal_report"]
    ):
        raise RuntimeError("terminal manifest does not bind report")
    bank_binding = terminal_manifest.get("source_identities", {}).get("bank")
    if not isinstance(bank_binding, Mapping) or any(
        bank_binding.get(name) != inputs["bank"].get(name)
        for name in ("path", "bytes", "sha256")
    ):
        raise RuntimeError("terminal manifest bank identity differs")
    if (
        bank_manifest.get("source_commit") != bank.get("source_commit")
        or terminals.get("bank_source_commit") != bank.get("source_commit")
        or terminal_report.get("bank_source_commit") != bank.get("source_commit")
        or terminal_report.get("terminal_pair_count") != 36
        or terminal_report.get("planned_pair_count") != 36
    ):
        raise RuntimeError("P51 source or report provenance differs")
    for commit in (
        bank.get("source_commit"),
        bank_manifest.get("source_commit"),
        producer,
    ):
        if not _is_commit(commit) or subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            cwd=root,
            check=False,
        ).returncode:
            raise RuntimeError("P51 source commit is not an ancestor")


def _tracked_input_commit(root: Path, path: Path, name: str) -> str:
    relative = path.relative_to(root).as_posix()
    subprocess.run(
        ("git", "ls-files", "--error-unmatch", relative),
        cwd=root,
        check=True,
        capture_output=True,
    )
    if subprocess.run(
        ("git", "diff", "--quiet", "HEAD", "--", relative),
        cwd=root,
        check=False,
    ).returncode:
        raise RuntimeError(f"{name} differs from committed bytes")
    commit = _git_output(root, ("log", "-1", "--format=%H", "--", relative))
    if not _is_commit(commit) or subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=root,
        check=False,
    ).returncode:
        raise RuntimeError(f"{name} commit is not an ancestor")
    return commit


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_identities(root: Path) -> dict[str, object]:
    result = {
        path.as_posix(): {
            "path": path.as_posix(),
            **_bytes_identity((root / path).read_bytes()),
        }
        for path in SOURCE_PATHS
    }
    result["frozen_document"] = _frozen_document_status(root)
    return result


def _require_clean_source(root: Path, identities: Mapping[str, object]) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P57 runner requires clean committed source")
    frozen = identities.get("frozen_document")
    if not isinstance(frozen, Mapping) or not all(
        frozen.get(name)
        for name in ("commit_is_ancestor", "content_matches", "blob_matches")
    ):
        raise RuntimeError("P57 frozen experiment document drifted")


def _frozen_document_status(root: Path) -> dict[str, object]:
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
        cwd=root,
        check=False,
    ).returncode == 0
    expected = subprocess.run(
        (
            "git",
            "show",
            f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH.as_posix()}",
        ),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    actual = (root / FROZEN_DOCUMENT_PATH).read_bytes()
    frozen_blob = _git_output(
        root,
        ("rev-parse", f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}"),
    )
    current_blob = _git_output(root, ("rev-parse", f"HEAD:{FROZEN_DOCUMENT_PATH}"))
    return {
        "commit": FROZEN_DOCUMENT_COMMIT,
        "commit_is_ancestor": ancestor,
        "content_matches": actual == expected,
        "blob_matches": current_blob == frozen_blob,
        "current": _bytes_identity(actual),
        "frozen": _bytes_identity(expected),
    }


def _public_identity(value: Mapping[str, object]) -> dict[str, object]:
    return {"bytes": value["bytes"], "sha256": value["sha256"]}


def _command(arguments) -> list[str]:
    return [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--bank",
        str(arguments.bank),
        "--terminals",
        str(arguments.terminals),
        "--output",
        str(arguments.output),
    ]


def _source_commit(root: Path) -> str:
    commit = _git_output(root, ("rev-parse", "HEAD"))
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P57 requires a full Git source commit")
    return commit


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _git_output(root: Path, arguments: Sequence[str]) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _bytes_identity(content: bytes) -> dict[str, object]:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
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


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
