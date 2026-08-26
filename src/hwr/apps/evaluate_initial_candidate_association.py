"""Run the frozen R0001-P68-E1 initial candidate association gate."""

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

from hwr.adapters.mujoco.candidate_acquisition import (
    CandidateAcquisitionDiagnostic,
    _run_acquisition_once,
)
from hwr.adapters.mujoco.candidate_association import (
    CandidateAssociationDiagnostic,
    summary_for_identity,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.eval.tool_kinematics import recursive_xml_input_identity
from hwr.eval.initial_candidate_association import (
    PROPOSAL_ID,
    analyze_episode_records,
    associate_candidates,
    canonical_bytes,
    canonical_sha256,
    classify_episode,
    reconstruct_candidate_support,
)
from hwr.eval.target_selection import TASK_IDS

MODULE_NAME = "hwr.apps.evaluate_initial_candidate_association"
MANIFEST_SCHEMA = "hwr.p68-initial-candidate-association-artifacts/v1"
EPISODES_SCHEMA = "hwr.p68-initial-candidate-association-episodes/v1"
FROZEN_DOCUMENT_COMMIT = "b1db1368a2321e79f16f673ef140232860be3001"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0013/03-experiment.md")
FORMAL_P50_INPUT = Path(
    "runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001"
)
FORMAL_MAPPING_INPUT = Path(
    "runs/research-loop/0012/r0012-p50-e4-mapping-s20265004"
)
FORMAL_INTERACTION_INPUT = Path(
    "runs/research-loop/0012/r0012-p61-interaction-contract-s20266101"
)
FORMAL_P72_INPUT = Path(
    "runs/research-loop/0013/r0013-p72-p61-mutation-s20267201"
)
FORMAL_OUTPUT = Path(
    "runs/research-loop/0013/r0013-p68-initial-association-s20266801"
)
EXPECTED_INPUTS = {
    FORMAL_P50_INPUT: {
        "capsules.json":
            "223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf",
        "plan.json":
            "5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab",
        "report.json":
            "b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0",
        "manifest.json":
            "cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86",
    },
    FORMAL_MAPPING_INPUT: {
        "tables.json":
            "88540ddb87e6df129eaaa9666b0011598a422629b1085c27b7b844e735918fbc",
        "report.json":
            "96fdc3abde155e09715bb7e1314c9b6cbef26dffdeda193f6dc336e4cd767402",
        "manifest.json":
            "7c950b5e132a0c24395a63ee6dec150f5244466cf5b3f38118a4efa00a2a3579",
    },
    FORMAL_INTERACTION_INPUT: {
        "transitions.json":
            "1cc139e7f8b02a6325d16282f9b7882e9736c40d03f56644c1739d79ee7bcc0a",
        "report.json":
            "d9a760eaa30198eda95d20e90a4ebf4c9d9f5bcd2e118b6e139446866545719a",
        "manifest.json":
            "6a019a7591a2614c6082dea102c29f9cb24e101f78da8ce21ce3725f60df221d",
    },
}
SOURCE_PATHS = (
    Path("src/hwr/adapters/mujoco/candidate_association.py"),
    Path("src/hwr/adapters/mujoco/candidate_acquisition.py"),
    Path("src/hwr/adapters/mujoco/entity_candidate_mapping.py"),
    Path("src/hwr/apps/evaluate_initial_candidate_association.py"),
    Path("src/hwr/eval/initial_candidate_association.py"),
    Path("src/hwr/eval/interaction_contract.py"),
    Path("src/hwr/eval/target_selection.py"),
    Path("configs/adapters/mujoco/formal_3d_v1.json"),
    Path("configs/eval/interaction_contract_v1.json"),
    Path("configs/tasks/formal_3d_v1.json"),
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
MAX_WALL_SECONDS = 30 * 60
MAX_RSS_BYTES = 8 * 1024**3
MAX_ARTIFACT_BYTES = 2 * 1024**3
MIN_DISK_FREE_BYTES = 20 * 1024**3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p50-input", type=Path, required=True)
    parser.add_argument("--mapping-input", type=Path, required=True)
    parser.add_argument("--interaction-input", type=Path, required=True)
    parser.add_argument("--p72-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    paths = _resolve_paths(root, arguments)
    _require_frozen_paths(root, paths)
    output = paths["output"]
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    source_commit = _git(root, "rev-parse", "HEAD")
    provenance = _provenance(root, paths, source_commit)
    _require_provenance(provenance)
    _require_disk(output)
    p72_report = _read_json(paths["p72"] / "report.json")
    if p72_report.get("p68_dependency_gate_passed") is not True:
        raise RuntimeError("P68 dependency gate did not pass")
    capsules = _read_json(paths["p50"] / "capsules.json")
    mapping = _read_json(paths["mapping"] / "tables.json")
    interactions = _read_json(paths["interaction"] / "transitions.json")
    records = execute_cohort(
        root,
        paths["p50"],
        capsules,
        mapping,
        interactions,
        started=started,
    )
    report = analyze_episode_records(records, TASK_IDS)
    report.update(
        {
            "source_commit": source_commit,
            "command": list(_command(arguments)),
            "p72_dependency_gate_passed": True,
            "all_episode_replays_valid": True,
        }
    )
    artifacts = {
        "episodes.json": _json_bytes(
            {
                "schema_version": EPISODES_SCHEMA,
                "proposal_id": PROPOSAL_ID,
                "episode_count": len(records),
                "records": records,
            }
        ),
        "report.json": _json_bytes(report),
    }
    elapsed = time.perf_counter() - started
    manifest = _manifest(
        source_commit,
        arguments,
        provenance,
        artifacts,
        elapsed,
        report["decision"],
    )
    artifacts["manifest.json"] = _json_bytes(manifest)
    _require_budget(elapsed, manifest["runtime"]["peak_rss_bytes"], artifacts)
    _write_atomic(output, artifacts)
    return {
        "output": str(output),
        "decision": report["decision"],
        "episodes_sha256": canonical_sha256(records),
        "manifest_sha256": _sha256(artifacts["manifest.json"]),
    }


def execute_cohort(
    root: Path,
    p50_path: Path,
    capsules: Mapping[str, object],
    mapping: Mapping[str, object],
    interactions: Mapping[str, object],
    *,
    started: float,
) -> list[dict[str, object]]:
    if capsules.get("capsule_count") != 24:
        raise RuntimeError("P50 capsule count differs")
    task_tables = {
        item["task_id"]: item for item in mapping["tasks"]
    }
    annotations = {
        item["task_id"]: item
        for item in interactions["initial_microinteraction"]
    }
    tasks, bindings = load_default_formal_household_catalogs(root)
    records = []
    for episode in capsules["episodes"]:
        if time.perf_counter() - started > MAX_WALL_SECONDS:
            raise RuntimeError("P68 wall-time budget exceeded")
        task_id = str(episode["task_id"])
        plan = _episode_plan(episode)
        baseline = _run_acquisition_once(
            CandidateAcquisitionDiagnostic(tasks[task_id], bindings[task_id]),
            plan,
            capture_persistence_enabled=False,
            backend_run_ordinal=0,
        )
        treatment = CandidateAssociationDiagnostic(
            tasks[task_id], bindings[task_id]
        ).run_episode(plan)
        identity_checks = _identity_checks(
            p50_path, episode, baseline, treatment
        )
        if not all(identity_checks.values()):
            failed = sorted(
                name for name, passed in identity_checks.items() if not passed
            )
            raise RuntimeError(
                f"P68 replay identity differs for "
                f"{episode['planned_episode_id']}: {', '.join(failed)}"
            )
        official, supports = reconstruct_candidate_support(
            treatment["keyframes"],
            acquisition_base_pose=treatment["acquisition_pose"],
            final_input=treatment["final_payload"],
        )
        if official.candidate_set_sha256 != episode["candidate_set"]["sha256"]:
            raise RuntimeError("P68 reconstructed candidate identity differs")
        allowed = frozenset(
            annotations[task_id]["allowed_entity_instance_or_roles"]
        )
        candidates = associate_candidates(
            supports,
            treatment["segmentations"],
            task_tables[task_id],
            allowed,
        )
        records.append(
            {
                **classify_episode(
                    task_id=task_id,
                    planned_episode_id=str(episode["planned_episode_id"]),
                    selected_index=int(episode["candidate_set"]["selected_index"]),
                    candidate_records=candidates,
                ),
                "cell_id": episode["cell_id"],
                "replicate_ordinal": episode["replicate_ordinal"],
                "environment_seed": episode["environment_seed"],
                "policy_rng_seed": episode["policy_rng_seed"],
                "candidate_set_sha256": official.candidate_set_sha256,
                "segmentation_sequence_sha256":
                    treatment["segmentation_sequence_sha256"],
                "identity_checks": {**identity_checks, "passed": True},
            }
        )
    return records


def _identity_checks(
    p50_path: Path,
    episode: Mapping[str, object],
    baseline: Mapping[str, object],
    treatment: Mapping[str, object],
) -> dict[str, bool]:
    historical = episode["primary_run"]
    baseline_summary = summary_for_identity(baseline)
    treatment_summary = summary_for_identity(treatment)
    historical_summary = {
        name: historical[name] for name in baseline_summary
    }
    checks = {
        "baseline_matches_historical": (
            canonical_bytes(baseline_summary)
            == canonical_bytes(historical_summary)
        ),
        "observer_on_off_identity": (
            canonical_bytes(baseline_summary)
            == canonical_bytes(treatment_summary)
        ),
        "candidate_bytes_match_historical": (
            treatment["candidate_bytes"]
            == (p50_path / episode["candidate_set"]["path"]).read_bytes()
        ),
        "capture_count_matches": (
            len(treatment["captures"]) == len(episode["captures"])
        ),
        "segmentation_count_matches": (
            len(treatment["segmentations"]) == len(episode["captures"])
        ),
    }
    for index, (capture, historical_capture) in enumerate(
        zip(treatment["captures"], episode["captures"], strict=True)
    ):
        prefix = f"capture_{index:02d}"
        policy = p50_path / historical_capture["policy_input"]["path"]
        visible = p50_path / historical_capture["candidate_visible_input"]["path"]
        checks[f"{prefix}_identity"] = (
            list(capture.observation_identity)
            == [
                historical_capture["observation_timestamp_ns"],
                historical_capture["sequence_id"],
            ]
        )
        checks[f"{prefix}_policy_bytes"] = (
            capture.policy_input_bytes == policy.read_bytes()
        )
        checks[f"{prefix}_candidate_visible_bytes"] = (
            capture.candidate_visible_bytes == visible.read_bytes()
        )
    return checks


def _episode_plan(episode: Mapping[str, object]) -> dict[str, object]:
    return {
        "environment_seed": episode["environment_seed"],
        "policy_rng_seed": episode["policy_rng_seed"],
        "sampled_observation_latency_steps":
            episode["planned_latency"]["observation_steps"],
        "sampled_action_latency_steps":
            episode["planned_latency"]["action_steps"],
    }


def _resolve_paths(root: Path, arguments: argparse.Namespace) -> dict[str, Path]:
    return {
        "p50": _resolve(root, arguments.p50_input),
        "mapping": _resolve(root, arguments.mapping_input),
        "interaction": _resolve(root, arguments.interaction_input),
        "p72": _resolve(root, arguments.p72_input),
        "output": _resolve(root, arguments.output),
    }


def _require_frozen_paths(root: Path, paths: Mapping[str, Path]) -> None:
    expected = {
        "p50": root / FORMAL_P50_INPUT,
        "mapping": root / FORMAL_MAPPING_INPUT,
        "interaction": root / FORMAL_INTERACTION_INPUT,
        "p72": root / FORMAL_P72_INPUT,
        "output": root / FORMAL_OUTPUT,
    }
    for name, path in paths.items():
        if path != expected[name].resolve():
            raise ValueError(f"{name} path differs from frozen path")


def _provenance(
    root: Path, paths: Mapping[str, Path], source_commit: str
) -> dict[str, object]:
    inputs = {
        f"{directory.as_posix()}/{name}": _file_identity(root / directory / name)
        for directory, files in EXPECTED_INPUTS.items()
        for name in files
    }
    p72_identity = _file_identity(paths["p72"] / "report.json")
    p72_report = _read_json(paths["p72"] / "report.json")
    p72_manifest = _read_json(paths["p72"] / "manifest.json")
    p72_paths = (
        paths["p72"] / "report.json",
        paths["p72"] / "manifest.json",
    )
    frozen_expected = subprocess.run(
        ("git", "show", f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    frozen_actual = (root / FROZEN_DOCUMENT_PATH).read_bytes()
    expected_hashes = {
        f"{directory.as_posix()}/{name}": expected
        for directory, files in EXPECTED_INPUTS.items()
        for name, expected in files.items()
    }
    sources = {
        path.as_posix(): {
            **_file_identity(root / path),
            "head_blob": _git(root, "rev-parse", f"HEAD:{path.as_posix()}"),
            "working_blob": _git(root, "hash-object", "--", path.as_posix()),
        }
        for path in SOURCE_PATHS
    }
    _, bindings = load_default_formal_household_catalogs(root)
    xml = {
        task_id: recursive_xml_input_identity(root, binding.model_path)
        for task_id, binding in bindings.items()
    }
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
            inputs[name]["sha256"] == expected
            for name, expected in expected_hashes.items()
        ),
        "source_files_match_head": all(
            item["head_blob"] == item["working_blob"] for item in sources.values()
        ),
        "historical_trees_match": all(
            _git(root, "rev-parse", f"HEAD:{path}") == expected
            for path, expected in HISTORICAL_TREES.items()
        ),
        "p72_dependency_report_bound": (
            p72_identity["sha256"]
            == p72_manifest["artifacts"]["report.json"]["sha256"]
            and p72_report["decision"]
            == "accepted as residual P61 contract gap evidence"
            and p72_report["p68_dependency_gate_passed"] is True
            and subprocess.run(
                (
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    p72_report["source_commit"],
                    "HEAD",
                ),
                cwd=root,
                check=False,
            ).returncode
            == 0
        ),
        "p72_evidence_tracked_and_matches_head": all(
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", path.relative_to(root)),
                cwd=root,
                check=False,
                capture_output=True,
            ).returncode
            == 0
            and _git(
                root, "hash-object", "--", path.relative_to(root).as_posix()
            )
            == _git(root, "rev-parse", f"HEAD:{path.relative_to(root)}")
            for path in p72_paths
        ),
    }
    return {
        "checks": {**checks, "passed": all(checks.values())},
        "inputs": inputs,
        "p72_report": p72_identity,
        "frozen_document": _file_identity(root / FROZEN_DOCUMENT_PATH),
        "sources": sources,
        "recursive_mujoco_xml": xml,
        "historical_trees": dict(HISTORICAL_TREES),
    }


def _require_provenance(value: Mapping[str, object]) -> None:
    checks = value["checks"]
    if checks["passed"] is not True:
        failed = sorted(
            name for name, passed in checks.items()
            if name != "passed" and not passed
        )
        raise RuntimeError(f"P68 provenance gate failed: {', '.join(failed)}")


def _manifest(
    source_commit: str,
    arguments: argparse.Namespace,
    provenance: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    elapsed: float,
    decision: str,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": "complete",
        "source_commit": source_commit,
        "decision": decision,
        "command": list(_command(arguments)),
        "provenance": provenance,
        "runtime": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "mujoco": importlib.metadata.version("mujoco"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "wall_seconds": elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
            "disk_free_bytes": shutil.disk_usage(
                Path(__file__).resolve().parents[3]
            ).free,
        },
        "artifacts": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in sorted(artifacts.items())
        },
        "training_executed": False,
        "policy_inference_executed": False,
        "capability_claim_allowed": False,
        "task_success_claim_allowed": False,
        "generalization_claim_allowed": False,
        "hardware_safety_claim_allowed": False,
    }


def _require_budget(
    elapsed: float, peak_rss: int, artifacts: Mapping[str, bytes]
) -> None:
    if elapsed > MAX_WALL_SECONDS:
        raise RuntimeError("P68 wall-time budget exceeded")
    if peak_rss > MAX_RSS_BYTES:
        raise RuntimeError("P68 RSS budget exceeded")
    if sum(len(value) for value in artifacts.values()) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("P68 artifact budget exceeded")


def _require_disk(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("P68 disk-free guard failed")


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
        "--p50-input",
        arguments.p50_input.as_posix(),
        "--mapping-input",
        arguments.mapping_input.as_posix(),
        "--interaction-input",
        arguments.interaction_input.as_posix(),
        "--p72-input",
        arguments.p72_input.as_posix(),
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
