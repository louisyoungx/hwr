"""Preflight the frozen R0001-P50-E3 entity mapping without Episodes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.adapters.mujoco.entity_candidate_mapping import (
    EntityCandidateMappingError,
    mujoco_runtime_version,
    preflight_entity_role_tables,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.apps import (
    candidate_artifact_manifest,
    candidate_commit_is_ancestor,
    candidate_file_identity,
    candidate_json_bytes,
    candidate_sha256,
    candidate_source_commit,
    create_candidate_output,
)
from hwr.eval.tool_kinematics import recursive_xml_input_identity

PROPOSAL_ID = "R0001-P50-E3"
FROZEN_DOCUMENT_COMMIT = "88992a773ee2b0f214dba7975cdddf25f282d679"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0011/03-experiment.md")
FROZEN_DOCUMENT_BLOB = "2ea08a0ab8fea5b0444d4ba7b162e4129e5765b8"
FORMAL_DIRECTORY = Path(
    "runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001"
)
FORMAL_PLAN = FORMAL_DIRECTORY / "plan.json"
FORMAL_CAPSULES = FORMAL_DIRECTORY / "capsules.json"
FORMAL_E1_REPORT = FORMAL_DIRECTORY / "report.json"
FORMAL_E1_MANIFEST = FORMAL_DIRECTORY / "manifest.json"
FORMAL_E2_REPORT = Path(
    "runs/research-loop/0010/"
    "r0010-p50-e2-funnel-s20265001/report.json"
)
FORMAL_OUTPUT = Path(
    "runs/research-loop/0011/"
    "r0011-p50-e3-entity-coverage-s20265003"
)
FROZEN_INPUTS = {
    "plan": (
        FORMAL_PLAN,
        "5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab",
    ),
    "capsules": (
        FORMAL_CAPSULES,
        "223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf",
    ),
    "e1_report": (
        FORMAL_E1_REPORT,
        "b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0",
    ),
    "e1_manifest": (
        FORMAL_E1_MANIFEST,
        "cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86",
    ),
    "e2_report": (
        FORMAL_E2_REPORT,
        "4c7f36d20356d2f0f9c83d024412da5ec3a95dea8714e9a04d91d0cd686d0e39",
    ),
}
TASK_IDS = (
    "tidy_living_room_3d/v1",
    "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
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
}
PROTECTED_PATHS = (
    "src/hwr/eval/target_selection.py",
    "configs/adapters/mujoco/formal_3d_v1.json",
    "configs/tasks/formal_3d_v1.json",
    "assets/mujoco",
)
SOURCE_PATHS = {
    "preflight_app": Path(
        "src/hwr/apps/evaluate_entity_candidate_coverage.py"
    ),
    "mapping": Path(
        "src/hwr/adapters/mujoco/entity_candidate_mapping.py"
    ),
    "formal_generator": Path("src/hwr/eval/target_selection.py"),
}
FAILURE_SCHEMA = "hwr.p50-e3-preflight-failure/v1"
MANIFEST_SCHEMA = "hwr.p50-e3-preflight-artifacts/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--historical-capsules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    _validate_arguments(root, arguments, output)
    source_commit = candidate_source_commit(root)
    source_gate = require_preflight_source(root, source_commit)
    inputs = read_frozen_inputs(root)
    validate_frozen_input_lineage(inputs)
    identities = source_identities(root)
    command = [
        sys.executable,
        "-m",
        "hwr.apps.evaluate_entity_candidate_coverage",
        "--plan",
        str(arguments.plan),
        "--historical-capsules",
        str(arguments.historical_capsules),
        "--output",
        str(arguments.output),
    ]
    try:
        _, bindings = load_default_formal_household_catalogs(root)
        preflight_entity_role_tables(bindings, TASK_IDS)
    except EntityCandidateMappingError as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "decision": "inconclusive_design_infeasible",
            "reason": "frozen_body_role_conflict",
            "error": str(error),
            "error_details": error.details,
            "episode_count": 0,
            "physical_acquisition_count": 0,
            "measurement_evidence_accepted": False,
        }
        artifacts = {"failure.json": candidate_json_bytes(failure)}
        manifest = artifact_manifest(
            root,
            source_commit,
            command,
            source_gate,
            inputs,
            identities,
            artifacts,
        )
        artifacts["manifest.json"] = candidate_json_bytes(manifest)
        create_candidate_output(output, artifacts)
        return {
            "output": str(output),
            "decision": failure["decision"],
            "failure_sha256": manifest["artifacts"]["failure.json"]["sha256"],
            "manifest_sha256": candidate_sha256(artifacts["manifest.json"]),
        }
    raise RuntimeError(
        "frozen mapping unexpectedly passed; full P50-E3 is not implemented"
    )


def require_preflight_source(
    root: Path,
    source_commit: str,
    *,
    runner=subprocess.run,
) -> dict[str, object]:
    status = runner(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P50-E3 preflight requires clean committed source")
    ancestor = runner(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            FROZEN_DOCUMENT_COMMIT,
            source_commit,
        ),
        cwd=root,
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RuntimeError("P50-E3 frozen document is not an ancestor")
    document = runner(
        ("git", "show", f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    blob = runner(
        (
            "git",
            "rev-parse",
            f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}",
        ),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        blob != FROZEN_DOCUMENT_BLOB
        or (root / FROZEN_DOCUMENT_PATH).read_bytes() != document
    ):
        raise RuntimeError("P50-E3 frozen document content drifted")
    protected = runner(
        (
            "git",
            "diff",
            "--quiet",
            FROZEN_DOCUMENT_COMMIT,
            source_commit,
            "--",
            *PROTECTED_PATHS,
        ),
        cwd=root,
        check=False,
    ).returncode == 0
    if not protected:
        raise RuntimeError("P50-E3 target/config/XML drifted")
    trees = {
        path: runner(
            ("git", "rev-parse", f"{source_commit}:{path}"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for path in HISTORICAL_TREES
    }
    if trees != HISTORICAL_TREES:
        raise RuntimeError("P50-E3 historical research trees drifted")
    return {
        "clean_committed_source": True,
        "frozen_document_ancestor": ancestor,
        "frozen_document_blob": blob,
        "target_config_xml_unchanged": protected,
        "historical_research_loop_trees": trees,
        "passed": True,
    }


def read_frozen_inputs(root: Path) -> dict[str, dict[str, object]]:
    records = {}
    for name, (relative, expected) in FROZEN_INPUTS.items():
        path = root / relative
        content = path.read_bytes()
        actual = candidate_sha256(content)
        if actual != expected:
            raise RuntimeError(f"P50-E3 frozen input drifted: {name}")
        records[name] = {
            "path": relative.as_posix(),
            "sha256": actual,
            "bytes": len(content),
            "content": content,
        }
    return records


def validate_frozen_input_lineage(
    inputs: Mapping[str, Mapping[str, object]]
) -> None:
    e1_report = _object(inputs["e1_report"]["content"])
    e1_manifest = _object(inputs["e1_manifest"]["content"])
    e2_report = _object(inputs["e2_report"]["content"])
    artifacts = e1_manifest.get("artifacts", {})
    checks = {
        "e1_report_accepted": e1_report.get("decision")
        == "accepted as immutable acquisition evidence contract",
        "e1_manifest_complete": e1_manifest.get("status") == "complete",
        "e2_report_accepted": e2_report.get("decision")
        == "accepted as candidate-funnel measurement evidence",
        "plan_bound": artifacts.get("plan.json", {}).get("sha256")
        == FROZEN_INPUTS["plan"][1],
        "capsules_bound": artifacts.get("capsules.json", {}).get("sha256")
        == FROZEN_INPUTS["capsules"][1],
        "report_bound": artifacts.get("report.json", {}).get("sha256")
        == FROZEN_INPUTS["e1_report"][1],
    }
    if not all(checks.values()):
        raise RuntimeError("P50-E3 frozen input lineage differs")


def source_identities(root: Path) -> dict[str, object]:
    _, bindings = load_default_formal_household_catalogs(root)
    return {
        "binding": candidate_file_identity(
            root, root / "configs/adapters/mujoco/formal_3d_v1.json"
        ),
        "task_config": candidate_file_identity(
            root, root / "configs/tasks/formal_3d_v1.json"
        ),
        "sources": {
            name: candidate_file_identity(root, root / path)
            for name, path in SOURCE_PATHS.items()
        },
        "recursive_xml": {
            task_id: recursive_xml_input_identity(
                root, bindings[task_id].model_path
            )
            for task_id in TASK_IDS
        },
        "frozen_document": candidate_file_identity(
            root, root / FROZEN_DOCUMENT_PATH
        ),
        "historical_research_loop_trees": dict(HISTORICAL_TREES),
    }


def artifact_manifest(
    root: Path,
    source_commit: str,
    command: Sequence[str],
    source_gate: Mapping[str, object],
    inputs: Mapping[str, Mapping[str, object]],
    identities: Mapping[str, object],
    artifacts: Mapping[str, bytes],
) -> dict[str, object]:
    return candidate_artifact_manifest(
        schema=MANIFEST_SCHEMA,
        proposal_id=PROPOSAL_ID,
        source_commit=source_commit,
        frozen_document_commit=FROZEN_DOCUMENT_COMMIT,
        command=command,
        identities=identities,
        artifacts=artifacts,
        runtime_versions={
            "mujoco": mujoco_runtime_version(),
            "hwr_platform": importlib.metadata.version("hwr-platform"),
        },
        status="failed",
        extra={
            "frozen_document_commit_is_ancestor": (
                candidate_commit_is_ancestor(
                    root, FROZEN_DOCUMENT_COMMIT, source_commit
                )
            ),
            "source_gate": dict(source_gate),
            "fixed_inputs": {
                name: {
                    key: value[key]
                    for key in ("path", "sha256", "bytes")
                }
                for name, value in inputs.items()
            },
            "preflight_only": True,
            "episode_count": 0,
            "physical_acquisition_count": 0,
            "measurement_evidence_accepted": False,
        },
    )


def _validate_arguments(
    root: Path,
    arguments: argparse.Namespace,
    output: Path,
) -> None:
    expected = (
        (root / FORMAL_PLAN).resolve(),
        (root / FORMAL_CAPSULES).resolve(),
        (root / FORMAL_OUTPUT).resolve(),
    )
    actual = (
        _resolve(root, arguments.plan),
        _resolve(root, arguments.historical_capsules),
        output,
    )
    if actual != expected:
        raise ValueError("P50-E3 preflight requires frozen paths")
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError("P50-E3 output already exists")


def _object(content: object) -> dict[str, object]:
    if not isinstance(content, (bytes, bytearray)):
        raise RuntimeError("P50-E3 frozen input bytes are missing")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise RuntimeError("P50-E3 frozen input is not an object")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
