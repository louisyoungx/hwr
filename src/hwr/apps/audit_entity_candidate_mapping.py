"""Audit the frozen R0001-P50-E4 exact-geom evaluator mapping contract."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
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

from hwr.adapters.mujoco.entity_candidate_mapping import (
    ALIAS_SCHEMA,
    EntityCandidateMappingError,
    TaskAliasContract,
    classify_segmentation_entity,
    load_entity_alias_contracts,
    preflight_exact_geom_role_tables,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.apps import (
    candidate_file_identity as _file_identity,
    candidate_json_bytes as _json_bytes,
    candidate_sha256 as _sha256,
    candidate_source_commit as _source_commit,
    create_candidate_output as _create_output,
    resolve_candidate_path as _resolve,
)
from hwr.eval.tool_kinematics import recursive_xml_input_identity

PROPOSAL_ID = "R0001-P50-E4"
MODULE_NAME = "hwr.apps.audit_entity_candidate_mapping"
TABLES_SCHEMA = "hwr.p50-e4-exact-geom-tables/v1"
REPORT_SCHEMA = "hwr.p50-e4-exact-geom-report/v1"
MANIFEST_SCHEMA = "hwr.p50-e4-exact-geom-artifacts/v1"
FROZEN_DOCUMENT_COMMIT = "a95dbbeacc80a974d7a234f2dc79442249eaf07b"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0012/03-experiment.md")
FROZEN_DOCUMENT_BLOB = "55219f9b42693a4c579a13919e754d24e94a15c8"
FROZEN_CONTEXT_PATH = Path("docs/research-loop/0012/00-context.md")
FROZEN_CONTEXT_BLOB = "f25965426679b0dda3e437f1ef697efdea000757"
ALIASES_PATH = Path("configs/eval/entity_candidate_aliases_v1.json")
BINDING_PATH = Path("configs/adapters/mujoco/formal_3d_v1.json")
TASK_PATH = Path("configs/tasks/formal_3d_v1.json")
FORMAL_OUTPUT = Path("runs/research-loop/0012/r0012-p50-e4-mapping-s20265004")
TASK_IDS = ("tidy_living_room_3d/v1", "clear_dining_table_3d/v1",
            "store_kitchen_items_3d/v1")
SOURCE_PATHS = {
    "mapping": Path("src/hwr/adapters/mujoco/entity_candidate_mapping.py"),
    "audit_app": Path("src/hwr/apps/audit_entity_candidate_mapping.py"),
}
PROTECTED_PATHS = (BINDING_PATH, TASK_PATH, Path("assets/mujoco"))
FORBIDDEN_ISOLATION_MODULES = frozenset(
    (
        "hwr.adapters.mujoco.entity_candidate_mapping",
        "hwr.apps.audit_entity_candidate_mapping",
    )
)
FORBIDDEN_ISOLATION_SYMBOLS = frozenset(
    (
        "EntityAlias",
        "TaskAliasContract",
        "TaskVisibleGeom",
        "build_exact_geom_role_table",
        "classify_segmentation_entity",
        "load_entity_alias_contracts",
        "preflight_exact_geom_role_tables",
    )
)
CLAIM_FLAGS = {
    "training_executed": False,
    "policy_inference_executed": False,
    "capability_claim_allowed": False,
    "task_success_claim_allowed": False,
    "generalization_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
    "entity_coverage_claim_allowed": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    aliases_path = _resolve(root, arguments.aliases)
    output = _resolve(root, arguments.output)
    _validate_arguments(root, aliases_path, output)
    started = time.perf_counter()
    tracemalloc.start()
    source_commit = _source_commit(root)
    _, bindings = load_default_formal_household_catalogs(root)
    identities = _source_identities(root, bindings, aliases_path)
    source_gate = _require_clean_source(root, identities)
    contracts = load_entity_alias_contracts(aliases_path)
    isolation = audit_alias_isolation(root)
    evaluation = evaluate_mapping_contract(bindings, contracts, isolation)
    artifacts = {
        "tables.json": _json_bytes(evaluation["tables"]),
        "report.json": _json_bytes(evaluation["report"]),
    }
    elapsed = time.perf_counter() - started
    _, traced_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss_bytes = _peak_rss_bytes()
    manifest = _manifest(
        source_commit,
        _command(arguments),
        source_gate,
        identities,
        artifacts,
        started,
        traced_peak_bytes,
        peak_rss_bytes,
    )
    artifacts["manifest.json"] = _json_bytes(manifest)
    _require_budgets(
        time.perf_counter() - started,
        max(traced_peak_bytes, peak_rss_bytes),
        sum(len(content) for content in artifacts.values()),
    )
    _create_output(output, artifacts)
    return {
        "output": str(output),
        "decision": evaluation["report"]["decision"],
        "tables_sha256": manifest["artifacts"]["tables.json"]["sha256"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": _sha256(artifacts["manifest.json"]),
    }


def evaluate_mapping_contract(
    bindings: Mapping[str, object],
    contracts: Mapping[str, TaskAliasContract],
    isolation: Mapping[str, object],
) -> dict[str, object]:
    try:
        first = preflight_exact_geom_role_tables(
            bindings, contracts, TASK_IDS
        )
        second = preflight_exact_geom_role_tables(
            bindings, contracts, TASK_IDS
        )
    except EntityCandidateMappingError as error:
        decision = (
            "rejected_design_not_expressive"
            if error.details.get("failure_kind")
            in (
                "alias_contract_semantic",
                "task_visible_inventory_mismatch",
            )
            else "invalid"
        )
        tables = {
            "schema_version": TABLES_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "task_order": list(TASK_IDS),
            "tasks": [],
        }
        report = {
            "schema_version": REPORT_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "decision": decision,
            "error": str(error),
            "error_details": error.details,
            "checks": {
                "three_scene_preflight": False,
                "all_mapping_contract_gates": False,
            },
            **CLAIM_FLAGS,
        }
        return {"tables": tables, "report": report}
    first_tables = _tables_document(first)
    second_tables = _tables_document(second)
    first_bytes = _json_bytes(first_tables)
    second_bytes = _json_bytes(second_tables)
    deterministic = first_bytes == second_bytes
    guards = negative_guard_audit(first)
    alias_count = sum(int(table["alias_count"]) for table in first.values())
    exact_conflicts = sum(
        int(table["exact_geom_role_conflict_count"])
        for table in first.values()
    )
    unknown_inventory = sum(
        int(table["task_visible_inventory_unknown_count"])
        for table in first.values()
    )
    semantic_checks = {
        "three_scene_preflight": list(first) == list(TASK_IDS),
        "eight_one_hop_same_body_aliases": alias_count == 8,
        "zero_exact_geom_role_conflicts": exact_conflicts == 0,
        "zero_task_visible_inventory_unknown": unknown_inventory == 0,
        "zero_negative_guard_mislabels": guards["mismatch_count"] == 0,
        "illegal_segmentation_object_types_fail_closed": guards[
            "illegal_object_types_fail_closed"
        ],
    }
    integrity_checks = {
        "tables_bit_identical": deterministic,
        "tables_hash_bit_identical": _sha256(first_bytes)
        == _sha256(second_bytes),
        "alias_schema_frozen": all(
            isinstance(value, TaskAliasContract)
            for value in contracts.values()
        ),
        "import_ast_isolation_passed": isolation.get("passed") is True,
    }
    if not all(semantic_checks.values()):
        decision = "rejected_design_not_expressive"
    elif not all(integrity_checks.values()):
        decision = "invalid"
    else:
        decision = "accepted as exact-geom evaluator mapping contract"
    report = {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "decision": decision,
        "task_count": len(first),
        "alias_count": alias_count,
        "exact_geom_role_conflict_count": exact_conflicts,
        "task_visible_inventory_unknown_count": unknown_inventory,
        "negative_guard_mislabel_count": guards["mismatch_count"],
        "determinism": {
            "first_tables": _bytes_identity(first_bytes),
            "reconstructed_tables": _bytes_identity(second_bytes),
            "bit_identical": deterministic,
        },
        "negative_guards": guards,
        "isolation": dict(isolation),
        "checks": {
            **semantic_checks,
            **integrity_checks,
            "all_mapping_contract_gates": (
                all(semantic_checks.values())
                and all(integrity_checks.values())
            ),
        },
        **CLAIM_FLAGS,
    }
    return {"tables": first_tables, "report": report}


def negative_guard_audit(
    tables: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    records = []

    def require(
        task_id: str,
        entity_kind: str,
        entity_name: str,
        actual_role: object,
        expected_role: str,
    ) -> None:
        records.append(
            {
                "task_id": task_id,
                "entity_kind": entity_kind,
                "entity_name": entity_name,
                "expected_role": expected_role,
                "actual_role": actual_role,
                "passed": actual_role == expected_role,
            }
        )

    for name in ("sideboard_top", "sideboard_body"):
        require(
            TASK_IDS[1],
            "geom",
            name,
            _geom(tables[TASK_IDS[1]], name)["role"],
            "other_furniture",
        )
    require(
        TASK_IDS[2],
        "geom",
        "drawer_handle_visual",
        _geom(tables[TASK_IDS[2]], "drawer_handle_visual")["role"],
        "articulation",
    )
    for name in (
        "drawer_bottom",
        "drawer_front",
        "drawer_back",
        "drawer_left",
        "drawer_right",
        "drawer_divider",
    ):
        require(
            TASK_IDS[2],
            "geom",
            name,
            _geom(tables[TASK_IDS[2]], name)["role"],
            "target_container",
        )
    for task_id, table in tables.items():
        for geom in table["geoms"]:
            name = geom["geom_name"]
            if task_id == TASK_IDS[2] and isinstance(name, str):
                if name.startswith("drawer_frame_"):
                    require(
                        task_id,
                        "geom",
                        name,
                        geom["role"],
                        "other_furniture",
                    )
            if geom["body_id"] == 0 and name and name != "floor":
                require(
                    task_id,
                    "world_geom",
                    name,
                    geom["role"],
                    "other_furniture",
                )
        for site in table["sites"]:
            require(
                task_id,
                "site",
                str(site["site_name"]),
                site["role"],
                "unknown_site",
            )
        background = table["background"]
        require(
            task_id,
            "background",
            "(-1,-1)",
            background["role"],
            "background",
        )
        illegal_type = 2_147_483_647
        illegal = classify_segmentation_entity(table, 0, illegal_type)
        require(
            task_id,
            "illegal_object_type",
            f"(0,{illegal_type})",
            illegal["role"],
            "unknown",
        )
    records.sort(
        key=lambda value: (
            value["task_id"],
            value["entity_kind"],
            value["entity_name"],
        )
    )
    mismatches = [value for value in records if not value["passed"]]
    illegal_cases = [
        value for value in records
        if value["entity_kind"] == "illegal_object_type"
    ]
    return {
        "guard_count": len(records),
        "mismatch_count": len(mismatches),
        "illegal_object_type_case_count": len(illegal_cases),
        "illegal_object_types_fail_closed": (
            len(illegal_cases) == len(tables)
            and all(value["passed"] for value in illegal_cases)
        ),
        "mismatches": mismatches,
        "records": records,
    }


def audit_alias_isolation(root: Path) -> dict[str, object]:
    paths = _isolation_paths(root)
    violations = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    if item.name in FORBIDDEN_ISOLATION_MODULES:
                        violations.append(
                            _violation(relative, node, "forbidden_import", item.name)
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported = {
                    f"{module}.{item.name}" if module else item.name
                    for item in node.names
                }
                if (
                    module in FORBIDDEN_ISOLATION_MODULES
                    or imported & FORBIDDEN_ISOLATION_MODULES
                    or any(
                        item.name in FORBIDDEN_ISOLATION_SYMBOLS
                        for item in node.names
                    )
                ):
                    violations.append(
                        _violation(relative, node, "forbidden_import", module)
                    )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if (
                    ALIASES_PATH.as_posix() in node.value
                    or ALIASES_PATH.name in node.value
                    or node.value in FORBIDDEN_ISOLATION_MODULES
                ):
                    violations.append(
                        _violation(
                            relative,
                            node,
                            "forbidden_alias_path",
                            node.value,
                        )
                    )
    violations.sort(
        key=lambda value: (
            value["path"],
            value["line"],
            value["column"],
            value["kind"],
            value["symbol"],
        )
    )
    return {
        "schema_version": "hwr.p50-e4-alias-isolation-audit/v1",
        "passed": not violations,
        "audited_source_count": len(paths),
        "audited_sources": [
            path.relative_to(root).as_posix() for path in paths
        ],
        "evaluator_only_exclusions": [
            SOURCE_PATHS["mapping"].as_posix(),
            SOURCE_PATHS["audit_app"].as_posix(),
            "src/hwr/apps/evaluate_entity_candidate_coverage.py",
        ],
        "forbidden_modules": sorted(FORBIDDEN_ISOLATION_MODULES),
        "forbidden_symbols": sorted(FORBIDDEN_ISOLATION_SYMBOLS),
        "forbidden_alias_path": ALIASES_PATH.as_posix(),
        "violations": violations,
    }


def _isolation_paths(root: Path) -> tuple[Path, ...]:
    excluded = {
        root / SOURCE_PATHS["mapping"],
        root / SOURCE_PATHS["audit_app"],
        root / "src/hwr/apps/evaluate_entity_candidate_coverage.py",
    }
    paths = tuple(
        path
        for path in sorted((root / "src/hwr").rglob("*.py"))
        if path not in excluded
    )
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError("P50-E4 isolation source inventory is incomplete")
    return paths


def _source_identities(
    root: Path,
    bindings: Mapping[str, object],
    aliases_path: Path,
) -> dict[str, object]:
    files = {
        "aliases": aliases_path,
        "binding": root / BINDING_PATH,
        "task_config": root / TASK_PATH,
        "frozen_document": root / FROZEN_DOCUMENT_PATH,
        "frozen_context": root / FROZEN_CONTEXT_PATH,
        **{
            name: root / path
            for name, path in SOURCE_PATHS.items()
        },
    }
    context = _frozen_file_status(root, FROZEN_CONTEXT_PATH, FROZEN_CONTEXT_BLOB)
    declared = _context_tree_inventory(
        (root / FROZEN_CONTEXT_PATH).read_text(encoding="utf-8")
    )
    actual = {
        path: _git_output(root, ("rev-parse", f"HEAD:{path}"))
        for path in declared
    }
    return {
        "files": {
            name: _file_identity(root, path)
            for name, path in sorted(files.items())
        },
        "recursive_xml": {
            task_id: recursive_xml_input_identity(
                root, bindings[task_id].model_path
            )
            for task_id in TASK_IDS
        },
        "frozen_document": _frozen_file_status(
            root, FROZEN_DOCUMENT_PATH, FROZEN_DOCUMENT_BLOB),
        "frozen_context": context,
        "historical_research_loop_trees": {
            "declared_by_frozen_context": declared,
            "actual": actual,
            "actual_matches_context": actual == declared,
        },
    }


def _require_clean_source(
    root: Path,
    identities: Mapping[str, object],
) -> dict[str, object]:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P50-E4 audit requires clean committed source")
    for name in ("frozen_document", "frozen_context"):
        frozen = identities.get(name)
        required = ("commit_is_ancestor", "content_matches", "blob_matches")
        if not isinstance(frozen, Mapping) or not all(
            frozen.get(field) for field in required
        ):
            raise RuntimeError(f"P50-E4 {name.replace('_', ' ')} drifted")
    protected = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            FROZEN_DOCUMENT_COMMIT,
            "HEAD",
            "--",
            *(path.as_posix() for path in PROTECTED_PATHS),
        ),
        cwd=root,
        check=False,
    ).returncode == 0
    if not protected:
        raise RuntimeError("P50-E4 binding, task, or XML inputs drifted")
    trees = identities.get("historical_research_loop_trees")
    if (
        not isinstance(trees, Mapping)
        or trees.get("actual_matches_context") is not True
    ):
        raise RuntimeError("P50-E4 historical research trees drifted")
    return {
        "clean_committed_source": True,
        "frozen_document_commit_is_ancestor": True,
        "frozen_document_content_matches": True,
        "frozen_document_blob_matches": True,
        "frozen_context_commit_is_ancestor": True,
        "frozen_context_content_matches": True,
        "frozen_context_blob_matches": True,
        "binding_task_xml_unchanged": True,
        "historical_research_loop_trees": trees,
        "passed": True,
    }


def _frozen_file_status(
    root: Path, path: Path, expected_blob: str
) -> dict[str, object]:
    ancestor = subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            FROZEN_DOCUMENT_COMMIT,
            "HEAD",
        ),
        cwd=root,
        check=False,
    ).returncode == 0
    expected = subprocess.run(
        (
            "git",
            "show",
            f"{FROZEN_DOCUMENT_COMMIT}:{path.as_posix()}",
        ),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    actual = (root / path).read_bytes()
    frozen_blob = _git_output(
        root, ("rev-parse", f"{FROZEN_DOCUMENT_COMMIT}:{path.as_posix()}")
    )
    current_blob = _git_output(
        root, ("rev-parse", f"HEAD:{path.as_posix()}")
    )
    return {
        "path": path.as_posix(),
        "commit": FROZEN_DOCUMENT_COMMIT,
        "blob": expected_blob,
        "commit_is_ancestor": ancestor,
        "content_matches": actual == expected,
        "blob_matches": current_blob == frozen_blob == expected_blob,
        "current": _bytes_identity(actual),
        "frozen": _bytes_identity(expected),
    }


def _context_tree_inventory(content: str) -> dict[str, str]:
    records = {}
    for line in content.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4:
            continue
        path = fields[1].strip("`").rstrip("/")
        tree = fields[2].strip("`")
        if not path.startswith("docs/research-loop/"):
            continue
        if path in records or len(tree) != 40 or any(
            value not in "0123456789abcdef" for value in tree
        ):
            raise RuntimeError("P50-E4 frozen context tree inventory is invalid")
        records[path] = tree
    expected = tuple(
        f"docs/research-loop/{index:04d}" for index in range(1, 12)
    )
    if tuple(records) != expected:
        raise RuntimeError("P50-E4 frozen context tree inventory is incomplete")
    return records


def _manifest(
    source_commit: str,
    command: Sequence[str],
    source_gate: Mapping[str, object],
    identities: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    started: float,
    traced_peak_bytes: int,
    peak_rss_bytes: int,
) -> dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": "complete",
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "command": list(command),
        "source_gate": dict(source_gate),
        "source_identities": identities,
        "input_identities": {
            "aliases": identities["files"]["aliases"],
            "binding": identities["files"]["binding"],
            "task_config": identities["files"]["task_config"],
            "frozen_context": identities["files"]["frozen_context"],
            "recursive_xml": identities["recursive_xml"],
        },
        "runtime": {
            "python": platform.python_version(),
            "mujoco": importlib.metadata.version("mujoco"),
            "numpy": np.__version__,
            "hwr_platform": importlib.metadata.version("hwr-platform"),
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
        "frozen_design": {
            "alias_schema": ALIAS_SCHEMA,
            "task_order": list(TASK_IDS),
            "expected_alias_count": 8,
            "exact_geom_claims_only": True,
            "alias_hops": 1,
            "same_body_alias_required": True,
            "episode_count": 0,
            "action_count": 0,
        },
        "artifacts": {
            name: _bytes_identity(content)
            for name, content in sorted(artifacts.items())
        },
        **CLAIM_FLAGS,
    }


def _tables_document(
    tables: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": TABLES_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "task_order": list(TASK_IDS),
        "tasks": [tables[task_id] for task_id in TASK_IDS],
    }


def _geom(table: Mapping[str, object], name: str) -> Mapping[str, object]:
    matches = [
        value for value in table["geoms"] if value["geom_name"] == name
    ]
    if len(matches) != 1:
        raise EntityCandidateMappingError(
            f"negative guard geom is not unique: {name}"
        )
    return matches[0]


def _validate_arguments(
    root: Path,
    aliases_path: Path,
    output: Path,
) -> None:
    if aliases_path != (root / ALIASES_PATH).resolve():
        raise ValueError("P50-E4 aliases path differs from frozen input")
    if output != (root / FORMAL_OUTPUT).resolve():
        raise ValueError("P50-E4 output path differs from frozen output")
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError("P50-E4 output or staging already exists")


def _require_budgets(
    wall_seconds: float,
    peak_rss_bytes: int,
    artifact_bytes: int,
) -> None:
    if wall_seconds >= 120.0:
        raise RuntimeError("P50-E4 wall-time budget exceeded")
    if peak_rss_bytes >= 1024**3:
        raise RuntimeError("P50-E4 RSS budget exceeded")
    if artifact_bytes >= 20 * 1024**2:
        raise RuntimeError("P50-E4 artifact budget exceeded")


def _command(arguments: argparse.Namespace) -> list[str]:
    return [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--aliases",
        str(arguments.aliases),
        "--output",
        str(arguments.output),
    ]


def _violation(
    path: str,
    node: ast.AST,
    kind: str,
    symbol: str,
) -> dict[str, object]:
    return {
        "path": path,
        "line": int(getattr(node, "lineno", 0)),
        "column": int(getattr(node, "col_offset", 0)),
        "kind": kind,
        "symbol": symbol,
    }


def _git_output(root: Path, arguments: Sequence[str]) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bytes_identity(content: bytes) -> dict[str, object]:
    return {"bytes": len(content), "sha256": _sha256(content)}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
