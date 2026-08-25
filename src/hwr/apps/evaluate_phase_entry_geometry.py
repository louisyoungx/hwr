"""Run the frozen R0001-P60 phase-entry geometry measurement."""

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
import tracemalloc
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.adapters.mujoco.phase_entry_geometry import PhaseEntryGeometryMujoco
from hwr.adapters.mujoco.training_catalog import load_default_formal_household_catalogs
from hwr.eval.phase_entry_geometry import (
    EPISODES_PER_CELL,
    EPISODES_SCHEMA,
    LATENCY_MATCH_LIMIT,
    PLAN_ID,
    PLAN_SCHEMA,
    PREFIX_STEPS,
    PROPOSAL_ID,
    RAW_SEED_LIMIT,
    SALT_COMMITMENT,
    SEED_AUDIT_SCHEMA,
    TASK_IDS,
    analyze_evidence,
    canonical_bytes,
    frozen_cells,
    raw_seed_record,
)
from hwr.eval.seed_contract import SEED_SCHEMA, read_seed_salt, require_seed_reveal
from hwr.eval.tool_kinematics import recursive_xml_input_identity

MODULE_NAME = "hwr.apps.evaluate_phase_entry_geometry"
REPORT_SCHEMA = "hwr.p60-phase-entry-report/v1"
MANIFEST_SCHEMA = "hwr.p60-phase-entry-artifacts/v1"
FAILURE_SCHEMA = "hwr.p60-phase-entry-failure/v1"
FROZEN_DOCUMENT_COMMIT = "a95dbbeacc80a974d7a234f2dc79442249eaf07b"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0012/03-experiment.md")
FORMAL_OUTPUT = Path("runs/research-loop/0012/r0012-p60-phase-entry-s20266001")
FORMAL_SALT_FILE = Path("runs/research-loop/0012/.host/p60-salt.txt")
BINDING_PATH = Path("configs/adapters/mujoco/formal_3d_v1.json")
TASK_PATH = Path("configs/tasks/formal_3d_v1.json")
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
SOURCE_PATHS = (
    Path("src/hwr/eval/phase_entry_geometry.py"),
    Path("src/hwr/adapters/mujoco/phase_entry_geometry.py"),
    Path("src/hwr/apps/evaluate_phase_entry_geometry.py"),
    Path("src/hwr/eval/seed_contract.py"),
    Path("src/hwr/eval/cartesian_convergence.py"),
    Path("src/hwr/eval/tool_kinematics.py"),
    Path("src/hwr/eval/target_selection.py"),
    Path("src/hwr/adapters/mujoco/cartesian_convergence.py"),
    Path("src/hwr/adapters/mujoco/target_selection_diagnostic.py"),
    Path("src/hwr/adapters/mujoco/formal_household_backend.py"),
    Path("src/hwr/adapters/mujoco/dual_arm_backend.py"),
    Path("src/hwr/adapters/mujoco/bindings.py"),
    Path("src/hwr/adapters/mujoco/entity_contact_graph.py"),
    Path("src/hwr/adapters/mujoco/contact_ledger.py"),
    Path("src/hwr/adapters/mujoco/model.py"),
    Path("src/hwr/adapters/mujoco/training_catalog.py"),
    Path("src/hwr/core/embodied.py"),
    Path("src/hwr/core/runtime.py"),
    Path("src/hwr/core/state_snapshot.py"),
    Path("src/hwr/core/types.py"),
    Path("src/hwr/safety/dual_arm.py"),
    Path("src/hwr/safety/supervisor.py"),
    Path("src/hwr/scenarios/formal3d.py"),
    FROZEN_DOCUMENT_PATH,
)
PROTECTED_PATHS = (
    "configs/adapters/mujoco/formal_3d_v1.json",
    "configs/tasks/formal_3d_v1.json",
    "assets/mujoco",
    "src/hwr/eval/seed_contract.py",
    "src/hwr/eval/cartesian_convergence.py",
    "src/hwr/eval/tool_kinematics.py",
    "src/hwr/eval/target_selection.py",
    "src/hwr/adapters/mujoco/cartesian_convergence.py",
    "src/hwr/adapters/mujoco/cartesian_convergence_provenance.py",
    "src/hwr/adapters/mujoco/target_selection_diagnostic.py",
    "src/hwr/adapters/mujoco/formal_household_backend.py",
    "src/hwr/adapters/mujoco/entity_contact_graph.py",
    "src/hwr/adapters/mujoco/contact_ledger.py",
    "src/hwr/safety",
)
MAX_WALL_SECONDS = 90 * 60
MAX_RSS_BYTES = 8 * 1024**3
MAX_ARTIFACT_BYTES = 2 * 1024**3
MIN_DISK_FREE_BYTES = 20 * 1024**3
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
    parser.add_argument("--salt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser
def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    salt_path = _resolve(root, arguments.salt_file)
    if output != (root / FORMAL_OUTPUT).resolve():
        raise ValueError("output path differs from frozen formal output")
    if salt_path != (root / FORMAL_SALT_FILE).resolve():
        raise ValueError("salt path differs from frozen formal input")
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = _command(arguments)
    started = time.perf_counter()
    tracemalloc.start()
    identities: dict[str, object] = {}
    artifacts: dict[str, bytes] = {}
    try:
        identities = _source_identities(root)
        _require_clean_source(root, identities)
        _require_disk_capacity(output)
        salt_payload = salt_path.read_bytes()
        salt = read_seed_salt(salt_path)
        require_seed_reveal(SALT_COMMITMENT, salt)
        plan, seed_audit, episodes, execution = execute_cohort(
            root,
            salt,
            source_commit,
            started=started,
        )
        analysis = (
            analyze_evidence(plan, seed_audit, episodes)
            if execution["decision"] == "cohort_complete"
            else None
        )
        repeated_analysis = (
            analyze_evidence(plan, seed_audit, episodes)
            if execution["decision"] == "cohort_complete"
            else None
        )
        report = _report(
            source_commit,
            execution,
            analysis,
            identities,
            deterministic_analysis=(
                canonical_bytes(analysis) == canonical_bytes(repeated_analysis)
            ),
        )
        artifacts = {
            "plan.json": _json_bytes(plan),
            "seed-audit.json": _json_bytes(seed_audit),
            "episodes.json": _json_bytes(episodes),
            "report.json": _json_bytes(report),
        }
        _require_artifact_budget(artifacts)
        manifest = _manifest(
            source_commit,
            command,
            identities,
            {
                "path": arguments.salt_file.as_posix(),
                **_bytes_identity(salt_payload),
            },
            artifacts,
            started,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _require_artifact_budget(artifacts)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "source_commit": source_commit,
            "decision": "invalid",
            "error_type": type(error).__name__,
            "error": str(error),
            **CLAIM_FLAGS,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _manifest(
            source_commit,
            command,
            identities,
            None,
            artifacts,
            started,
            status="failed",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
        raise
    finally:
        tracemalloc.stop()
    return {
        "output": str(output),
        "decision": report["decision"],
        "strict_diagnostic": report["strict_diagnostic"],
        "nominal_diagnostic": report["nominal_diagnostic"],
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
    }
def execute_cohort(
    root: Path,
    salt: str,
    source_commit: str,
    *,
    started: float,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    _require_catalogs(tasks, bindings)
    audits: list[dict[str, object]] = []
    physical_records: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    infeasible_cells = []
    hard_stop = None
    artifact_bytes_estimate = 0
    for cell in frozen_cells():
        bridge = PhaseEntryGeometryMujoco(
            tasks[cell.task_id],
            bindings[cell.task_id],
        )
        eligible: list[dict[str, object]] = []
        matched_count = 0
        raw_count = 0
        for raw_ordinal in range(RAW_SEED_LIMIT):
            budget_failure = _runtime_budget_failure(root, started)
            if budget_failure is not None:
                hard_stop = {
                    "reason": budget_failure,
                    "planned_episode_id": None,
                    "cell_id": cell.cell_id,
                }
                break
            raw_count = raw_ordinal + 1
            seed = raw_seed_record(salt, cell, raw_ordinal)
            sampled = bridge.sample_latencies(int(seed["environment_seed"]))
            matched = sampled == (
                cell.observation_latency_steps,
                cell.action_latency_steps,
            )
            audit = {
                **cell.to_dict(),
                **seed,
                "sampled_observation_latency_steps": sampled[0],
                "sampled_action_latency_steps": sampled[1],
                "latency_matched": matched,
                "physical_prefix_executed": matched,
                "eligibility_reason": (
                    "pending_prefix_evaluation"
                    if matched
                    else "natural_latency_mismatch"
                ),
            }
            if matched:
                matched_count += 1
                measured = bridge.inspect_prefix(
                    int(seed["environment_seed"]),
                    int(seed["policy_rng_seed"]),
                )
                record = {
                    **cell.to_dict(),
                    **seed,
                    "sampled_observation_latency_steps": sampled[0],
                    "sampled_action_latency_steps": sampled[1],
                    "latency_matched": True,
                    **measured,
                }
                record["episode_ordinal"] = None
                physical_records.append(record)
                artifact_bytes_estimate += len(_json_bytes(record))
                runtime_matches = (
                    measured["runtime_observation_latency_steps"],
                    measured["runtime_action_latency_steps"],
                ) == sampled and measured["latency_override_inactive"] is True
                if not runtime_matches:
                    measured["eligible"] = False
                    measured["eligibility_reason"] = "runtime_latency_mismatch"
                    record["eligible"] = False
                    record["eligibility_reason"] = "runtime_latency_mismatch"
                    hard_stop = {
                        "reason": "runtime_latency_mismatch",
                        "planned_episode_id": seed["planned_episode_id"],
                        "cell_id": cell.cell_id,
                    }
                audit["eligibility_reason"] = measured["eligibility_reason"]
                audit["eligible"] = measured["eligible"]
                audit["raw_prefix_trace_sha256"] = measured["raw_prefix_trace_sha256"]
                if measured["hard_safety_failure"] or measured[
                    "eligibility_reason"
                ] == "nonfinite_geometry":
                    hard_stop = {
                        "reason": measured["eligibility_reason"],
                        "planned_episode_id": seed["planned_episode_id"],
                        "cell_id": cell.cell_id,
                    }
                budget_failure = _runtime_budget_failure(root, started)
                if hard_stop is None and budget_failure is not None:
                    hard_stop = {
                        "reason": budget_failure,
                        "planned_episode_id": seed["planned_episode_id"],
                        "cell_id": cell.cell_id,
                    }
                if (
                    hard_stop is None
                    and artifact_bytes_estimate > MAX_ARTIFACT_BYTES
                ):
                    hard_stop = {
                        "reason": "artifact_budget_exceeded",
                        "planned_episode_id": seed["planned_episode_id"],
                        "cell_id": cell.cell_id,
                    }
                if measured["eligible"]:
                    record["episode_ordinal"] = len(eligible)
                    eligible.append(record)
            else:
                audit["eligible"] = False
            audits.append(audit)
            if hard_stop is not None or len(eligible) == EPISODES_PER_CELL:
                break
            if matched_count == LATENCY_MATCH_LIMIT:
                break
        if hard_stop is not None:
            break
        if len(eligible) != EPISODES_PER_CELL:
            infeasible_cells.append(
                {
                    **cell.to_dict(),
                    "eligible_count": len(eligible),
                    "latency_matched_count": matched_count,
                    "raw_seed_count": raw_count,
                }
            )
            break
        selected.extend(eligible)
    complete = (
        hard_stop is None
        and not infeasible_cells
        and len(selected) == len(frozen_cells()) * EPISODES_PER_CELL
    )
    published = selected if complete else []
    plan = {
        "schema_version": PLAN_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "plan_id": PLAN_ID,
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "salt_commitment": SALT_COMMITMENT,
        "seed_schema": SEED_SCHEMA,
        "natural_evaluation_latency_rejection": True,
        "reset_latency_override_used": False,
        "prefix_only": True,
        "b2_action_allowed": False,
        "complete_case_deletion_allowed": False,
        "raw_seed_limit_per_cell": RAW_SEED_LIMIT,
        "latency_match_limit_per_cell": LATENCY_MATCH_LIMIT,
        "episodes_per_cell": EPISODES_PER_CELL,
        "prefix_steps_per_episode": PREFIX_STEPS,
        "cells": [cell.to_dict() for cell in frozen_cells()],
        "planned_episode_count": len(published),
        "episodes": [_plan_episode(row) for row in published],
        "infeasible_cells": infeasible_cells,
        "hard_stop": hard_stop,
    }
    seed_audit = {
        "schema_version": SEED_AUDIT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "plan_id": PLAN_ID,
        "source_commit": source_commit,
        "salt_commitment": SALT_COMMITMENT,
        "salt_reveal": salt,
        "record_count": len(audits),
        "records": audits,
    }
    episodes = {
        "schema_version": EPISODES_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "plan_id": PLAN_ID,
        "source_commit": source_commit,
        "physical_prefix_count": len(physical_records),
        "eligible_physical_prefix_count": sum(
            bool(row["eligible"]) for row in physical_records
        ),
        "selected_episode_count": len(published),
        "records": physical_records,
    }
    decision = (
        "cohort_complete"
        if complete
        else ("invalid" if hard_stop is not None else "inconclusive_design_infeasible")
    )
    return plan, seed_audit, episodes, {
        "decision": decision,
        "infeasible_cells": infeasible_cells,
        "hard_stop": hard_stop,
    }
def _report(
    source_commit,
    execution,
    analysis,
    identities,
    *,
    deterministic_analysis,
) -> dict[str, object]:
    if execution["decision"] == "cohort_complete":
        if analysis is None or analysis["decision"] == "invalid":
            raise RuntimeError(
                None if analysis is None else analysis["validation_error"]
            )
        decision = analysis["decision"]
        strict = analysis["strict_diagnostic"]
        nominal = analysis["nominal_diagnostic"]
        summary = analysis["summary"]
        checks = {
            **analysis["checks"],
            "source_commit_bound": True,
            "frozen_document_ancestor_content_blob": all(
                identities["frozen_document"].get(name)
                for name in (
                    "commit_is_ancestor",
                    "content_matches",
                    "blob_matches",
                )
            ),
            "protected_source_model_identity": (
                identities["protected_frozen_source"].get("passed") is True
            ),
            "historical_tree_identity": True,
            "deterministic_analysis_bit_identical": deterministic_analysis,
        }
        checks["passed"] = all(
            value for name, value in checks.items() if name != "passed"
        )
        if not checks["passed"]:
            decision = "invalid"
            strict = nominal = None
    else:
        decision = execution["decision"]
        strict = nominal = summary = None
        checks = {"passed": False}
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "plan_id": PLAN_ID,
        "source_commit": source_commit,
        "decision": decision,
        "strict_diagnostic": strict,
        "nominal_diagnostic": nominal,
        "sample_unit": "Episode",
        "estimands_separate": True,
        "estimand": (
            "strict arm-chain outer-envelope exclusion and nominal finite-horizon "
            "B2 command support at the acquisition+B0+B1 entry state"
        ),
        "checks": checks,
        "summary": summary,
        "infeasible_cells": execution["infeasible_cells"],
        "hard_stop": execution["hard_stop"],
        "strict_reachability_claim_allowed": False,
        "ik_collision_path_dynamics_claim_allowed": False,
        **CLAIM_FLAGS,
    }


def _manifest(
    source_commit,
    command,
    identities,
    salt_identity,
    artifacts,
    started,
    *,
    status,
):
    elapsed = time.perf_counter() - started
    traced_peak = tracemalloc.get_traced_memory()[1] if tracemalloc.is_tracing() else 0
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": status,
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "command": list(command),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": importlib.metadata.version("mujoco"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "wall_seconds": elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
            "tracemalloc_peak_bytes": traced_peak,
            "disk_free_bytes": shutil.disk_usage(
                Path(__file__).resolve().parents[3]
            ).free,
        },
        "salt_input_identity": salt_identity,
        "source_identities": identities,
        "historical_research_loop_trees": dict(HISTORICAL_TREES),
        "frozen_design": _frozen_design(),
        "artifacts": {
            name: _bytes_identity(content)
            for name, content in sorted(artifacts.items())
        },
        **CLAIM_FLAGS,
    }


def _frozen_design() -> dict[str, object]:
    return {
        "plan_id": PLAN_ID,
        "salt_commitment": SALT_COMMITMENT,
        "cell_count": len(frozen_cells()),
        "episodes_per_cell": EPISODES_PER_CELL,
        "episode_count": len(frozen_cells()) * EPISODES_PER_CELL,
        "raw_seed_limit_per_cell": RAW_SEED_LIMIT,
        "latency_match_limit_per_cell": LATENCY_MATCH_LIMIT,
        "prefix_steps_per_episode": PREFIX_STEPS,
        "b2_action_allowed": False,
        "strict_pair_rule": "any_arm_strict_outer_impossible",
        "nominal_pair_rule": "any_arm_nominal_b2_support_deficit",
        "strict_and_nominal_decisions_separate": True,
    }


def _source_identities(root: Path) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    _require_catalogs(tasks, bindings)
    values = {
        path.as_posix(): _file_identity(root, root / path)
        for path in SOURCE_PATHS
    }
    values[BINDING_PATH.as_posix()] = _file_identity(root, root / BINDING_PATH)
    values[TASK_PATH.as_posix()] = _file_identity(root, root / TASK_PATH)
    values["recursive_mujoco_xml"] = {
        task: recursive_xml_input_identity(root, bindings[task].model_path)
        for task in TASK_IDS
    }
    values["frozen_document"] = _frozen_document_status(root)
    values["protected_frozen_source"] = _protected_source_status(root)
    values["git_trees"] = {
        path: _git_output(root, ("rev-parse", f"HEAD:{path}"))
        for path in ("src/hwr", "configs", "assets")
    }
    return values


def _require_clean_source(root: Path, identities: Mapping[str, object]) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P60 runner requires clean committed source")
    frozen = identities["frozen_document"]
    if not all(
        frozen.get(name)
        for name in ("commit_is_ancestor", "content_matches", "blob_matches")
    ):
        raise RuntimeError("P60 frozen experiment document drifted")
    protected = identities["protected_frozen_source"]
    if protected.get("passed") is not True:
        raise RuntimeError("P60 protected source/model inputs drifted")
    actual_trees = {
        path: _git_output(root, ("rev-parse", f"HEAD:{path}"))
        for path in HISTORICAL_TREES
    }
    if actual_trees != HISTORICAL_TREES:
        raise RuntimeError("P60 historical research-loop tree drifted")


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


def _protected_source_status(root: Path) -> dict[str, object]:
    frozen = _git_tree_entries(root, FROZEN_DOCUMENT_COMMIT, PROTECTED_PATHS)
    current = _git_tree_entries(root, "HEAD", tuple(sorted(frozen)))
    blob_changes = {
        path
        for path in set(frozen) | set(current)
        if frozen.get(path) != current.get(path)
    }
    changed_paths = _git_output(
        root,
        ("diff", "--name-only", FROZEN_DOCUMENT_COMMIT, "HEAD", "--", *PROTECTED_PATHS),
    ).splitlines()
    changed = sorted(blob_changes | set(changed_paths))
    return {
        "base_commit": FROZEN_DOCUMENT_COMMIT,
        "checked_paths": sorted(frozen),
        "changed_paths": changed,
        "passed": bool(frozen) and not changed,
    }


def _git_tree_entries(root, commit, pathspecs):
    output = _git_output(root, ("ls-tree", "-r", commit, "--", *pathspecs))
    result = {}
    for line in output.splitlines():
        metadata, path = line.split("\t", 1)
        _, kind, blob = metadata.split()
        if kind == "blob":
            result[path] = blob
    return result


def _plan_episode(row):
    fields = (
        "ordinal",
        "cell_id",
        "task_id",
        "observation_latency_steps",
        "action_latency_steps",
        "raw_seed_ordinal",
        "planned_episode_id",
        "environment_seed",
        "policy_rng_seed",
        "sampled_observation_latency_steps",
        "sampled_action_latency_steps",
    )
    result = {name: row[name] for name in fields}
    result["episode_ordinal"] = row["episode_ordinal"]
    return result


def _require_catalogs(tasks, bindings) -> None:
    if tuple(tasks) != TASK_IDS or tuple(bindings) != TASK_IDS:
        raise RuntimeError("P60 task/binding catalog order differs")


def _runtime_budget_failure(root: Path, started: float) -> str | None:
    if time.perf_counter() - started > MAX_WALL_SECONDS:
        return "wall_time_budget_exceeded"
    if _peak_rss_bytes() > MAX_RSS_BYTES:
        return "rss_budget_exceeded"
    if shutil.disk_usage(root).free < MIN_DISK_FREE_BYTES:
        return "disk_free_below_20_gib"
    return None


def _require_disk_capacity(output: Path) -> None:
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("P60 requires at least 20 GiB free")


def _require_artifact_budget(artifacts: Mapping[str, bytes]) -> None:
    if sum(len(content) for content in artifacts.values()) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("P60 artifact budget exceeded")


def _command(arguments) -> list[str]:
    return [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--salt-file",
        arguments.salt_file.as_posix(),
        "--output",
        arguments.output.as_posix(),
    ]


def _source_commit(root: Path) -> str:
    commit = _git_output(root, ("rev-parse", "HEAD"))
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P60 requires a full Git source commit")
    return commit


def _git_output(root: Path, arguments: Sequence[str]) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_identity(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        **_bytes_identity(path.read_bytes()),
    }


def _bytes_identity(content: bytes) -> dict[str, object]:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        for name, content in sorted(artifacts.items()):
            temporary = staging / f"{name}.tmp"
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, staging / name)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()

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
