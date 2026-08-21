"""Run the frozen R0001-P41-E2 target-index-only interaction diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.adapters.mujoco.training_catalog import load_default_formal_household_catalogs
from hwr.adapters.mujoco.target_selection_diagnostic import (
    BranchConfiguration,
    TargetSelectionDiagnostic,
)
from hwr.eval.seed_contract import (
    SEED_SCHEMA,
    derive_domain_seed,
    planned_episode_id,
    seed_commitment,
    verify_seed_reveal,
)
from hwr.eval.target_selection import (
    CANDIDATE_SCHEMA,
    INPUT_SCHEMA,
    PHASES,
    PLAN_SCHEMA,
    PLANNED_HORIZON,
    POWER_SCHEMA,
    PROPOSAL_ID,
    TASK_IDS,
    TERMINAL_SCHEMA,
    select_control_index,
)
from hwr.eval.target_selection_safety import (
    evaluate_safety_guards,
    evaluate_synthetic_power,
    paired_primary_statistics,
)
MODULE_NAME = "hwr.apps.evaluate_target_selection"
REPORT_SCHEMA = "hwr.p41-target-selection-report/v1"
MANIFEST_SCHEMA = "hwr.p41-target-selection-artifacts/v1"
FAILURE_SCHEMA = "hwr.p41-target-selection-failure/v1"
SMOKE_SALT = "R0001-P41-E2-smoke-s20264101"
FORMAL_SALT = "R0001-P41-E2-formal-s20264102"
FROZEN_DOCUMENT_COMMIT = "565a881a6e09d3136bbc0f311d386b418b7b55fe"
FROZEN_PARENT_COMMIT = "4c4efda16759577cb05098a7628f29d3bfbef890"
P40_REPORT = Path("runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002/report.json")
P40_MANIFEST = Path("runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002/manifest.json")
P40_REPORT_SHA256 = "987a2217cf9f5c6eb08b018b3bf13164917c75bccbfa77140a8786c80987f841"
P40_MANIFEST_SHA256 = "fdb847a41f55a7a3bb362d650baa2d131e2a5178ac73166336d57368ba60546b"
BINDING_PATH = Path("configs/adapters/mujoco/formal_3d_v1.json")
BINDING_SHA256 = "7984ef2544bb618269681d274257a598b02621371a26de002bfdd8bbf7decab6"
BINDING_BYTES = 3051
TASK_PATH = Path("configs/tasks/formal_3d_v1.json")
TASK_SHA256 = "fa180803a86b42bc633dbf119fa596dd74d3b5c18bf4c2e4f75be97dcccb2a7d"
TASK_BYTES = 5480
CLAIM_FLAGS = {
    "training_executed": False,
    "policy_inference_executed": False,
    "capability_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
    "action_causality_claim_allowed": False,
    "task_success_claim_allowed": False,
    "generalization_claim_allowed": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    _validate_mode(arguments)
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--output",
        str(arguments.output),
        "--salt",
        arguments.salt,
    ] + (["--smoke"] if arguments.smoke else [])
    try:
        identities = _source_identities(root)
        _require_clean_source(root, identities)
        power = evaluate_synthetic_power()
        plan = build_plan(
            root,
            arguments.salt,
            smoke=arguments.smoke,
            selected_pair_count=power["selected_pair_count"],
        )
        terminals = execute_plan(root, plan, smoke=arguments.smoke)
        analysis = analyze_terminals(terminals, smoke=arguments.smoke)
        report = _build_report(
            source_commit, command, plan, terminals, power, analysis, arguments.smoke
        )
        artifacts = {
            "report.json": _json_bytes(report),
            "plan.json": _json_bytes(plan),
            "terminals.json": _json_bytes(terminals),
            "power.json": _json_bytes(power),
        }
        manifest = _manifest(
            source_commit, command, identities, plan, terminals, artifacts,
            smoke=arguments.smoke, status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "source_commit": source_commit,
            "mode": "smoke" if arguments.smoke else "formal",
            "error_type": type(error).__name__,
            "error": str(error),
            **CLAIM_FLAGS,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _manifest(
            source_commit,
            command,
            _source_identities(root),
            None,
            None,
            artifacts,
            smoke=arguments.smoke,
            status="failed",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
        raise
    return {
        "output": str(output),
        "decision": report["decision"],
        "mode": report["mode"],
        "selected_pair_count": power["selected_pair_count"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
    }


def build_plan(
    root: Path,
    salt: str,
    *,
    smoke: bool,
    selected_pair_count: int | None,
) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    if set(tasks) != set(TASK_IDS):
        raise RuntimeError("P41 task catalog differs from frozen contract")
    commitment = seed_commitment(salt)
    cells = [
        {
            "task_id": task,
            "observation_latency_steps": observation,
            "action_latency_steps": action,
            "domain": "supported" if observation in (1, 2) else "challenge",
            "replicate_count": 3 if observation in (1, 2) else 2,
        }
        for task in TASK_IDS
        for observation in (1, 2, 3)
        for action in (1, 2, 3)
    ]
    if smoke:
        cells = [
            {
                "task_id": task,
                "observation_latency_steps": latency,
                "action_latency_steps": latency,
                "domain": "smoke",
                "replicate_count": 1,
            }
            for task in TASK_IDS
            for latency in (1, 2)
        ]
    if not smoke and selected_pair_count != 54:
        return _unavailable_plan(salt, cells, selected_pair_count)
    pairs, rejected = [], []
    for cell_ordinal, cell in enumerate(cells):
        diagnostic = TargetSelectionDiagnostic(
            tasks[cell["task_id"]], bindings[cell["task_id"]]
        )
        replicate_count = int(cell["replicate_count"])
        accepted = 0
        candidate_ordinal = 0
        while accepted < replicate_count:
            identity = planned_episode_id(
                f"{PROPOSAL_ID}-{'smoke' if smoke else 'formal'}",
                str(cell["task_id"]),
                f"cell-{cell_ordinal}",
                candidate_ordinal,
            )
            environment_seed = derive_domain_seed(salt, "environment", identity)
            policy_seed = derive_domain_seed(salt, "policy", identity)
            sampled_observation, sampled_action = diagnostic.sample_latencies(
                environment_seed
            )
            audit = {
                "candidate_ordinal": candidate_ordinal,
                "planned_episode_id": identity,
                "environment_seed": environment_seed,
                "policy_rng_seed": policy_seed,
                "sampled_observation_latency_steps": sampled_observation,
                "sampled_action_latency_steps": sampled_action,
                "accepted": bool(
                    sampled_observation
                    == int(cell["observation_latency_steps"])
                    and sampled_action == int(cell["action_latency_steps"])
                ),
            }
            if audit["accepted"]:
                pair_id = hashlib.sha256(
                    f"{identity}|accepted-{accepted}".encode()
                ).hexdigest()
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "task_id": cell["task_id"],
                        "cell_ordinal": cell_ordinal,
                        "replicate_ordinal": accepted,
                        "domain": cell["domain"],
                        "observation_latency_steps": sampled_observation,
                        "action_latency_steps": sampled_action,
                        "environment_seed": environment_seed,
                        "policy_rng_seed": policy_seed,
                        "seed_commitment": commitment,
                        "roles": ["candidate", "control"],
                        "forced_same_index": smoke,
                    }
                )
                accepted += 1
            else:
                rejected.append({"cell_ordinal": cell_ordinal, **audit})
            candidate_ordinal += 1
            if candidate_ordinal > 100_000:
                raise RuntimeError("natural latency rejection exhausted")
    return {
        "schema_version": PLAN_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "mode": "smoke" if smoke else "formal",
        "salt_commitment": commitment,
        "salt_reveal": salt,
        "commitment_verified": verify_seed_reveal(commitment, salt),
        "seed_schema": SEED_SCHEMA,
        "role_enters_seed_derivation": False,
        "natural_evaluation_latency_rejection": True,
        "reset_latency_override_used": False,
        "replacement_seed_allowed": False,
        "complete_case_deletion_allowed": False,
        "cells": cells,
        "planned_pair_count": len(pairs),
        "execution_count": 2 * len(pairs),
        "pairs": pairs,
        "rejected_seed_audit": rejected,
    }


def execute_plan(
    root: Path,
    plan: Mapping[str, object],
    *,
    smoke: bool,
) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    records = []
    for pair in plan["pairs"]:
        diagnostic = TargetSelectionDiagnostic(
            tasks[pair["task_id"]], bindings[pair["task_id"]]
        )
        configuration = BranchConfiguration(
            int(pair["environment_seed"]),
            int(pair["policy_rng_seed"]),
            supported_domain=pair["domain"] != "challenge",
        )
        try:
            candidate = diagnostic.run_branch(configuration)
        except Exception as error:
            records.append(_unresolved_terminal(pair, "candidate", error))
            continue
        candidate_count = int(candidate["candidate_count"])
        if smoke:
            control_index = int(candidate["selected_index"])
        else:
            control_index = _control_index_from_branch(
                candidate, int(pair["policy_rng_seed"])
            )
        try:
            control = diagnostic.run_branch(
                BranchConfiguration(
                    configuration.environment_seed,
                    configuration.policy_rng_seed,
                    forced_index=control_index,
                    supported_domain=configuration.supported_domain,
                )
            )
        except Exception as error:
            records.append(_unresolved_terminal(pair, "control", error))
            continue
        candidate_hash_equal = (
            candidate["candidate_set_sha256"] == control["candidate_set_sha256"]
            and candidate["_candidate_bytes"] == control["_candidate_bytes"]
        )
        same_index = candidate["selected_index"] == control["selected_index"]
        bit_identity = _same_index_identity(candidate, control) if same_index else None
        resolved = bool(
            candidate["resolved"]
            and control["resolved"]
            and candidate_hash_equal
            and (not same_index or bit_identity)
        )
        candidate = _persistable_branch(candidate)
        control = _persistable_branch(control)
        records.append(
            {
                "schema_version": TERMINAL_SCHEMA,
                "pair_id": pair["pair_id"],
                "task_id": pair["task_id"],
                "domain": pair["domain"],
                "observation_latency_steps": pair["observation_latency_steps"],
                "action_latency_steps": pair["action_latency_steps"],
                "environment_seed": pair["environment_seed"],
                "policy_rng_seed": pair["policy_rng_seed"],
                "candidate_event": int(candidate["main_event"]),
                "control_event": int(control["main_event"]),
                "candidate_index": candidate["selected_index"],
                "control_index": control["selected_index"],
                "candidate_count": candidate_count,
                "candidate_set_sha256": candidate["candidate_set_sha256"],
                "candidate_hash_equal": candidate_hash_equal,
                "same_index": same_index,
                "same_index_bit_identical": bit_identity,
                "resolved": resolved,
                "candidate": candidate,
                "control": control,
            }
        )
    return {
        "schema_version": TERMINAL_SCHEMA,
        "mode": plan["mode"],
        "planned_pair_count": plan["planned_pair_count"],
        "terminal_pair_count": len(records),
        "records": records,
    }


def analyze_terminals(
    terminals: Mapping[str, object], *, smoke: bool
) -> dict[str, object]:
    records = terminals["records"]
    missing = max(0, int(terminals["planned_pair_count"]) - len(records))
    duplicate_ids = len({value["pair_id"] for value in records}) != len(records)
    unresolved = (
        sum(not bool(value["resolved"]) for value in records)
        + missing
        + int(duplicate_ids)
    )
    resolved = [value for value in records if value["resolved"]]
    safety = evaluate_safety_guards(
        resolved, formal=not smoke and unresolved == 0
    )
    common = {
        "planned_pair_count": terminals["planned_pair_count"],
        "terminal_pair_count": len(records),
        "planned_terminal_identity_complete": (
            missing == 0 and not duplicate_ids
        ),
        "unresolved_infrastructure": unresolved,
        "candidate_hash_equal": all(
            value["candidate_hash_equal"] for value in records
        ),
        "same_index_bit_identity": all(
            not value["same_index"] or value["same_index_bit_identical"]
            for value in records
        ),
        "candidate_set_nonempty": all(
            value["resolved"] and int(value["candidate_count"]) > 0
            for value in records
        ),
        **safety,
    }
    if smoke:
        return {
            **common,
            "selector_comparison_executed": False,
            "passed": all(
                (
                    unresolved == 0,
                    common["candidate_hash_equal"],
                    common["same_index_bit_identity"],
                    common["candidate_set_nonempty"],
                    common["hard_guard"]["passed"],
                )
            ),
        }
    if unresolved:
        return {
            **common,
            "selector_comparison_executed": True,
            "primary": None,
            "primary_passed": False,
        }
    primary = paired_primary_statistics(records)
    primary_passed = (
        primary["planned_pair_count"] == 54
        and primary["one_sided_exact_mcnemar_p"] <= 0.05
        and primary["discordant_candidate_probability_exact_95_lower"] > 0.5
        and primary["delta_itt"] >= 0.20
        and all(
            row["candidate_only"] >= row["control_only"]
            for row in primary["by_task"].values()
        )
        and all(
            row["candidate_only"] >= row["control_only"]
            for row in primary["by_observation_latency"].values()
        )
    )
    return {
        **common,
        "selector_comparison_executed": True,
        "primary": primary,
        "primary_passed": primary_passed,
    }


def _build_report(
    source_commit,
    command,
    plan,
    terminals,
    power,
    analysis,
    smoke,
):
    if smoke:
        decision = (
            "accepted as target-selection smoke contract evidence"
            if power["selected_pair_count"] == 54 and analysis["passed"]
            else "inconclusive"
        )
    elif power["selected_pair_count"] != 54:
        decision = "inconclusive_power"
    elif analysis["unresolved_infrastructure"] or not analysis[
        "target_contact_intensity_guard"
    ]["supported"]:
        decision = "inconclusive"
    elif (
        analysis["primary_passed"]
        and analysis["hard_guard"]["passed"]
        and analysis["non_target_allowed_contact_guard"]["passed"]
        and analysis["target_contact_intensity_guard"]["passed"]
    ):
        decision = "accepted as target-selection interaction-yield evidence"
    else:
        decision = "rejected"
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "mode": "smoke" if smoke else "formal",
        "decision": decision,
        "main_event_name": "same_entity_dual_arm_contact_associated_motion",
        "main_event_is_contact_associated_not_controlled": True,
        "selector_comparison_executed": not smoke,
        "plan_summary": {
            "planned_pair_count": plan["planned_pair_count"],
            "rejected_seed_count": len(plan["rejected_seed_audit"]),
        },
        "terminal_summary": {
            "terminal_pair_count": terminals["terminal_pair_count"]
        },
        "power": power,
        "analysis": analysis,
        **CLAIM_FLAGS,
    }


def _manifest(
    source_commit,
    command,
    identities,
    plan,
    terminals,
    artifacts,
    *,
    smoke,
    status,
):
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": status,
        "mode": "smoke" if smoke else "formal",
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "frozen_document_commit_is_ancestor": status == "complete",
        "command": list(command),
        "p40_e2": identities["p40_e2"],
        "binding": identities["binding"],
        "task_config": identities["task_config"],
        "seed_lineage": (
            None
            if plan is None
            else {
                "schema_version": SEED_SCHEMA,
                "commitment": plan["salt_commitment"],
                "reveal": plan["salt_reveal"],
                "commitment_verified": plan["commitment_verified"],
                "role_enters_seed_derivation": False,
            }
        ),
        "schemas": {
            "policy_input": INPUT_SCHEMA,
            "candidate": CANDIDATE_SCHEMA,
            "plan": PLAN_SCHEMA,
            "terminal": TERMINAL_SCHEMA,
            "power": POWER_SCHEMA,
        },
        "primitive_constants": {
            "phases": [list(value) for value in PHASES],
            "planned_horizon": PLANNED_HORIZON,
            "action_validity_ns": 100_000_000,
        },
        "planned_pair_count": (
            None if plan is None else plan["planned_pair_count"]
        ),
        "terminal_pair_count": (
            None if terminals is None else terminals["terminal_pair_count"]
        ),
        **CLAIM_FLAGS,
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for name, content in artifacts.items()
        },
    }


def _control_index_from_branch(branch, policy_seed):
    candidate_set = branch["_candidate_set"]
    if candidate_set is None:
        return -1
    return select_control_index(candidate_set, policy_seed)


def _same_index_identity(candidate, control) -> bool:
    return (
        candidate["_full_trace_sha256"] == control["_full_trace_sha256"]
        and candidate["_full_graph_sha256"] == control["_full_graph_sha256"]
        and candidate["_full_ledger_sha256"] == control["_full_ledger_sha256"]
        and candidate["_candidate_bytes"] == control["_candidate_bytes"]
        and candidate["main_event"] == control["main_event"]
    )


def _persistable_branch(branch):
    return {
        key: value for key, value in branch.items() if not key.startswith("_")
    }


def _unresolved_terminal(pair, role, error):
    return {
        "schema_version": TERMINAL_SCHEMA,
        "pair_id": pair["pair_id"],
        "task_id": pair["task_id"],
        "domain": pair["domain"],
        "observation_latency_steps": pair["observation_latency_steps"],
        "action_latency_steps": pair["action_latency_steps"],
        "environment_seed": pair["environment_seed"],
        "policy_rng_seed": pair["policy_rng_seed"],
        "candidate_event": 0,
        "control_event": 0,
        "candidate_index": None,
        "control_index": None,
        "candidate_count": None,
        "candidate_set_sha256": None,
        "candidate_hash_equal": False,
        "same_index": False,
        "same_index_bit_identical": None,
        "resolved": False,
        "infrastructure_failure": {
            "role": role,
            "error_type": type(error).__name__,
            "error": str(error),
        },
        "candidate": None,
        "control": None,
    }


def _unavailable_plan(salt, cells, selected):
    return {
        "schema_version": PLAN_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "mode": "formal",
        "decision": "inconclusive_power",
        "selected_pair_count": selected,
        "salt_commitment": seed_commitment(salt),
        "salt_reveal": salt,
        "commitment_verified": verify_seed_reveal(seed_commitment(salt), salt),
        "seed_schema": SEED_SCHEMA,
        "role_enters_seed_derivation": False,
        "natural_evaluation_latency_rejection": True,
        "reset_latency_override_used": False,
        "replacement_seed_allowed": False,
        "complete_case_deletion_allowed": False,
        "cells": cells,
        "planned_pair_count": 0,
        "execution_count": 0,
        "pairs": [],
        "rejected_seed_audit": [],
    }


def _source_identities(root):
    return {
        "p40_e2": {
            "report": _file_identity(root, root / P40_REPORT),
            "manifest": _file_identity(root, root / P40_MANIFEST),
        },
        "binding": _file_identity(root, root / BINDING_PATH),
        "task_config": _file_identity(root, root / TASK_PATH),
    }


def _require_clean_source(root, identities):
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("P41-E2 runner requires clean committed source")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("P41-E2 frozen document commit is not an ancestor")
    history = tuple(f"docs/research-loop/{index:04d}" for index in range(1, 8))
    unchanged = subprocess.run(
        ("git", "diff", "--quiet", FROZEN_PARENT_COMMIT, "HEAD", "--", *history),
        cwd=root,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("P41-E2 historical research-loop documents drifted")
    if (
        identities["p40_e2"]["report"]["sha256"] != P40_REPORT_SHA256
        or identities["p40_e2"]["manifest"]["sha256"] != P40_MANIFEST_SHA256
    ):
        raise RuntimeError("P40-E2 artifact identity differs from frozen contract")
    report = json.loads((root / P40_REPORT).read_text())
    if report["decision"] != "accepted as entity-contact measurement contract evidence":
        raise RuntimeError("P40-E2 is not accepted")
    binding = identities["binding"]
    if (
        binding["sha256"] != BINDING_SHA256
        or binding["bytes"] != BINDING_BYTES
    ):
        raise RuntimeError("P41-E2 binding identity differs from frozen contract")
    task = identities["task_config"]
    if task["sha256"] != TASK_SHA256 or task["bytes"] != TASK_BYTES:
        raise RuntimeError("P41-E2 task identity differs from frozen contract")


def _validate_mode(arguments):
    expected = SMOKE_SALT if arguments.smoke else FORMAL_SALT
    if arguments.salt != expected:
        raise ValueError(f"P41-E2 mode requires frozen salt {expected}")


def _source_commit(root):
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P41-E2 runner requires a full Git source commit")
    return commit


def _file_identity(root, path):
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _create_output(output, artifacts):
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir()
    try:
        for name, content in artifacts.items():
            _atomic_write(staging / name, content)
        os.replace(staging, output)
    except BaseException:
        for path in staging.glob("*"):
            path.unlink()
        if staging.exists():
            staging.rmdir()
        raise


def _atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _resolve(root, path):
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
