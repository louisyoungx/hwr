"""Run frozen R0001-P50-E1 acquisition and R0001-P50-E2 offline funnel."""
import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from hwr.apps import (
    aggregate_candidate_funnels,
    analyze_candidate_capsule_directory,
    candidate_artifact_manifest,
    candidate_commit_is_ancestor,
    candidate_file_identity as _file_identity,
    candidate_git_tree as _git_tree,
    candidate_json_bytes as _json_bytes,
    candidate_sha256 as _sha256,
    candidate_source_commit as _source_commit,
    create_candidate_output as _create_output,
    persist_candidate_episode,
    read_bound_blob as _read_bound_blob,
    require_candidate_disk_capacity as _require_disk_capacity,
    resolve_candidate_path as _resolve,
    validate_candidate_terminal_ledger,
    validate_candidate_record_set,
)
from hwr.adapters.mujoco.candidate_acquisition import (
    CAPSULE_SCHEMA,
    EPISODE_SCHEMA,
    AcquisitionCapsule,
    AcquisitionContractError,
    AcquisitionEpisodeResult,
    CandidateAcquisitionDiagnostic,
    mujoco_runtime_version,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.eval.seed_contract import (
    SEED_SCHEMA,
    derive_domain_seed,
    planned_episode_id,
    read_seed_salt,
    require_seed_reveal,
)
from hwr.eval.candidate_funnel import (
    CandidateFunnelContractError,
    candidate_gate_source_identity,
)
from hwr.eval.target_selection import (
    ACQUISITION_STEPS,
    CANDIDATE_SCHEMA,
    INPUT_SCHEMA,
    TASK_IDS,
)
from hwr.eval.tool_kinematics import recursive_xml_input_identity
MODULE_NAME = "hwr.apps.evaluate_candidate_acquisition"
PROPOSAL_ID = "R0001-P50-E1"
PLAN_ID = "R0001-P50-E1-formal"
PLAN_SCHEMA = "hwr.p50-acquisition-plan/v1"
CAPSULE_INDEX_SCHEMA = "hwr.p50-acquisition-capsule-index/v1"
REPORT_SCHEMA = "hwr.p50-acquisition-report/v1"
MANIFEST_SCHEMA = "hwr.p50-acquisition-artifacts/v1"
FAILURE_SCHEMA = "hwr.p50-acquisition-failure/v1"
SALT_COMMITMENT = "ed945b2dcfe90c6aab639164da32cc8a1a905df56534c42a443d1bd4753e16a4"
FROZEN_DOCUMENT_COMMIT = "2d1752f2c0c8b9e39d7f3ebaa8e9ff0ec1d13f38"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0010/03-experiment.md")
FORMAL_OUTPUT = Path("runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001")
FORMAL_SALT_FILE = Path("runs/research-loop/0010/.host/p50-e1-salt.txt")
FUNNEL_OUTPUT = Path("runs/research-loop/0010/r0010-p50-e2-funnel-s20265001")
FUNNEL_INPUT = FORMAL_OUTPUT
FUNNEL_REPORT_SCHEMA = "hwr.p50-candidate-funnel-report/v1"
FUNNEL_MANIFEST_SCHEMA = "hwr.p50-candidate-funnel-artifacts/v1"
FUNNEL_FAILURE_SCHEMA = "hwr.p50-candidate-funnel-failure/v1"
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
}
SOURCE_PATHS = {
    "p50_app": Path("src/hwr/apps/evaluate_candidate_acquisition.py"),
    "p50_app_helpers": Path("src/hwr/apps/__init__.py"),
    "p50_bridge": Path("src/hwr/adapters/mujoco/candidate_acquisition.py"),
    "p50_funnel": Path("src/hwr/eval/candidate_funnel.py"),
    "formal_generator": Path("src/hwr/eval/target_selection.py"),
    "p41_bridge": Path("src/hwr/adapters/mujoco/target_selection_diagnostic.py"),
    "formal_backend": Path(
        "src/hwr/adapters/mujoco/formal_household_backend.py"
    ),
}
PROTECTED_PATHS = (
    "assets/mujoco",
    BINDING_PATH.as_posix(),
    TASK_PATH.as_posix(),
    "src/hwr/eval/target_selection.py",
    "src/hwr/adapters/mujoco/target_selection_diagnostic.py",
    "src/hwr/adapters/mujoco/formal_household_backend.py",
    "src/hwr/safety",
)
FROZEN_CELLS = tuple(
    (task_id, observation_latency, action_latency)
    for task_id in TASK_IDS
    for observation_latency in (1, 2)
    for action_latency in (1, 2)
)
CLAIM_FLAGS = {
    "training_executed": False,
    "policy_inference_executed": False,
    "capability_claim_allowed": False,
    "task_success_claim_allowed": False,
    "generalization_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
}
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("acquisition", "funnel"), default="acquisition")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path)
    parser.add_argument("--capsules", type=Path)
    return parser
def run(arguments: argparse.Namespace) -> dict[str, object]:
    _validate_arguments(arguments)
    if arguments.mode == "funnel":
        return _run_funnel(arguments)
    return _run_acquisition(arguments)
def _run_acquisition(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    salt_file = _resolve(root, arguments.salt_file)
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--output",
        arguments.output.as_posix(),
        "--salt-file",
        arguments.salt_file.as_posix(),
    ]
    started = time.perf_counter()
    tracemalloc.start()
    identities: Mapping[str, object] = {}
    try:
        identities = _source_identities(root)
        _require_clean_source(root, identities)
        _require_disk_capacity(output)
        salt = read_seed_salt(salt_file)
        require_seed_reveal(SALT_COMMITMENT, salt)
        plan = build_plan(root, salt)
        execution = execute_plan(root, plan)
        report = analyze_acquisition(plan, execution)
        report.update(
            {
                "source_commit": source_commit,
                "command": command,
                "wall_time_seconds": time.perf_counter() - started,
                "peak_tracemalloc_bytes": tracemalloc.get_traced_memory()[1],
                **CLAIM_FLAGS,
            }
        )
        artifacts = {
            "plan.json": _json_bytes(plan),
            "capsules.json": _json_bytes(execution["capsules"]),
            "report.json": _json_bytes(report),
            **execution["binary_artifacts"],
        }
        manifest = _manifest(
            source_commit,
            command,
            identities,
            plan,
            execution,
            artifacts,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "source_commit": source_commit,
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
            None,
            artifacts,
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
        "planned_episode_count": plan["planned_episode_count"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": _sha256(artifacts["manifest.json"]),
    }
def _run_funnel(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    capsules = _resolve(root, arguments.capsules)
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = [
        ".venv/bin/python", "-m", MODULE_NAME, "--mode", "funnel",
        "--capsules", arguments.capsules.as_posix(),
        "--output", arguments.output.as_posix(),
    ]
    started = time.perf_counter()
    tracemalloc.start()
    identities: Mapping[str, object] = {}
    input_identity: Mapping[str, object] | None = None
    try:
        identities = _source_identities(root)
        _require_clean_source(root, identities)
        first, input_identity = analyze_candidate_capsule_directory(
            root, capsules, identities, FROZEN_DOCUMENT_COMMIT
        )
        second, replay_identity = analyze_candidate_capsule_directory(
            root, capsules, identities, FROZEN_DOCUMENT_COMMIT
        )
        deterministic = _json_bytes(first) == _json_bytes(second)
        if input_identity != replay_identity:
            raise CandidateFunnelContractError("E1 input identity changed during replay")
        checks = {
            **first["aggregate"]["checks"],
            "report_bit_identical": deterministic,
            "e1_accepted": True,
        }
        passed = all(checks.values())
        report = {
            "schema_version": FUNNEL_REPORT_SCHEMA,
            "proposal_id": "R0001-P50-E2",
            "source_commit": source_commit,
            "command": command,
            "decision": (
                "accepted as candidate-funnel measurement evidence"
                if passed else "invalid"
            ),
            "source_acquisition": input_identity,
            "gate_source_identity": candidate_gate_source_identity(),
            "episodes": first["episodes"],
            "aggregate": {**first["aggregate"], "checks": {**checks, "passed": passed}},
            "report_replay_bit_identical": deterministic,
            "wall_time_seconds": time.perf_counter() - started,
            "peak_tracemalloc_bytes": tracemalloc.get_traced_memory()[1],
            "descriptive_stage_is_not_causal_improvement_evidence": True,
            **CLAIM_FLAGS,
        }
        artifacts = {"report.json": _json_bytes(report)}
        manifest = _funnel_manifest(
            source_commit, command, identities, input_identity, artifacts,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FUNNEL_FAILURE_SCHEMA,
            "proposal_id": "R0001-P50-E2",
            "source_commit": source_commit,
            "error_type": type(error).__name__,
            "error": str(error),
            **CLAIM_FLAGS,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _funnel_manifest(
            source_commit, command, identities, input_identity, artifacts,
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
        "episode_count": report["aggregate"]["episode_count"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": _sha256(artifacts["manifest.json"]),
    }
def build_plan(root: Path, salt: str) -> dict[str, object]:
    require_seed_reveal(SALT_COMMITMENT, salt)
    tasks, bindings = load_default_formal_household_catalogs(root)
    if tuple(task for task in TASK_IDS) != tuple(
        task for task in TASK_IDS if task in tasks and task in bindings
    ):
        raise AcquisitionContractError("formal task catalog differs")
    cells, episodes, rejected = [], [], []
    checked_environment: set[int] = set()
    checked_policy: set[int] = set()
    for cell_ordinal, frozen in enumerate(FROZEN_CELLS):
        task_id, observation_latency, action_latency = frozen
        cell_id = (
            f"cell-{cell_ordinal:02d}-obs-{observation_latency}"
            f"-action-{action_latency}"
        )
        cell = {
            "cell_id": cell_id,
            "cell_ordinal": cell_ordinal,
            "task_id": task_id,
            "observation_latency_steps": observation_latency,
            "action_latency_steps": action_latency,
            "replicate_count": 2,
        }
        cells.append(cell)
        sampler = _sampler_for_cell(
            tasks[task_id],
            bindings[task_id],
            observation_latency,
            action_latency,
        )
        accepted = 0
        for candidate_ordinal in range(96):
            identity = planned_episode_id(
                PLAN_ID, task_id, cell_id, candidate_ordinal
            )
            environment_seed = derive_domain_seed(salt, "environment", identity)
            policy_seed = derive_domain_seed(salt, "policy", identity)
            if (
                environment_seed in checked_environment
                or policy_seed in checked_policy
                or environment_seed in checked_policy
                or policy_seed in checked_environment
            ):
                raise AcquisitionContractError("checked seed identity collided")
            checked_environment.add(environment_seed)
            checked_policy.add(policy_seed)
            sampled_observation, sampled_action = sampler.sample_latencies(
                environment_seed
            )
            record = {
                "candidate_ordinal": candidate_ordinal,
                "planned_episode_id": identity,
                "cell_id": cell_id,
                "cell_ordinal": cell_ordinal,
                "task_id": task_id,
                "environment_seed": environment_seed,
                "policy_rng_seed": policy_seed,
                "sampled_observation_latency_steps": sampled_observation,
                "sampled_action_latency_steps": sampled_action,
            }
            if (sampled_observation, sampled_action) == (
                observation_latency,
                action_latency,
            ):
                episodes.append(
                    {
                        **record,
                        "replicate_ordinal": accepted,
                        "replacement": False,
                    }
                )
                accepted += 1
                if accepted == 2:
                    break
            else:
                rejected.append(
                    {
                        **record,
                        "accepted": False,
                        "rejection_reason": "natural_latency_cell_mismatch",
                    }
                )
        if accepted != 2:
            raise AcquisitionContractError(
                f"{cell_id} did not find two matches by candidate ordinal 95"
            )
    return {
        "schema_version": PLAN_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "plan_id": PLAN_ID,
        "salt_commitment": SALT_COMMITMENT,
        "salt_reveal": salt,
        "commitment_verified": True,
        "seed_schema": SEED_SCHEMA,
        "role_enters_seed_derivation": False,
        "natural_evaluation_latency_rejection": True,
        "reset_latency_override_used": False,
        "replacement_seed_allowed": False,
        "maximum_candidate_ordinal": 95,
        "maximum_latency_sampler_calls": 1_152,
        "planned_control_steps_per_episode": ACQUISITION_STEPS,
        "planned_episode_count": len(episodes),
        "planned_control_step_count": len(episodes) * ACQUISITION_STEPS,
        "validation_replay_count": len(episodes),
        "maximum_physical_acquisition_count": 2 * len(episodes),
        "maximum_control_step_count": 2 * len(episodes) * ACQUISITION_STEPS,
        "cells": cells,
        "episodes": episodes,
        "rejected_seed_audit": rejected,
    }
def execute_plan(
    root: Path, plan: Mapping[str, object]
) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    terminals: list[dict[str, object]] = []
    capsule_records: list[dict[str, object]] = []
    binary: dict[str, bytes] = {}
    for episode in plan["episodes"]:
        try:
            result = CandidateAcquisitionDiagnostic(
                tasks[episode["task_id"]], bindings[episode["task_id"]]
            ).run_episode(episode)
            terminal, capsule, blobs = persist_candidate_episode(result)
            terminals.append(terminal)
            capsule_records.append(capsule)
            binary.update(blobs)
        except Exception as error:
            terminals.append(
                {
                    "schema_version": EPISODE_SCHEMA,
                    "planned_episode_id": episode["planned_episode_id"],
                    "task_id": episode["task_id"],
                    "cell_id": episode["cell_id"],
                    "cell_ordinal": episode["cell_ordinal"],
                    "replicate_ordinal": episode["replicate_ordinal"],
                    "candidate_ordinal": episode["candidate_ordinal"],
                    "environment_seed": episode["environment_seed"],
                    "policy_rng_seed": episode["policy_rng_seed"],
                    "planned_latency": {
                        "observation_steps": episode[
                            "sampled_observation_latency_steps"
                        ],
                        "action_steps": episode["sampled_action_latency_steps"],
                    },
                    "replacement": False,
                    "resolved": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "error_details": getattr(error, "details", {}),
                }
            )
    return {
        "terminals": terminals,
        "capsules": {
            "schema_version": CAPSULE_INDEX_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "planned_episode_count": plan["planned_episode_count"],
            "capsule_count": len(capsule_records),
            "episodes": capsule_records,
            "terminals": terminals,
        },
        "binary_artifacts": binary,
    }
def analyze_acquisition(
    plan: Mapping[str, object], execution: Mapping[str, object]
) -> dict[str, object]:
    terminals = execution["terminals"]
    capsules = execution["capsules"]["episodes"]
    terminal_ledger = validate_candidate_terminal_ledger(plan, terminals)
    capsule_ledger = validate_candidate_record_set(plan, capsules)
    by_id = {value["planned_episode_id"]: value for value in capsules}
    capsule_complete = len(by_id) == len(plan["episodes"]) == 24
    checks = {
        "planned_terminal_ledger": bool(terminal_ledger["passed"]),
        "planned_capsule_ledger": bool(capsule_ledger["passed"]),
        "twelve_cells_complete": (
            len(plan["cells"]) == 12
            and all(
                sum(
                    row["cell_id"] == cell["cell_id"]
                    for row in plan["episodes"]
                )
                == 2
                for cell in plan["cells"]
            )
        ),
        "capsules_complete": capsule_complete,
        "candidate_offline_replay": capsule_complete
        and all(
            value["offline_candidate_replay_bit_identical"] in (None, True)
            for value in capsules
        ),
        "same_seed_replay": capsule_complete
        and all(bool(value["same_seed_validation_replay"]) for value in capsules),
        "capture_enabled_disabled_identity": capsule_complete
        and all(
            bool(value["capture_enabled_disabled_identity"]) for value in capsules
        ),
        "anchor_blobs_complete": capsule_complete
        and all(bool(value["anchor_blobs_complete"]) for value in capsules),
        "action_bounds": all(
            bool(value.get("action_bounds_valid")) for value in terminals
        ),
        "safety_intervention_zero": all(
            int(value.get("safety_intervention_count", 1)) == 0
            and int(value.get("validation_replay", {}).get(
                "safety_intervention_count", 1
            )) == 0
            for value in terminals
        ),
        "validation_action_bounds": all(
            value.get("validation_replay", {}).get("action_bounds_valid") is True
            for value in terminals
        ),
        "planned_step_terminal_semantics": all(
            0 < int(value.get("trace_step_count", 0)) <= ACQUISITION_STEPS
            and (
                bool(value.get("runtime_terminal"))
                or int(value.get("trace_step_count", 0)) == ACQUISITION_STEPS
            )
            and 0 < int(value.get("validation_replay", {}).get(
                "trace_step_count", 0
            )) <= ACQUISITION_STEPS
            and (
                bool(value.get("validation_replay", {}).get("runtime_terminal"))
                or int(value.get("validation_replay", {}).get(
                    "trace_step_count", 0
                )) == ACQUISITION_STEPS
            )
            for value in terminals
        ),
        "runtime_latency_matches_plan": all(
            value.get("planned_latency") == {
                "observation_steps": value.get("runtime_latency", {}).get(
                    "observation_steps"
                ),
                "action_steps": value.get("runtime_latency", {}).get(
                    "action_steps"
                ),
            }
            and value.get("runtime_latency", {}).get("override_inactive") is True
            for value in terminals
        ),
        "stale_action_zero": all(
            int(value.get("stale_action_applied_count", 1)) == 0
            and int(value.get("validation_replay", {}).get(
                "stale_action_applied_count", 1
            )) == 0
            for value in terminals
        ),
        "severe_collision_zero": all(
            int(value.get("severe_collision_count", 1)) == 0
            and int(value.get("validation_replay", {}).get(
                "severe_collision_count", 1
            )) == 0
            for value in terminals
        ),
        "invalid_force_zero": all(
            int(value.get("invalid_force_count", 1)) == 0
            and int(value.get("validation_replay", {}).get(
                "invalid_force_count", 1
            )) == 0
            for value in terminals
        ),
        "p40_conservation": all(
            float(value.get("p40_conservation_maximum_difference", 1.0))
            <= 1.0e-12
            and float(value.get("validation_replay", {}).get(
                "p40_conservation_maximum_difference", 1.0
            )) <= 1.0e-12
            for value in terminals
        ),
    }
    valid = all(checks.values())
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "decision": (
            "accepted as immutable acquisition evidence contract"
            if valid
            else "invalid"
        ),
        "contract_valid": valid,
        "checks": {**checks, "passed": valid},
        "ledger": {
            "terminals": terminal_ledger,
            "capsules": capsule_ledger,
        },
        "planned_episode_count": plan["planned_episode_count"],
        "planned_acquisition_count": plan["planned_episode_count"],
        "validation_replay_count": plan["planned_episode_count"],
        "maximum_physical_acquisition_count": 2 * plan["planned_episode_count"],
        "terminal_episode_count": len(terminals),
        "capsule_count": len(capsules),
        "acquisition_failure_count": sum(
            bool(value.get("acquisition_failure")) for value in terminals
        ),
        "empty_candidate_episode_count": sum(
            int(value.get("candidate_count", 0)) == 0 for value in terminals
        ),
        "post_selection_executed": False,
    }
def _sampler_for_cell(
    task,
    binding,
    observation_latency_steps: int,
    action_latency_steps: int,
):
    del observation_latency_steps, action_latency_steps
    return CandidateAcquisitionDiagnostic(task, binding)
def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.mode == "acquisition":
        if arguments.output != FORMAL_OUTPUT:
            raise ValueError(f"R0001-P50-E1 requires frozen output {FORMAL_OUTPUT}")
        if arguments.salt_file != FORMAL_SALT_FILE or arguments.capsules is not None:
            raise ValueError(f"R0001-P50-E1 requires frozen salt file {FORMAL_SALT_FILE}")
    elif (
        arguments.output != FUNNEL_OUTPUT
        or arguments.capsules != FUNNEL_INPUT
        or arguments.salt_file is not None
    ):
        raise ValueError(
            f"R0001-P50-E2 requires frozen capsules {FUNNEL_INPUT} "
            f"and output {FUNNEL_OUTPUT}"
        )
def _source_identities(root: Path) -> dict[str, object]:
    _, bindings = load_default_formal_household_catalogs(root)
    return {
        "binding": _file_identity(root, root / BINDING_PATH),
        "task_config": _file_identity(root, root / TASK_PATH),
        "recursive_xml": {
            task_id: recursive_xml_input_identity(root, bindings[task_id].model_path)
            for task_id in TASK_IDS
        },
        "sources": {
            name: _file_identity(root, root / path)
            for name, path in SOURCE_PATHS.items()
        },
        "historical_research_loop_trees": {
            path: _git_tree(root, path) for path in HISTORICAL_TREES
        },
        "frozen_document": _file_identity(root, root / FROZEN_DOCUMENT_PATH),
    }
def _require_clean_source(
    root: Path, identities: Mapping[str, object]
) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P50-E1 runner requires clean committed source")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("P50 frozen document commit is not an ancestor")
    protected = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            FROZEN_DOCUMENT_COMMIT,
            "HEAD",
            "--",
            *PROTECTED_PATHS,
        ),
        cwd=root,
        check=False,
    )
    if protected.returncode != 0:
        raise RuntimeError("P50 source/config/XML anchors drifted")
    if identities.get("historical_research_loop_trees") != HISTORICAL_TREES:
        raise RuntimeError("P50 historical research-loop documents drifted")
    frozen = subprocess.run(
        ("git", "show", f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if identities["frozen_document"] != {
        "path": FROZEN_DOCUMENT_PATH.as_posix(),
        "sha256": _sha256(frozen),
        "bytes": len(frozen),
    }:
        raise RuntimeError("P50 frozen document content drifted")
def _manifest(
    source_commit,
    command,
    identities,
    plan,
    execution,
    artifacts,
    *,
    status,
):
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
        status=status,
        extra={
            "frozen_document_commit_is_ancestor": (
                candidate_commit_is_ancestor(
                    Path(__file__).resolve().parents[3],
                    FROZEN_DOCUMENT_COMMIT,
                    source_commit,
                )
            ),
            "seed_lineage": None if plan is None else {
                "schema_version": SEED_SCHEMA,
                "plan_id": PLAN_ID,
                "commitment": SALT_COMMITMENT,
                "reveal": plan["salt_reveal"],
                "commitment_verified": plan["commitment_verified"],
                "role_enters_seed_derivation": False,
            },
            "schemas": {
                "policy_input": INPUT_SCHEMA,
                "candidate": CANDIDATE_SCHEMA,
                "capsule": CAPSULE_SCHEMA,
                "capsule_index": CAPSULE_INDEX_SCHEMA,
                "terminal": EPISODE_SCHEMA,
            },
            "planned_episode_count": (
                None if plan is None else plan["planned_episode_count"]
            ),
            "terminal_episode_count": (
                None if execution is None else len(execution["terminals"])
            ),
            **CLAIM_FLAGS,
        },
    )
def _funnel_manifest(
    source_commit: str,
    command: Sequence[str],
    identities: Mapping[str, object],
    input_identity: Mapping[str, object] | None,
    artifacts: Mapping[str, bytes],
    *,
    status: str,
) -> dict[str, object]:
    return candidate_artifact_manifest(
        schema=FUNNEL_MANIFEST_SCHEMA,
        proposal_id="R0001-P50-E2",
        source_commit=source_commit,
        frozen_document_commit=FROZEN_DOCUMENT_COMMIT,
        command=command,
        identities=identities,
        artifacts=artifacts,
        runtime_versions={
            "mujoco": mujoco_runtime_version(),
            "hwr_platform": importlib.metadata.version("hwr-platform"),
        },
        status=status,
        extra={
            "frozen_document_commit_is_ancestor": (
                candidate_commit_is_ancestor(
                    Path(__file__).resolve().parents[3],
                    FROZEN_DOCUMENT_COMMIT,
                    source_commit,
                )
            ),
            "source_acquisition": input_identity,
            "gate_source_identity": candidate_gate_source_identity(),
            "report_only": True,
            "formal_candidate_output_modified": False,
            "unique_observation_shadow_enters_candidate_output": False,
            **CLAIM_FLAGS,
        },
    )
def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2

if __name__ == "__main__":
    raise SystemExit(main())
