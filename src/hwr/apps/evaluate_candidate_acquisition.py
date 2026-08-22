"""Run frozen R0001-P50-E1 acquisition and R0001-P50-E2 offline funnel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.adapters.mujoco.candidate_acquisition import (
    CAPSULE_SCHEMA,
    EPISODE_SCHEMA,
    AcquisitionCapsule,
    AcquisitionContractError,
    AcquisitionEpisodeResult,
    CandidateAcquisitionDiagnostic,
    mujoco_runtime_version,
    persist_episode,
    validate_terminal_ledger,
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
FROZEN_DOCUMENT_COMMIT = "5fad6cec27e8f797c31a202497745a5616ab220b"
FORMAL_OUTPUT = Path(
    "runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001"
)
FORMAL_SALT_FILE = Path("runs/research-loop/0010/.host/p50-e1-salt.txt")
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
    "p50_bridge": Path("src/hwr/adapters/mujoco/candidate_acquisition.py"),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    _validate_arguments(arguments)
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
            terminal, capsule, blobs = persist_episode(result)
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
                    "replicate_ordinal": episode["replicate_ordinal"],
                    "candidate_ordinal": episode["candidate_ordinal"],
                    "replacement": False,
                    "resolved": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
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
    ledger = validate_terminal_ledger(plan, terminals)
    by_id = {value["planned_episode_id"]: value for value in capsules}
    capsule_complete = len(by_id) == len(plan["episodes"]) == 24
    checks = {
        "planned_terminal_ledger": bool(ledger["passed"]),
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
        and all(bool(value["offline_candidate_replay_bit_identical"]) for value in capsules),
        "same_seed_replay": capsule_complete
        and all(bool(value["same_seed_lockstep_replay"]) for value in capsules),
        "capture_enabled_disabled_identity": capsule_complete
        and all(
            bool(value["capture_enabled_disabled_identity"]) for value in capsules
        ),
        "anchor_blobs_complete": capsule_complete
        and all(bool(value["anchor_blobs_complete"]) for value in capsules),
        "action_bounds": all(
            bool(value.get("action_bounds_valid")) for value in terminals
        ),
        "stale_action_zero": all(
            int(value.get("stale_action_applied_count", 1)) == 0
            for value in terminals
        ),
        "severe_collision_zero": all(
            int(value.get("severe_collision_count", 1)) == 0
            for value in terminals
        ),
        "invalid_force_zero": all(
            int(value.get("invalid_force_count", 1)) == 0
            for value in terminals
        ),
        "p40_conservation": all(
            float(value.get("p40_conservation_maximum_difference", 1.0))
            <= 1.0e-12
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
        "ledger": ledger,
        "planned_episode_count": plan["planned_episode_count"],
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
    if arguments.output != FORMAL_OUTPUT:
        raise ValueError(f"R0001-P50-E1 requires frozen output {FORMAL_OUTPUT}")
    if arguments.salt_file != FORMAL_SALT_FILE:
        raise ValueError(
            f"R0001-P50-E1 requires frozen salt file {FORMAL_SALT_FILE}"
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
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": status,
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "frozen_document_commit_is_ancestor": status == "complete",
        "command": list(command),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco_runtime_version(),
            "hwr_platform": importlib.metadata.version("hwr-platform"),
        },
        "source_identities": identities,
        "seed_lineage": None
        if plan is None
        else {
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
        "artifacts": {
            name: {"sha256": _sha256(content), "bytes": len(content)}
            for name, content in sorted(artifacts.items())
        },
    }


def _require_disk_capacity(output: Path) -> None:
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 5 * 1024**3:
        raise RuntimeError("P50-E1 requires at least 5GiB free on the data volume")


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P50 runner requires a full Git source commit")
    return commit


def _git_tree(root: Path, path: str) -> str:
    result = subprocess.run(
        ("git", "rev-parse", f"HEAD:{path}"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _file_identity(root: Path, path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(content),
        "bytes": len(content),
    }


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir()
    try:
        for name, content in sorted(artifacts.items()):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, content)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
