"""Build and evaluate the frozen R0001-P51-E1 convergence experiment."""

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

import numpy as np

from hwr.adapters.mujoco.cartesian_convergence import (
    CartesianConvergenceMujoco,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.eval.cartesian_convergence import (
    BANK_SCHEMA,
    B2_STEPS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    LATENCY_MATCH_LIMIT,
    PAIR_COUNT_PER_CELL,
    PLAN_ID,
    PROPOSAL_ID,
    RAW_SEED_LIMIT,
    SALT_COMMITMENT,
    SEED_AUDIT_SCHEMA,
    TASK_IDS,
    TERMINAL_SCHEMA,
    CartesianConvergenceContractError,
    analyze_terminals,
    frozen_cells,
    pair_identity,
    raw_seed_record,
    role_order,
    validate_bank,
)
from hwr.eval.seed_contract import (
    SEED_SCHEMA,
    read_seed_salt,
    require_seed_reveal,
)
from hwr.eval.tool_kinematics import recursive_xml_input_identity


MODULE_NAME = "hwr.apps.evaluate_cartesian_convergence"
REPORT_SCHEMA = "hwr.p51-cartesian-convergence-report/v1"
MANIFEST_SCHEMA = "hwr.p51-cartesian-convergence-artifacts/v1"
FAILURE_SCHEMA = "hwr.p51-cartesian-convergence-failure/v1"
FROZEN_DOCUMENT_COMMIT = "2d1752f2c0c8b9e39d7f3ebaa8e9ff0ec1d13f38"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0010/03-experiment.md")
BINDING_PATH = Path("configs/adapters/mujoco/formal_3d_v1.json")
TASK_PATH = Path("configs/tasks/formal_3d_v1.json")
SOURCE_PATHS = (
    Path("src/hwr/eval/cartesian_convergence.py"),
    Path("src/hwr/eval/cartesian_convergence_validation.py"),
    Path("src/hwr/eval/seed_contract.py"),
    Path("src/hwr/adapters/mujoco/cartesian_convergence.py"),
    Path("src/hwr/adapters/mujoco/cartesian_convergence_provenance.py"),
    Path("src/hwr/apps/evaluate_cartesian_convergence.py"),
    Path("src/hwr/eval/target_selection.py"),
    Path("src/hwr/eval/target_selection_safety.py"),
    Path("src/hwr/eval/tool_kinematics.py"),
    Path("src/hwr/adapters/mujoco/target_selection_diagnostic.py"),
    Path("src/hwr/adapters/mujoco/formal_household_backend.py"),
    Path("src/hwr/adapters/mujoco/dual_arm_backend.py"),
    Path("src/hwr/adapters/mujoco/bindings.py"),
    Path("src/hwr/adapters/mujoco/contact_ledger.py"),
    Path("src/hwr/adapters/mujoco/entity_contact_graph.py"),
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
SOURCE_TREES = (Path("src/hwr"), Path("configs"), Path("assets"))
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
CLAIM_FLAGS = {
    "training_executed": False,
    "policy_inference_executed": False,
    "capability_claim_allowed": False,
    "task_success_claim_allowed": False,
    "generalization_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
    "candidate_correctness_claim_allowed": False,
    "contact_or_grasp_claim_allowed": False,
}
UNCHANGED_FLAGS = {
    "candidate_generator_changed": False,
    "candidate_bytes_changed": False,
    "selector_changed": False,
    "acquisition_changed": False,
    "b0_b1_prefix_changed": False,
    "phase_changed": False,
    "target_formula_changed": False,
    "velocity_cap_changed": False,
    "gripper_changed": False,
    "fk_changed": False,
    "backend_changed": False,
    "safety_changed": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build-bank", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--salt-file", type=Path)
    parser.add_argument(
        "--salt-commitment",
        choices=(SALT_COMMITMENT,),
        default=SALT_COMMITMENT,
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = _command(arguments)
    started = time.perf_counter()
    identities: dict[str, object] = {}
    artifacts: dict[str, bytes] = {}
    try:
        _validate_arguments(arguments)
        identities = _source_identities(root)
        _require_clean_source(root, identities)
        if arguments.mode == "build-bank":
            _require_disk_capacity(output, 5 * 1024**3)
            salt_path = _resolve(root, arguments.salt_file)
            salt = read_seed_salt(salt_path)
            require_seed_reveal(SALT_COMMITMENT, salt)
            bank, seed_audit = build_bank(root, salt, source_commit)
            report = _bank_report(bank, seed_audit)
            artifacts = {
                "seed-audit.json": _json_bytes(seed_audit),
                "bank.json": _json_bytes(bank),
                "report.json": _json_bytes(report),
            }
        else:
            bank_path = _resolve(root, arguments.bank)
            bank_identity = _require_committed_bank(root, bank_path)
            bank = _read_json(bank_path)
            validate_bank(bank)
            _require_bank_provenance(root, bank, identities)
            terminals = evaluate_bank(root, bank)
            analysis = analyze_terminals(terminals, bank)
            report = _evaluation_report(bank, terminals, analysis)
            identities["bank"] = bank_identity
            artifacts = {
                "terminals.json": _json_bytes(terminals),
                "report.json": _json_bytes(report),
            }
        manifest = _manifest(
            arguments.mode,
            source_commit,
            command,
            identities,
            artifacts,
            started,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "mode": arguments.mode,
            "source_commit": source_commit,
            "decision": "invalid",
            "error_type": type(error).__name__,
            "error": str(error),
            **CLAIM_FLAGS,
            **UNCHANGED_FLAGS,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _manifest(
            arguments.mode,
            source_commit,
            command,
            identities,
            artifacts,
            started,
            status="failed",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
        raise
    return {
        "output": str(output),
        "mode": arguments.mode,
        "decision": report["decision"],
        "manifest_sha256": hashlib.sha256(
            artifacts["manifest.json"]
        ).hexdigest(),
    }


def build_bank(
    root: Path, salt: str, source_commit: str
) -> tuple[dict[str, object], dict[str, object]]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    _require_catalogs(tasks, bindings)
    audits: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    infeasible_cells = []
    for cell in frozen_cells():
        bridge = CartesianConvergenceMujoco(
            tasks[cell.task_id], bindings[cell.task_id]
        )
        cell_pairs = []
        latency_matched = 0
        for ordinal in range(RAW_SEED_LIMIT):
            seed = raw_seed_record(salt, cell, ordinal)
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
                "acquisition_executed": matched,
                "eligibility_reason": (
                    "pending_prefix_evaluation"
                    if matched
                    else "natural_latency_mismatch"
                ),
            }
            if matched:
                latency_matched += 1
                prefix = bridge.inspect_prefix(
                    int(seed["environment_seed"]),
                    int(seed["policy_rng_seed"]),
                )
                audit.update(prefix)
                if prefix["eligible"]:
                    pair_id = pair_identity(str(seed["planned_episode_id"]))
                    order_seed, order = role_order(salt, pair_id)
                    cell_pairs.append(
                        {
                            **audit,
                            "pair_id": pair_id,
                            "replicate_ordinal": len(cell_pairs),
                            "role_order_domain_seed": order_seed,
                            "role_order": list(order),
                        }
                    )
            audits.append(audit)
            if len(cell_pairs) == PAIR_COUNT_PER_CELL:
                break
            if latency_matched == LATENCY_MATCH_LIMIT:
                break
        if len(cell_pairs) != PAIR_COUNT_PER_CELL:
            infeasible_cells.append(
                {
                    **cell.to_dict(),
                    "eligible_count": len(cell_pairs),
                    "latency_matched_count": latency_matched,
                    "raw_seed_count": ordinal + 1,
                }
            )
        accepted.extend(cell_pairs)
    feasible = not infeasible_cells
    pairs = accepted if feasible else []
    seed_audit = {
        "schema_version": SEED_AUDIT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "plan_id": PLAN_ID,
        "salt_commitment": SALT_COMMITMENT,
        "salt_reveal": salt,
        "records": audits,
    }
    bank = {
        "schema_version": BANK_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "plan_id": PLAN_ID,
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "salt_commitment": SALT_COMMITMENT,
        "salt_reveal": salt,
        "commitment_verified": True,
        "seed_schema": SEED_SCHEMA,
        "role_enters_seed_derivation": False,
        "natural_evaluation_latency_rejection": True,
        "reset_latency_override_used": False,
        "replacement_seed_allowed": False,
        "complete_case_deletion_allowed": False,
        "cells": [cell.to_dict() for cell in frozen_cells()],
        "eligible_pair_count": len(pairs),
        "infeasible_cells": infeasible_cells,
        "seed_audit": audits,
        "source_identities": _source_identities(root),
        "pairs": pairs,
    }
    if feasible:
        validate_bank(bank)
    return bank, seed_audit


def evaluate_bank(
    root: Path, bank: Mapping[str, object]
) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    _require_catalogs(tasks, bindings)
    records = []
    for pair in bank["pairs"]:
        bridge = CartesianConvergenceMujoco(
            tasks[pair["task_id"]], bindings[pair["task_id"]]
        )
        try:
            record = bridge.evaluate_pair(pair)
        except CartesianConvergenceContractError:
            raise
        except Exception as error:
            record = _unresolved_record(pair, error)
        records.append(record)
        if record.get("hard_safety_stop") or not record.get("resolved"):
            break
    return {
        "schema_version": TERMINAL_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "bank_source_commit": bank["source_commit"],
        "planned_pair_count": len(bank["pairs"]),
        "terminal_pair_count": len(records),
        "records": records,
    }


def _bank_report(bank, seed_audit) -> dict[str, object]:
    feasible = not bank["infeasible_cells"]
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "mode": "build-bank",
        "decision": (
            "bank_complete" if feasible else "inconclusive_design_infeasible"
        ),
        "plan_id": PLAN_ID,
        "cell_count": len(bank["cells"]),
        "eligible_pair_count": bank["eligible_pair_count"],
        "raw_seed_count": len(seed_audit["records"]),
        "latency_matched_count": sum(
            bool(value["latency_matched"])
            for value in seed_audit["records"]
        ),
        "infeasible_cells": bank["infeasible_cells"],
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
    }


def _evaluation_report(bank, terminals, analysis) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "mode": "evaluate",
        "decision": analysis["decision"],
        "plan_id": PLAN_ID,
        "bank_source_commit": bank["source_commit"],
        "planned_pair_count": terminals["planned_pair_count"],
        "terminal_pair_count": terminals["terminal_pair_count"],
        "analysis": analysis,
        "estimand": (
            "frame-fixed versus legacy B2 tool-to-preposition convergence "
            "within the frozen eligible natural-latency cohort"
        ),
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
    }


def _manifest(
    mode,
    source_commit,
    command,
    identities,
    artifacts,
    started,
    *,
    status,
):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "mode": mode,
        "status": status,
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "frozen_document_commit_is_ancestor": status == "complete",
        "command": list(command),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": importlib.metadata.version("mujoco"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "device": "cpu",
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_platform_units": usage.ru_maxrss,
            "disk_free_bytes": shutil.disk_usage(
                Path(__file__).resolve().parents[3]
            ).free,
        },
        "frozen_design": {
            "plan_id": PLAN_ID,
            "cell_count": len(frozen_cells()),
            "pairs_per_cell": PAIR_COUNT_PER_CELL,
            "pair_count": len(frozen_cells()) * PAIR_COUNT_PER_CELL,
            "latency_match_limit_per_cell": LATENCY_MATCH_LIMIT,
            "raw_seed_limit_per_cell": RAW_SEED_LIMIT,
            "b2_steps_per_arm": B2_STEPS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "salt_commitment": SALT_COMMITMENT,
            "seed_schema": SEED_SCHEMA,
            "seed_derivation": {
                "planned_episode_id": (
                    "SHA256(schema || plan_id || task_id || cell_id || "
                    "candidate_ordinal)"
                ),
                "environment_seed": (
                    "int63(SHA256(salt || environment || planned_episode_id))"
                ),
                "policy_rng_seed": (
                    "int63(SHA256(salt || policy || planned_episode_id))"
                ),
                "role_enters_seed_derivation": False,
            },
        },
        "source_identities": identities,
        "historical_round_trees": dict(HISTORICAL_TREES),
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
        "artifacts": {
            name: _bytes_identity(content)
            for name, content in artifacts.items()
        },
    }


def _source_identities(root: Path) -> dict[str, object]:
    values = {
        path.as_posix(): _file_identity(root, root / path)
        for path in SOURCE_PATHS
    }
    values[BINDING_PATH.as_posix()] = _file_identity(
        root, root / BINDING_PATH
    )
    values[TASK_PATH.as_posix()] = _file_identity(root, root / TASK_PATH)
    tasks, bindings = load_default_formal_household_catalogs(root)
    _require_catalogs(tasks, bindings)
    values["recursive_mujoco_xml"] = {
        task: recursive_xml_input_identity(root, bindings[task].model_path)
        for task in TASK_IDS
    }
    robot = root / "assets/mujoco/common/robot_body.xml"
    values["robot_model"] = _file_identity(root, robot)
    values["git_trees"] = {
        path.as_posix(): _git_output(root, ("rev-parse", f"HEAD:{path}"))
        for path in SOURCE_TREES
    }
    return values


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
        raise RuntimeError("P51-E1 runner requires clean committed source")
    _require_frozen_document(root)
    for path, expected in HISTORICAL_TREES.items():
        actual = _git_output(root, ("rev-parse", f"HEAD:{path}"))
        if actual != expected:
            raise RuntimeError(f"historical research-loop tree drifted: {path}")
    if not identities:
        raise RuntimeError("P51-E1 source provenance is empty")


def _require_frozen_document(root: Path) -> None:
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
    if (root / FROZEN_DOCUMENT_PATH).read_bytes() != expected:
        raise RuntimeError("P51-E1 frozen experiment document content drifted")


def _require_committed_bank(root: Path, bank_path: Path) -> dict[str, object]:
    try:
        relative = bank_path.relative_to(root)
    except ValueError as error:
        raise ValueError("bank must be inside the source repository") from error
    subprocess.run(
        ("git", "ls-files", "--error-unmatch", relative.as_posix()),
        cwd=root,
        check=True,
        capture_output=True,
    )
    if subprocess.run(
        ("git", "diff", "--quiet", "HEAD", "--", relative.as_posix()),
        cwd=root,
        check=False,
    ).returncode:
        raise RuntimeError("bank differs from committed bytes")
    commit = _git_output(
        root, ("log", "-1", "--format=%H", "--", relative.as_posix())
    )
    if not commit:
        raise RuntimeError("bank has no committed provenance")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=root,
        check=False,
    ).returncode:
        raise RuntimeError("bank commit is not an ancestor of evaluation source")
    return {**_file_identity(root, bank_path), "commit": commit}


def _require_bank_provenance(root, bank, current) -> None:
    source = str(bank.get("source_commit", ""))
    if bank.get("frozen_document_commit") != FROZEN_DOCUMENT_COMMIT:
        raise RuntimeError("bank frozen document commit differs")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", source, "HEAD"),
        cwd=root,
        check=False,
    ).returncode:
        raise RuntimeError("bank source commit is not an ancestor")
    if bank.get("source_identities") != current:
        raise RuntimeError("source, task, binding, XML, or robot identity drifted")


def _unresolved_record(pair, error) -> dict[str, object]:
    return {
        "pair_id": pair["pair_id"],
        "planned_episode_id": pair["planned_episode_id"],
        "task_id": pair["task_id"],
        "cell_id": pair["cell_id"],
        "replicate_ordinal": pair["replicate_ordinal"],
        "observation_latency_steps": pair["observation_latency_steps"],
        "action_latency_steps": pair["action_latency_steps"],
        "environment_seed": pair["environment_seed"],
        "policy_rng_seed": pair["policy_rng_seed"],
        "role_order": pair["role_order"],
        "pair_identity_valid": None,
        "continuation_identity_equal": None,
        "first_treatment_guard": pair["first_treatment_guard"],
        "resolved": False,
        "hard_safety_stop": False,
        "infrastructure_failure": {
            "error_type": type(error).__name__,
            "error": str(error),
        },
        "arms": {},
        "safety_identity_equal": None,
        "cap_identity_equal": None,
        "gripper_identity_equal": None,
        "phase_identity_equal": None,
        "target_identity_equal": None,
        "fk_identity_equal": None,
        "backend_identity_equal": None,
    }


def _require_catalogs(tasks, bindings) -> None:
    if tuple(tasks) != TASK_IDS or tuple(bindings) != TASK_IDS:
        raise RuntimeError("P51-E1 task/binding catalog order differs")


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.salt_commitment != SALT_COMMITMENT:
        raise ValueError("salt commitment differs from frozen contract")
    if arguments.mode == "build-bank":
        if arguments.salt_file is None or arguments.bank is not None:
            raise ValueError("build-bank requires only --salt-file")
    elif arguments.bank is None or arguments.salt_file is not None:
        raise ValueError("evaluate requires only --bank and never reads salt")


def _require_disk_capacity(output: Path, required_bytes: int) -> None:
    existing = output.parent
    while not existing.exists():
        if existing.parent == existing:
            raise FileNotFoundError(output.parent)
        existing = existing.parent
    if shutil.disk_usage(existing).free < required_bytes:
        raise RuntimeError("P51-E1 bank requires at least 5 GiB free")


def _command(arguments: argparse.Namespace) -> list[str]:
    result = [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--mode",
        arguments.mode,
        "--output",
        str(arguments.output),
    ]
    if arguments.bank is not None:
        result.extend(("--bank", str(arguments.bank)))
    if arguments.salt_file is not None:
        result.extend(("--salt-file", str(arguments.salt_file)))
    result.extend(("--salt-commitment", arguments.salt_commitment))
    return result


def _source_commit(root: Path) -> str:
    commit = _git_output(root, ("rev-parse", "HEAD"))
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P51-E1 requires a full Git source commit")
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
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        **_bytes_identity(content),
    }


def _bytes_identity(content: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
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
    return 0 if result["decision"].startswith(("accepted", "bank_complete")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
