"""Evaluate the frozen R0001-P36-E2 contract without policy inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.adapters.mujoco import (
    MujocoFormalHouseholdDualArmBackend,
    load_default_formal_household_catalogs,
)
from hwr.eval.factorial_benchmark import (
    ACTION_LATENCIES,
    EXPERIMENT_ID,
    FactorialBenchmarkContractError,
    OBSERVATION_LATENCIES,
    TASKS,
    ZERO_HASH,
    PowerDesign,
    aggregate_terminal_ledger,
    benchmark_cells,
    build_planned_ledger,
    evaluate_synthetic_power,
    ledger_contract,
    make_terminal_record,
    validate_planned_ledger,
)
from hwr.eval.seed_contract import SEED_SCHEMA, seed_commitment, verify_seed_reveal


MODULE_NAME = "hwr.apps.evaluate_factorial_benchmark_contract"
REPORT_SCHEMA = "hwr.factorial-benchmark-contract-report/v1"
MANIFEST_SCHEMA = "hwr.factorial-benchmark-contract-artifacts/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    if output.exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--output",
        str(output),
        "--salt",
        arguments.salt,
    ]
    power = evaluate_synthetic_power(PowerDesign())
    selected_n = power["selected_n"]
    planned = (
        build_planned_ledger(int(selected_n), arguments.salt)
        if selected_n is not None
        else _unavailable_plan(arguments.salt)
    )
    reset_smoke = _reset_only_smoke(root)
    fault_injection = _runner_integrity_diagnostics(arguments.salt)
    seed_lineage = {
        "schema_version": SEED_SCHEMA,
        "commitment": seed_commitment(arguments.salt),
        "reveal": arguments.salt,
        "commitment_verified": verify_seed_reveal(
            seed_commitment(arguments.salt), arguments.salt
        ),
        "environment_seed_mode": "derived",
        "role_enters_seed_derivation": False,
    }
    reset_passed = bool(reset_smoke["passed"])
    integrity_passed = all(fault_injection.values())
    power_passed = selected_n is not None
    decision = (
        "accepted as balanced benchmark contract evidence"
        if power_passed and reset_passed and integrity_passed
        else (
            "inconclusive_power"
            if not power_passed and reset_passed and integrity_passed
            else "rejected"
        )
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "proposal_id": "R0001-P36-E2",
        "source_commit": source_commit,
        "invocation": {"module": MODULE_NAME, "command": command, "output": str(output)},
        "decision": decision,
        "formal_seed_bank": False,
        "capability_claim_allowed": False,
        "closed_loop_success_available": False,
        "primary_ledger": "complete_challenge",
        "full_profile_supported": False,
        "policy_inference_executed": False,
        "complete_episode_executed": False,
        "action_applied": False,
        "diagnostic_seed_lineage": seed_lineage,
        "diagnostic_planned_ledger_available": bool(
            planned["diagnostic_plan_available"]
        ),
        "formal_capability_plan_usable": False,
        "synthetic_power": power,
        "ledger_contract": ledger_contract(),
        "runner_integrity_fault_injection": fault_injection,
        "reset_only_smoke": reset_smoke,
    }
    artifacts = {
        "report.json": _json_bytes(report),
        "planned-ledger.json": _json_bytes(planned),
        "reset-only-smoke.json": _json_bytes(reset_smoke),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source_commit": source_commit,
        "command": command,
        "formal_seed_bank": False,
        "capability_claim_allowed": False,
        "closed_loop_success_available": False,
        "diagnostic_seed_lineage": seed_lineage,
        "artifacts": {
            name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
            for name, value in artifacts.items()
        },
    }
    manifest_bytes = _json_bytes(manifest)
    artifacts["manifest.json"] = manifest_bytes
    _create_output(output, artifacts)
    return {
        "output": str(output),
        "decision": decision,
        "selected_n": selected_n,
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "planned_ledger_sha256": manifest["artifacts"]["planned-ledger.json"]["sha256"],
        "reset_smoke_sha256": manifest["artifacts"]["reset-only-smoke.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_bytes": len(manifest_bytes),
    }


def _runner_integrity_diagnostics(salt: str) -> dict[str, bool]:
    planned = build_planned_ledger(1, salt, available=False)
    pair = planned["pairs"][0]
    provenance = {"deployment_sha256": "a" * 64, "source_commit": "b" * 40}
    baseline = make_terminal_record(
        pair, "baseline", "valid_success", "task_success", ZERO_HASH
    )
    candidate = make_terminal_record(
        pair, "candidate", "valid_failure", "policy_exception",
        str(baseline["record_sha256"]), policy_provenance=provenance,
    )
    aggregate = aggregate_terminal_ledger(planned, [baseline, candidate], salt)
    checks = {
        "missing_exit_is_unresolved": aggregate["missing_terminal_records"] > 0,
        "policy_failure_is_valid_failure": aggregate["valid_failure"] == 1,
        "duplicate_rejected": _raises_contract(
            lambda: aggregate_terminal_ledger(planned, [baseline, baseline], salt)
        ),
        "unknown_classification_rejected": _raises_contract(
            lambda: make_terminal_record(
                pair, "candidate", "unresolved_infrastructure", "unknown", ZERO_HASH
            )
        ),
    }
    for name, field, value in (
        ("corruption_rejected", "environment_seed", int(pair["environment_seed"]) + 1),
        ("out_of_range_cell_rejected", "cell_ordinal", 27),
        ("replacement_seed_rejected", "policy_rng_seed", int(pair["policy_rng_seed"]) + 1),
    ):
        mutation = json.loads(json.dumps(planned))
        mutation["pairs"][0][field] = value
        checks[name] = _raises_contract(
            lambda mutation=mutation: validate_planned_ledger(mutation, salt)
        )
    infrastructure = make_terminal_record(
        pair, "candidate", "unresolved_infrastructure",
        "unattributed_exception", ZERO_HASH,
    )
    unresolved = aggregate_terminal_ledger(planned, [infrastructure], salt)
    checks["infrastructure_unknown_is_unresolved"] = (
        unresolved["decision"] == "inconclusive"
    )
    checks["planned_identity_holds"] = aggregate["identity_holds"] is True
    return checks


def _raises_contract(callback) -> bool:
    try:
        callback()
    except FactorialBenchmarkContractError:
        return True
    return False


def _unavailable_plan(salt: str) -> dict[str, object]:
    return {
        "schema_version": "hwr.factorial-benchmark-plan/v1",
        "experiment_id": EXPERIMENT_ID,
        "plan_id": "R0001-P36-E2-diagnostic-plan",
        "diagnostic_plan_available": False,
        "formal_capability_plan_usable": False,
        "decision": "inconclusive_power",
        "replicate_count_per_slot_cell": None,
        "training_seed_slots": [0, 1, 2],
        "cells": benchmark_cells(),
        "cell_count": 27,
        "pair_count": 0,
        "execution_count": 0,
        "roles": ["baseline", "candidate"],
        "role_enters_seed_derivation": False,
        "cell_label_policy_visible": False,
        "policy_visible_fields": [],
        "replacement_seed_allowed": False,
        "complete_case_deletion_allowed": False,
        "formal_seed_bank": False,
        "capability_claim_allowed": False,
        "closed_loop_success_available": False,
        "seed_commitment": seed_commitment(salt),
        "seed_schema": SEED_SCHEMA,
        "environment_seed_mode": "derived",
        "pairs": [],
        "ledger_contract": ledger_contract(),
    }


def _reset_only_smoke(root: Path) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    records: list[dict[str, object]] = []
    for task_id in TASKS:
        seed = 20_263_602 + TASKS.index(task_id)
        backend = MujocoFormalHouseholdDualArmBackend(
            tasks[task_id],
            bindings[task_id],
            camera_width=16,
            camera_height=12,
            evaluation_profile=True,
        )
        backend.set_camera_rendering(False)
        try:
            for observation_latency in OBSERVATION_LATENCIES:
                for action_latency in ACTION_LATENCIES:
                    observation = backend.reset_for_latency_pair_diagnostic(
                        seed=seed,
                        task_id=task_id,
                        observation_latency_steps=observation_latency,
                        action_latency_steps=action_latency,
                    )
                    audit = backend.task_audit()
                    randomization = audit["randomization"]
                    provenance = audit["latency_pair_diagnostic"]
                    records.append(
                        {
                            "task_id": task_id,
                            "seed": seed,
                            "observation_latency_steps": observation_latency,
                            "action_latency_steps": action_latency,
                            "instruction": observation.instruction.text,
                            "instruction_sha256": _canonical_sha256(
                                observation.instruction.text
                            ),
                            "physical_state_sha256": _physical_state_sha256(backend),
                            "camera_calibration_sha256": _canonical_sha256(
                                [
                                    {
                                        "camera_id": value.camera_id,
                                        "intrinsics": value.intrinsics,
                                        "robot_from_camera": value.robot_from_camera,
                                    }
                                    for value in observation.camera_calibrations
                                ]
                            ),
                            "randomization": randomization,
                            "provenance": provenance,
                        }
                    )
        finally:
            backend.close()
    checks = _reset_checks(records)
    return {
        "schema_version": "hwr.factorial-reset-only-smoke/v1",
        "reset_count": len(records),
        "policy_inference_executed": False,
        "complete_episode_executed": False,
        "action_applied": False,
        "checks": checks,
        "passed": all(checks.values()),
        "records": records,
    }


def _reset_checks(records: Sequence[Mapping[str, object]]) -> dict[str, bool]:
    allowed = {(task, observation, action) for task in TASKS
               for observation in OBSERVATION_LATENCIES
               for action in ACTION_LATENCIES}
    actual = {
        (
            str(value["task_id"]),
            int(value["observation_latency_steps"]),
            int(value["action_latency_steps"]),
        )
        for value in records
    }
    per_task = [[value for value in records if value["task_id"] == task] for task in TASKS]
    return {
        "exact_27_cell_coverage": len(records) == 27 and actual == allowed,
        "only_latency_pair_changed": all(
            value["provenance"]["verified_only_latency_pair_changed"] is True
            for value in records
        ),
        "other_randomization_constant_per_task": all(
            len({value["provenance"]["other_randomization_sha256"] for value in values})
            == 1
            for values in per_task
        ),
        "sampled_randomization_constant_per_task": all(
            len({value["provenance"]["sampled_randomization_sha256"] for value in values})
            == 1
            for values in per_task
        ),
        "instruction_constant_per_task": all(
            len({value["instruction_sha256"] for value in values}) == 1
            for values in per_task
        ),
        "physical_state_constant_per_task": all(
            len({value["physical_state_sha256"] for value in values}) == 1
            for values in per_task
        ),
        "camera_calibration_constant_per_task": all(
            len({value["camera_calibration_sha256"] for value in values}) == 1
            for values in per_task
        ),
        "effective_latencies_match_cells": all(
            value["provenance"]["effective_observation_latency_steps"]
            == value["observation_latency_steps"]
            and value["provenance"]["effective_action_latency_steps"]
            == value["action_latency_steps"]
            for value in records
        ),
    }


def _physical_state_sha256(backend: MujocoFormalHouseholdDualArmBackend) -> str:
    snapshot = backend.capture_state_snapshot()
    value = {
        "task_id": snapshot.task_id,
        "backend_fingerprint": snapshot.backend_fingerprint,
        "generalized_positions": snapshot.generalized_positions,
        "generalized_velocities": snapshot.generalized_velocities,
        "actuator_controls": snapshot.actuator_controls,
    }
    return _canonical_sha256(value)


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        raise FileExistsError(staging)
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


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_commit(root: Path) -> str:
    git_path = root / ".git"
    if git_path.is_file():
        marker = git_path.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise RuntimeError("P36 source Git metadata is invalid")
        git_path = (root / marker.removeprefix("gitdir: ")).resolve()
    head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = head.removeprefix("ref: ")
        loose = git_path / reference
        head = (
            loose.read_text(encoding="utf-8").strip()
            if loose.is_file()
            else _packed_reference(git_path, reference)
        )
    if len(head) != 40 or any(value not in "0123456789abcdef" for value in head):
        raise RuntimeError("P36 requires a full Git source commit")
    return head


def _packed_reference(git_path: Path, reference: str) -> str:
    for line in (git_path / "packed-refs").read_text(encoding="utf-8").splitlines():
        fields = line.split(" ", 1)
        if len(fields) == 2 and fields[1] == reference:
            return fields[0]
    raise RuntimeError("P36 source Git reference is unresolved")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
