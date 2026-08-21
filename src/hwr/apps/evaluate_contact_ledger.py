"""Run the frozen R0001-P40-E1 report-only contact ledger contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.adapters.mujoco import (
    CONTACT_CATEGORIES,
    MujocoFormalHouseholdDualArmBackend,
    load_default_formal_household_catalogs,
    run_timestep_stability_fixture,
)
from hwr.core.embodied import DualArmAction, DualArmActionFrame


MODULE_NAME = "hwr.apps.evaluate_contact_ledger"
PROPOSAL_ID = "R0001-P40-E1"
REPORT_SCHEMA = "hwr.contact-ledger-contract-report/v1"
MANIFEST_SCHEMA = "hwr.contact-ledger-contract-artifacts/v1"
TASK_IDS = (
    "tidy_living_room_3d/v1",
    "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
)
BASE_SEED = 20_264_001
SEED_STRIDE = 104_729
CONTROL_STEP_LIMIT = 32
FROZEN_PARENT_COMMIT = "a722f3522cdb8f12c1a78c56ce8c1d7c873e9190"
FROZEN_DOCUMENT_COMMIT = "4ef5f3b728e3e13aa18552ea6cb744121ccce71f"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    _require_clean_source(root)
    source_commit = _source_commit(root)
    binding_path = root / "configs/adapters/mujoco/formal_3d_v1.json"
    binding_identity = _file_identity(binding_path)
    command = [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--output",
        str(arguments.output),
    ]
    try:
        evaluation = _evaluate_contract(root)
        report = _build_report(source_commit, command, evaluation)
        artifacts = {"report.json": _json_bytes(report)}
        manifest = _manifest(
            source_commit,
            command,
            binding_identity,
            evaluation,
            artifacts,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": "hwr.contact-ledger-contract-failure/v1",
            "proposal_id": PROPOSAL_ID,
            "source_commit": source_commit,
            "error_type": type(error).__name__,
            "error": str(error),
            "measurement_only": True,
            "capability_claim_allowed": False,
            "hardware_safety_claim_allowed": False,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _manifest(
            source_commit,
            command,
            binding_identity,
            None,
            artifacts,
            status="failed",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
        raise
    return {
        "output": str(output),
        "decision": report["decision"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
    }


def _evaluate_contract(root: Path) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    if set(tasks) != set(TASK_IDS):
        raise RuntimeError("P40 task catalog differs from the frozen contract")
    fixture = run_timestep_stability_fixture()
    task_reports = []
    for task_index, task_id in enumerate(TASK_IDS):
        seed = BASE_SEED + task_index * SEED_STRIDE
        disabled = _run_trace(
            tasks[task_id], bindings[task_id], seed=seed, enabled=False
        )
        enabled = _run_trace(
            tasks[task_id], bindings[task_id], seed=seed, enabled=True
        )
        traces_identical = disabled["trace"] == enabled["trace"]
        ledger = enabled["contact_ledger"]
        task_reports.append(
            {
                "task_id": task_id,
                "seed": seed,
                "fixed_hold_action": enabled["fixed_hold_action"],
                "camera_rendering_enabled": False,
                "control_step_limit": CONTROL_STEP_LIMIT,
                "disabled_legacy_trace": disabled["trace"],
                "enabled_legacy_trace": enabled["trace"],
                "disabled_trace_sha256": _canonical_sha256(disabled["trace"]),
                "enabled_trace_sha256": _canonical_sha256(enabled["trace"]),
                "legacy_trace_bit_identical": traces_identical,
                "contact_ledger": ledger,
                "physics": enabled["physics"],
                "allowed_robot_contact_roles": {
                    role: sorted(names)
                    for role, names in bindings[
                        task_id
                    ].allowed_robot_contact_roles.items()
                },
                "legacy_allowed_robot_contact_geoms": sorted(
                    bindings[task_id].allowed_robot_contact_geoms
                ),
            }
        )
    checks = {
        "timestep_stability_passed": fixture["passed"] is True,
        "legacy_traces_bit_identical": all(
            value["legacy_trace_bit_identical"] for value in task_reports
        ),
        "all_ledgers_valid": all(
            value["contact_ledger"]["contract_valid"] for value in task_reports
        ),
        "all_categories_published": all(
            set(value["contact_ledger"]["categories"]) == set(CONTACT_CATEGORIES)
            for value in task_reports
        ),
        "all_periods_complete": all(
            value["contact_ledger"]["control_period_count"]
            == len(value["enabled_legacy_trace"])
            for value in task_reports
        ),
        "role_union_matches_legacy_allow_list": all(
            set().union(*map(set, value["allowed_robot_contact_roles"].values()))
            == set(value["legacy_allowed_robot_contact_geoms"])
            for value in task_reports
        ),
    }
    return {
        "timestep_fixture": fixture,
        "tasks": task_reports,
        "checks": checks,
        "passed": all(checks.values()),
        "physics": {
            value["task_id"]: value["physics"] for value in task_reports
        },
    }


def _run_trace(task, binding, *, seed: int, enabled: bool) -> dict[str, object]:
    backend = MujocoFormalHouseholdDualArmBackend(
        task,
        binding,
        camera_width=16,
        camera_height=12,
        evaluation_profile=True,
    )
    trace: list[dict[str, object]] = []
    try:
        backend.contact_ledger.set_enabled(enabled)
        observation = backend.reset(seed=seed, task_id=task.task_id)
        backend.set_camera_rendering(False)
        action = DualArmAction(
            0.0,
            0.0,
            (0.0,) * 6,
            (0.0,) * 6,
            observation.proprioception.left_gripper_position,
            observation.proprioception.right_gripper_position,
        )
        for step in range(CONTROL_STEP_LIMIT):
            timestamp = observation.timestamp_ns
            outcome = backend.apply(
                DualArmActionFrame(
                    timestamp,
                    timestamp,
                    timestamp + 250_000_000,
                    "R0001-P40-E1-fixed-hold",
                    action,
                )
            )
            observation = outcome.observation
            audit = backend.task_audit()
            result = backend.result()
            applied = outcome.info["applied_action"]
            trace.append(
                {
                    "step": step,
                    "applied_action_vector": list(applied.action.vector()),
                    "proprioception": list(observation.proprioception.vector()),
                    "reward": outcome.reward,
                    "terminated": outcome.terminated,
                    "truncated": outcome.truncated,
                    "success": None if result is None else result.success,
                    "reason": None if result is None else result.reason,
                    "severe_collision_count": audit["severe_collision_count"],
                    "maximum_forbidden_force": audit["maximum_forbidden_force"],
                    "maximum_forbidden_pair": audit["maximum_forbidden_pair"],
                    "safety_intervened": outcome.info["safety_intervened"],
                }
            )
            if outcome.terminated or outcome.truncated:
                break
        ledger = backend.contact_ledger.report()
        physics = {
            "timestep": float(backend.model.opt.timestep),
            "solver": int(backend.model.opt.solver),
            "iterations": int(backend.model.opt.iterations),
            "tolerance": float(backend.model.opt.tolerance),
            "control_hz": float(task.control_hz),
            "substeps_per_control_period": int(backend._substeps),
        }
    finally:
        backend.close()
    return {
        "trace": trace,
        "contact_ledger": ledger,
        "fixed_hold_action": list(action.vector()),
        "physics": physics,
    }


def _build_report(
    source_commit: str,
    command: Sequence[str],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    passed = bool(evaluation["passed"])
    legacy_unchanged = bool(
        evaluation["checks"]["legacy_traces_bit_identical"]
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "decision": (
            "accepted as safety measurement contract evidence"
            if passed
            else "rejected"
        ),
        "capability_claim_allowed": False,
        "hardware_safety_claim_allowed": False,
        "measurement_only": True,
        "legacy_safety_decision_unchanged": legacy_unchanged,
        "forbidden_force_threshold_newtons": 220.0,
        "forbidden_force_threshold_is_hardware_safety_threshold": False,
        "policy_inference_executed": False,
        "closed_loop_capability_episode_executed": False,
        **evaluation,
    }


def _manifest(
    source_commit: str,
    command: Sequence[str],
    binding_identity: Mapping[str, object],
    evaluation: Mapping[str, object] | None,
    artifacts: Mapping[str, bytes],
    *,
    status: str,
) -> dict[str, object]:
    legacy_unchanged = bool(
        evaluation is not None
        and evaluation["checks"]["legacy_traces_bit_identical"]
    )
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": status,
        "source_commit": source_commit,
        "frozen_parent_commit": FROZEN_PARENT_COMMIT,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "command": list(command),
        "binding": dict(binding_identity),
        "physics": None if evaluation is None else evaluation["physics"],
        "timestep_fixture": (
            None
            if evaluation is None
            else {
                "schema_version": evaluation["timestep_fixture"]["schema_version"],
                "fixture_xml_sha256": evaluation[
                    "timestep_fixture"
                ]["fixture_xml_sha256"],
            }
        ),
        "constants": {
            "base_seed": BASE_SEED,
            "seed_stride": SEED_STRIDE,
            "control_step_limit": CONTROL_STEP_LIMIT,
            "task_ids": list(TASK_IDS),
            "categories": list(CONTACT_CATEGORIES),
            "forbidden_force_threshold_newtons": 220.0,
        },
        "capability_claim_allowed": False,
        "hardware_safety_claim_allowed": False,
        "measurement_only": True,
        "legacy_safety_decision_unchanged": legacy_unchanged,
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for name, content in artifacts.items()
        },
    }


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
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


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_clean_source(root: Path) -> None:
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("P40 formal runner requires clean committed source")
    for commit in (FROZEN_PARENT_COMMIT, FROZEN_DOCUMENT_COMMIT):
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            cwd=root,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("P40 frozen document commit is not an ancestor")
    history = tuple(
        f"docs/research-loop/{index:04d}" for index in range(1, 7)
    )
    result = subprocess.run(
        ("git", "diff", "--quiet", FROZEN_PARENT_COMMIT, "HEAD", "--", *history),
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("P40 historical research-loop documents drifted")


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
        raise RuntimeError("P40 formal runner requires a full Git source commit")
    return commit


def _file_identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(Path(__file__).resolve().parents[3]).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _canonical_sha256(value: object) -> str:
    content = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


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
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
