"""Run the R0019 paired baseline/privileged-teacher basket evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.bimanual_teacher import (
    BASELINE_SOURCE,
    BASKET_TASK_ID,
    TEACHER_SOURCE,
    BasketTeacherError,
    GenericBasketPrimitiveBaseline,
    PrivilegedBasketTeacher,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_bimanual_training_catalogs,
)
from hwr.train.bimanual_runtime import dual_arm_action_frame


DEVELOPMENT_SEEDS = (19001,)
CONFIRMATION_SEED_START = 91_900_001
CONFIRMATION_SEED_STRIDE = 104_729
CONFIRMATION_EPISODES = 100
SUCCESS_THRESHOLD = 80
SEVERE_COLLISION_THRESHOLD = 0
WALL_TIME_LIMIT_SECONDS = 1_800.0
CONFIRMATION_OUTPUT = Path(
    "runs/research-loop/0019/confirmation/paired-100.json"
)
REQUIRED_TEACHER_TASK_PHASES = frozenset(
    {
        "approach",
        "acquire",
        "secure",
        "lift",
        "target_transport",
        "place",
        "release",
        "stabilize",
    }
)
SOURCE_PATHS = (
    Path("assets/mujoco/bimanual/living_basket.xml"),
    Path("assets/mujoco/common/robot_assets.xml"),
    Path("assets/mujoco/common/robot_defaults.xml"),
    Path("assets/mujoco/common/robot_body.xml"),
    Path("assets/mujoco/common/robot_contacts.xml"),
    Path("assets/mujoco/common/robot_actuators.xml"),
    Path("assets/mujoco/common/robot_sensors.xml"),
    Path("src/hwr/adapters/mujoco/bimanual_teacher.py"),
    Path("src/hwr/apps/evaluate_bimanual_teacher.py"),
    Path("src/hwr/adapters/mujoco/bimanual_backend.py"),
    Path("src/hwr/adapters/mujoco/bimanual_bindings.py"),
    Path("src/hwr/adapters/mujoco/dual_arm_backend.py"),
    Path("src/hwr/adapters/mujoco/training_catalog.py"),
    Path("src/hwr/core/embodied.py"),
    Path("src/hwr/safety/dual_arm.py"),
    Path("src/hwr/tasks/bimanual.py"),
    Path("src/hwr/train/bimanual_runtime.py"),
    Path("configs/tasks/bimanual_household_v1.json"),
    Path("configs/adapters/mujoco/bimanual_household_v1.json"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("development", "confirmation"),
        default="development",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--controller",
        choices=("baseline", "teacher", "paired"),
        default="paired",
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--qualification-report", type=Path)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _output_path(root, arguments.mode, arguments.output)
    seeds = _seeds(arguments)
    controllers = (
        ("baseline", "teacher")
        if arguments.controller == "paired"
        else (arguments.controller,)
    )
    source_worktree_dirty = _source_worktree_dirty(root)
    source_commit = _source_commit(root)
    source_files = {
        str(path): _sha256(root / path) for path in SOURCE_PATHS
    }
    _require_confirmation_preconditions(
        arguments.mode,
        controllers,
        source_worktree_dirty,
        output=output,
        root=root,
        output_exists=output.exists(),
    )
    qualification = _confirmation_qualification(
        mode=arguments.mode,
        report_path=arguments.qualification_report,
        root=root,
        source_commit=source_commit,
        source_files=source_files,
    )
    started = time.perf_counter()
    episodes = []
    run_stop_reason = None
    episode_plan = (
        (seed, controller)
        for seed in seeds
        for controller in controllers
    )
    for seed, controller in episode_plan:
        try:
            episode = _run_episode(root, seed, controller)
        except Exception as error:
            episode = _infrastructure_failure_episode(
                seed=seed,
                controller=controller,
                error=error,
            )
            run_stop_reason = "infrastructure_error"
        episodes.append(episode)
        if run_stop_reason is not None:
            break
        if time.perf_counter() - started > WALL_TIME_LIMIT_SECONDS:
            run_stop_reason = "wall_time_budget_exceeded"
            break
    report = _report(
        mode=arguments.mode,
        controllers=controllers,
        seeds=seeds,
        episodes=episodes,
        source_commit=source_commit,
        source_files=source_files,
        source_worktree_dirty=source_worktree_dirty,
        elapsed_seconds=time.perf_counter() - started,
        confirmation_qualification=qualification,
        run_stop_reason=run_stop_reason,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return report


def _output_path(root: Path, mode: str, output: Path | None) -> Path:
    selected = (
        output
        or CONFIRMATION_OUTPUT
        if mode == "confirmation"
        else output or Path("runs/research-loop/0019/development/latest.json")
    )
    return selected if selected.is_absolute() else root / selected


def _seeds(arguments: argparse.Namespace) -> tuple[int, ...]:
    if arguments.mode == "confirmation":
        if arguments.seed:
            raise ValueError("confirmation seeds are frozen and cannot be overridden")
        if arguments.controller != "paired":
            raise ValueError("confirmation mode requires paired controllers")
        return tuple(
            CONFIRMATION_SEED_START + index * CONFIRMATION_SEED_STRIDE
            for index in range(CONFIRMATION_EPISODES)
        )
    return tuple(arguments.seed or DEVELOPMENT_SEEDS)


def _source_worktree_dirty(root: Path) -> bool:
    return bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _require_confirmation_preconditions(
    mode: str,
    controllers: Sequence[str],
    source_worktree_dirty: bool,
    *,
    output: Path,
    root: Path,
    output_exists: bool,
) -> None:
    if mode != "confirmation":
        return
    errors = _confirmation_preflight_errors(
        controllers=controllers,
        source_worktree_dirty=source_worktree_dirty,
        output=output,
        root=root,
        output_exists=output_exists,
    )
    if errors:
        raise RuntimeError(
            "confirmation preconditions failed: " + ", ".join(errors)
        )


def _confirmation_preflight_errors(
    *,
    controllers: Sequence[str],
    source_worktree_dirty: bool,
    output: Path | None,
    root: Path | None,
    output_exists: bool,
) -> list[str]:
    errors = []
    if tuple(controllers) != ("baseline", "teacher"):
        errors.append("confirmation_controllers_not_paired")
    if source_worktree_dirty:
        errors.append("source_worktree_dirty")
    if output_exists:
        errors.append("confirmation_output_already_exists")
    if output is not None and root is not None:
        expected_output = (root / CONFIRMATION_OUTPUT).resolve()
        if output.resolve() != expected_output:
            errors.append("confirmation_output_path_mismatch")
    missing_phases = sorted(
        REQUIRED_TEACHER_TASK_PHASES
        - PrivilegedBasketTeacher.implemented_task_phases
    )
    if missing_phases:
        errors.append("teacher_missing_task_phases:" + ",".join(missing_phases))
    return errors


def _confirmation_qualification(
    *,
    mode: str,
    report_path: Path | None,
    root: Path,
    source_commit: str,
    source_files: Mapping[str, str],
) -> dict[str, object]:
    if mode != "confirmation":
        if report_path is not None:
            raise ValueError(
                "qualification report is only valid in confirmation mode"
            )
        return {"status": "not_applicable"}
    if report_path is None:
        raise RuntimeError(
            "confirmation requires --qualification-report from a clean "
            "development run with at least one teacher success"
        )
    resolved = (
        report_path if report_path.is_absolute() else root / report_path
    ).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError("confirmation qualification report must be inside repo")
    report = json.loads(resolved.read_text(encoding="utf-8"))
    errors = []
    if report.get("mode") != "development":
        errors.append("qualification_not_development")
    if report.get("task_id") != BASKET_TASK_ID:
        errors.append("qualification_task_mismatch")
    if report.get("source_commit") != source_commit:
        errors.append("qualification_source_commit_mismatch")
    if bool(report.get("source_worktree_dirty", True)):
        errors.append("qualification_source_worktree_dirty")
    if report.get("source_files") != dict(source_files):
        errors.append("qualification_source_files_mismatch")
    if report.get("implementation_evidence", {}).get("valid") is not True:
        errors.append("qualification_teacher_implementation_invalid")
    if report.get("decision") != "validated_development":
        errors.append("qualification_decision_invalid")
    teacher = report.get("summaries", {}).get("teacher", {})
    if int(teacher.get("successes", 0)) < 1:
        errors.append("qualification_teacher_has_no_success")
    if report.get("confirmation_evidence", {}).get("status") != "not_run":
        errors.append("qualification_confirmation_status_invalid")
    if report.get("run_status", {}).get("completed") is not True:
        errors.append("qualification_run_incomplete")
    if errors:
        raise RuntimeError(
            "confirmation qualification failed: " + ", ".join(errors)
        )
    return {
        "status": "validated",
        "path": str(resolved.relative_to(root)),
        "sha256": _sha256(resolved),
    }


def _run_episode(root: Path, seed: int, controller_name: str) -> dict[str, object]:
    tasks, bindings = load_default_bimanual_training_catalogs(root)
    backend = MujocoBimanualTaskBackend(
        tasks[BASKET_TASK_ID],
        bindings[BASKET_TASK_ID],
        camera_width=256,
        camera_height=192,
    )
    controller = (
        GenericBasketPrimitiveBaseline(backend, seed=seed)
        if controller_name == "baseline"
        else PrivilegedBasketTeacher(backend, seed=seed)
    )
    started = time.perf_counter()
    event_counts: Counter[str] = Counter()
    stage_rows: list[dict[str, object]] = []
    safety_interventions = 0
    minimum_reach = {"left": float("inf"), "right": float("inf")}
    first_contact = {"left": None, "right": None, "bilateral": None}
    controller_failure = None
    executed_steps = 0
    try:
        observation = backend.reset(seed=seed, task_id=BASKET_TASK_ID)
        backend.set_camera_rendering(False)
        if isinstance(controller, GenericBasketPrimitiveBaseline):
            controller.reset(observation)
        for step in range(tasks[BASKET_TASK_ID].max_steps):
            try:
                output = controller.action(observation)
            except BasketTeacherError as error:
                controller_failure = str(error)
                controller.failure_stage = "grasp_planning_failed"
                break
            outcome = backend.apply(
                dual_arm_action_frame(
                    observation.timestamp_ns,
                    output.action,
                    source=(
                        BASELINE_SOURCE
                        if controller_name == "baseline"
                        else TEACHER_SOURCE
                    ),
                )
            )
            executed_steps += 1
            observation = outcome.observation
            applied = outcome.info["applied_action"].action
            if isinstance(controller, GenericBasketPrimitiveBaseline):
                controller.record_applied(applied)
            safety_interventions += int(outcome.info["safety_intervened"])
            for event in outcome.events:
                reason = str(event.details.get("reason", ""))
                event_counts[f"{event.event_type}:{reason}"] += 1
            audit = backend.task_audit()
            metrics = audit["metrics"]
            minimum_reach["left"] = min(
                minimum_reach["left"], float(metrics["left_reach_distance"])
            )
            minimum_reach["right"] = min(
                minimum_reach["right"], float(metrics["right_reach_distance"])
            )
            contacts = (
                bool(metrics["left_contact"]),
                bool(metrics["right_contact"]),
            )
            if first_contact["left"] is None and contacts[0]:
                first_contact["left"] = step
            if first_contact["right"] is None and contacts[1]:
                first_contact["right"] = step
            if first_contact["bilateral"] is None and all(contacts):
                first_contact["bilateral"] = step
            if not stage_rows or stage_rows[-1]["stage"] != output.stage:
                stage_rows.append({"stage": output.stage, "start_step": step})
            stage_rows[-1].update(
                {
                    "end_step": step,
                    "left_reach_distance_m": metrics["left_reach_distance"],
                    "right_reach_distance_m": metrics["right_reach_distance"],
                    "maximum_concurrent_steps": audit["maximum_concurrent_steps"],
                    "target_distance_m": metrics["target_distance"],
                    "safety_interventions": safety_interventions,
                }
            )
            if outcome.terminated or outcome.truncated:
                break
        result = backend.result()
        audit = backend.task_audit()
        failure_stage = getattr(controller, "failure_stage", None)
        if (
            controller_name == "teacher"
            and failure_stage is None
            and not bool(result and result.success)
        ):
            failure_stage = f"{controller.stage}_timeout"
        return {
            "controller": controller_name,
            "seed": seed,
            "steps": executed_steps,
            "episode_completed": (
                result is not None or controller_failure is not None
            ),
            "physics_episode_completed": result is not None,
            "valid_episode_result": True,
            "controller_failure": controller_failure,
            "elapsed_seconds": time.perf_counter() - started,
            "success": bool(result and result.success),
            "termination_reason": (
                "controller_failure"
                if controller_failure is not None
                else None
                if result is None
                else result.reason
            ),
            "first_contact_step": first_contact,
            "minimum_reach_distance_m": {
                arm: None if value == float("inf") else value
                for arm, value in minimum_reach.items()
            },
            "safety_intervention_count": safety_interventions,
            "event_counts": dict(sorted(event_counts.items())),
            "stage_records": stage_rows,
            "teacher_failure_stage": failure_stage,
            "audit": audit,
        }
    finally:
        backend.close()


def _infrastructure_failure_episode(
    *,
    seed: int,
    controller: str,
    error: Exception,
) -> dict[str, object]:
    return {
        "controller": controller,
        "seed": seed,
        "steps": 0,
        "episode_completed": False,
        "physics_episode_completed": False,
        "valid_episode_result": False,
        "controller_failure": None,
        "infrastructure_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "elapsed_seconds": 0.0,
        "success": False,
        "termination_reason": "infrastructure_error",
        "first_contact_step": {
            "left": None,
            "right": None,
            "bilateral": None,
        },
        "minimum_reach_distance_m": {
            "left": None,
            "right": None,
        },
        "safety_intervention_count": 0,
        "event_counts": {},
        "stage_records": [],
        "teacher_failure_stage": None,
        "audit": {
            "severe_collision_count": 0,
            "maximum_concurrent_steps": 0,
        },
    }


def _report(
    *,
    mode: str,
    controllers: Sequence[str],
    seeds: Sequence[int],
    episodes: Sequence[Mapping[str, object]],
    source_commit: str,
    source_files: Mapping[str, str],
    source_worktree_dirty: bool,
    elapsed_seconds: float,
    confirmation_qualification: Mapping[str, object] | None = None,
    run_stop_reason: str | None = None,
) -> dict[str, object]:
    summaries = {}
    for controller in controllers:
        rows = [row for row in episodes if row["controller"] == controller]
        summaries[controller] = {
            "episodes": len(rows),
            "valid_episode_results": sum(
                bool(row.get("valid_episode_result")) for row in rows
            ),
            "successes": sum(bool(row["success"]) for row in rows),
            "controller_failures": sum(
                row.get("controller_failure") is not None for row in rows
            ),
            "infrastructure_failures": sum(
                row.get("infrastructure_error") is not None for row in rows
            ),
            "actual_severe_collisions": sum(
                int(row["audit"]["severe_collision_count"]) for row in rows
            ),
            "safety_interventions": sum(
                int(row["safety_intervention_count"]) for row in rows
            ),
            "episodes_with_bilateral_contact": sum(
                row["first_contact_step"]["bilateral"] is not None for row in rows
            ),
            "maximum_concurrent_steps": max(
                (int(row["audit"]["maximum_concurrent_steps"]) for row in rows),
                default=0,
            ),
            "termination_reasons": dict(
                Counter(str(row["termination_reason"]) for row in rows)
            ),
        }
    confirmation_errors = _confirmation_validation_errors(
        mode=mode,
        controllers=controllers,
        seeds=seeds,
        episodes=episodes,
        source_worktree_dirty=source_worktree_dirty,
        confirmation_qualification=confirmation_qualification,
        run_stop_reason=run_stop_reason,
    )
    teacher = summaries.get("teacher")
    passed = bool(
        mode == "confirmation"
        and not confirmation_errors
        and teacher is not None
        and teacher["successes"] >= SUCCESS_THRESHOLD
        and teacher["actual_severe_collisions"] == SEVERE_COLLISION_THRESHOLD
    )
    implementation_errors = _teacher_implementation_errors(controllers)
    development_supported = bool(
        mode == "development"
        and run_stop_reason is None
        and not implementation_errors
        and teacher is not None
        and (
            teacher["successes"] >= 1
            and teacher["actual_severe_collisions"]
            == SEVERE_COLLISION_THRESHOLD
        )
    )
    confirmation_status = (
        "not_run"
        if mode != "confirmation"
        else "incomplete"
        if confirmation_errors
        else "completed"
    )
    decision = (
        "invalid"
        if implementation_errors
        or confirmation_errors
        or run_stop_reason is not None
        else "validated_development"
        if passed or development_supported
        else "abandoned"
    )
    return {
        "schema_version": "hwr.r0019-basket-teacher-evaluation/v2",
        "mode": mode,
        "task_id": BASKET_TASK_ID,
        "source_commit": source_commit,
        "source_files": dict(source_files),
        "source_worktree_dirty": source_worktree_dirty,
        "seed_domain": {
            "kind": mode,
            "seeds": list(seeds),
        },
        "controllers": list(controllers),
        "frozen_confirmation_design": {
            "paired": True,
            "episode_count_per_controller": CONFIRMATION_EPISODES,
            "seed_start": CONFIRMATION_SEED_START,
            "seed_stride": CONFIRMATION_SEED_STRIDE,
            "teacher_success_threshold": SUCCESS_THRESHOLD,
            "actual_severe_collision_threshold": SEVERE_COLLISION_THRESHOLD,
            "wall_time_limit_seconds": WALL_TIME_LIMIT_SECONDS,
            "task_max_steps": 1_200,
            "safety_threshold_newtons": 220.0,
        },
        "elapsed_seconds": elapsed_seconds,
        "run_status": {
            "completed": run_stop_reason is None,
            "stop_reason": run_stop_reason,
        },
        "summaries": summaries,
        "episodes": list(episodes),
        "confirmation_qualification": dict(
            confirmation_qualification or {"status": "not_applicable"}
        ),
        "implementation_evidence": {
            "valid": not implementation_errors,
            "errors": implementation_errors,
            "implemented_teacher_task_phases": sorted(
                PrivilegedBasketTeacher.implemented_task_phases
            ),
            "required_teacher_task_phases": sorted(
                REQUIRED_TEACHER_TASK_PHASES
            ),
        },
        "confirmation_evidence": {
            "status": confirmation_status,
            "valid": (
                None
                if confirmation_status == "not_run"
                else not confirmation_errors
            ),
            "errors": confirmation_errors,
        },
        "l0_gate_passed": passed,
        "decision": decision,
        "allowed_claim": (
            "task and control chain have a feasible privileged teacher ceiling"
            if passed
            else "implemented teacher subgoal behavior and failure localization only"
            if implementation_errors
            else "development behavior and failure localization only"
        ),
        "disallowed_claim": (
            "full-task teacher ceiling, deployable policy capability, visual "
            "generalization, or hardware transfer"
        ),
    }


def _teacher_implementation_errors(controllers: Sequence[str]) -> list[str]:
    if "teacher" not in controllers:
        return []
    missing_phases = sorted(
        REQUIRED_TEACHER_TASK_PHASES
        - PrivilegedBasketTeacher.implemented_task_phases
    )
    if not missing_phases:
        return []
    return ["teacher_missing_task_phases:" + ",".join(missing_phases)]


def _confirmation_validation_errors(
    *,
    mode: str,
    controllers: Sequence[str],
    seeds: Sequence[int],
    episodes: Sequence[Mapping[str, object]],
    source_worktree_dirty: bool,
    confirmation_qualification: Mapping[str, object] | None,
    run_stop_reason: str | None,
) -> list[str]:
    if mode != "confirmation":
        return []
    errors = []
    if (confirmation_qualification or {}).get("status") != "validated":
        errors.append("confirmation_qualification_invalid")
    if run_stop_reason is not None:
        errors.append(f"confirmation_run_stopped:{run_stop_reason}")
    expected_seeds = tuple(
        CONFIRMATION_SEED_START + index * CONFIRMATION_SEED_STRIDE
        for index in range(CONFIRMATION_EPISODES)
    )
    if tuple(seeds) != expected_seeds:
        errors.append("confirmation_seed_domain_mismatch")
    errors.extend(
        _confirmation_preflight_errors(
            controllers=controllers,
            source_worktree_dirty=source_worktree_dirty,
            output=None,
            root=None,
            output_exists=False,
        )
    )
    actual_pairs = Counter(
        (int(row["seed"]), str(row["controller"])) for row in episodes
    )
    expected_pairs = Counter(
        (seed, controller)
        for seed in expected_seeds
        for controller in ("baseline", "teacher")
    )
    if actual_pairs != expected_pairs:
        errors.append("confirmation_episode_pairs_incomplete")
    if any(
        not bool(row.get("valid_episode_result"))
        for row in episodes
    ):
        errors.append("confirmation_episode_execution_incomplete")
    return errors


def _source_commit(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "l0_gate_passed": report["l0_gate_passed"],
                "run_status": report["run_status"],
                "summaries": report["summaries"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return _exit_code(report)


def _exit_code(report: Mapping[str, object]) -> int:
    return 0 if report["decision"] == "validated_development" else 2


if __name__ == "__main__":
    raise SystemExit(main())
