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
SOURCE_PATHS = (
    Path("src/hwr/adapters/mujoco/bimanual_teacher.py"),
    Path("src/hwr/apps/evaluate_bimanual_teacher.py"),
    Path("src/hwr/adapters/mujoco/bimanual_backend.py"),
    Path("src/hwr/adapters/mujoco/dual_arm_backend.py"),
    Path("src/hwr/safety/dual_arm.py"),
    Path("src/hwr/tasks/bimanual.py"),
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
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output_argument = arguments.output or Path(
        f"runs/research-loop/0019/{arguments.mode}/latest.json"
    )
    output = output_argument if output_argument.is_absolute() else root / output_argument
    seeds = _seeds(arguments)
    controllers = (
        ("baseline", "teacher")
        if arguments.controller == "paired"
        else (arguments.controller,)
    )
    source_worktree_dirty = _source_worktree_dirty(root)
    _require_clean_confirmation_source(arguments.mode, source_worktree_dirty)
    started = time.perf_counter()
    episodes = []
    for seed in seeds:
        for controller in controllers:
            episodes.append(_run_episode(root, seed, controller))
            if time.perf_counter() - started > WALL_TIME_LIMIT_SECONDS:
                raise RuntimeError("R0019 wall-time budget exceeded")
    report = _report(
        mode=arguments.mode,
        controllers=controllers,
        seeds=seeds,
        episodes=episodes,
        source_commit=_source_commit(root),
        source_files={
            str(path): _sha256(root / path) for path in SOURCE_PATHS
        },
        source_worktree_dirty=source_worktree_dirty,
        elapsed_seconds=time.perf_counter() - started,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return report


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


def _require_clean_confirmation_source(
    mode: str,
    source_worktree_dirty: bool,
) -> None:
    if mode == "confirmation" and source_worktree_dirty:
        raise RuntimeError(
            "confirmation requires a clean committed source worktree"
        )


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
    try:
        observation = backend.reset(seed=seed, task_id=BASKET_TASK_ID)
        backend.set_camera_rendering(False)
        if isinstance(controller, GenericBasketPrimitiveBaseline):
            controller.reset(observation)
        for step in range(tasks[BASKET_TASK_ID].max_steps):
            output = controller.action(observation)
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
            "steps": step + 1,
            "elapsed_seconds": time.perf_counter() - started,
            "success": bool(result and result.success),
            "termination_reason": None if result is None else result.reason,
            "first_contact_step": first_contact,
            "minimum_reach_distance_m": minimum_reach,
            "safety_intervention_count": safety_interventions,
            "event_counts": dict(sorted(event_counts.items())),
            "stage_records": stage_rows,
            "teacher_failure_stage": failure_stage,
            "audit": audit,
        }
    finally:
        backend.close()


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
) -> dict[str, object]:
    summaries = {}
    for controller in controllers:
        rows = [row for row in episodes if row["controller"] == controller]
        summaries[controller] = {
            "episodes": len(rows),
            "successes": sum(bool(row["success"]) for row in rows),
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
    )
    teacher = summaries.get("teacher")
    passed = bool(
        mode == "confirmation"
        and not confirmation_errors
        and teacher is not None
        and teacher["successes"] >= SUCCESS_THRESHOLD
        and teacher["actual_severe_collisions"] == SEVERE_COLLISION_THRESHOLD
    )
    return {
        "schema_version": "hwr.r0019-basket-teacher-evaluation/v1",
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
        "summaries": summaries,
        "episodes": list(episodes),
        "confirmation_evidence": {
            "valid": not confirmation_errors,
            "errors": confirmation_errors,
        },
        "decision": (
            "validated_development"
            if passed
            else "invalid"
            if confirmation_errors
            else "inconclusive_capability"
            if mode == "confirmation"
            else "validated_development"
        ),
        "allowed_claim": (
            "task and control chain have a feasible privileged teacher ceiling"
            if passed
            else "development behavior and failure localization only"
        ),
        "disallowed_claim": (
            "deployable policy capability, visual generalization, or hardware transfer"
        ),
    }


def _confirmation_validation_errors(
    *,
    mode: str,
    controllers: Sequence[str],
    seeds: Sequence[int],
    episodes: Sequence[Mapping[str, object]],
    source_worktree_dirty: bool,
) -> list[str]:
    if mode != "confirmation":
        return []
    errors = []
    expected_seeds = tuple(
        CONFIRMATION_SEED_START + index * CONFIRMATION_SEED_STRIDE
        for index in range(CONFIRMATION_EPISODES)
    )
    if tuple(controllers) != ("baseline", "teacher"):
        errors.append("confirmation_controllers_not_paired")
    if tuple(seeds) != expected_seeds:
        errors.append("confirmation_seed_domain_mismatch")
    if source_worktree_dirty:
        errors.append("source_worktree_dirty")
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
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
