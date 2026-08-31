"""Run the R0020 development-only joint basket teacher experiment."""

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
from hwr.adapters.mujoco.bimanual_teacher import BASKET_TASK_ID, BasketTeacherError
from hwr.adapters.mujoco.joint_basket_teacher import (
    JOINT_TEACHER_SOURCE,
    JointBasketMotionTeacher,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_bimanual_training_catalogs,
)
from hwr.train.bimanual_runtime import dual_arm_action_frame


DEFAULT_SEEDS = (19_001,)
COHORT_SEEDS = (19_001, 19_002, 19_003, 19_004)
DEFAULT_OUTPUT = Path("runs/research-loop/0020/development/seed-19001.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    seeds = tuple(arguments.seed or DEFAULT_SEEDS)
    if not seeds or any(seed < 0 or seed >= 1_000_000 for seed in seeds):
        raise ValueError("R0020 only accepts explicit development seeds")
    output = (
        arguments.output
        if arguments.output.is_absolute()
        else root / arguments.output
    )
    started = time.perf_counter()
    episodes = [_run_episode(root, seed) for seed in seeds]
    report = _report(
        seeds=seeds,
        episodes=episodes,
        source_commit=_source_commit(root),
        source_worktree_dirty=_source_worktree_dirty(root),
        elapsed_seconds=time.perf_counter() - started,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return report


def _run_episode(root: Path, seed: int) -> dict[str, object]:
    tasks, bindings = load_default_bimanual_training_catalogs(root)
    backend = MujocoBimanualTaskBackend(
        tasks[BASKET_TASK_ID],
        bindings[BASKET_TASK_ID],
        camera_width=32,
        camera_height=24,
    )
    teacher = JointBasketMotionTeacher(backend, seed=seed)
    started = time.perf_counter()
    event_counts: Counter[str] = Counter()
    stage_rows: list[dict[str, object]] = []
    safety_interventions = 0
    executed_steps = 0
    controller_failure = None
    try:
        observation = backend.reset(seed=seed, task_id=BASKET_TASK_ID)
        backend.set_camera_rendering(False)
        initial_payload = backend.data.xpos[
            backend.task_ids.payload_body
        ].copy()
        maximum_lift = 0.0
        try:
            for step in range(backend.task.max_steps):
                output = teacher.action(observation)
                outcome = backend.apply(
                    dual_arm_action_frame(
                        observation.timestamp_ns,
                        output.action,
                        source=JOINT_TEACHER_SOURCE,
                    )
                )
                executed_steps += 1
                observation = outcome.observation
                safety_interventions += int(outcome.info["safety_intervened"])
                for event in outcome.events:
                    reason = str(event.details.get("reason", ""))
                    event_counts[f"{event.event_type}:{reason}"] += 1
                audit = backend.task_audit()
                metrics = audit["metrics"]
                payload = backend.data.xpos[backend.task_ids.payload_body]
                maximum_lift = max(
                    maximum_lift,
                    float(payload[2] - initial_payload[2]),
                )
                if not stage_rows or stage_rows[-1]["stage"] != output.stage:
                    stage_rows.append(
                        {"stage": output.stage, "start_step": step}
                    )
                stage_rows[-1].update(
                    {
                        "end_step": step,
                        "payload_position_m": [
                            float(value) for value in payload
                        ],
                        "target_distance_m": float(metrics["target_distance"]),
                        "left_contact": bool(metrics["left_contact"]),
                        "right_contact": bool(metrics["right_contact"]),
                        "maximum_concurrent_steps": int(
                            audit["maximum_concurrent_steps"]
                        ),
                        "stable_steps": int(audit["stable_steps"]),
                        "safety_interventions": safety_interventions,
                    }
                )
                if outcome.terminated or outcome.truncated:
                    break
        except BasketTeacherError as error:
            controller_failure = str(error)
            teacher.failure_stage = "joint_planning_failed"
            for _ in range(executed_steps, backend.task.max_steps):
                outcome = backend.apply(
                    dual_arm_action_frame(
                        observation.timestamp_ns,
                        teacher._hold(observation),
                        source=JOINT_TEACHER_SOURCE,
                    )
                )
                executed_steps += 1
                observation = outcome.observation
                if outcome.terminated or outcome.truncated:
                    break
        result = backend.result()
        audit = backend.task_audit()
        plan = teacher.grasp_plan
        return {
            "seed": seed,
            "steps": executed_steps,
            "elapsed_seconds": time.perf_counter() - started,
            "valid_episode_result": result is not None,
            "physics_episode_completed": result is not None,
            "controller_failure": controller_failure,
            "success": bool(result and result.success),
            "termination_reason": (
                "controller_failure"
                if controller_failure is not None
                else None
                if result is None
                else result.reason
            ),
            "teacher_failure_stage": teacher.failure_stage,
            "stages_reached": [row["stage"] for row in stage_rows],
            "stage_records": stage_rows,
            "maximum_lift_m": maximum_lift,
            "safety_intervention_count": safety_interventions,
            "event_counts": dict(sorted(event_counts.items())),
            "grasp_plan": (
                None
                if plan is None
                else {
                    "base_x": plan.base_x,
                    "maximum_pad_distance_m": plan.maximum_pad_distance,
                    "path_minimum_clearance_m": plan.path_minimum_clearance,
                    "waypoint_count": len(plan.waypoints),
                }
            ),
            "audit": audit,
        }
    finally:
        backend.close()


def _report(
    *,
    seeds: Sequence[int],
    episodes: Sequence[Mapping[str, object]],
    source_commit: str,
    source_worktree_dirty: bool,
    elapsed_seconds: float,
) -> dict[str, object]:
    successes = sum(bool(row["success"]) for row in episodes)
    severe = sum(
        int(row["audit"]["severe_collision_count"]) for row in episodes
    )
    infrastructure_valid = all(
        bool(row["valid_episode_result"]) for row in episodes
    )
    single_seed_passed = (
        tuple(seeds) == DEFAULT_SEEDS
        and successes == 1
        and severe == 0
        and infrastructure_valid
    )
    cohort_complete = tuple(seeds) == COHORT_SEEDS
    l0_gate_passed = bool(
        cohort_complete
        and successes >= 3
        and severe == 0
        and infrastructure_valid
    )
    development_supported = single_seed_passed or l0_gate_passed
    decision = (
        "invalid"
        if not infrastructure_valid
        else "validated_development"
        if development_supported
        else "abandoned"
    )
    return {
        "schema_version": "hwr.r0020-joint-basket-teacher/v1",
        "mode": "development",
        "task_id": BASKET_TASK_ID,
        "source_commit": source_commit,
        "source_worktree_dirty": source_worktree_dirty,
        "seed_domain": {"kind": "development", "seeds": list(seeds)},
        "controller": JOINT_TEACHER_SOURCE,
        "elapsed_seconds": elapsed_seconds,
        "episodes": list(episodes),
        "summary": {
            "episodes": len(episodes),
            "successes": successes,
            "actual_severe_collisions": severe,
            "safety_interventions": sum(
                int(row["safety_intervention_count"]) for row in episodes
            ),
        },
        "single_seed_gate_passed": single_seed_passed,
        "cohort_gate_evaluated": cohort_complete,
        "l0_gate_passed": l0_gate_passed,
        "confirmation_evidence": {"status": "not_run", "valid": None},
        "sealed_final_evidence": {"status": "not_run", "valid": None},
        "decision": decision,
        "allowed_claim": (
            "stable L0 oracle ceiling on the declared development cohort"
            if l0_gate_passed
            else "single development seed is physically solvable by the teacher"
            if single_seed_passed
            else "development behavior and failure localization only"
        ),
        "disallowed_claim": (
            "confirmation, deployable policy capability, visual generalization, "
            "or hardware transfer"
        ),
    }


def _source_commit(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
                "single_seed_gate_passed": report[
                    "single_seed_gate_passed"
                ],
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "validated_development" else 2


if __name__ == "__main__":
    raise SystemExit(main())
