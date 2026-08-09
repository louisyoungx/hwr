"""Replay trained benchmark policies and render synchronized closed-loop video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from hwr.render import VideoConfig, capture_rollout, write_rollout_video
from hwr.render.rollout import RolloutTrace
from hwr.scenarios import household_task_registry
from hwr.sim import HouseholdTaskSpec, RobotSpec
from hwr.train import load_policy


@dataclass(frozen=True)
class ReplayCase:
    task: HouseholdTaskSpec
    model_path: Path
    seed: int
    report_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--reports-root", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/benchmark-rollouts.mp4"),
    )
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--intro-seconds", type=float, default=1.0)
    parser.add_argument("--outro-seconds", type=float, default=2.0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--allow-failure", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_case(report_path: Path, workspace_root: Path, seed_index: int) -> ReplayCase:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    registry = household_task_registry()
    task_id = report.get("task_id")
    if task_id not in registry:
        raise ValueError(f"unknown task_id in {report_path}: {task_id}")
    seed_range = report.get("configuration", {}).get("evaluation_seeds")
    if not isinstance(seed_range, list) or len(seed_range) != 2:
        raise ValueError(f"missing evaluation seed range in {report_path}")
    first_seed, last_seed = (int(value) for value in seed_range)
    seed = first_seed + seed_index
    if seed_index < 0 or seed > last_seed:
        raise ValueError(f"seed index {seed_index} is outside the evaluation range")
    model_value = report.get("artifacts", {}).get("model_path")
    if not isinstance(model_value, str):
        raise ValueError(f"missing model path in {report_path}")
    model_path = _resolve_path(Path(model_value), workspace_root)
    if not model_path.is_dir():
        raise FileNotFoundError(f"trained model is unavailable: {model_path}")
    return ReplayCase(registry[task_id], model_path, seed, report_path)


def _discover_cases(arguments: argparse.Namespace) -> tuple[ReplayCase, ...]:
    workspace_root = arguments.workspace_root.resolve()
    if arguments.report:
        report_paths = tuple(_resolve_path(path, workspace_root) for path in arguments.report)
    else:
        reports_root = _resolve_path(arguments.reports_root, workspace_root)
        report_paths = tuple(sorted(reports_root.glob("*.json")))
    if not report_paths:
        raise FileNotFoundError("no benchmark reports were found")
    cases = tuple(
        _load_case(path, workspace_root, arguments.seed_index) for path in report_paths
    )
    task_order = {task_id: index for index, task_id in enumerate(household_task_registry())}
    return tuple(sorted(cases, key=lambda case: task_order[case.task.task_id]))


def _capture_cases(
    cases: tuple[ReplayCase, ...],
    robot_spec: RobotSpec,
    *,
    device: str,
) -> tuple[RolloutTrace, ...]:
    traces: list[RolloutTrace] = []
    for case in cases:
        policy = load_policy(case.model_path, device=device)
        try:
            traces.append(capture_rollout(case.task, robot_spec, policy, seed=case.seed))
        finally:
            policy.close()
    return tuple(traces)


def run_render(arguments: argparse.Namespace) -> dict[str, object]:
    cases = _discover_cases(arguments)
    robot_spec = RobotSpec()
    traces = _capture_cases(cases, robot_spec, device=arguments.device)
    failed = [trace.task_id for trace in traces if not trace.result.success]
    if failed and not arguments.allow_failure:
        raise RuntimeError(f"replay failed before rendering: {', '.join(failed)}")
    workspace_root = arguments.workspace_root.resolve()
    output_path = _resolve_path(arguments.output_path, workspace_root)
    config = VideoConfig(
        frames_per_second=arguments.fps,
        playback_speed=arguments.playback_speed,
        intro_seconds=arguments.intro_seconds,
        outro_seconds=arguments.outro_seconds,
        panel_width=arguments.panel_width,
        height=arguments.height,
    )
    video = write_rollout_video(
        output_path,
        traces,
        tuple(case.task for case in cases),
        robot_spec,
        config=config,
    )
    result: dict[str, object] = {
        "schema_version": "hwr.replay-video/v1",
        "video": {
            "path": str(video.path),
            "sha256": _sha256(video.path),
            "width": video.width,
            "height": video.height,
            "frames": video.frame_count,
            "frames_per_second": config.frames_per_second,
            "playback_speed": config.playback_speed,
            "duration_seconds": video.duration_seconds,
        },
        "rollouts": [
            {
                "task_id": trace.task_id,
                "policy_id": trace.policy_id,
                "model_path": str(case.model_path),
                "benchmark_report": str(case.report_path),
                "seed": trace.seed,
                "success": trace.result.success,
                "reason": trace.result.reason,
                "steps": trace.result.metrics.get("steps", 0.0),
                "collisions": trace.result.metrics.get("collisions", 0.0),
            }
            for case, trace in zip(cases, traces, strict=True)
        ],
    }
    _write_json_atomic(output_path.with_suffix(".json"), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_render(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
