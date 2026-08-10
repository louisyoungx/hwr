"""Evaluate one reloaded visual checkpoint on isolated formal 3D seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from hwr.adapters.mujoco import MujocoHouseholdBackend, load_mujoco_task_bindings
from hwr.eval import evaluate_formal_visual_policy
from hwr.scenarios.formal3d import load_formal_3d_tasks
from hwr.train import load_visual_knn_policy, load_visual_policy


ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    tasks = load_formal_3d_tasks(ROOT / "configs/tasks/formal_3d_v1.json")
    bindings = load_mujoco_task_bindings(
        ROOT / "configs/adapters/mujoco/formal_3d_v1.json", root=ROOT
    )
    task = tasks[arguments.task_id]
    binding = bindings[arguments.task_id]
    manifest = json.loads(
        (arguments.checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["schema_version"] == "hwr.visual-knn-policy/v1":
        policy = load_visual_knn_policy(arguments.checkpoint)
        config = policy.config
    else:
        policy = load_visual_policy(arguments.checkpoint, device=arguments.device)
        config = policy.model.config
    seeds = tuple(range(arguments.seed, arguments.seed + arguments.episodes))
    dataset_seeds = set(manifest["dataset"]["seeds"])
    overlap = sorted(dataset_seeds.intersection(seeds))
    if overlap:
        policy.close()
        raise ValueError(f"evaluation seeds overlap training data: {overlap}")
    report = evaluate_formal_visual_policy(
        task.task_id,
        task.max_steps,
        lambda: MujocoHouseholdBackend(
            task,
            binding,
            camera_width=config.image_width,
            camera_height=config.image_height,
        ),
        policy,
        seeds,
    ).to_dict()
    report["training_seeds"] = sorted(dataset_seeds)
    report["evaluation_seeds"] = list(seeds)
    report["checkpoint_sha256"] = manifest["checkpoint_sha256"]
    _write_json_atomic(arguments.report, report)
    policy.close()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = evaluate(build_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
