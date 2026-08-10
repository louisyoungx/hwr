"""Collect formal visual DAgger labels on learned-policy visited states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hwr.adapters.mujoco import (
    MujocoHouseholdBackend,
    PrivilegedHouseholdExpert,
    load_mujoco_task_bindings,
)
from hwr.data import aggregate_visual_policy_dataset, verify_visual_dataset
from hwr.scenarios.formal3d import load_formal_3d_tasks
from hwr.train import load_visual_policy


ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    tasks = load_formal_3d_tasks(ROOT / "configs/tasks/formal_3d_v1.json")
    bindings = load_mujoco_task_bindings(
        ROOT / "configs/adapters/mujoco/formal_3d_v1.json", root=ROOT
    )
    task = tasks[arguments.task_id]
    binding = bindings[arguments.task_id]
    policy = load_visual_policy(arguments.checkpoint, device=arguments.device)
    config = policy.model.config
    try:
        path = aggregate_visual_policy_dataset(
            arguments.output_root,
            arguments.dataset_id,
            arguments.base_dataset,
            task,
            lambda: MujocoHouseholdBackend(
                task,
                binding,
                camera_width=config.image_width,
                camera_height=config.image_height,
            ),
            lambda backend: PrivilegedHouseholdExpert(backend),
            policy,
            range(arguments.seed, arguments.seed + arguments.episodes),
            max_steps=arguments.max_steps,
            sample_stride=arguments.sample_stride,
        )
    finally:
        policy.close()
    print(json.dumps(verify_visual_dataset(path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
