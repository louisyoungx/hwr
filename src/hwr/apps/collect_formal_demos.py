"""Collect contact-valid formal 3D demonstrations into visual-only datasets."""

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
from hwr.data import generate_visual_expert_dataset, verify_visual_dataset
from hwr.scenarios.formal3d import load_formal_3d_tasks


ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--image-width", type=int, default=48)
    parser.add_argument("--image-height", type=int, default=36)
    parser.add_argument("--action-history", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=1)
    return parser


def collect(arguments: argparse.Namespace) -> dict[str, object]:
    tasks = load_formal_3d_tasks(ROOT / "configs/tasks/formal_3d_v1.json")
    bindings = load_mujoco_task_bindings(
        ROOT / "configs/adapters/mujoco/formal_3d_v1.json", root=ROOT
    )
    task_ids = sorted(tasks) if not arguments.task_id else arguments.task_id
    unknown = sorted(set(task_ids) - set(tasks))
    if unknown:
        raise ValueError(f"unknown formal tasks: {unknown}")
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, object] = {}
    for task_index, task_id in enumerate(task_ids):
        task = tasks[task_id]
        binding = bindings[task_id]
        dataset_id = f"{task_id.replace('/', '_')}-expert-s{arguments.seed}"
        start = arguments.seed + task_index * 10_000
        seeds = range(start, start + arguments.episodes)
        path = generate_visual_expert_dataset(
            arguments.output_root,
            dataset_id,
            task,
            lambda task=task, binding=binding: MujocoHouseholdBackend(
                task,
                binding,
                camera_width=arguments.image_width,
                camera_height=arguments.image_height,
            ),
            lambda backend: PrivilegedHouseholdExpert(backend),
            seeds,
            image_size=(arguments.image_width, arguments.image_height),
            action_history=arguments.action_history,
            sample_stride=arguments.sample_stride,
        )
        manifests[task_id] = verify_visual_dataset(path)
    return {
        "schema_version": "hwr.formal-demo-collection/v1",
        "datasets": manifests,
    }


def main(argv: Sequence[str] | None = None) -> int:
    report = collect(build_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
