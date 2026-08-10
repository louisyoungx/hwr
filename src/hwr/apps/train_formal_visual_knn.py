"""Fit, hash, save, and reload one local visual kNN policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from hwr.data import load_visual_dataset
from hwr.train import load_visual_knn_policy, save_visual_knn_policy


ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--neighbors", type=int, default=5)
    return parser


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def train(arguments: argparse.Namespace) -> dict[str, object]:
    dataset = load_visual_dataset(arguments.dataset)
    task_id = str(dataset.manifest["task_id"])
    instruction = str(dataset.manifest["instruction"])
    model_path = save_visual_knn_policy(
        arguments.output_root / "models/formal-knn-v1",
        task_id.replace("/", "_"),
        arguments.run_id,
        dataset,
        task_instructions={task_id: (0, instruction)},
        control_hz=float(dataset.manifest["metadata"]["control_hz"]),
        neighbors=arguments.neighbors,
    )
    reloaded = load_visual_knn_policy(model_path)
    reloaded.reset(task_id=task_id, seed=0)
    reloaded.close()
    manifest = json.loads(
        (model_path / "manifest.json").read_text(encoding="utf-8")
    )
    report: dict[str, object] = {
        "schema_version": "hwr.formal-visual-knn-training-run/v1",
        "run_id": arguments.run_id,
        "task_id": task_id,
        "training_seeds": dataset.manifest["seeds"],
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_samples": len(dataset),
        "phase_count": len(dataset.phase_names),
        "neighbors": arguments.neighbors,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_reloaded": True,
        "model_path": str(model_path),
    }
    _write_json_atomic(
        arguments.output_root / "runs/formal-knn-v1" / arguments.run_id / "training.json",
        report,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = train(build_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
