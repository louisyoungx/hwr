"""Train, register, and reload one local formal visual policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from hwr.data import load_visual_dataset
from hwr.train import (
    VisualTrainingConfig,
    load_visual_policy,
    save_visual_training_result,
    train_visual_policy,
)


ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
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
    if int(dataset.manifest.get("metadata", {}).get("sample_stride", 0)) != 1:
        raise ValueError("formal contact training requires full-rate sample_stride=1 data")
    task_id = str(dataset.manifest["task_id"])
    instruction = str(dataset.manifest["instruction"])
    config = VisualTrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        device=arguments.device,
    )
    result = train_visual_policy(dataset, config)
    model_id = task_id.replace("/", "_")
    model_path = save_visual_training_result(
        arguments.output_root / "models/formal-v1",
        model_id,
        arguments.run_id,
        result,
        dataset_manifest=dataset.manifest,
        task_instructions={task_id: (0, instruction)},
        control_hz=float(dataset.manifest["metadata"]["control_hz"]),
    )
    reloaded = load_visual_policy(model_path)
    reloaded.reset(task_id=task_id, seed=arguments.seed)
    reloaded.close()
    manifest = json.loads((model_path / "manifest.json").read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "schema_version": "hwr.formal-visual-training-run/v1",
        "run_id": arguments.run_id,
        "task_id": task_id,
        "training_seeds": dataset.manifest["seeds"],
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_samples": len(dataset),
        "training_device": result.device,
        "epochs": arguments.epochs,
        "first_train_loss": result.history[0]["train_loss"],
        "last_train_loss": result.history[-1]["train_loss"],
        "best_validation_loss": result.best_validation_loss,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_reloaded": True,
        "model_path": str(model_path),
    }
    _write_json_atomic(
        arguments.output_root / "runs/formal-v1" / arguments.run_id / "training.json",
        report,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = train(build_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
