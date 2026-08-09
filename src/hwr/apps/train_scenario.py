"""Generate data, train a policy, and evaluate one household scenario."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

from hwr.data import BehaviorDataset, aggregate_policy_dataset, generate_expert_dataset
from hwr.eval import evaluate_policy
from hwr.scenarios import PickPlaceExpert, household_task_registry
from hwr.sim import Household2DEnv, RobotSpec
from hwr.policy import NeuralPolicy
from hwr.train import TrainingConfig, load_policy, save_training_result, train_behavior_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", choices=sorted(household_task_registry()))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(128, 128))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--aggregation-rounds", type=int, default=0)
    parser.add_argument("--aggregation-episodes", type=int, default=20)
    parser.add_argument("--expert-action-probability", type=float, default=0.2)
    return parser


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_training(arguments: argparse.Namespace) -> dict[str, object]:
    if min(arguments.episodes, arguments.eval_episodes, arguments.epochs) <= 0:
        raise ValueError("episode and epoch counts must be positive")
    started = time.monotonic()
    robot = RobotSpec()
    task = household_task_registry()[arguments.task_id]
    output_root = arguments.output_root
    datasets_root = output_root / "datasets"
    models_root = output_root / "models"
    runs_root = output_root / "runs"
    datasets_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    dataset_id = f"{arguments.run_id}-expert"
    training_seeds = range(arguments.seed, arguments.seed + arguments.episodes)
    dataset_path = generate_expert_dataset(
        datasets_root,
        dataset_id,
        task,
        lambda: Household2DEnv(robot, task),
        PickPlaceExpert(robot),
        training_seeds,
    )
    dataset = BehaviorDataset.load(dataset_path)
    training_config = TrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        hidden_dims=tuple(arguments.hidden_dims),
        device=arguments.device,
    )
    training_result = train_behavior_policy(dataset, training_config)
    aggregation_history: list[dict[str, object]] = []
    for round_index in range(arguments.aggregation_rounds):
        aggregation_policy = NeuralPolicy(
            training_result.model,
            training_result.normalization,
            policy_version=f"{arguments.run_id}:aggregation-{round_index + 1}",
            control_hz=robot.control_hz,
            device="cpu",
        )
        aggregation_start = arguments.seed + 20_000 + round_index * arguments.aggregation_episodes
        aggregated_id = f"{arguments.run_id}-aggregation-{round_index + 1}"
        aggregated_path = aggregate_policy_dataset(
            datasets_root,
            aggregated_id,
            dataset,
            task,
            lambda: Household2DEnv(robot, task),
            PickPlaceExpert(robot),
            aggregation_policy,
            range(aggregation_start, aggregation_start + arguments.aggregation_episodes),
            expert_action_probability=arguments.expert_action_probability,
        )
        dataset_path = aggregated_path
        dataset = BehaviorDataset.load(aggregated_path)
        training_result = train_behavior_policy(dataset, training_config)
        aggregation_history.append(
            {
                "round": round_index + 1,
                "dataset_id": dataset.manifest["dataset_id"],
                "episodes": dataset.manifest["episode_count"],
                "samples": dataset.manifest["sample_count"],
                "best_validation_loss": training_result.best_validation_loss,
            }
        )
    model_path = save_training_result(
        models_root,
        arguments.task_id.replace("/", "_"),
        arguments.run_id,
        training_result,
        dataset_manifest=dataset.manifest,
        control_hz=robot.control_hz,
    )
    policy = load_policy(model_path)
    evaluation_seeds = range(
        arguments.seed + 10_000,
        arguments.seed + 10_000 + arguments.eval_episodes,
    )
    evaluation = evaluate_policy(
        task,
        lambda: Household2DEnv(robot, task),
        policy,
        evaluation_seeds,
    )
    report: dict[str, object] = {
        "schema_version": "hwr.training-run/v1",
        "run_id": arguments.run_id,
        "task_id": task.task_id,
        "scene_id": task.scene.scene_id,
        "configuration": {
            "episodes": arguments.episodes,
            "eval_episodes": arguments.eval_episodes,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "hidden_dims": list(arguments.hidden_dims),
            "device": arguments.device,
            "seed": arguments.seed,
            "training_seeds": [arguments.seed, arguments.seed + arguments.episodes - 1],
            "evaluation_seeds": [
                arguments.seed + 10_000,
                arguments.seed + 10_000 + arguments.eval_episodes - 1,
            ],
            "aggregation_rounds": arguments.aggregation_rounds,
            "aggregation_episodes": arguments.aggregation_episodes,
            "expert_action_probability": arguments.expert_action_probability,
        },
        "dataset": {
            "dataset_id": dataset.manifest["dataset_id"],
            "episodes": dataset.manifest["episode_count"],
            "samples": dataset.manifest["sample_count"],
            "checksum": dataset.manifest["checksum"],
        },
        "training": {
            "device": training_result.device,
            "epochs": arguments.epochs,
            "best_validation_loss": training_result.best_validation_loss,
            "first_train_loss": training_result.history[0]["train_loss"],
            "last_train_loss": training_result.history[-1]["train_loss"],
            "aggregation": aggregation_history,
        },
        "evaluation": evaluation.to_dict(),
        "artifacts": {
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    run_report = runs_root / arguments.run_id / "report.json"
    _write_json_atomic(run_report, report)
    if arguments.report_path is not None:
        _write_json_atomic(arguments.report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_training(arguments)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
