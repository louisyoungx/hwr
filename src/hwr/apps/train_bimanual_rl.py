"""Train all three physical bimanual tasks locally without demonstrations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from hwr.adapters.mujoco import (
    MujocoBimanualBackendFactory,
    load_default_bimanual_training_catalogs,
)
from hwr.train import (
    BimanualRLTrainingConfig,
    BimanualTrainingRunner,
    fork_bimanual_training_run,
    resume_bimanual_training_run,
    save_bimanual_live_progress,
)
from hwr.train.bimanual_registry import save_bimanual_training_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume --run-id and treat --episodes as the new total",
    )
    parser.add_argument(
        "--fork-from",
        type=Path,
        help="start a new audited run from a verified no-demonstration checkpoint",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/bimanual-rl"))
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument(
        "--episode-steps",
        type=int,
        help="optional diagnostic cap; defaults to each task's physical horizon",
    )
    parser.add_argument("--replay-capacity", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--actor-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--final-actor-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--actor-learning-rate-decay-updates", type=int, default=6500)
    parser.add_argument("--initial-random-episodes", type=int, default=9)
    parser.add_argument("--random-action-hold-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-width", type=int, default=64)
    parser.add_argument("--raw-height", type=int, default=48)
    parser.add_argument("--image-width", type=int, default=32)
    parser.add_argument("--image-height", type=int, default=24)
    parser.add_argument("--point-count", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--exploration-noise", type=float, default=0.18)
    parser.add_argument("--exploration-correlation", type=float, default=0.85)
    parser.add_argument("--action-smoothing", type=float, default=0.65)
    parser.add_argument("--gripper-exploration", type=float, default=0.35)
    parser.add_argument("--gripper-hold-steps", type=int, default=16)
    parser.add_argument("--policy-gripper-hold-steps", type=int, default=12)
    parser.add_argument("--reflection-coupling", type=float, default=0.60)
    parser.add_argument("--paired-gripper-exploration", type=float, default=0.60)
    parser.add_argument("--global-random-burst", type=float, default=0.01)
    parser.add_argument("--global-random-burst-steps", type=int, default=8)
    parser.add_argument("--actuator-dwell", type=float, default=0.0)
    parser.add_argument("--actuator-dwell-steps", type=int, default=240)
    parser.add_argument("--actuator-initial-dwell", type=float, default=0.0)
    parser.add_argument("--actuator-dwell-closed", type=float, default=0.50)
    parser.add_argument("--frontier-reset", type=float, default=0.50)
    parser.add_argument("--frontier-capacity", type=int, default=16)
    parser.add_argument("--frontier-signature-uniform", type=float, default=0.20)
    parser.add_argument("--frontier-source-capacity", type=int, default=2)
    parser.add_argument("--frontier-contact-stability", type=int, default=40)
    parser.add_argument("--frontier-reset-validation-steps", type=int, default=40)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    return parser


def _source_commit(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if arguments.resume and arguments.fork_from is not None:
        raise ValueError("--resume and --fork-from are mutually exclusive")
    root = Path(__file__).resolve().parents[3]
    tasks, bindings = load_default_bimanual_training_catalogs(root)
    config = BimanualRLTrainingConfig(
        episodes=arguments.episodes,
        episode_step_limit=arguments.episode_steps,
        replay_capacity=arguments.replay_capacity,
        batch_size=arguments.batch_size,
        learning_starts=arguments.learning_starts,
        actor_learning_rate=arguments.actor_learning_rate,
        final_actor_learning_rate=arguments.final_actor_learning_rate,
        actor_learning_rate_decay_updates=(
            arguments.actor_learning_rate_decay_updates
        ),
        initial_random_episodes=arguments.initial_random_episodes,
        random_action_hold_steps=arguments.random_action_hold_steps,
        seed=arguments.seed,
        device=arguments.device,
        raw_image_width=arguments.raw_width,
        raw_image_height=arguments.raw_height,
        image_width=arguments.image_width,
        image_height=arguments.image_height,
        point_count=arguments.point_count,
        hidden_dim=arguments.hidden_dim,
        exploration_noise=arguments.exploration_noise,
        exploration_correlation=arguments.exploration_correlation,
        action_smoothing=arguments.action_smoothing,
        gripper_exploration_probability=arguments.gripper_exploration,
        gripper_exploration_hold_steps=arguments.gripper_hold_steps,
        policy_gripper_hold_steps=arguments.policy_gripper_hold_steps,
        reflection_coupled_exploration_probability=arguments.reflection_coupling,
        paired_gripper_exploration_probability=(
            arguments.paired_gripper_exploration
        ),
        global_random_burst_probability=arguments.global_random_burst,
        global_random_burst_steps=arguments.global_random_burst_steps,
        actuator_dwell_probability=arguments.actuator_dwell,
        actuator_dwell_steps=arguments.actuator_dwell_steps,
        actuator_initial_dwell_probability=arguments.actuator_initial_dwell,
        actuator_dwell_closed_probability=arguments.actuator_dwell_closed,
        frontier_reset_probability=arguments.frontier_reset,
        frontier_capacity_per_task=arguments.frontier_capacity,
        frontier_signature_uniform_fraction=(
            arguments.frontier_signature_uniform
        ),
        frontier_max_entries_per_source_signature=(
            arguments.frontier_source_capacity
        ),
        frontier_minimum_contact_stability_steps=(
            arguments.frontier_contact_stability
        ),
        frontier_reset_validation_steps=(
            arguments.frontier_reset_validation_steps
        ),
    )
    output_root = (
        arguments.output_root
        if arguments.output_root.is_absolute()
        else root / arguments.output_root
    )
    path = output_root / arguments.run_id
    source_commit = _source_commit(root)
    runner = BimanualTrainingRunner(
        tasks, MujocoBimanualBackendFactory(bindings), config
    )
    parent_training_run = None
    if arguments.resume:
        resume_bimanual_training_run(path, runner)
    elif path.exists():
        raise FileExistsError(f"training run already exists: {path}")
    elif arguments.fork_from is not None:
        parent_path = (
            arguments.fork_from
            if arguments.fork_from.is_absolute()
            else root / arguments.fork_from
        )
        parent_training_run = fork_bimanual_training_run(parent_path, runner)
    created = path.exists()

    def save_progress(result) -> None:
        nonlocal created
        episode_count = len(result.records)
        checkpoint_due = (
            not created
            or episode_count % arguments.checkpoint_interval == 0
            or episode_count == config.episodes
        )
        if checkpoint_due:
            save_bimanual_training_run(
                output_root,
                arguments.run_id,
                result,
                source_commit=source_commit,
                overwrite=created,
                parent_training_run=parent_training_run,
            )
            created = True
        else:
            save_bimanual_live_progress(path, result)

    result = runner.train(on_episode=save_progress)
    if not created:
        save_progress(result)
    return {
        "run_path": str(path),
        "episodes": len(result.records),
        "successes": sum(record.success for record in result.records),
        "updates": result.trainer.update_count,
        "replay_size": result.replay.size,
    }


def main(argv: Sequence[str] | None = None) -> int:
    value = run(build_parser().parse_args(argv))
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
