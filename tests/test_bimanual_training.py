from __future__ import annotations

from pathlib import Path

from hwr.apps.train_bimanual_rl import build_parser
from hwr.train import (
    BimanualRLTrainingConfig,
    BimanualTrainingRunner,
    load_default_bimanual_training_catalogs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bimanual_training_cli_exposes_bounded_replay_capacity() -> None:
    arguments = build_parser().parse_args(
        ["--run-id", "capacity-smoke", "--replay-capacity", "12000"]
    )

    assert arguments.replay_capacity == 12_000


def test_local_training_collects_all_three_tasks_without_action_labels() -> None:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config = BimanualRLTrainingConfig(
        episodes=3,
        episode_step_limit=3,
        replay_capacity=128,
        batch_size=4,
        learning_starts=100,
        initial_random_episodes=3,
        raw_image_width=16,
        raw_image_height=12,
        image_width=8,
        image_height=6,
        point_count=8,
        language_dim=16,
        hidden_dim=32,
    )

    result = BimanualTrainingRunner(tasks, bindings, config).train()

    assert {record.task_id for record in result.records} == set(tasks)
    assert result.replay.episode_count == 3
    assert all(size > 0 for size in result.replay.task_sizes().values())
    assert result.replay.hindsight_count == 9
    assert result.replay.failure_size > 0
    assert result.trainer.config.behavior_regularization == 0.0
    assert not hasattr(result, "expert")
    assert not hasattr(result, "demonstrations")
    assert result.task_sampler.audit()["action_outputs"] is False


def test_local_training_executes_actor_critic_update_from_random_experience() -> None:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config = BimanualRLTrainingConfig(
        episodes=1,
        episode_step_limit=4,
        replay_capacity=64,
        batch_size=4,
        learning_starts=4,
        updates_per_environment_step=0.5,
        initial_random_episodes=1,
        raw_image_width=16,
        raw_image_height=12,
        image_width=8,
        image_height=6,
        point_count=8,
        language_dim=16,
        hidden_dim=32,
    )

    result = BimanualTrainingRunner(tasks, bindings, config).train()

    assert result.trainer.update_count == 2
    assert result.records[0].updates == 2
    assert result.records[0].actor_updates == 0
    assert result.records[0].mean_critic_loss > 0.0
    assert result.records[0].mean_safety_loss > 0.0
    assert result.replay.size == 16
