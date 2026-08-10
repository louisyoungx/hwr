from __future__ import annotations

from pathlib import Path

from hwr.apps.train_bimanual_rl import build_parser
from hwr.adapters.mujoco import (
    MujocoBimanualBackendFactory,
    MujocoBimanualTaskBackend,
    load_default_bimanual_training_catalogs,
)
from hwr.train import (
    BimanualRLTrainingConfig,
    BimanualTrainingRunner,
    FrontierOutcome,
)
from hwr.train.bimanual_training import _bilateral_near_statistics


ROOT = Path(__file__).resolve().parents[1]


def test_bimanual_training_cli_exposes_bounded_replay_capacity() -> None:
    arguments = build_parser().parse_args(
        [
            "--run-id",
            "capacity-smoke",
            "--replay-capacity",
            "12000",
            "--global-random-burst",
            "0.02",
            "--global-random-burst-steps",
            "6",
            "--actuator-dwell",
            "0.003",
            "--actuator-dwell-steps",
            "180",
            "--frontier-reset",
            "0.4",
            "--frontier-capacity",
            "12",
            "--frontier-signature-uniform",
            "0.25",
        ]
    )

    assert arguments.replay_capacity == 12_000
    assert arguments.global_random_burst == 0.02
    assert arguments.global_random_burst_steps == 6
    assert arguments.actuator_dwell == 0.003
    assert arguments.actuator_dwell_steps == 180
    assert arguments.frontier_reset == 0.4
    assert arguments.frontier_capacity == 12
    assert arguments.frontier_signature_uniform == 0.25
    assert arguments.episode_steps is None


def test_formal_training_defaults_to_each_tasks_full_physical_horizon() -> None:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    runner = BimanualTrainingRunner(
        tasks,
        MujocoBimanualBackendFactory(bindings),
        BimanualRLTrainingConfig(episodes=1),
    )

    assert runner.config.episode_step_limit is None
    assert {task.max_steps for task in tasks.values()} == {1200, 1600}


def test_task_schedule_rng_is_independent_from_action_exploration() -> None:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config = BimanualRLTrainingConfig(episodes=1, seed=731)
    first = BimanualTrainingRunner(
        tasks, MujocoBimanualBackendFactory(bindings), config
    )
    second = BimanualTrainingRunner(
        tasks, MujocoBimanualBackendFactory(bindings), config
    )

    for _ in range(100):
        first.explorer.sample_random()
    first.frontier_rng.random(100)
    first_schedule = [
        first.task_sampler.sample(first.task_rng)[0] for _ in range(20)
    ]
    second_schedule = [
        second.task_sampler.sample(second.task_rng)[0] for _ in range(20)
    ]

    assert first_schedule == second_schedule
    assert first.task_rng is not first.frontier_rng
    assert first.task_rng is not first.exploration_rng
    assert first.frontier_rng is not first.exploration_rng


def test_bilateral_near_statistics_require_both_sides_in_same_state() -> None:
    def state(left: float, right: float) -> tuple[float, ...]:
        return (*([0.0] * 24), left, right)

    total, longest = _bilateral_near_statistics(
        [
            state(0.05, 0.20),
            state(0.20, 0.05),
            state(0.08, 0.09),
            state(0.07, 0.10),
            state(0.08, 0.11),
            state(0.09, 0.09),
        ]
    )

    assert total == 3
    assert longest == 2


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

    result = BimanualTrainingRunner(
        tasks, MujocoBimanualBackendFactory(bindings), config
    ).train()

    assert {record.task_id for record in result.records} == set(tasks)
    assert result.replay.episode_count == 3
    assert all(size > 0 for size in result.replay.task_sizes().values())
    assert result.replay.hindsight_count == 9
    assert result.replay.failure_size > 0
    assert result.trainer.config.behavior_regularization == 0.0
    assert not hasattr(result, "expert")
    assert not hasattr(result, "demonstrations")
    assert result.task_sampler.audit()["action_outputs"] is False
    assert result.task_sampler.audit()["reach_metric"] == (
        "minimum_over_time_of_worst_side_distance"
    )
    assert result.frontier.audit()["action_outputs"] is False
    assert all(
        record.minimum_worst_side_reach_distance
        >= max(
            record.minimum_left_reach_distance,
            record.minimum_right_reach_distance,
        )
        for record in result.records
    )
    assert all(
        record.bilateral_near_steps >= record.maximum_bilateral_near_steps
        for record in result.records
    )


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

    result = BimanualTrainingRunner(
        tasks, MujocoBimanualBackendFactory(bindings), config
    ).train()

    assert result.trainer.update_count == 2
    assert result.records[0].updates == 2
    assert result.records[0].actor_updates == 0
    assert result.records[0].mean_critic_loss > 0.0
    assert result.records[0].mean_safety_loss > 0.0
    assert result.replay.size == 16
    partition = result.replay.partitions[result.records[0].task_id]
    discounts = partition.regular.all().bootstrap_discounts
    assert discounts is not None
    assert float(discounts.max()) <= result.trainer.config.discount


def test_training_episode_can_start_from_an_autonomous_frontier_snapshot() -> None:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config = BimanualRLTrainingConfig(
        episodes=1,
        episode_step_limit=2,
        replay_capacity=32,
        batch_size=4,
        learning_starts=100,
        initial_random_episodes=0,
        frontier_reset_probability=1.0,
        raw_image_width=16,
        raw_image_height=12,
        image_width=8,
        image_height=6,
        point_count=8,
        language_dim=16,
        hidden_dim=32,
    )
    runner = BimanualTrainingRunner(
        tasks, MujocoBimanualBackendFactory(bindings), config
    )
    task_id = runner.task_ids[0]
    backend = MujocoBimanualTaskBackend(
        tasks[task_id], bindings[task_id], camera_width=16, camera_height=12
    )
    try:
        backend.reset(seed=91, task_id=task_id)
        snapshot = backend.capture_state_snapshot()
    finally:
        backend.close()
    assert runner.frontier.consider(
        task_id,
        snapshot,
        FrontierOutcome(0.05, 0.12, False, False),
        source_episode=17,
        source_step=23,
    )

    result = runner.train()

    assert result.records[0].frontier_reset is True
    assert result.records[0].frontier_source_episode == 17
    assert result.records[0].frontier_source_step == 23
    assert result.frontier.reset_count == 1
