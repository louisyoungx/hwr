from __future__ import annotations

from pathlib import Path

import torch

from hwr.apps.train_bimanual_rl import build_parser
from hwr.adapters.mujoco import (
    MujocoBimanualBackendFactory,
    MujocoBimanualTaskBackend,
    load_default_bimanual_training_catalogs,
)
from hwr.train import (
    BimanualRLTrainingConfig,
    BimanualTrainingRunner,
    LearningFrontierCandidate,
    LearningSignal,
)
from hwr.train.bimanual_metrics import (
    bilateral_near_statistics,
    physical_progress_statistics,
    transition_safety_cost,
)


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
            "--actuator-initial-dwell",
            "0.8",
            "--actuator-dwell-closed",
            "0.75",
            "--frontier-reset",
            "0.4",
            "--frontier-capacity",
            "12",
            "--frontier-signature-uniform",
            "0.25",
            "--frontier-source-capacity",
            "3",
            "--visual-contrastive-weight",
            "0.08",
            "--augmentation-consistency-weight",
            "0.2",
        ]
    )

    assert arguments.replay_capacity == 12_000
    assert arguments.global_random_burst == 0.02
    assert arguments.global_random_burst_steps == 6
    assert arguments.actuator_dwell == 0.003
    assert arguments.actuator_dwell_steps == 180
    assert arguments.actuator_initial_dwell == 0.8
    assert arguments.actuator_dwell_closed == 0.75
    assert arguments.frontier_reset == 0.4
    assert arguments.frontier_capacity == 12
    assert arguments.frontier_signature_uniform == 0.25
    assert arguments.frontier_source_capacity == 3
    assert arguments.visual_contrastive_weight == 0.08
    assert arguments.augmentation_consistency_weight == 0.2
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

    total, longest = bilateral_near_statistics(
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


def test_physical_progress_statistics_report_real_task_motion() -> None:
    def state(
        payload_x: float,
        articulation: float,
        controlled_target: float,
        controlled_articulation: float,
    ) -> tuple[float, ...]:
        values = [0.0] * 62
        values[0:3] = [payload_x, 0.0, 0.0]
        values[6] = articulation
        values[12:15] = [1.0, 0.0, 0.0]
        values[60] = controlled_target
        values[61] = controlled_articulation
        return tuple(values)

    summary = physical_progress_statistics(
        [state(0.1, 0.02, 0.0, 0.0), state(0.7, 0.31, 0.24, 0.20)]
    )

    assert abs(summary.minimum_target_distance - 0.3) < 1e-9
    assert summary.maximum_articulation_position == 0.31
    assert summary.maximum_controlled_target_progress == 0.24
    assert summary.maximum_controlled_articulation_progress == 0.20


def test_transition_safety_cost_includes_observed_severe_collision() -> None:
    safe = {"severe_collisions": 0.0}
    severe = {"severe_collisions": 1.0}

    assert not transition_safety_cost({"safety_intervened": False}, safe)
    assert transition_safety_cost({"safety_intervened": True}, safe)
    assert transition_safety_cost({"safety_intervened": False}, severe)


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
    assert result.replay.size == 9
    assert result.replay.legacy_discarded_hindsight_count == 0
    assert result.replay.failure_size > 0
    assert result.trainer.config.behavior_regularization == 0.0
    assert not hasattr(result, "expert")
    assert not hasattr(result, "demonstrations")
    assert result.task_sampler.audit()["action_outputs"] is False
    assert result.task_sampler.audit()["distance_thresholds"] is False
    assert result.task_sampler.audit()["task_semantic_fields"] == []
    assert result.frontier.audit()["action_outputs"] is False
    for task_id, partition in result.replay.partitions.items():
        indices = partition.regular.all().augmentation_transform_indices
        expected = 1 if tasks[task_id].legal_transforms else 0
        assert torch.all(indices == expected)
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
    assert result.replay.size == 4
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
        state = backend.privileged_training_state().critic_state
    finally:
        backend.close()
    assert runner.frontier.consider_episode(
        task_id,
        (
            LearningFrontierCandidate(
                snapshot,
                state,
                LearningSignal(1.0, 1.0, 0.0, 1.0),
                source_episode=17,
                source_step=23,
            ),
        ),
    ) == 1

    result = runner.train()

    assert result.records[0].frontier_reset is True
    assert result.records[0].frontier_source_episode == 17
    assert result.records[0].frontier_source_step == 23
    assert result.records[0].environment_reset_seed == (
        config.seed + 17 * 104729
    )
    assert result.frontier.reset_count == 1


def test_frontier_reset_validates_snapshot_without_generating_probe_actions() -> None:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config = BimanualRLTrainingConfig(
        episodes=1,
        episode_step_limit=2,
        replay_capacity=32,
        batch_size=4,
        learning_starts=100,
        initial_random_episodes=0,
        paired_gripper_exploration_probability=1.0,
        actuator_initial_dwell_probability=1.0,
        actuator_dwell_closed_probability=1.0,
        actuator_dwell_steps=1,
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
        state = backend.privileged_training_state().critic_state
    finally:
        backend.close()
    assert runner.frontier.consider_episode(
        task_id,
        (
            LearningFrontierCandidate(
                snapshot,
                state,
                LearningSignal(1.0, 1.0, 0.0, 1.0),
                source_episode=17,
                source_step=23,
            ),
        ),
    ) == 1

    result = runner.train()

    assert result.records[0].frontier_source_signature >= 0
    assert result.records[0].frontier_reset_validated is True
    assert result.records[0].frontier_reset_reproduced is True
    assert result.records[0].frontier_reset_applied is True
    assert result.frontier.audit()["action_outputs"] is False
    assert result.records[0].steps == 2
    assert result.environment_steps == 2
