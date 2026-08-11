from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hwr.train import (
    AutonomousEpisode,
    AutonomousReplayBuffer,
    AutomaticCurriculum,
    CurriculumConfig,
    transform_batch,
)
from hwr.train.asymmetric_replay import AsymmetricReplayBuffer
from hwr.train.asymmetric_rl import AsymmetricRLBatch
from hwr.train.environment_augmentation import LATERAL_REFLECTION
from hwr.train.autonomous_replay import _sample_time_augmentation
from tests.test_asymmetric_rl import _actor_inputs


def _episode(
    *, success: bool = False, legal_transforms: tuple[str, ...] = (LATERAL_REFLECTION,)
) -> AutonomousEpisode:
    count = 4
    state = torch.zeros(count, 62)
    next_state = torch.zeros(count, 62)
    state[:, 0] = torch.arange(count, dtype=torch.float32) * 0.1
    next_state[:, 0] = state[:, 0] + 0.05
    inputs = _actor_inputs(count)
    batch = AsymmetricRLBatch(
        actor_inputs=inputs,
        next_actor_inputs={name: value.clone() for name, value in inputs.items()},
        privileged_state=state,
        next_privileged_state=next_state,
        action_chunks=torch.zeros(count, 3, 16),
        stop_decisions=torch.zeros(count, 3),
        rewards=torch.full((count,), -0.1),
        done=torch.zeros(count),
        augmentation_transform_indices=torch.zeros(count, dtype=torch.int64),
    )
    return AutonomousEpisode(
        batch,
        success=success,
        legal_transforms=legal_transforms,
    )


def test_mirror_swaps_arms_wrists_actions_and_continuous_goal_sides() -> None:
    batch = _episode().batch
    batch.actor_inputs["left_wrist_rgb"].fill_(1.0)
    batch.actor_inputs["right_wrist_rgb"].fill_(2.0)
    batch.action_chunks[:, :, 2:8] = torch.arange(1.0, 7.0)
    batch.action_chunks[:, :, 8:14] = torch.arange(7.0, 13.0)
    batch = replace(batch, proposed_action_chunks=batch.action_chunks + 1.0)
    batch.action_chunks[:, :, 14] = 0.25
    batch.action_chunks[:, :, 15] = 0.75
    batch.privileged_state[:, 1] = 0.2
    batch.privileged_state[:, 24] = 1.0
    batch.privileged_state[:, 25] = 2.0

    mirrored = transform_batch(batch, LATERAL_REFLECTION)

    assert torch.all(mirrored.actor_inputs["left_wrist_rgb"] == 2.0)
    assert torch.all(mirrored.actor_inputs["right_wrist_rgb"] == 1.0)
    assert torch.all(
        mirrored.action_chunks[:, :, 2:8]
        == torch.tensor((7, -8, 9, -10, 11, -12))
    )
    assert torch.all(
        mirrored.action_chunks[:, :, 8:14]
        == torch.tensor((1, -2, 3, -4, 5, -6))
    )
    assert torch.all(
        mirrored.proposed_action_chunks[:, :, 2:8]
        == torch.tensor((8, -9, 10, -11, 12, -13))
    )
    assert torch.all(mirrored.action_chunks[:, :, 14] == 0.75)
    assert torch.all(mirrored.action_chunks[:, :, 15] == 0.25)
    assert torch.all(mirrored.privileged_state[:, 1] == -0.2)
    assert torch.all(mirrored.privileged_state[:, 24] == 2.0)
    assert torch.all(mirrored.privileged_state[:, 25] == 1.0)

    restored = transform_batch(mirrored, LATERAL_REFLECTION)
    assert torch.equal(restored.action_chunks, batch.action_chunks)
    assert torch.equal(restored.proposed_action_chunks, batch.proposed_action_chunks)
    assert torch.equal(
        restored.augmentation_transform_indices,
        batch.augmentation_transform_indices,
    )


def test_failed_episode_stores_only_autonomous_transitions() -> None:
    replay = AutonomousReplayBuffer(64, seed=7)

    result = replay.add_episode(_episode(success=False))
    sampled = replay.sample(8, failure_fraction=0.5)

    assert result.original_count == 4
    assert result.augmentation_count == 4
    assert result.failure_return_count == 4
    assert replay.size == replay.failure_size == 4
    assert replay.discovery_size == 2
    assert sampled.rewards.shape == (8,)
    assert sampled.actor_weights is not None
    assert torch.all(sampled.augmentation_transform_indices == 1)
    assert torch.all(sampled.actor_weights == 1.0)


def test_legal_transform_is_generated_at_sample_time_without_scene_logic() -> None:
    episode = _episode()
    actions = episode.batch.action_chunks.clone()
    actions[:, :, 2:8] = torch.arange(1.0, 7.0)
    actions[:, :, 8:14] = torch.arange(7.0, 13.0)
    eligible = replace(
        episode.batch,
        action_chunks=actions,
        augmentation_transform_indices=torch.ones(4, dtype=torch.int64),
    )

    augmented = _sample_time_augmentation(
        eligible, torch.Generator().manual_seed(0)
    )

    changed = torch.any(augmented.action_chunks != eligible.action_chunks, dim=(1, 2))
    assert 0 < int(changed.sum()) < 4
    assert torch.all(augmented.augmentation_transform_indices == 1)


def test_expanded_legacy_replay_is_split_by_actor_eligibility() -> None:
    replay = AutonomousReplayBuffer(64, seed=23)
    replay.add_episode(_episode())
    state = replay.state_dict()
    expanded = AsymmetricReplayBuffer(64, seed=24)
    autonomous = replay.regular.all()
    critic_only = replace(
        autonomous,
        actor_weights=torch.zeros_like(autonomous.actor_weights),
    )
    expanded.add(autonomous)
    expanded.add(critic_only)
    state.pop("autonomous_replay_storage_schema")
    state["regular"] = expanded.state_dict()
    state["failures"] = expanded.state_dict()
    state["hindsight_count"] = 4
    restored = AutonomousReplayBuffer(64, seed=25)

    restored.load_state_dict(state)

    assert restored.size == 4
    assert torch.all(restored.regular.all().actor_weights > 0)
    assert torch.all(restored.failures.all().actor_weights > 0)
    assert restored.legacy_discarded_hindsight_count == 4


def test_runtime_interventions_receive_a_dedicated_safety_replay_quota() -> None:
    episode = _episode(success=False, legal_transforms=())
    batch = replace(
        episode.batch,
        proposed_action_chunks=episode.batch.action_chunks.clone(),
        safety_costs=torch.ones(4),
    )
    replay = AutonomousReplayBuffer(64, seed=11)

    replay.add_episode(replace(episode, batch=batch))
    sampled = replay.sample(
        8,
        failure_fraction=0.0,
        discovery_fraction=0.0,
        safety_fraction=0.5,
    )

    assert replay.safety_size == 4
    assert sampled.safety_costs is not None
    assert int((sampled.safety_costs > 0.5).sum()) >= 4
    assert torch.all(sampled.actor_weights[-4:] == 1.0)


def test_geometry_fields_do_not_control_state_novelty_replay() -> None:
    episode = _episode(success=False, legal_transforms=())
    near_state = episode.batch.next_privileged_state.clone()
    near_state[:, 24] = 0.04
    near_state[:, 25] = 0.20
    batch = replace(episode.batch, next_privileged_state=near_state)
    replay = AutonomousReplayBuffer(64, seed=13)

    replay.add_episode(replace(episode, batch=batch))
    sampled = replay.sample(
        4,
        failure_fraction=0.0,
        discovery_fraction=0.75,
        safety_fraction=0.0,
    )

    assert replay.discovery_size == 2
    assert torch.all(sampled.actor_weights[-2:] == 1.0)

    outside_state = near_state.clone()
    outside_state[:, 24] = 0.061
    outside = AutonomousReplayBuffer(64, seed=14)
    outside.add_episode(
        replace(
            episode,
            batch=replace(episode.batch, next_privileged_state=outside_state),
        )
    )
    assert outside.discovery_size == replay.discovery_size


def test_joint_reach_values_do_not_create_a_training_branch() -> None:
    episode = _episode(success=False, legal_transforms=())
    jointly_near = episode.batch.next_privileged_state.clone()
    jointly_near[:, 24] = 0.08
    jointly_near[:, 25] = 0.09
    replay = AutonomousReplayBuffer(64, seed=15)

    replay.add_episode(
        replace(
            episode,
            batch=replace(episode.batch, next_privileged_state=jointly_near),
        )
    )

    assert replay.discovery_size == 2
    outside = jointly_near.clone()
    outside[:, 25] = 0.101
    rejected = AutonomousReplayBuffer(64, seed=16)
    rejected.add_episode(
        replace(
            episode,
            batch=replace(episode.batch, next_privileged_state=outside),
        )
    )
    assert rejected.discovery_size == replay.discovery_size


def test_positive_local_reward_improvements_receive_a_ranked_replay_quota() -> None:
    episode = _episode(success=False, legal_transforms=())
    state = episode.batch.privileged_state.clone()
    next_state = episode.batch.next_privileged_state.clone()
    next_state[:, 60] = torch.tensor((0.0, 0.01, 0.02, 0.03))
    actions = episode.batch.action_chunks.clone()
    actions[:, :, 2] = 0.01
    replay = AutonomousReplayBuffer(64, seed=17)

    replay.add_episode(
        replace(
            episode,
            batch=replace(
                episode.batch,
                privileged_state=state,
                next_privileged_state=next_state,
                action_chunks=actions,
                rewards=torch.tensor((4.0, 4.0, 4.5, 4.0)),
            ),
            reward_improvements=torch.tensor((0.0, 0.0, 0.5, -0.05)),
        )
    )
    sampled = replay.sample(
        4,
        failure_fraction=0.0,
        discovery_fraction=0.0,
        progress_fraction=0.75,
        safety_fraction=0.0,
    )

    assert replay.progress_size == 1
    assert int((sampled.rewards == 4.5).sum()) >= 1
    assert torch.all(sampled.actor_weights[-1:] == 1.0)

    legacy = replay.state_dict()
    old_priority_size = legacy["progress_events"]["size"]
    legacy.pop("progress_event_schema")
    restored = AutonomousReplayBuffer(64, seed=18)
    restored.load_state_dict(legacy)
    audit = restored.priority_migration_audit()
    assert audit is not None
    assert audit["discarded_legacy_priority_rows"] == old_priority_size
    assert restored.legacy_discarded_reward_priority_count == old_priority_size
    assert restored.progress_size > 0

    passive = AutonomousReplayBuffer(64, seed=20)
    passive.add_episode(
        replace(
            episode,
            batch=replace(
                episode.batch,
                privileged_state=state,
                next_privileged_state=next_state,
            ),
        )
    )
    assert passive.progress_size == 0

    far_next = next_state.clone()
    far_next[:, 0] = 4.0
    far = AutonomousReplayBuffer(64, seed=19)
    far.add_episode(
        replace(
            episode,
            batch=replace(
                episode.batch,
                privileged_state=state,
                next_privileged_state=far_next,
                action_chunks=actions,
            ),
        )
    )
    assert far.progress_size == 0


def test_reward_priority_ignores_a_high_stationary_reward_plateau() -> None:
    episode = _episode(success=False, legal_transforms=())
    replay = AutonomousReplayBuffer(64, seed=21)

    replay.add_episode(
        replace(
            episode,
            batch=replace(
                episode.batch,
                rewards=torch.tensor((8.0, 8.0, 8.0, 8.0)),
            ),
            reward_improvements=torch.zeros(4),
        )
    )

    assert replay.progress_size == 0


def test_automatic_curriculum_expands_only_after_safe_success_window() -> None:
    curriculum = AutomaticCurriculum(
        ("basket",),
        CurriculumConfig(window=4, initial_level=0.1, step=0.2),
    )

    for _ in range(4):
        update = curriculum.record("basket", success=True, severe_collision=False)
    assert update.level == pytest.approx(0.3)
    assert update.changed

    for _ in range(4):
        update = curriculum.record("basket", success=True, severe_collision=True)
    assert update.level == pytest.approx(0.1)
    assert update.severe_collision_rate == 1.0


def test_automatic_curriculum_discards_only_changed_task_window() -> None:
    curriculum = AutomaticCurriculum(
        ("basket", "tray"),
        CurriculumConfig(window=4, initial_level=0.1, step=0.2),
    )
    curriculum.record("basket", success=True, severe_collision=False)
    curriculum.record("tray", success=True, severe_collision=False)

    discarded = curriculum.discard_tasks(("tray",))

    assert discarded["tray"] == {
        "level": 0.1,
        "success_count": 1,
        "severe_count": 1,
    }
    state = curriculum.state_dict()
    assert state["success"]["basket"] == [True]
    assert state["success"]["tray"] == []


def test_curriculum_level_contracts_and_expands_backend_randomization() -> None:
    from pathlib import Path

    from hwr.adapters.mujoco import (
        MujocoBimanualTaskBackend,
        load_bimanual_mujoco_bindings,
    )
    from hwr.tasks import load_bimanual_task_specs

    root = Path(__file__).resolve().parents[1]
    tasks = load_bimanual_task_specs(root / "configs/tasks/bimanual_household_v1.json")
    bindings = load_bimanual_mujoco_bindings(
        root / "configs/adapters/mujoco/bimanual_household_v1.json", root=root
    )
    task_id = "carry_living_room_basket/v1"
    backend = MujocoBimanualTaskBackend(
        tasks[task_id], bindings[task_id], camera_width=8, camera_height=8
    )
    try:
        backend.set_curriculum_level(0.0)
        backend.reset(seed=1, task_id=task_id)
        easy = backend.task_audit()["randomization"]
        backend.set_curriculum_level(1.0)
        backend.reset(seed=1, task_id=task_id)
        full = backend.task_audit()["randomization"]
    finally:
        backend.close()

    assert set(easy.values()) == {1.0}
    assert any(value != 1.0 for value in full.values())
