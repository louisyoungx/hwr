from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hwr.tasks import BIMANUAL_GOAL_DIM
from hwr.train import (
    AutomaticCurriculum,
    CurriculumConfig,
    GoalConditionedReplayBuffer,
    GoalEpisode,
    hindsight_relabel,
    mirror_batch,
)
from hwr.train.asymmetric_rl import AsymmetricRLBatch
from tests.test_asymmetric_rl import _actor_inputs


def _episode(*, success: bool = False, mirrorable: bool = True) -> GoalEpisode:
    count = 4
    achieved = torch.zeros(count, BIMANUAL_GOAL_DIM)
    achieved[:, 0] = torch.arange(count, dtype=torch.float32) * 0.1
    achieved[:, 1] = 0.2
    next_achieved = achieved.clone()
    next_achieved[:, 0] += 0.05
    desired = torch.zeros_like(achieved)
    desired[:, 0] = 1.0
    desired[:, 7:10] = 1.0
    state = torch.zeros(count, 60)
    next_state = torch.zeros(count, 60)
    state[:, :BIMANUAL_GOAL_DIM] = achieved
    state[:, BIMANUAL_GOAL_DIM : 2 * BIMANUAL_GOAL_DIM] = desired
    next_state[:, :BIMANUAL_GOAL_DIM] = next_achieved
    next_state[:, BIMANUAL_GOAL_DIM : 2 * BIMANUAL_GOAL_DIM] = desired
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
    )
    return GoalEpisode(
        batch,
        achieved,
        next_achieved,
        desired,
        success=success,
        mirrorable=mirrorable,
    )


def test_hindsight_relabels_only_critic_goal_and_zero_weights_actor() -> None:
    episode = _episode()
    relabeled = hindsight_relabel(episode, torch.Generator().manual_seed(3))
    desired = relabeled.privileged_state[
        :, BIMANUAL_GOAL_DIM : 2 * BIMANUAL_GOAL_DIM
    ]

    assert not torch.equal(desired, episode.desired_goals)
    assert torch.equal(relabeled.actor_inputs["instruction_embedding"], episode.batch.actor_inputs["instruction_embedding"])
    assert torch.count_nonzero(relabeled.actor_weights) == 0
    assert torch.isfinite(relabeled.rewards).all()


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

    mirrored = mirror_batch(batch)

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

    restored = mirror_batch(mirrored)
    assert torch.equal(restored.action_chunks, batch.action_chunks)
    assert torch.equal(restored.proposed_action_chunks, batch.proposed_action_chunks)


def test_failed_episode_returns_original_her_and_mirrors_to_priority_replay() -> None:
    replay = GoalConditionedReplayBuffer(64, seed=7)

    result = replay.add_episode(_episode(success=False, mirrorable=True))
    sampled = replay.sample(8, failure_fraction=0.5)

    assert result.original_count == 4
    assert result.hindsight_count == 4
    assert result.mirror_count == 8
    assert result.failure_return_count == 16
    assert replay.size == replay.failure_size == 16
    assert replay.discovery_size == 8
    assert sampled.rewards.shape == (8,)
    assert sampled.actor_weights is not None
    assert set(sampled.actor_weights.tolist()).issubset({0.0, 1.0})


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
