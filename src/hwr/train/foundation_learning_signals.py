"""Episode-local task-blind curriculum signals from learned posterior states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from hwr.data.foundation_loading import FoundationSequenceBatchLoader
from hwr.train.foundation_trainer import FoundationWorldModelTrainer
from hwr.world_model.distributions import reward_expectation


@dataclass(frozen=True)
class EpisodeLearningSignals:
    state_novelty: float
    td_error: float
    window_count: int


def evaluate_episode_learning_signals(
    trainer: FoundationWorldModelTrainer,
    loader: FoundationSequenceBatchLoader,
    episode_ids: Sequence[str],
    *,
    maximum_windows: int,
) -> dict[str, EpisodeLearningSignals]:
    """Evaluate each Episode independently without task or geometry features."""
    identities = tuple(dict.fromkeys(episode_ids))
    if not identities or any(not value for value in identities) or maximum_windows <= 0:
        raise ValueError("Episode learning signal request is invalid")
    selected = _episode_window_indices(loader, identities, maximum_windows)
    modules: tuple[nn.Module, ...] = (
        trainer.visual_student,
        trainer.world_model,
        trainer.value,
        trainer.imagination.slow_value,
    )
    modes = tuple(module.training for module in modules)
    for module in modules:
        module.eval()
    try:
        with torch.inference_mode():
            return {
                episode_id: _evaluate_indices(trainer, loader, indices)
                for episode_id, indices in selected.items()
            }
    finally:
        for module, training in zip(modules, modes, strict=True):
            module.train(training)


def posterior_state_change_novelty(features: torch.Tensor) -> torch.Tensor:
    """Cosine state change, independent of state coordinates and scale."""
    if features.ndim != 3 or features.shape[1] < 2:
        raise ValueError("posterior novelty requires batch-time-feature states")
    current = nn.functional.normalize(features[:, :-1], dim=-1)
    following = nn.functional.normalize(features[:, 1:], dim=-1)
    return (1.0 - (current * following).sum(dim=-1)).clamp_min(0.0)


def observed_one_step_td_error(
    features: torch.Tensor,
    rewards: torch.Tensor,
    continues: torch.Tensor,
    value: nn.Module,
    slow_value: nn.Module,
    *,
    discount: float,
    symlog_limit: float,
) -> torch.Tensor:
    """Bellman error on observed transitions under current learned value heads."""
    if (
        features.ndim != 3
        or rewards.shape != features.shape[:2][:-1] + (features.shape[1] - 1,)
        or continues.shape != rewards.shape
        or not 0.0 < discount <= 1.0
    ):
        raise ValueError("observed TD-error tensors are invalid")
    current = reward_expectation(
        value(features[:, :-1]), limit=symlog_limit
    )
    following = reward_expectation(
        slow_value(features[:, 1:]), limit=symlog_limit
    )
    target = rewards + discount * continues * following
    return (current - target).abs()


def _episode_window_indices(
    loader: FoundationSequenceBatchLoader,
    episode_ids: tuple[str, ...],
    maximum_windows: int,
) -> dict[str, tuple[int, ...]]:
    candidates = {episode_id: [] for episode_id in episode_ids}
    for index in range(len(loader)):
        metadata = loader.window_metadata(index)
        episode_id = str(metadata["episode_id"])
        start = int(metadata["transition_start"])
        if episode_id in candidates and start % loader.windows.transitions == 0:
            candidates[episode_id].append(index)
    return {
        episode_id: _spread_indices(indices, maximum_windows)
        for episode_id, indices in candidates.items()
    }


def _spread_indices(indices: list[int], maximum: int) -> tuple[int, ...]:
    if len(indices) <= maximum:
        return tuple(indices)
    positions = torch.linspace(0, len(indices) - 1, steps=maximum).round().long()
    return tuple(indices[int(position)] for position in positions)


def _evaluate_indices(
    trainer: FoundationWorldModelTrainer,
    loader: FoundationSequenceBatchLoader,
    indices: tuple[int, ...],
) -> EpisodeLearningSignals:
    if not indices:
        return EpisodeLearningSignals(0.0, 0.0, 0)
    batch = loader.build(indices)
    visual = trainer.visual_student(batch.student_inputs).pooled_state.reshape(
        batch.sequence_batch_size,
        batch.observation_count,
        trainer.world_model.config.visual_dimension,
    )
    world = trainer.world_model.observe(
        visual,
        batch.language_features,
        batch.proprioception,
        batch.executed_actions,
    )
    novelty = posterior_state_change_novelty(world.features)
    config = trainer.imagination.config
    td_error = observed_one_step_td_error(
        world.features,
        batch.rewards,
        batch.continues,
        trainer.value,
        trainer.imagination.slow_value,
        discount=config.discount,
        symlog_limit=config.value_symlog_limit,
    )
    return EpisodeLearningSignals(
        float(novelty.mean().cpu()),
        float(td_error.mean().cpu()),
        len(indices),
    )
