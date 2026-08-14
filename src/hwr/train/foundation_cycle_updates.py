"""Replay-loader orchestration for normal cycles and Actor warm starts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from hwr.data.foundation_cache import FoundationFeatureCache
from hwr.data.foundation_loading import (
    FoundationPreparedFeatures,
    FoundationSequenceBatchLoader,
)
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.train.foundation_metrics import mean_metrics
from hwr.train.foundation_actor_warmup import (
    ActorWarmupCriteria,
    FoundationActorWarmupResult,
    assess_actor_warmup,
)
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_trainer import FoundationWorldModelTrainer
from hwr.train.foundation_update_cycle import (
    ShardLocalWindowSampler,
    run_foundation_update_cycle,
)


def run_replay_updates(
    trainer: FoundationWorldModelTrainer,
    replay_path: Path,
    cache: FoundationFeatureCache,
    preprocessor: HighResolutionVisionPreprocessor,
    prepared: FoundationPreparedFeatures,
    rng: np.random.Generator,
    config: FoundationOnlineTrainingConfig,
    *,
    updates: int,
    train_task_actor: bool,
    train_exploration_actor: bool,
    progress: Callable[[Mapping[str, float], int], None] | None = None,
) -> dict[str, float]:
    """Run bounded updates from one replay contract without changing sampling."""
    loader = FoundationSequenceBatchLoader(
        replay_path,
        cache,
        preprocessor,
        trainer.visual_student.config,
        prepared,
        transitions=config.sequence_transitions,
        device=str(next(trainer.actor.parameters()).device),
    )

    def publish(values: list[Mapping[str, float]]) -> None:
        if progress is not None:
            progress(mean_metrics(values), len(values))

    return run_foundation_update_cycle(
        trainer,
        loader,
        rng,
        updates=updates,
        batch_size=config.batch_size,
        augmentation_probability=config.augmentation_probability,
        severe_collision_batch_fraction=config.severe_collision_batch_fraction,
        train_task_actor=train_task_actor,
        train_exploration_actor=train_exploration_actor,
        progress_interval=min(config.metrics_publish_interval_updates, updates),
        progress=publish,
    )


def warm_start_actor(
    trainer: FoundationWorldModelTrainer,
    replay_path: Path,
    cache: FoundationFeatureCache,
    preprocessor: HighResolutionVisionPreprocessor,
    prepared: FoundationPreparedFeatures,
    rng: np.random.Generator,
    config: FoundationOnlineTrainingConfig,
    *,
    train_task_actor: bool,
) -> FoundationActorWarmupResult:
    """Train an admitted Actor before it becomes a collection source."""
    loader = FoundationSequenceBatchLoader(
        replay_path,
        cache,
        preprocessor,
        trainer.visual_student.config,
        prepared,
        transitions=config.sequence_transitions,
        device=str(next(trainer.actor.parameters()).device),
    )
    sampler = ShardLocalWindowSampler(loader)
    criteria = ActorWarmupCriteria.from_config(config)
    metrics: list[dict[str, float]] = []
    windows: list[dict[str, float]] = []
    actor_kind = "task" if train_task_actor else "exploration"
    for _ in range(criteria.maximum_updates):
        indices = sampler.sample(rng, config.batch_size)
        batch = loader.build(indices, include_visual_targets=False)
        metrics.append(
            trainer.actor_warmup_step(
                batch, train_task_actor=train_task_actor
            )
        )
        if len(metrics) % criteria.window_updates:
            continue
        windows.append(mean_metrics(metrics[-criteria.window_updates :]))
        assessment = assess_actor_warmup(
            windows, actor_kind, criteria, update_count=len(metrics)
        )
        if assessment["passed"] is True:
            return FoundationActorWarmupResult(
                len(metrics), mean_metrics(metrics), assessment
            )
    return FoundationActorWarmupResult(
        len(metrics), mean_metrics(metrics), assessment
    )
