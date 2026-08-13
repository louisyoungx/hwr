"""One bounded task-blind optimization cycle for foundation training."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from hwr.data.foundation_loading import FoundationSequenceBatchLoader
from hwr.train.accelerator_memory import release_accelerator_memory_after_step
from hwr.train.foundation_augmentation import transform_foundation_batch
from hwr.train.foundation_trainer import FoundationWorldModelTrainer


def run_foundation_update_cycle(
    trainer: FoundationWorldModelTrainer,
    loader: FoundationSequenceBatchLoader,
    rng: np.random.Generator,
    *,
    updates: int,
    batch_size: int,
    augmentation_probability: float,
    train_task_actor: bool,
    train_exploration_actor: bool,
    progress_interval: int,
    progress: Callable[[list[Mapping[str, float]]], None],
) -> dict[str, float]:
    if min(updates, batch_size, progress_interval) <= 0:
        raise ValueError("foundation update cycle dimensions are invalid")
    metrics: list[dict[str, float]] = []
    for _ in range(updates):
        indices = rng.integers(0, len(loader), size=batch_size)
        batch = loader.build([int(value) for value in indices])
        transforms = [
            _sample_transform(loader, int(value), rng, augmentation_probability)
            for value in indices
        ]
        batch = transform_foundation_batch(batch, transforms)
        metrics.append(
            trainer.train_step(
                batch,
                train_task_actor=train_task_actor,
                train_exploration_actor=train_exploration_actor,
            )
        )
        release_accelerator_memory_after_step(len(metrics))
        if len(metrics) % progress_interval == 0:
            progress(metrics)
    names = metrics[0]
    return {
        name: float(sum(item[name] for item in metrics) / len(metrics))
        for name in names
    }


def _sample_transform(
    loader: FoundationSequenceBatchLoader,
    index: int,
    rng: np.random.Generator,
    probability: float,
) -> str | None:
    legal = loader.legal_transform_ids(index)
    if not legal or rng.random() >= probability:
        return None
    return str(rng.choice(legal))
