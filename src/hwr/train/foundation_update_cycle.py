"""One bounded task-blind optimization cycle for foundation training."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from hwr.data.foundation_loading import FoundationSequenceBatchLoader
from hwr.train.accelerator_memory import release_accelerator_memory_after_step
from hwr.train.foundation_augmentation import transform_foundation_batch
from hwr.train.foundation_metrics import mean_metrics
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
    sampler = ShardLocalWindowSampler(loader)
    metrics: list[dict[str, float]] = []
    for _ in range(updates):
        indices = sampler.sample(rng, batch_size)
        batch = loader.build(
            indices, include_visual_targets=trainer.visual_update_due
        )
        transforms = [
            _sample_transform(loader, int(value), rng, augmentation_probability)
            for value in indices
        ]
        batch = transform_foundation_batch(batch, transforms)
        step_metrics = trainer.train_step(
            batch,
            train_task_actor=train_task_actor,
            train_exploration_actor=train_exploration_actor,
        )
        step_metrics["trainer/replay_shards_per_batch"] = float(
            len({loader.window_shard_index(value) for value in indices})
        )
        metrics.append(step_metrics)
        release_accelerator_memory_after_step(len(metrics))
        if len(metrics) % progress_interval == 0:
            progress(metrics)
    return mean_metrics(metrics)


class ShardLocalWindowSampler:
    """Uniform window sampling with one decompressed Episode per batch."""

    def __init__(self, loader: FoundationSequenceBatchLoader) -> None:
        grouped: dict[int, list[int]] = {}
        for index in range(len(loader)):
            grouped.setdefault(loader.window_shard_index(index), []).append(index)
        self.shards = tuple(sorted(grouped))
        self.indices = tuple(tuple(grouped[shard]) for shard in self.shards)
        counts = np.asarray([len(values) for values in self.indices], np.float64)
        self.probabilities = counts / counts.sum()

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[int, ...]:
        if batch_size <= 0:
            raise ValueError("foundation replay batch size must be positive")
        shard = int(rng.choice(len(self.indices), p=self.probabilities))
        values = rng.choice(self.indices[shard], size=batch_size, replace=True)
        return tuple(int(value) for value in values)


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
