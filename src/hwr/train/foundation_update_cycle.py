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
    severe_collision_batch_fraction: float,
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
        indices = sampler.sample(
            rng,
            batch_size,
            severe_collision_fraction=severe_collision_batch_fraction,
        )
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
        metadata = getattr(loader, "window_metadata", None)
        collision_groups: dict[int, list[int]] = {}
        for index in range(len(loader)):
            if callable(metadata) and _is_severe_collision_window(metadata(index)):
                collision_groups.setdefault(
                    loader.window_shard_index(index), []
                ).append(index)
        self.collision_shards = tuple(sorted(collision_groups))
        self.collision_indices = tuple(
            tuple(collision_groups[shard]) for shard in self.collision_shards
        )

    def sample(
        self,
        rng: np.random.Generator,
        batch_size: int,
        *,
        severe_collision_fraction: float = 0.0,
    ) -> tuple[int, ...]:
        if batch_size <= 0 or not 0.0 <= severe_collision_fraction <= 1.0:
            raise ValueError("foundation replay batch size must be positive")
        if self.collision_shards and rng.random() < severe_collision_fraction:
            groups = rng.choice(
                len(self.collision_indices),
                size=batch_size,
                replace=len(self.collision_indices) < batch_size,
            )
            return tuple(
                int(rng.choice(self.collision_indices[int(group)]))
                for group in groups
            )
        shard = int(rng.choice(len(self.indices), p=self.probabilities))
        values = rng.choice(self.indices[shard], size=batch_size, replace=True)
        return tuple(int(value) for value in values)


def _is_severe_collision_window(metadata: Mapping[str, object]) -> bool:
    episode = metadata.get("metadata", {})
    return (
        isinstance(episode, Mapping)
        and episode.get("result_reason") == "severe_collision"
        and int(metadata.get("transition_stop", -1))
        == int(metadata.get("transition_count", -2))
    )


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
