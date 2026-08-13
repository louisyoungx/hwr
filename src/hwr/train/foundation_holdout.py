"""Task-balanced autonomous holdout data for action-causality audits."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable, Mapping

import numpy as np

from hwr.core.runtime import RuntimeBackend
from hwr.data.autonomous_trajectory import AppendableAutonomousTrajectoryStore
from hwr.data.foundation_loading import FoundationSequenceBatchLoader
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.policy.latent_actions import LatentActionScaling
from hwr.train.foundation_batch import FoundationTrainingBatch
from hwr.train.foundation_collection import (
    AutonomousCollectionConfig,
    AutonomousEpisodeCollector,
)
from hwr.train.foundation_exploration import (
    RandomRLActionSource,
    RandomRLExplorationConfig,
)


HOLDOUT_COLLECTOR = "foundation-causality-holdout/v1"


def collect_causality_holdout(
    store: AppendableAutonomousTrajectoryStore,
    environments: Mapping[str, RuntimeBackend],
    maximum_steps: Mapping[str, int],
    preprocessor: HighResolutionVisionPreprocessor,
    action_scaling: LatentActionScaling,
    *,
    exploration_config: RandomRLExplorationConfig,
    episodes_per_task: int,
    base_seed: int,
    source_commit: str,
) -> None:
    """Collect fixed random-RL Episodes that are never exposed to optimizers."""
    task_ids = tuple(sorted(maximum_steps))
    if (
        episodes_per_task <= 0
        or base_seed < 0
        or set(environments) != set(task_ids)
    ):
        raise ValueError("causality holdout collection configuration is invalid")
    expected = {
        (task_id, _holdout_seed(base_seed, task_index, episode_index))
        for task_index, task_id in enumerate(task_ids)
        for episode_index in range(episodes_per_task)
    }
    existing = {
        (str(shard["task_id"]), int(shard["seed"]))
        for shard in store.manifest["shards"]
    }
    if not existing <= expected:
        raise ValueError("causality holdout contains unexpected task seeds")
    for task_index, task_id in enumerate(task_ids):
        collector = AutonomousEpisodeCollector(
            preprocessor,
            AutonomousCollectionConfig(
                "mujoco-bimanual-runtime/v2",
                source_commit,
                int(maximum_steps[task_id]),
            ),
        )
        for episode_index in range(episodes_per_task):
            seed = _holdout_seed(base_seed, task_index, episode_index)
            if (task_id, seed) in existing:
                continue
            episode = collector.collect(
                environments[task_id],
                RandomRLActionSource(action_scaling, exploration_config),
                task_id=task_id,
                seed=seed,
            )
            metadata = {**episode.metadata, "collector": HOLDOUT_COLLECTOR}
            store.append(replace(episode, metadata=metadata))
            existing.add((task_id, seed))
    if existing != expected:
        raise RuntimeError("causality holdout collection is incomplete")
    _verify_holdout(store, expected, source_commit)


def select_causality_windows(
    loader: FoundationSequenceBatchLoader,
    task_ids: tuple[str, ...],
    *,
    windows_per_task: int,
    selection_seed: int,
) -> dict[str, tuple[int, ...]]:
    """Select deterministic non-overlapping windows across every Episode."""
    if windows_per_task <= 0 or selection_seed < 0 or not task_ids:
        raise ValueError("causality window selection configuration is invalid")
    grouped: dict[str, dict[str, list[tuple[bytes, int]]]] = {
        task_id: {} for task_id in task_ids
    }
    transitions = loader.windows.transitions
    for index in range(len(loader)):
        metadata = loader.window_metadata(index)
        task_id = str(metadata["task_id"])
        start = int(metadata["transition_start"])
        if task_id not in grouped or start % transitions:
            continue
        episode_id = str(metadata["episode_id"])
        digest = hashlib.sha256(
            f"{selection_seed}:{episode_id}:{start}".encode()
        ).digest()
        grouped[task_id].setdefault(episode_id, []).append((digest, index))
    selected = {}
    for task_id, episodes in grouped.items():
        queues = [
            [index for _, index in sorted(values)]
            for _, values in sorted(episodes.items())
        ]
        chosen = []
        while len(chosen) < windows_per_task and any(queues):
            for queue in queues:
                if queue and len(chosen) < windows_per_task:
                    chosen.append(queue.pop())
        if len(chosen) != windows_per_task:
            raise ValueError(f"causality holdout lacks windows for {task_id}")
        selected[task_id] = tuple(
            sorted(
                chosen,
                key=lambda value: (
                    str(loader.window_metadata(value)["episode_id"]),
                    int(loader.window_metadata(value)["transition_start"]),
                ),
            )
        )
    return selected


def causality_window_manifest(
    loader: FoundationSequenceBatchLoader,
    selected: Mapping[str, tuple[int, ...]],
) -> list[dict[str, object]]:
    return [
        {
            "task_id": task_id,
            "episode_id": str(metadata["episode_id"]),
            "seed": int(metadata["seed"]),
            "transition_start": int(metadata["transition_start"]),
            "transition_stop": int(metadata["transition_stop"]),
        }
        for task_id in sorted(selected)
        for index in selected[task_id]
        for metadata in (loader.window_metadata(index),)
    ]


def causality_batches_by_task(
    loader: FoundationSequenceBatchLoader,
    selected: Mapping[str, tuple[int, ...]],
    *,
    batch_size: int,
) -> dict[str, Iterable[FoundationTrainingBatch]]:
    if batch_size <= 0 or any(len(values) % batch_size for values in selected.values()):
        raise ValueError("causality audit batch size must divide every task partition")

    def batches(indices: tuple[int, ...]) -> Iterable[FoundationTrainingBatch]:
        for start in range(0, len(indices), batch_size):
            yield loader.build(
                indices[start : start + batch_size],
                include_visual_targets=False,
            )

    return {task_id: batches(indices) for task_id, indices in selected.items()}


def _holdout_seed(base_seed: int, task_index: int, episode_index: int) -> int:
    return base_seed + 500_000_003 + task_index * 104_729 + episode_index * 1_009


def _verify_holdout(
    store: AppendableAutonomousTrajectoryStore,
    expected: set[tuple[str, int]],
    source_commit: str,
) -> None:
    verified = set()
    for shard in store.manifest["shards"]:
        identity = (str(shard["task_id"]), int(shard["seed"]))
        metadata = shard.get("metadata", {})
        if (
            identity not in expected
            or str(shard["source_commit"]) != source_commit
            or metadata.get("collector") != HOLDOUT_COLLECTOR
            or metadata.get("action_process", {}).get("schema_version")
            != "hwr.correlated-random-rl/v1"
        ):
            raise ValueError("causality holdout provenance differs")
        with np.load(store.path / str(shard["path"]), allow_pickle=False) as arrays:
            sources = {str(value) for value in arrays["action_source"]}
        if sources != {"random_rl_exploration"}:
            raise ValueError("causality holdout contains non-random action sources")
        verified.add(identity)
    if verified != expected:
        raise ValueError("causality holdout task-seed coverage differs")
