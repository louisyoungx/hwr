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
from hwr.train.foundation_sequence_reservoir import slice_episode_sequence


HOLDOUT_COLLECTOR = "foundation-causality-holdout/v5"
SYSTEM_IDENTIFICATION_PHASE = "system_identification"
COLLISION_VALIDATION_PHASE = "collision_validation"
SYSTEM_IDENTIFICATION_CORRELATIONS = (0.0, 0.50, 0.90, 0.96)


def collect_causality_holdout(
    store: AppendableAutonomousTrajectoryStore,
    environments: Mapping[str, RuntimeBackend],
    maximum_steps: Mapping[str, int],
    preprocessor: HighResolutionVisionPreprocessor,
    action_scaling: LatentActionScaling,
    *,
    exploration_config: RandomRLExplorationConfig,
    episodes_per_task: int,
    windows_per_episode: int,
    sequence_transitions: int,
    retained_transitions_per_episode: int,
    maximum_attempts_per_episode: int,
    base_seed: int,
    source_commit: str,
    holdout_phase: str = SYSTEM_IDENTIFICATION_PHASE,
    collision_balanced: bool = False,
    collision_positive_episodes: int | None = None,
) -> None:
    """Collect one deterministic compact holdout phase without policy data."""
    task_ids = tuple(sorted(maximum_steps))
    if (
        min(
            episodes_per_task,
            windows_per_episode,
            sequence_transitions,
            retained_transitions_per_episode,
            maximum_attempts_per_episode,
        )
        <= 0
        or base_seed < 0
        or set(environments) != set(task_ids)
        or holdout_phase not in {
            SYSTEM_IDENTIFICATION_PHASE,
            COLLISION_VALIDATION_PHASE,
        }
        or (
            collision_balanced
            and collision_positive_episodes is None
            and episodes_per_task % 2
        )
        or (
            collision_positive_episodes is not None
            and not 0 <= collision_positive_episodes <= episodes_per_task
        )
    ):
        raise ValueError("causality holdout collection configuration is invalid")
    expected_slots = {
        (holdout_phase, task_id, episode_index)
        for task_index, task_id in enumerate(task_ids)
        for episode_index in range(episodes_per_task)
    }
    existing = _existing_slots(store)
    phase_slots = {slot for slot in existing if slot[0] == holdout_phase}
    if not phase_slots <= expected_slots:
        raise ValueError("foundation holdout contains unexpected phase slots")
    minimum_transitions = windows_per_episode * sequence_transitions
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
            slot = (holdout_phase, task_id, episode_index)
            if slot in existing:
                continue
            collision_target = (
                _collision_balance_target(episode_index, episodes_per_task)
                if collision_balanced
                else None
            )
            if collision_positive_episodes is not None:
                collision_target = (
                    "positive"
                    if episode_index < collision_positive_episodes
                    else "negative"
                )
            for attempt in range(maximum_attempts_per_episode):
                seed = _holdout_seed(
                    base_seed,
                    task_index,
                    episode_index,
                    attempt,
                    holdout_phase=holdout_phase,
                )
                episode = collector.collect(
                    environments[task_id],
                    RandomRLActionSource(
                        action_scaling,
                        replace(
                            exploration_config,
                            motion_correlation=(
                                SYSTEM_IDENTIFICATION_CORRELATIONS[
                                    episode_index
                                    % len(SYSTEM_IDENTIFICATION_CORRELATIONS)
                                ]
                            ),
                        ),
                    ),
                    task_id=task_id,
                    seed=seed,
                )
                transitions = len(episode.arrays["executed_action"])
                collision_class = _episode_collision_class(episode.metadata)
                if transitions < minimum_transitions or (
                    collision_target is not None
                    and collision_class != collision_target
                ):
                    continue
                retained = min(retained_transitions_per_episode, transitions)
                compact = slice_episode_sequence(
                    episode,
                    start=transitions - retained,
                    transitions=retained,
                    slot=0,
                    slot_count=1,
                )
                metadata = {
                    **compact.metadata,
                    "collector": HOLDOUT_COLLECTOR,
                    "holdout_phase": holdout_phase,
                    "holdout_slot": episode_index,
                    "seed_attempt": attempt,
                    "minimum_transitions": minimum_transitions,
                    "windows_per_episode": windows_per_episode,
                    "collision_balance_target": (
                        collision_target or "unconstrained"
                    ),
                    "collision_class": collision_class,
                    "retained_transitions": retained,
                    "system_identification_excitation": {
                        "motion_correlation": (
                            SYSTEM_IDENTIFICATION_CORRELATIONS[
                                episode_index
                                % len(SYSTEM_IDENTIFICATION_CORRELATIONS)
                            ]
                        ),
                        "task_conditioned": False,
                    },
                }
                store.append(replace(compact, metadata=metadata))
                existing[slot] = seed
                break
            if slot not in existing:
                raise RuntimeError(
                    f"causality holdout could not fill usable slot {slot}"
                )
    if {slot for slot in existing if slot[0] == holdout_phase} != expected_slots:
        raise RuntimeError("foundation holdout phase collection is incomplete")
    _verify_holdout(
        store,
        expected_slots,
        source_commit,
        minimum_transitions=minimum_transitions,
        windows_per_episode=windows_per_episode,
        episodes_per_task=episodes_per_task,
        holdout_phase=holdout_phase,
        collision_balanced=collision_balanced,
        collision_positive_episodes=collision_positive_episodes,
    )


def select_causality_windows(
    loader: FoundationSequenceBatchLoader,
    task_ids: tuple[str, ...],
    *,
    windows_per_task: int,
    selection_seed: int,
) -> dict[str, tuple[int, ...]]:
    """Select the same number of non-overlapping windows from every Episode."""
    if windows_per_task <= 0 or selection_seed < 0 or not task_ids:
        raise ValueError("causality window selection configuration is invalid")
    grouped: dict[str, dict[str, list[tuple[bytes, int]]]] = {
        task_id: {} for task_id in task_ids
    }
    transitions = loader.windows.transitions
    for index in range(len(loader)):
        metadata = loader.window_metadata(index)
        episode_metadata = metadata.get("metadata", {})
        task_id = str(metadata["task_id"])
        start = int(metadata["transition_start"])
        if (
            task_id not in grouped
            or start % transitions
            or not isinstance(episode_metadata, Mapping)
            or episode_metadata.get("holdout_phase")
            != SYSTEM_IDENTIFICATION_PHASE
        ):
            continue
        episode_id = str(metadata["episode_id"])
        digest = hashlib.sha256(
            f"{selection_seed}:{episode_id}:{start}".encode()
        ).digest()
        grouped[task_id].setdefault(episode_id, []).append((digest, index))
    selected = {}
    for task_id, episodes in grouped.items():
        if not episodes or windows_per_task % len(episodes):
            raise ValueError(
                f"causality holdout cannot balance windows for {task_id}"
            )
        per_episode = windows_per_task // len(episodes)
        chosen = []
        for episode_id, values in sorted(episodes.items()):
            queue = [index for _, index in sorted(values)]
            if len(queue) < per_episode:
                raise ValueError(
                    f"causality holdout Episode {episode_id} lacks windows"
                )
            chosen.extend(queue[-per_episode:])
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


def holdout_phase_manifest(
    manifest: Mapping[str, object], phase: str
) -> dict[str, object]:
    if phase not in {SYSTEM_IDENTIFICATION_PHASE, COLLISION_VALIDATION_PHASE}:
        raise ValueError("unknown foundation holdout phase")
    shards = [
        shard
        for shard in manifest.get("shards", ())
        if shard.get("metadata", {}).get("holdout_phase") == phase
    ]
    return {
        **manifest,
        "shards": shards,
        "episode_count": len(shards),
        "transition_count": sum(int(item["transition_count"]) for item in shards),
    }


def _holdout_seed(
    base_seed: int,
    task_index: int,
    episode_index: int,
    attempt: int,
    *,
    holdout_phase: str = SYSTEM_IDENTIFICATION_PHASE,
) -> int:
    phase_offset = 0 if holdout_phase == SYSTEM_IDENTIFICATION_PHASE else 200_000_033
    return (
        base_seed
        + 500_000_003
        + task_index * 10_000_019
        + episode_index * 100_003
        + attempt * 1_009
        + phase_offset
    )


def _collision_balance_target(
    episode_index: int, episodes_per_task: int
) -> str | None:
    if episodes_per_task == 1:
        return None
    return "positive" if episode_index < episodes_per_task // 2 else "negative"


def _episode_collision_class(metadata: Mapping[str, object]) -> str:
    return "positive" if metadata.get("result_reason") == "severe_collision" else "negative"


def _existing_slots(
    store: AppendableAutonomousTrajectoryStore,
) -> dict[tuple[str, str, int], int]:
    result: dict[tuple[str, str, int], int] = {}
    for shard in store.manifest["shards"]:
        metadata = shard.get("metadata", {})
        if metadata.get("collector") != HOLDOUT_COLLECTOR:
            raise ValueError("causality holdout collector version differs")
        slot = (
            str(metadata.get("holdout_phase", "")),
            str(shard["task_id"]),
            int(metadata.get("holdout_slot", -1)),
        )
        if slot in result or slot[0] not in {
            SYSTEM_IDENTIFICATION_PHASE,
            COLLISION_VALIDATION_PHASE,
        } or slot[2] < 0:
            raise ValueError("causality holdout task slot is invalid")
        result[slot] = int(shard["seed"])
    return result


def _verify_holdout(
    store: AppendableAutonomousTrajectoryStore,
    expected_slots: set[tuple[str, str, int]],
    source_commit: str,
    *,
    minimum_transitions: int,
    windows_per_episode: int,
    episodes_per_task: int,
    holdout_phase: str,
    collision_balanced: bool,
    collision_positive_episodes: int | None,
) -> None:
    verified = set()
    for shard in store.manifest["shards"]:
        metadata = shard.get("metadata", {})
        if metadata.get("holdout_phase") != holdout_phase:
            continue
        identity = (
            holdout_phase,
            str(shard["task_id"]),
            int(metadata.get("holdout_slot", -1)),
        )
        target = (
            _collision_balance_target(identity[2], episodes_per_task)
            if collision_balanced
            else None
        )
        if collision_positive_episodes is not None:
            target = (
                "positive"
                if identity[2] < collision_positive_episodes
                else "negative"
            )
        if (
            identity not in expected_slots
            or str(shard["source_commit"]) != source_commit
            or metadata.get("collector") != HOLDOUT_COLLECTOR
            or int(shard["transition_count"]) < minimum_transitions
            or int(metadata.get("minimum_transitions", -1)) != minimum_transitions
            or int(metadata.get("windows_per_episode", -1)) != windows_per_episode
            or metadata.get("collision_balance_target")
            != (target or "unconstrained")
            or metadata.get("collision_class")
            != _episode_collision_class(metadata)
            or (target is not None and metadata.get("collision_class") != target)
            or metadata.get("action_process", {}).get("schema_version")
            != "hwr.correlated-random-rl/v1"
        ):
            raise ValueError("causality holdout provenance differs")
        with np.load(store.path / str(shard["path"]), allow_pickle=False) as arrays:
            sources = {str(value) for value in arrays["action_source"]}
        if sources != {"random_rl_exploration"}:
            raise ValueError("causality holdout contains non-random action sources")
        verified.add(identity)
    if verified != expected_slots:
        raise ValueError("causality holdout task-slot coverage differs")
