"""Compact task-blind sequence evidence retained from autonomous Episodes."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from hwr.data.autonomous_trajectory import (
    OBSERVATION_ARRAY_FIELDS,
    TRANSITION_ARRAY_FIELDS,
    AppendableAutonomousTrajectoryStore,
    AutonomousEpisode,
)


SEQUENCE_RESERVOIR_SCHEMA = "hwr.foundation-sequence-reservoir/v1"


def append_episode_sequence_evidence(
    store: AppendableAutonomousTrajectoryStore,
    episode: AutonomousEpisode,
    *,
    sequence_transitions: int,
    windows_per_episode: int,
) -> tuple[AutonomousEpisode, ...]:
    """Retain bounded continuous windows without inspecting task semantics."""
    starts = select_sequence_evidence_starts(
        episode.arrays,
        sequence_transitions=sequence_transitions,
        windows_per_episode=windows_per_episode,
    )
    excerpts = tuple(
        slice_episode_sequence(
            episode,
            start=start,
            transitions=sequence_transitions,
            slot=slot,
            slot_count=len(starts),
        )
        for slot, start in enumerate(starts)
    )
    for excerpt in excerpts:
        store.append(excerpt)
    return excerpts


def select_sequence_evidence_starts(
    arrays: Mapping[str, np.ndarray],
    *,
    sequence_transitions: int,
    windows_per_episode: int,
) -> tuple[int, ...]:
    """Select spread, salient and terminal non-overlapping sequence windows."""
    count = len(arrays["executed_action"])
    if min(sequence_transitions, windows_per_episode) <= 0 or count < sequence_transitions:
        raise ValueError("sequence evidence dimensions are invalid")
    maximum = max(1, count // sequence_transitions)
    target = min(windows_per_episode, maximum)
    candidates = tuple(range(0, count - sequence_transitions + 1))
    terminal = count - sequence_transitions
    preferred = [0, terminal]
    scores = _physical_salience_scores(arrays, sequence_transitions)
    ranked = sorted(candidates, key=lambda start: (scores[start], start), reverse=True)
    spread = np.linspace(0, terminal, num=target).round().astype(int).tolist()
    selected: list[int] = []
    for start in (*preferred, *spread, *ranked):
        if any(abs(start - existing) < sequence_transitions for existing in selected):
            continue
        selected.append(int(start))
        if len(selected) == target:
            break
    if len(selected) < target:
        raise RuntimeError("sequence evidence selector could not fill its bounded reservoir")
    return tuple(sorted(selected))


def slice_episode_sequence(
    episode: AutonomousEpisode,
    *,
    start: int,
    transitions: int,
    slot: int,
    slot_count: int,
) -> AutonomousEpisode:
    stop = start + transitions
    count = len(episode.arrays["executed_action"])
    if start < 0 or stop > count or min(transitions, slot_count) <= 0:
        raise ValueError("sequence evidence slice is invalid")
    arrays = {
        name: np.asarray(episode.arrays[name][start : stop + 1]).copy()
        for name in OBSERVATION_ARRAY_FIELDS
    }
    arrays.update(
        {
            name: np.asarray(episode.arrays[name][start:stop]).copy()
            for name in TRANSITION_ARRAY_FIELDS
        }
    )
    metadata = {
        **episode.metadata,
        "sequence_reservoir": {
            "schema_version": SEQUENCE_RESERVOIR_SCHEMA,
            "source_episode_id": episode.episode_id,
            "source_transition_count": count,
            "transition_start": start,
            "transition_stop": stop,
            "slot": slot,
            "slot_count": slot_count,
            "selector": "task-blind-physical-salience/v1",
        },
    }
    return replace(
        episode,
        episode_id=f"{episode.episode_id}--sequence-{slot:02d}-{start:06d}",
        arrays=arrays,
        metadata=metadata,
    )


def source_episode_id(shard: Mapping[str, object]) -> str:
    metadata = shard.get("metadata", {})
    reservoir = metadata.get("sequence_reservoir", {}) if isinstance(metadata, Mapping) else {}
    if isinstance(reservoir, Mapping):
        source = str(reservoir.get("source_episode_id", ""))
        if source:
            return source
    return str(shard.get("episode_id", ""))


def count_source_episodes(manifest: Mapping[str, object]) -> int:
    sources = tuple(
        source_episode_id(shard) for shard in manifest.get("shards", ())
    )
    if any(not value for value in sources):
        raise ValueError("sequence reservoir source Episode identity is missing")
    return len(set(sources))


def _physical_salience_scores(
    arrays: Mapping[str, np.ndarray], transitions: int
) -> np.ndarray:
    proprio = np.asarray(arrays["proprioception"], np.float64)
    actions = np.asarray(arrays["executed_action"], np.float64)
    intervention = np.asarray(arrays["safety_intervention"], np.float64)
    motion = np.linalg.norm(np.diff(proprio, axis=0), axis=1)
    innovation = np.linalg.norm(np.diff(actions, axis=0, prepend=actions[:1]), axis=1)
    signal = _robust_unit(motion) + _robust_unit(innovation) + (intervention > 0.0)
    return np.convolve(signal, np.ones(transitions, np.float64), mode="valid")


def _robust_unit(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, np.float64)
    scale = float(np.quantile(array, 0.95)) if array.size else 0.0
    if scale <= 1.0e-12:
        return np.zeros_like(array)
    return np.clip(array / scale, 0.0, 1.0)
