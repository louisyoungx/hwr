"""Continuous sequence windows over verified autonomous Episode shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hwr.data.autonomous_trajectory import (
    OBSERVATION_ARRAY_FIELDS,
    STATIC_ARRAY_FIELDS,
    TRANSITION_ARRAY_FIELDS,
    verify_autonomous_trajectory_dataset,
)


@dataclass(frozen=True)
class TrajectoryWindowIndex:
    shard_index: int
    transition_start: int


class AutonomousTrajectoryWindows:
    def __init__(self, path: Path, *, transitions: int, stride: int = 1) -> None:
        if transitions <= 0 or stride <= 0:
            raise ValueError("trajectory window and stride must be positive")
        self.path = path
        self.manifest = verify_autonomous_trajectory_dataset(path)
        self.transitions = transitions
        self.indices: list[TrajectoryWindowIndex] = []
        for shard_index, shard in enumerate(self.manifest["shards"]):
            count = int(shard["transition_count"])
            self.indices.extend(
                TrajectoryWindowIndex(shard_index, start)
                for start in range(0, max(0, count - transitions + 1), stride)
            )
        if not self.indices:
            raise ValueError("trajectory dataset has no complete sequence windows")
        self._cached_shard = -1
        self._cached_arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        location = self.indices[index]
        arrays = self._load_shard(location.shard_index)
        start = location.transition_start
        stop = start + self.transitions
        result = {
            name: arrays[name][start : stop + 1].copy()
            for name in OBSERVATION_ARRAY_FIELDS
        }
        result.update(
            {
                name: arrays[name][start:stop].copy()
                for name in TRANSITION_ARRAY_FIELDS
            }
        )
        result.update({name: arrays[name].copy() for name in STATIC_ARRAY_FIELDS})
        return result

    def shard_metadata(self, index: int) -> dict[str, object]:
        location = self.indices[index]
        return dict(self.manifest["shards"][location.shard_index])

    def _load_shard(self, index: int) -> dict[str, np.ndarray]:
        if self._cached_shard != index:
            shard = self.manifest["shards"][index]
            with np.load(self.path / shard["path"], allow_pickle=False) as arrays:
                self._cached_arrays = {name: arrays[name].copy() for name in arrays.files}
            self._cached_shard = index
        return self._cached_arrays
