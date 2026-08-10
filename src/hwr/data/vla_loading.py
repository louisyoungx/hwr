"""Verified VLA dataset loading and episode-safe splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hwr.data.vla_dataset import verify_vla_dataset
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS


@dataclass(frozen=True)
class LoadedVLADataset:
    path: Path
    manifest: dict[str, object]
    inputs: dict[str, np.ndarray]
    action_chunks: np.ndarray
    valid_steps: np.ndarray
    step_indices: np.ndarray
    episode_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.action_chunks.shape[0])

    def split_by_episode(
        self, *, validation_fraction: float = 0.25, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation fraction must be between zero and one")
        episodes = np.unique(self.episode_ids)
        if len(episodes) < 2:
            raise ValueError("VLA training requires at least two episodes")
        shuffled = episodes.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        validation_count = max(1, round(len(episodes) * validation_fraction))
        validation = set(shuffled[:validation_count])
        mask = np.asarray([episode in validation for episode in self.episode_ids])
        return np.flatnonzero(~mask), np.flatnonzero(mask)


def load_vla_dataset(path: Path) -> LoadedVLADataset:
    manifest = verify_vla_dataset(path)
    input_parts = {name: [] for name in VLA_POLICY_INPUT_FIELDS}
    action_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    for shard in manifest["shards"]:
        with np.load(path / shard["path"], allow_pickle=False) as arrays:
            for name in VLA_POLICY_INPUT_FIELDS:
                input_parts[name].append(arrays[f"input__{name}"].copy())
            actions = arrays["label__action_chunk"].astype(np.float32, copy=True)
            valid = arrays["label__valid_steps"].astype(np.int64, copy=True)
            steps = arrays["step_index"].astype(np.int32, copy=True)
        count = int(shard["sample_count"])
        action_parts.append(actions)
        valid_parts.append(valid)
        step_parts.append(steps)
        episode_parts.append(np.full(count, shard["episode_id"], dtype=object))
    return LoadedVLADataset(
        path=path,
        manifest=manifest,
        inputs={name: np.concatenate(parts) for name, parts in input_parts.items()},
        action_chunks=np.concatenate(action_parts),
        valid_steps=np.concatenate(valid_parts),
        step_indices=np.concatenate(step_parts),
        episode_ids=np.concatenate(episode_parts).astype(str),
    )
