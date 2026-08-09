"""Load verified visual behavior shards for episode-safe training splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hwr.data.visual import POLICY_INPUT_FIELDS, verify_visual_dataset


@dataclass(frozen=True)
class LoadedVisualDataset:
    path: Path
    manifest: dict[str, object]
    inputs: dict[str, np.ndarray]
    actions: np.ndarray
    step_indices: np.ndarray
    episode_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def split_by_episode(
        self,
        *,
        validation_fraction: float = 0.25,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation fraction must be between zero and one")
        episodes = np.unique(self.episode_ids)
        if len(episodes) < 2:
            raise ValueError("visual training needs at least two episodes")
        rng = np.random.default_rng(seed)
        shuffled = episodes.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, round(len(episodes) * validation_fraction))
        validation = set(shuffled[:validation_count])
        mask = np.asarray([episode in validation for episode in self.episode_ids])
        return np.flatnonzero(~mask), np.flatnonzero(mask)


def load_visual_dataset(path: Path) -> LoadedVisualDataset:
    manifest = verify_visual_dataset(path)
    input_parts = {name: [] for name in POLICY_INPUT_FIELDS}
    action_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    for shard in manifest["shards"]:
        with np.load(path / shard["path"], allow_pickle=False) as arrays:
            for name in POLICY_INPUT_FIELDS:
                input_parts[name].append(arrays[f"input__{name}"].copy())
            actions = arrays["label__action"].astype(np.float32, copy=True)
            steps = arrays["step_index"].astype(np.int32, copy=True)
        count = int(shard["sample_count"])
        action_parts.append(actions)
        step_parts.append(steps)
        episode_parts.append(np.full(count, shard["episode_id"], dtype=object))
    return LoadedVisualDataset(
        path=path,
        manifest=manifest,
        inputs={name: np.concatenate(parts) for name, parts in input_parts.items()},
        actions=np.concatenate(action_parts),
        step_indices=np.concatenate(step_parts),
        episode_ids=np.concatenate(episode_parts).astype(str),
    )
