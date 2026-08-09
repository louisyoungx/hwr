"""Parquet-backed behavior dataset and immutable manifest."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DATASET_SCHEMA = "hwr.behavior-dataset/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class BehaviorSample:
    episode_id: str
    step_index: int
    task_id: str
    observation: np.ndarray
    action: np.ndarray


class BehaviorDataset:
    """In-memory view backed by an immutable Parquet artifact."""

    def __init__(
        self,
        path: Path,
        observations: np.ndarray,
        actions: np.ndarray,
        episode_ids: np.ndarray,
        task_ids: np.ndarray,
        step_indices: np.ndarray,
        manifest: dict[str, Any],
    ) -> None:
        self.path = path
        self.observations = observations.astype(np.float32, copy=False)
        self.actions = actions.astype(np.float32, copy=False)
        self.episode_ids = episode_ids.astype(str, copy=False)
        self.task_ids = task_ids.astype(str, copy=False)
        self.step_indices = step_indices.astype(np.int64, copy=False)
        self.manifest = manifest

    def __len__(self) -> int:
        return self.observations.shape[0]

    def sample(self, index: int) -> BehaviorSample:
        return BehaviorSample(
            episode_id=str(self.episode_ids[index]),
            step_index=int(self.step_indices[index]),
            task_id=str(self.task_ids[index]),
            observation=self.observations[index],
            action=self.actions[index],
        )

    def split_by_episode(
        self,
        *,
        validation_fraction: float = 0.2,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        episodes = np.unique(self.episode_ids)
        if len(episodes) < 2:
            raise ValueError("at least two episodes are required for a split")
        rng = np.random.default_rng(seed)
        shuffled = episodes.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation_episodes = set(shuffled[:validation_count])
        validation_mask = np.asarray(
            [episode_id in validation_episodes for episode_id in self.episode_ids],
            dtype=bool,
        )
        return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)

    @classmethod
    def load(cls, path: Path, *, verify_checksum: bool = True) -> "BehaviorDataset":
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        parquet_path = path / "samples.parquet"
        if verify_checksum and _sha256(parquet_path) != manifest["checksum"]:
            raise ValueError("behavior dataset checksum mismatch")
        table = pq.read_table(parquet_path)
        observations = np.asarray(table["observation"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        return cls(
            path=path,
            observations=observations,
            actions=actions,
            episode_ids=np.asarray(table["episode_id"].to_pylist()),
            task_ids=np.asarray(table["task_id"].to_pylist()),
            step_indices=np.asarray(table["step_index"].to_pylist(), dtype=np.int64),
            manifest=manifest,
        )


def write_behavior_dataset(
    root: Path,
    dataset_id: str,
    samples: Sequence[BehaviorSample],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    if not dataset_id or not samples:
        raise ValueError("dataset id and samples are required")
    path = root / dataset_id
    path.mkdir(parents=True, exist_ok=False)
    observation_dim = int(samples[0].observation.shape[0])
    action_dim = int(samples[0].action.shape[0])
    if any(sample.observation.shape != (observation_dim,) for sample in samples):
        raise ValueError("all observations must share one vector dimension")
    if any(sample.action.shape != (action_dim,) for sample in samples):
        raise ValueError("all actions must share one vector dimension")
    table = pa.table(
        {
            "episode_id": [sample.episode_id for sample in samples],
            "step_index": [sample.step_index for sample in samples],
            "task_id": [sample.task_id for sample in samples],
            "observation": [sample.observation.tolist() for sample in samples],
            "action": [sample.action.tolist() for sample in samples],
        }
    )
    parquet_path = path / "samples.parquet"
    pq.write_table(table, parquet_path, compression="zstd")
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": dataset_id,
        "sample_count": len(samples),
        "episode_count": len({sample.episode_id for sample in samples}),
        "task_ids": sorted({sample.task_id for sample in samples}),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "checksum": _sha256(parquet_path),
        "metadata": {} if metadata is None else metadata,
    }
    _write_json_atomic(path / "manifest.json", manifest)
    return path

