"""Auditable sequence storage for actions executed by random or current RL actors."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hwr.core.embodied import DUAL_ARM_ACTION_DIM


AUTONOMOUS_TRAJECTORY_SCHEMA = "hwr.autonomous-trajectory/v2"
ALLOWED_ACTION_SOURCES = frozenset({"random_rl_exploration", "rl_actor"})
OBSERVATION_ARRAY_FIELDS = frozenset(
    {
        "rgb_uint8",
        "raw_head_depth_m",
        "head_depth_valid",
        "camera_validity",
        "frame_timestamps_ns",
        "proprioception",
        "observation_source_sha256",
        "intrinsics",
        "robot_from_camera",
    }
)
TRANSITION_ARRAY_FIELDS = frozenset(
    {
        "actor_proposal",
        "executed_action",
        "reward",
        "terminated",
        "truncated",
        "safety_cost",
        "action_source",
    }
)
STATIC_ARRAY_FIELDS = frozenset()
TRAJECTORY_ARRAY_FIELDS = (
    OBSERVATION_ARRAY_FIELDS | TRANSITION_ARRAY_FIELDS | STATIC_ARRAY_FIELDS
)
FORBIDDEN_LINEAGE_KEYS = frozenset(
    {
        "expert",
        "demonstration",
        "behavior_clone",
        "teacher_action",
        "action_label",
        "waypoint",
        "skill",
        "task_stage",
        "object_token",
        "target_token",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(
                normalized == forbidden
                or normalized.startswith(forbidden + "_")
                or normalized.endswith("_" + forbidden)
                for forbidden in FORBIDDEN_LINEAGE_KEYS
            ):
                return True
            if _contains_forbidden_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


@dataclass(frozen=True)
class AutonomousEpisode:
    episode_id: str
    task_id: str
    seed: int
    instruction: str
    locale: str
    environment_version: str
    source_commit: str
    preprocess_fingerprint: str
    legal_transform_ids: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identities = (
            self.episode_id,
            self.task_id,
            self.instruction,
            self.locale,
            self.environment_version,
            self.source_commit,
        )
        if any(not value for value in identities) or self.seed < 0:
            raise ValueError("autonomous episode identity fields are required")
        if len(self.preprocess_fingerprint) != 64:
            raise ValueError("autonomous episode requires a preprocess fingerprint")
        transforms = tuple(str(value) for value in self.legal_transform_ids)
        if len(set(transforms)) != len(transforms):
            raise ValueError("legal environment transforms must be unique")
        arrays = {name: np.asarray(value) for name, value in self.arrays.items()}
        if frozenset(arrays) != TRAJECTORY_ARRAY_FIELDS:
            raise ValueError("autonomous trajectory arrays violate the field whitelist")
        metadata = dict(self.metadata)
        if _contains_forbidden_key(metadata):
            raise ValueError("forbidden action supervision entered trajectory metadata")
        _validate_arrays(arrays)
        object.__setattr__(self, "legal_transform_ids", transforms)
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "metadata", metadata)


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    rgb = arrays["rgb_uint8"]
    if rgb.ndim != 5 or rgb.shape[1] != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("trajectory RGB must be observation-camera-height-width-RGB uint8")
    observations = rgb.shape[0]
    transitions = observations - 1
    if transitions <= 0:
        raise ValueError("autonomous episode requires at least one transition")
    height, width = rgb.shape[2:4]
    proprioception_dim = arrays["proprioception"].shape[-1]
    expected = {
        "raw_head_depth_m": (observations, height, width),
        "head_depth_valid": (observations, height, width),
        "camera_validity": (observations, 4),
        "frame_timestamps_ns": (observations, 4),
        "proprioception": (observations, proprioception_dim),
        "observation_source_sha256": (observations,),
        "actor_proposal": (transitions, DUAL_ARM_ACTION_DIM),
        "executed_action": (transitions, DUAL_ARM_ACTION_DIM),
        "reward": (transitions,),
        "terminated": (transitions,),
        "truncated": (transitions,),
        "safety_cost": (transitions,),
        "action_source": (transitions,),
        "intrinsics": (observations, 4, 4),
        "robot_from_camera": (observations, 4, 4, 4),
    }
    mismatches = {
        name: (arrays[name].shape, shape)
        for name, shape in expected.items()
        if arrays[name].shape != shape
    }
    if mismatches:
        raise ValueError(f"autonomous trajectory tensor shapes are invalid: {mismatches}")
    floating = (
        "proprioception",
        "actor_proposal",
        "executed_action",
        "reward",
        "safety_cost",
        "intrinsics",
        "robot_from_camera",
    )
    if not all(np.isfinite(arrays[name]).all() for name in floating):
        raise ValueError("autonomous trajectory finite fields contain invalid values")
    depth_valid = arrays["head_depth_valid"].astype(bool)
    if not np.isfinite(arrays["raw_head_depth_m"][depth_valid]).all():
        raise ValueError("valid trajectory depth values must be finite")
    sources = set(arrays["action_source"].astype(str))
    if not sources <= ALLOWED_ACTION_SOURCES:
        raise ValueError("trajectory action source is not autonomous RL")
    digests = arrays["observation_source_sha256"].astype(str)
    if any(len(value) != 64 for value in digests):
        raise ValueError("trajectory observation source hashes are invalid")
    terminal = arrays["terminated"].astype(bool) | arrays["truncated"].astype(bool)
    if bool(terminal[:-1].any()):
        raise ValueError("trajectory cannot continue after a terminal transition")
    if np.any(arrays["safety_cost"] < 0.0):
        raise ValueError("trajectory safety cost cannot be negative")


class AutonomousTrajectoryDatasetBuilder:
    def __init__(self, root: Path, dataset_id: str) -> None:
        if not dataset_id:
            raise ValueError("autonomous trajectory dataset id is required")
        self.path = root / dataset_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.shards: list[dict[str, Any]] = []
        self._sealed = False

    def write_episode(self, episode: AutonomousEpisode) -> Path:
        if self._sealed:
            raise ValueError("autonomous trajectory dataset is already sealed")
        if any(value["episode_id"] == episode.episode_id for value in self.shards):
            raise ValueError("autonomous trajectory episode id must be unique")
        filename = f"{episode.episode_id}.npz"
        path = self.path / filename
        temporary = path.with_name(f".{filename}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **episode.arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.shards.append(
            {
                "episode_id": episode.episode_id,
                "task_id": episode.task_id,
                "seed": episode.seed,
                "instruction": episode.instruction,
                "locale": episode.locale,
                "environment_version": episode.environment_version,
                "source_commit": episode.source_commit,
                "preprocess_fingerprint": episode.preprocess_fingerprint,
                "legal_transform_ids": list(episode.legal_transform_ids),
                "metadata": episode.metadata,
                "path": filename,
                "observation_count": int(episode.arrays["rgb_uint8"].shape[0]),
                "transition_count": int(episode.arrays["executed_action"].shape[0]),
                "sha256": _sha256(path),
            }
        )
        return path

    def seal(self) -> Path:
        if self._sealed or not self.shards:
            raise ValueError("autonomous trajectory dataset must contain unsealed data")
        manifest = {
            "schema_version": AUTONOMOUS_TRAJECTORY_SCHEMA,
            "dataset_id": self.path.name,
            "array_fields": sorted(TRAJECTORY_ARRAY_FIELDS),
            "allowed_action_sources": sorted(ALLOWED_ACTION_SOURCES),
            "episode_count": len(self.shards),
            "transition_count": sum(value["transition_count"] for value in self.shards),
            "shards": self.shards,
        }
        _write_json_atomic(self.path / "manifest.json", manifest)
        self._sealed = True
        return self.path


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_autonomous_trajectory_dataset(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != AUTONOMOUS_TRAJECTORY_SCHEMA:
        raise ValueError("autonomous trajectory dataset schema mismatch")
    if frozenset(manifest.get("array_fields", ())) != TRAJECTORY_ARRAY_FIELDS:
        raise ValueError("autonomous trajectory manifest field whitelist changed")
    if frozenset(manifest.get("allowed_action_sources", ())) != ALLOWED_ACTION_SOURCES:
        raise ValueError("autonomous trajectory action source whitelist changed")
    if _contains_forbidden_key(manifest):
        raise ValueError("forbidden action supervision entered trajectory manifest")
    for shard in manifest.get("shards", ()):
        shard_path = path / shard["path"]
        if _sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"autonomous trajectory checksum mismatch: {shard['path']}")
        with np.load(shard_path, allow_pickle=False) as arrays:
            copied = {name: arrays[name].copy() for name in arrays.files}
        if frozenset(copied) != TRAJECTORY_ARRAY_FIELDS:
            raise ValueError("autonomous trajectory shard fields changed")
        _validate_arrays(copied)
    return manifest
