"""Train, hash, save, and reload visual kNN policies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from hwr.data.visual_loading import LoadedVisualDataset
from hwr.policy.visual_knn import (
    VisualKnnConfig,
    VisualKnnPolicy,
    visual_knn_feature_scale,
    visual_knn_features,
)


VISUAL_KNN_SCHEMA = "hwr.visual-knn-policy/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _phase_action_mask(dataset: LoadedVisualDataset) -> np.ndarray:
    rows = []
    for phase in range(len(dataset.phase_names)):
        actions = dataset.actions[dataset.phase_indices == phase, :8]
        rows.append(np.max(np.abs(actions), axis=0) > 1e-4)
    return np.asarray(rows, dtype=bool)


def save_visual_knn_policy(
    root: Path,
    model_id: str,
    version: str,
    dataset: LoadedVisualDataset,
    *,
    task_instructions: Mapping[str, tuple[int, str]],
    control_hz: float,
    neighbors: int = 5,
) -> Path:
    path = root / model_id / version
    path.mkdir(parents=True, exist_ok=False)
    width, height = dataset.manifest["image_size"]
    config = VisualKnnConfig(
        int(width), int(height), int(dataset.manifest["action_history"]), neighbors
    )
    raw = visual_knn_features(dataset.inputs)
    mean = raw.mean(axis=0).astype(np.float32)
    std = np.maximum(raw.std(axis=0), 0.02).astype(np.float32)
    scale = visual_knn_feature_scale(config.action_history)
    references = ((raw - mean) / std * scale).astype(np.float32)
    checkpoint = path / "checkpoint.npz"
    np.savez_compressed(
        checkpoint,
        references=references,
        actions=dataset.actions.astype(np.float32),
        phases=dataset.phase_indices.astype(np.int32),
        feature_mean=mean,
        feature_std=std,
        feature_scale=scale,
        phase_action_mask=_phase_action_mask(dataset),
    )
    manifest = {
        "schema_version": VISUAL_KNN_SCHEMA,
        "model_id": model_id,
        "version": version,
        "config": config.to_dict(),
        "dataset": dataset.manifest,
        "phase_names": list(dataset.phase_names),
        "task_instructions": {
            task_id: {"instruction_id": value[0], "text": value[1]}
            for task_id, value in task_instructions.items()
        },
        "control_hz": control_hz,
        "checkpoint_sha256": _sha256(checkpoint),
    }
    _write_json(path / "manifest.json", manifest)
    return path


def load_visual_knn_policy(path: Path) -> VisualKnnPolicy:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VISUAL_KNN_SCHEMA:
        raise ValueError("visual kNN checkpoint schema mismatch")
    checkpoint = path / "checkpoint.npz"
    if _sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("visual kNN checkpoint checksum mismatch")
    with np.load(checkpoint, allow_pickle=False) as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    config = VisualKnnConfig(**manifest["config"])
    instructions = {
        task_id: (int(value["instruction_id"]), str(value["text"]))
        for task_id, value in manifest["task_instructions"].items()
    }
    return VisualKnnPolicy(
        config,
        values["references"],
        values["actions"],
        values["phases"],
        values["feature_mean"],
        values["feature_std"],
        values["feature_scale"],
        manifest["phase_names"],
        values["phase_action_mask"],
        policy_version=f"{manifest['model_id']}:{manifest['version']}",
        task_instructions=instructions,
        control_hz=float(manifest["control_hz"]),
    )
