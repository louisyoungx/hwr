"""Content-verified local model registry."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from hwr.policy.model import BehaviorMLP, ModelConfig
from hwr.policy.neural import NeuralPolicy, Normalization
from hwr.train.trainer import TrainingResult


MODEL_SCHEMA = "hwr.behavior-model/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_training_result(
    root: Path,
    model_id: str,
    version: str,
    result: TrainingResult,
    *,
    dataset_manifest: dict[str, Any],
    control_hz: float,
) -> Path:
    if not model_id or not version:
        raise ValueError("model id and version are required")
    path = root / model_id / version
    path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = path / "checkpoint.pt"
    torch.save(result.model.state_dict(), checkpoint_path)
    manifest = {
        "schema_version": MODEL_SCHEMA,
        "model_id": model_id,
        "version": version,
        "model_config": result.model_config.to_dict(),
        "training_config": result.training_config.to_dict(),
        "normalization": result.normalization.to_dict(),
        "best_validation_loss": result.best_validation_loss,
        "history": result.history,
        "training_device": result.device,
        "dataset": dataset_manifest,
        "control_hz": control_hz,
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    _write_json_atomic(path / "manifest.json", manifest)
    return path


def load_policy(path: Path, *, device: str = "cpu") -> NeuralPolicy:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = path / "checkpoint.pt"
    if _sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("model checkpoint checksum mismatch")
    model_values = manifest["model_config"]
    model_config = ModelConfig(
        observation_dim=model_values["observation_dim"],
        continuous_action_dim=model_values["continuous_action_dim"],
        hidden_dims=tuple(model_values["hidden_dims"]),
    )
    model = BehaviorMLP(model_config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return NeuralPolicy(
        model,
        Normalization.from_dict(manifest["normalization"]),
        policy_version=f"{manifest['model_id']}:{manifest['version']}",
        control_hz=float(manifest["control_hz"]),
        device=device,
    )

