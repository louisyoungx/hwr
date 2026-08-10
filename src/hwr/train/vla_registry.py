"""Content-verified deployable VLA Actor registry."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from hwr.policy.vla_model import VLAActorConfig, VLAActorModel
from hwr.policy.vla_runtime import DeployableVLAActor, VLANormalization
from hwr.train.vla_trainer import VLABehaviorTrainingResult


VLA_ACTOR_MODEL_SCHEMA = "hwr.vla-actor-model/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_vla_behavior_result(
    root: Path,
    model_id: str,
    version: str,
    result: VLABehaviorTrainingResult,
    *,
    dataset_manifest: Mapping[str, Any],
) -> Path:
    return save_vla_actor_checkpoint(
        root,
        model_id,
        version,
        result.model,
        result.normalization,
        dataset_manifest=dataset_manifest,
        training_metadata={
            "training_kind": "behavior_cloning",
            "training_config": result.training_config.to_dict(),
            "best_validation_loss": result.best_validation_loss,
            "history": result.history,
            "training_device": result.device,
        },
    )


def save_vla_actor_checkpoint(
    root: Path,
    model_id: str,
    version: str,
    model: VLAActorModel,
    normalization: VLANormalization,
    *,
    dataset_manifest: Mapping[str, Any],
    training_metadata: Mapping[str, Any],
) -> Path:
    if not model_id or not version:
        raise ValueError("VLA model id and version are required")
    if any("critic" in str(key).lower() for key in training_metadata):
        raise ValueError("deployable Actor metadata cannot contain critic state")
    path = root / model_id / version
    path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = path / "actor.pt"
    torch.save(model.state_dict(), checkpoint_path)
    manifest = {
        "schema_version": VLA_ACTOR_MODEL_SCHEMA,
        "model_id": model_id,
        "version": version,
        "model_config": model.config.to_dict(),
        "normalization": normalization.to_dict(),
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_schema": dataset_manifest["schema_version"],
        "dataset_manifest_sha256": _mapping_sha256(dataset_manifest),
        "preprocess_fingerprint": dataset_manifest["preprocess_fingerprint"],
        "language_encoder_id": dataset_manifest["language_encoder_id"],
        "language_weights_sha256": dataset_manifest["language_weights_sha256"],
        "actor_sha256": _sha256(checkpoint_path),
        **dict(training_metadata),
    }
    _write_json_atomic(path / "manifest.json", manifest)
    return path


def _model_config(values: Mapping[str, Any]) -> VLAActorConfig:
    return VLAActorConfig(
        visual_history=int(values["visual_history"]),
        action_history=int(values["action_history"]),
        proprioception_dim=int(values["proprioception_dim"]),
        language_dim=int(values["language_dim"]),
        point_count=int(values["point_count"]),
        action_chunk_size=int(values["action_chunk_size"]),
        hidden_dim=int(values["hidden_dim"]),
        attention_heads=int(values["attention_heads"]),
        transformer_layers=int(values["transformer_layers"]),
        dropout=float(values["dropout"]),
        action_dim=int(values["action_dim"]),
    )


def load_deployable_vla_actor(
    path: Path, *, device: str = "cpu"
) -> DeployableVLAActor:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VLA_ACTOR_MODEL_SCHEMA:
        raise ValueError("VLA Actor checkpoint schema mismatch")
    checkpoint_path = path / "actor.pt"
    if _sha256(checkpoint_path) != manifest["actor_sha256"]:
        raise ValueError("VLA Actor checkpoint checksum mismatch")
    if any("critic" in key.lower() for key in manifest):
        raise ValueError("deployable VLA checkpoint contains critic metadata")
    model = VLAActorModel(_model_config(manifest["model_config"]))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if any("critic" in key.lower() for key in state):
        raise ValueError("deployable VLA checkpoint contains critic weights")
    model.load_state_dict(state)
    return DeployableVLAActor(
        model,
        VLANormalization.from_dict(manifest["normalization"]),
        preprocess_fingerprint=manifest["preprocess_fingerprint"],
        language_encoder_id=manifest["language_encoder_id"],
        language_weights_sha256=manifest["language_weights_sha256"],
        device=device,
    )
