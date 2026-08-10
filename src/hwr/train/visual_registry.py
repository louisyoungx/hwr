"""Content-verified registry for learned formal visual policies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from hwr.policy.visual_model import HouseholdVisualPolicyModel, VisualModelConfig
from hwr.policy.visual_policy import LearnedVisualPolicy, VisualNormalization
from hwr.train.visual_trainer import VisualTrainingResult


VISUAL_MODEL_SCHEMA = "hwr.visual-policy-model/v3"


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


def save_visual_training_result(
    root: Path,
    model_id: str,
    version: str,
    result: VisualTrainingResult,
    *,
    dataset_manifest: Mapping[str, Any],
    task_instructions: Mapping[str, tuple[int, str]],
    control_hz: float,
) -> Path:
    path = root / model_id / version
    path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = path / "checkpoint.pt"
    torch.save(result.model.state_dict(), checkpoint_path)
    manifest = {
        "schema_version": VISUAL_MODEL_SCHEMA,
        "model_id": model_id,
        "version": version,
        "model_config": result.model_config.to_dict(),
        "training_config": result.training_config.to_dict(),
        "normalization": result.normalization.to_dict(),
        "best_validation_loss": result.best_validation_loss,
        "history": result.history,
        "training_device": result.device,
        "dataset": dict(dataset_manifest),
        "phase_names": list(dataset_manifest["phase_names"]),
        "phase_action_mask": [list(row) for row in result.phase_action_mask],
        "phase_step_limits": [list(row) for row in result.phase_step_limits],
        "navigation_routes": {
            name: [list(pose) for pose in route]
            for name, route in result.navigation_routes.items()
        },
        "task_instructions": {
            task_id: {"instruction_id": value[0], "text": value[1]}
            for task_id, value in task_instructions.items()
        },
        "control_hz": control_hz,
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    _write_json_atomic(path / "manifest.json", manifest)
    return path


def load_visual_policy(path: Path, *, device: str = "cpu") -> LearnedVisualPolicy:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VISUAL_MODEL_SCHEMA:
        raise ValueError("visual checkpoint schema mismatch")
    checkpoint_path = path / "checkpoint.pt"
    if _sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("visual checkpoint checksum mismatch")
    values = manifest["model_config"]
    config = VisualModelConfig(
        image_width=int(values["image_width"]),
        image_height=int(values["image_height"]),
        action_history=int(values["action_history"]),
        instruction_count=int(values["instruction_count"]),
        phase_count=int(values["phase_count"]),
        proprioception_dim=int(values["proprioception_dim"]),
        action_dim=int(values["action_dim"]),
        visual_channels=tuple(values["visual_channels"]),
        hidden_dim=int(values["hidden_dim"]),
        phase_embedding_dim=int(values["phase_embedding_dim"]),
    )
    model = HouseholdVisualPolicyModel(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    instructions = {
        task_id: (int(value["instruction_id"]), value["text"])
        for task_id, value in manifest["task_instructions"].items()
    }
    return LearnedVisualPolicy(
        model,
        VisualNormalization.from_dict(manifest["normalization"]),
        policy_version=f"{manifest['model_id']}:{manifest['version']}",
        control_hz=float(manifest["control_hz"]),
        task_instructions=instructions,
        phase_names=tuple(manifest["phase_names"]),
        phase_action_mask=tuple(
            tuple(bool(value) for value in row)
            for row in manifest["phase_action_mask"]
        ),
        phase_step_limits=tuple(
            tuple(int(value) for value in row)
            for row in manifest.get(
                "phase_step_limits",
                [[0, 2**31 - 1] for _ in manifest["phase_names"]],
            )
        ),
        navigation_routes={
            name: tuple(tuple(float(value) for value in pose) for pose in route)
            for name, route in manifest.get("navigation_routes", {}).items()
        },
        device=device,
    )
