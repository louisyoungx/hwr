"""Content-verified resumable state for asymmetric simulation training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from hwr.train.asymmetric_replay import AsymmetricReplayBuffer
from hwr.train.asymmetric_rl import AsymmetricActorCriticTrainer


ASYMMETRIC_TRAINING_SCHEMA = "hwr.asymmetric-training-state/v1"


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


def save_asymmetric_training_checkpoint(
    path: Path,
    trainer: AsymmetricActorCriticTrainer,
    replay: AsymmetricReplayBuffer,
    *,
    run_metadata: Mapping[str, Any] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    state_path = path / "training-state.pt"
    torch.save(
        {"trainer": trainer.state_dict(), "replay": replay.state_dict()}, state_path
    )
    manifest = {
        "schema_version": ASYMMETRIC_TRAINING_SCHEMA,
        "update_count": trainer.update_count,
        "replay_size": replay.size,
        "rl_config": trainer.config.to_dict(),
        "critic_config": trainer.critic_config.to_dict(),
        "run_metadata": dict(run_metadata or {}),
        "state_sha256": _sha256(state_path),
    }
    _write_json_atomic(path / "manifest.json", manifest)
    return path


def load_asymmetric_training_checkpoint(
    path: Path,
    trainer: AsymmetricActorCriticTrainer,
    replay: AsymmetricReplayBuffer,
) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ASYMMETRIC_TRAINING_SCHEMA:
        raise ValueError("asymmetric training checkpoint schema mismatch")
    state_path = path / "training-state.pt"
    if _sha256(state_path) != manifest["state_sha256"]:
        raise ValueError("asymmetric training checkpoint checksum mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    trainer.load_state_dict(state["trainer"])
    replay.load_state_dict(state["replay"])
    return manifest
