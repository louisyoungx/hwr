"""Auditable manifests and resumable checkpoints for no-demonstration training."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS
from hwr.policy.vla_model import VLAActorConfig, VLAActorModel
from hwr.train.bimanual_training import (
    BimanualRLTrainingConfig,
    BimanualTrainingResult,
    BimanualTrainingRunner,
)


BIMANUAL_RUN_SCHEMA = "hwr.bimanual-rl-run/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_episode_records(path: Path, result: BimanualTrainingResult) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_bimanual_live_progress(
    path: Path,
    result: BimanualTrainingResult,
) -> Path:
    """Publish episode metrics without rewriting the replay checkpoint."""
    if not path.is_dir():
        raise FileNotFoundError(f"training run does not exist: {path}")
    progress_path = path / "live-episodes.jsonl"
    _write_episode_records(progress_path, result)
    return progress_path


def _save_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_bimanual_training_run(
    root: Path,
    run_id: str,
    result: BimanualTrainingResult,
    *,
    source_commit: str,
    overwrite: bool = False,
) -> Path:
    if not run_id or not source_commit:
        raise ValueError("training run and source commit identities are required")
    path = root / run_id
    path.mkdir(parents=True, exist_ok=overwrite)
    checkpoint_path = path / "training-checkpoint.pt"
    actor_path = path / "actor.pt"
    _save_torch(
        checkpoint_path,
        {
            "trainer": result.trainer.state_dict(),
            "replay": result.replay.state_dict(),
            "curriculum": result.curriculum.state_dict(),
            "records": [asdict(record) for record in result.records],
            "environment_steps": result.environment_steps,
            "numpy_rng_state": result.numpy_rng_state,
            "torch_rng_state": result.torch_rng_state,
        },
    )
    _save_torch(actor_path, result.trainer.actor.state_dict())
    episodes_path = path / "episodes.jsonl"
    _write_episode_records(episodes_path, result)
    _write_episode_records(path / "live-episodes.jsonl", result)
    lineage = {
        "schema_version": "hwr.no-demonstration-lineage/v1",
        "initialization": "random_actor",
        "action_label_sources": [],
        "expert_policies": [],
        "demonstration_datasets": [],
        "teleoperation_sessions": [],
        "behavior_cloning": False,
        "teacher_policy": False,
        "updates": "goal-conditioned asymmetric off-policy actor-critic",
        "hindsight_actor_weight": 0.0,
    }
    _write_json(path / "lineage.json", lineage)
    actor_audit = {
        "schema_version": "hwr.actor-input-audit/v1",
        "allowed_fields": sorted(VLA_POLICY_INPUT_FIELDS),
        "forbidden_fields": [
            "achieved_goal",
            "critic_state",
            "desired_goal",
            "object_token",
            "privileged_state",
            "skill_plan",
            "target_token",
            "task_stage",
        ],
        "preprocess_fingerprint": result.preprocess_fingerprint,
        "language_encoder_id": result.language_encoder.encoder_id,
        "language_weights_sha256": result.language_encoder.weights_sha256,
    }
    _write_json(path / "actor-input-audit.json", actor_audit)
    model_manifest = {
        "schema_version": "hwr.bimanual-actor/v1",
        "actor_config": result.actor_config.to_dict(),
        "rl_action_scaling": asdict(result.rl_config.action_scaling()),
        "actor_sha256": _sha256(actor_path),
        "deployable_only": True,
        "contains_critic": False,
    }
    _write_json(path / "model-manifest.json", model_manifest)
    replay_manifest = {
        "schema_version": "hwr.goal-replay/v1",
        "size": result.replay.size,
        "failure_size": result.replay.failure_size,
        "discovery_size": result.replay.discovery_size,
        "episode_count": result.replay.episode_count,
        "hindsight_transition_count": result.replay.hindsight_count,
        "mirror_transition_count": result.replay.mirror_count,
        "action_labels": False,
        "failure_return": True,
    }
    _write_json(path / "replay-manifest.json", replay_manifest)
    files = (
        checkpoint_path,
        actor_path,
        episodes_path,
        path / "lineage.json",
        path / "actor-input-audit.json",
        path / "model-manifest.json",
        path / "replay-manifest.json",
    )
    manifest = {
        "schema_version": BIMANUAL_RUN_SCHEMA,
        "run_id": run_id,
        "source_commit": source_commit,
        "training_config": result.config.to_dict(),
        "rl_config": result.rl_config.to_dict(),
        "record_count": len(result.records),
        "success_count": sum(record.success for record in result.records),
        "update_count": result.trainer.update_count,
        "artifacts": {
            item.name: {"sha256": _sha256(item), "bytes": item.stat().st_size}
            for item in files
        },
    }
    _write_json(path / "manifest.json", manifest)
    return path


def verify_bimanual_training_run(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BIMANUAL_RUN_SCHEMA:
        raise ValueError("bimanual training run schema differs")
    for filename, expected in manifest["artifacts"].items():
        artifact = path / filename
        if not artifact.is_file() or _sha256(artifact) != expected["sha256"]:
            raise ValueError(f"bimanual training artifact differs: {filename}")
    lineage = json.loads((path / "lineage.json").read_text(encoding="utf-8"))
    if any(
        lineage[name]
        for name in (
            "action_label_sources",
            "expert_policies",
            "demonstration_datasets",
            "teleoperation_sessions",
        )
    ):
        raise ValueError("prohibited action supervision entered training lineage")
    if lineage["behavior_cloning"] or lineage["teacher_policy"]:
        raise ValueError("prohibited teacher training entered lineage")
    return manifest


def load_bimanual_actor(path: Path, *, device: str = "cpu") -> VLAActorModel:
    manifest = json.loads(
        (path / "model-manifest.json").read_text(encoding="utf-8")
    )
    actor_path = path / "actor.pt"
    if _sha256(actor_path) != manifest["actor_sha256"]:
        raise ValueError("bimanual Actor checkpoint checksum differs")
    actor = VLAActorModel(VLAActorConfig(**manifest["actor_config"]))
    actor.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True))
    return actor.to(device).eval()


def resume_bimanual_training_run(
    path: Path,
    runner: BimanualTrainingRunner,
) -> None:
    """Verify and restore a run; only its total episode target may increase."""
    manifest = verify_bimanual_training_run(path)
    saved = dict(manifest["training_config"])
    requested = runner.config.to_dict()
    saved.setdefault(
        "discovery_replay_fraction", requested["discovery_replay_fraction"]
    )
    saved.pop("episodes")
    requested.pop("episodes")
    if saved != requested:
        raise ValueError("resume training configuration differs")
    checkpoint = torch.load(
        path / "training-checkpoint.pt",
        map_location=runner.trainer.device,
        weights_only=False,
    )
    runner.load_training_state(checkpoint)
