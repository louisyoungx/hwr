"""Atomic training checkpoints and stripped deployment exports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.train.foundation_trainer import FoundationWorldModelTrainer
from hwr.world_model.deploy import DeployableWorldModelStateFilter
from hwr.world_model.model import ActionConditionedWorldModel


TRAINING_CHECKPOINT_SCHEMA = "hwr.foundation-training-checkpoint/v1"
DEPLOYMENT_SCHEMA = "hwr.foundation-deployment/v1"
ACTION_CAUSALITY_SCHEMA = "hwr.foundation-action-causality/v5"
_FORBIDDEN_DEPLOYMENT_NAMES = frozenset(
    {
        "continue_head",
        "critic",
        "reward_head",
        "safety_head",
        "teacher",
        "value",
        "visual_head",
    }
)


@dataclass(frozen=True)
class FoundationDeploymentComponents:
    visual_student: VisualStudentModel
    state_filter: DeployableWorldModelStateFilter
    actor: LatentActor
    action_scaling: LatentActionScaling
    manifest: Mapping[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_versioned_artifacts(root: Path, retain: int) -> tuple[Path, ...]:
    """Retain the newest immutable update directories under one artifact root."""
    if retain <= 0:
        raise ValueError("foundation artifact retention must be positive")
    if not root.exists():
        return ()
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and path.name.startswith("update-")
        and path.name.removeprefix("update-").isdigit()
    )
    removed = candidates[:-retain]
    resolved_root = root.resolve()
    for path in removed:
        resolved = path.resolve()
        if resolved.parent != resolved_root:
            raise ValueError("foundation artifact path escaped its version root")
        shutil.rmtree(resolved)
    return tuple(removed)


def _atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_verified_manifest(path: Path, schema: str) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != schema:
        raise ValueError("foundation artifact schema is invalid")
    artifact = path / str(manifest["artifact_file"])
    if file_sha256(artifact) != manifest.get("artifact_sha256"):
        raise ValueError("foundation artifact hash verification failed")
    return manifest


def _json_compatible(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def foundation_lineage(source_commit: str) -> dict[str, object]:
    """Return the one exact no-expert lineage accepted by formal training."""
    if not source_commit:
        raise ValueError("foundation checkpoint source commit is required")
    return {
        "source_commit": source_commit,
        "initialization": "random_project_owned_models",
        "action_sources": ["random_rl_exploration", "rl_actor"],
        "expert_policies": [],
        "demonstration_datasets": [],
        "behavior_cloning": False,
        "teacher_actions": False,
        "action_search": False,
        "legacy_p_series_parent": None,
    }


def require_foundation_lineage(
    value: object, *, source_commit: str
) -> dict[str, object]:
    expected = foundation_lineage(source_commit)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("foundation no-expert lineage differs")
    return expected


def save_foundation_training_checkpoint(
    path: Path,
    trainer: FoundationWorldModelTrainer,
    *,
    source_commit: str,
    data_manifest_sha256: str,
    training_diagnostics: Mapping[str, object],
) -> Path:
    """Save all trainable state; this artifact is never loadable by deployment."""
    if len(data_manifest_sha256) != 64:
        raise ValueError("foundation data manifest requires a SHA-256 identity")
    _validate_training_diagnostics(training_diagnostics)
    path.mkdir(parents=True, exist_ok=False)
    artifact_path = path / "training-state.pt"
    state = {
        "visual_student": trainer.visual_student.state_dict(),
        "visual_objective": trainer.visual_objective.state_dict(),
        "world_model": trainer.world_model.state_dict(),
        "actor": trainer.actor.state_dict(),
        "value": trainer.value.state_dict(),
        "optimizers": trainer.optimizer_state_dict(),
    }
    _atomic_torch_save(artifact_path, state)
    manifest = {
        "schema_version": TRAINING_CHECKPOINT_SCHEMA,
        "artifact_file": artifact_path.name,
        "artifact_sha256": file_sha256(artifact_path),
        "data_manifest_sha256": data_manifest_sha256,
        "training_diagnostics": dict(training_diagnostics),
        "update_count": trainer.update_count,
        "configs": {
            "visual_student": trainer.visual_student.config.to_dict(),
            "visual_objective": trainer.visual_objective.config.to_dict(),
            "world_model": trainer.world_model.config.to_dict(),
            "world_objective": trainer.world_objective.loss_config.to_dict(),
            "actor": trainer.actor.config.to_dict(),
            "imagination": trainer.imagination.config.to_dict(),
            "trainer": trainer.config.to_dict(),
        },
        "lineage": foundation_lineage(source_commit),
        "deployable": False,
    }
    _atomic_json(path / "manifest.json", manifest)
    return path


def load_foundation_training_checkpoint(
    path: Path, trainer: FoundationWorldModelTrainer
) -> Mapping[str, Any]:
    manifest = _read_verified_manifest(path, TRAINING_CHECKPOINT_SCHEMA)
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("foundation training checkpoint lineage is missing")
    require_foundation_lineage(
        lineage, source_commit=str(lineage.get("source_commit", ""))
    )
    expected = {
        "visual_student": trainer.visual_student.config.to_dict(),
        "visual_objective": trainer.visual_objective.config.to_dict(),
        "world_model": trainer.world_model.config.to_dict(),
        "world_objective": trainer.world_objective.loss_config.to_dict(),
        "actor": trainer.actor.config.to_dict(),
        "imagination": trainer.imagination.config.to_dict(),
        "trainer": trainer.config.to_dict(),
    }
    if manifest.get("configs") != _json_compatible(expected):
        raise ValueError("foundation training checkpoint configuration differs")
    state = torch.load(
        path / str(manifest["artifact_file"]), map_location="cpu", weights_only=True
    )
    trainer.visual_student.load_state_dict(state["visual_student"])
    trainer.visual_objective.load_state_dict(state["visual_objective"])
    trainer.world_model.load_state_dict(state["world_model"])
    trainer.actor.load_state_dict(state["actor"])
    trainer.value.load_state_dict(state["value"])
    trainer.load_optimizer_state_dict(state["optimizers"])
    return manifest


def _deployment_state(
    visual_student: VisualStudentModel,
    world_model: ActionConditionedWorldModel,
    actor: LatentActor,
) -> tuple[dict[str, object], DeployableWorldModelStateFilter]:
    state_filter = DeployableWorldModelStateFilter.from_world_model(world_model)
    state = {
        "visual_student": visual_student.state_dict(),
        "state_filter": state_filter.state_dict(),
        "actor": actor.state_dict(),
    }
    names = {name.lower() for component in state.values() for name in component}
    violations = sorted(
        name
        for name in names
        if any(forbidden in name for forbidden in _FORBIDDEN_DEPLOYMENT_NAMES)
    )
    if violations:
        raise ValueError(f"deployment state contains training-only weights: {violations}")
    return state, state_filter


def export_foundation_deployment(
    path: Path,
    visual_student: VisualStudentModel,
    world_model: ActionConditionedWorldModel,
    actor: LatentActor,
    action_scaling: LatentActionScaling,
    *,
    source_commit: str,
    training_checkpoint_sha256: str,
    training_diagnostics: Mapping[str, object],
    preprocessing: Mapping[str, Any],
    language_cache: Mapping[str, Any],
) -> Path:
    if not source_commit:
        raise ValueError("deployment source commit is required")
    if len(training_checkpoint_sha256) != 64:
        raise ValueError("deployment requires its training checkpoint SHA-256")
    _validate_training_diagnostics(training_diagnostics, require_passed=True)
    if not preprocessing or not language_cache:
        raise ValueError("deployment preprocessing and language cache are required")
    path.mkdir(parents=True, exist_ok=False)
    state, _ = _deployment_state(visual_student, world_model, actor)
    artifact_path = path / "deployable-state.pt"
    _atomic_torch_save(artifact_path, state)
    manifest = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "artifact_file": artifact_path.name,
        "artifact_sha256": file_sha256(artifact_path),
        "training_checkpoint_sha256": training_checkpoint_sha256,
        "training_diagnostics": dict(training_diagnostics),
        "source_commit": source_commit,
        "visual_student_config": visual_student.config.to_dict(),
        "world_model_config": world_model.config.to_dict(),
        "actor_config": actor.config.to_dict(),
        "action_scaling": action_scaling.to_dict(),
        "preprocessing": dict(preprocessing),
        "language_cache": dict(language_cache),
        "contains": ["visual_student", "state_filter", "actor"],
        "forbidden_components": sorted(_FORBIDDEN_DEPLOYMENT_NAMES),
        "deployable_only": True,
    }
    _atomic_json(path / "manifest.json", manifest)
    return path


def _validate_training_diagnostics(
    value: Mapping[str, object], *, require_passed: bool = False
) -> None:
    expected = {
        "action_causality_report_sha256",
        "action_causality_passed",
        "actor_readiness_unlocked",
        "task_actor_update_count",
    }
    if set(value) != expected:
        raise ValueError("foundation training diagnostic fields differ")
    digest = str(value["action_causality_report_sha256"])
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("action causality report requires a SHA-256 identity")
    if not isinstance(value["action_causality_passed"], bool):
        raise ValueError("action causality pass state must be boolean")
    if not isinstance(value["actor_readiness_unlocked"], bool):
        raise ValueError("Actor readiness state must be boolean")
    updates = value["task_actor_update_count"]
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
        raise ValueError("task Actor update count must be a non-negative integer")
    if require_passed and (
        value["action_causality_passed"] is not True
        or value["actor_readiness_unlocked"] is not True
        or updates <= 0
    ):
        raise ValueError("deployment requires causal and trained Actor evidence")


def foundation_deployment_qualified(value: Mapping[str, object]) -> bool:
    _validate_training_diagnostics(value)
    return bool(
        value["action_causality_passed"] is True
        and value["actor_readiness_unlocked"] is True
        and int(value["task_actor_update_count"]) > 0
    )


def load_foundation_deployment(
    path: Path, *, device: str = "cpu"
) -> FoundationDeploymentComponents:
    manifest = _read_verified_manifest(path, DEPLOYMENT_SCHEMA)
    if manifest.get("contains") != ["visual_student", "state_filter", "actor"]:
        raise ValueError("deployment component whitelist differs")
    visual_values = dict(manifest["visual_student_config"])
    visual_values["backbone_dimensions"] = tuple(visual_values["backbone_dimensions"])
    visual_values["backbone_depths"] = tuple(visual_values["backbone_depths"])
    visual_config = VisualStudentConfig(**visual_values)
    from hwr.world_model.config import WorldModelConfig

    world_config = WorldModelConfig(**manifest["world_model_config"])
    actor_config = LatentActorConfig(**manifest["actor_config"])
    visual_student = VisualStudentModel(visual_config)
    state_filter = DeployableWorldModelStateFilter(world_config)
    actor = LatentActor(actor_config)
    state = torch.load(
        path / str(manifest["artifact_file"]), map_location=device, weights_only=True
    )
    if set(state) != {"visual_student", "state_filter", "actor"}:
        raise ValueError("deployment state component whitelist differs")
    visual_student.load_state_dict(state["visual_student"])
    state_filter.load_state_dict(state["state_filter"])
    actor.load_state_dict(state["actor"])
    scaling = LatentActionScaling(**manifest["action_scaling"])
    return FoundationDeploymentComponents(
        visual_student.to(device).eval(),
        state_filter.to(device).eval(),
        actor.to(device).eval(),
        scaling,
        manifest,
    )
