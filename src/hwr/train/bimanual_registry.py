"""Auditable manifests and resumable checkpoints for no-demonstration training."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS
from hwr.policy.vla_model import VLAActorConfig, VLAActorModel
from hwr.train.asymmetric_replay import (
    COMPRESSIBLE_ACTOR_VISUAL_FIELDS,
    REPLAY_STORAGE_SCHEMA,
)
from hwr.train.bimanual_config import BimanualRLTrainingConfig
from hwr.train.bimanual_training import BimanualTrainingResult, BimanualTrainingRunner


BIMANUAL_RUN_SCHEMA = "hwr.bimanual-rl-run/v1"
ACTOR_INPUT_SHAPE_FIELDS = frozenset(
    {
        "raw_image_width",
        "raw_image_height",
        "image_width",
        "image_height",
        "point_count",
    }
)
FORKABLE_TRAINING_FIELDS = frozenset(
    {
        "episodes",
        "replay_capacity",
        "exploration_noise",
        "exploration_correlation",
        "action_smoothing",
        "gripper_exploration_probability",
        "gripper_exploration_hold_steps",
        "policy_gripper_hold_steps",
        "reflection_coupled_exploration_probability",
        "paired_gripper_exploration_probability",
        "global_random_burst_probability",
        "global_random_burst_steps",
        "actuator_dwell_probability",
        "actuator_dwell_steps",
        "actuator_initial_dwell_probability",
        "actuator_dwell_closed_probability",
        "frontier_reset_probability",
        "frontier_signature_uniform_fraction",
        "frontier_max_entries_per_source_signature",
        "task_sampling_temperature",
        "task_sampling_maximum_probability",
        "failure_replay_fraction",
        "discovery_replay_fraction",
        "progress_replay_fraction",
        "safety_replay_fraction",
        "visual_temporal_contrastive_weight",
        "augmentation_consistency_weight",
        *ACTOR_INPUT_SHAPE_FIELDS,
    }
)


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
    parent_training_run: Mapping[str, Any] | None = None,
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
            "frontier": result.frontier.state_dict(),
            "task_sampler": result.task_sampler.state_dict(),
            "records": [asdict(record) for record in result.records],
            "environment_steps": result.environment_steps,
            "task_rng_state": result.task_rng_state,
            "frontier_rng_state": result.frontier_rng_state,
            "exploration_rng_state": result.exploration_rng_state,
            "torch_rng_state": result.torch_rng_state,
        },
    )
    _save_torch(actor_path, result.trainer.actor.state_dict())
    episodes_path = path / "episodes.jsonl"
    _write_episode_records(episodes_path, result)
    _write_episode_records(path / "live-episodes.jsonl", result)
    lineage = {
        "schema_version": "hwr.no-demonstration-lineage/v1",
        "initialization": (
            "audited-no-demonstration-checkpoint"
            if parent_training_run
            else "random_actor"
        ),
        "action_label_sources": [],
        "expert_policies": [],
        "demonstration_datasets": [],
        "teleoperation_sessions": [],
        "behavior_cloning": False,
        "teacher_policy": False,
        "updates": "maximum-entropy asymmetric off-policy actor-critic",
        "actor_distribution": "reparameterized-squashed-gaussian",
        "safety_constraint": "privileged intervention cost critic",
        "initial_state_curriculum": (
            "autonomous-physical-frontier-resets-without-action-labels"
        ),
        "hindsight_goal_relabeling": False,
        "parent_training_run": (
            dict(parent_training_run) if parent_training_run else None
        ),
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
            "safety_cost",
            "safety_intervention",
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
        "schema_version": "hwr.autonomous-replay/v1",
        "storage": {
            "schema_version": REPLAY_STORAGE_SCHEMA,
            "compressed_actor_fields": sorted(
                COMPRESSIBLE_ACTOR_VISUAL_FIELDS
            ),
            "storage_dtype": "float16",
            "sample_compute_dtype": "float32",
        },
        "size": result.replay.size,
        "failure_size": result.replay.failure_size,
        "state_novelty_size": result.replay.discovery_size,
        "reward_improvement_size": result.replay.progress_size,
        "safety_event_size": result.replay.safety_size,
        "episode_count": result.replay.episode_count,
        "hindsight_enabled": False,
        "legacy_discarded_hindsight_transition_count": (
            result.replay.legacy_discarded_hindsight_count
        ),
        "legacy_discarded_reward_priority_transition_count": (
            result.replay.legacy_discarded_reward_priority_count
        ),
        "augmentation_eligible_transition_count": result.replay.augmentation_count,
        "stored_transform_copies": False,
        "sample_time_augmentation_probability": 0.50,
        "environment_augmentation_consistency": {
            "weight": result.config.augmentation_consistency_weight,
            "eligibility": "runtime-declared-legal-transform",
            "target": "same-actor-group-equivariance-without-action-labels",
            "actor_input_field": False,
            "privileged_actor_field": False,
        },
        "action_labels": False,
        "failure_return": True,
        "task_agnostic_priority_replay": {
            "state_novelty": "ranked-normalized-critic-state-change",
            "reward_improvement": (
                "ranked-positive-episode-local-reward-improvement-speed"
            ),
            "distance_thresholds": False,
            "task_semantic_fields": [],
            "action_labels": False,
        },
        "proposed_actions_for_safety_cost": True,
        "safety_cost_labels": (
            "deterministic_runtime_intervention_or_observed_severe_collision"
        ),
        "task_partition_sizes": result.replay.task_sizes(),
        "task_sampling": result.task_sampler.audit(),
        "frontier_curriculum": result.frontier.audit(),
        "action_exploration": dict(result.exploration_audit),
        "random_streams": {
            "schema_version": "hwr.independent-rng-streams/v1",
            "shared": False,
            "task_sampling": "independent",
            "frontier_selection": "independent",
            "exploration": ["random_actor", "temporal_action_exploration"],
        },
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
        "critic_config": result.trainer.critic_config.to_dict(),
        "record_count": len(result.records),
        "success_count": sum(record.success for record in result.records),
        "update_count": result.trainer.update_count,
        "artifacts": {
            item.name: {"sha256": _sha256(item), "bytes": item.stat().st_size}
            for item in files
        },
        "parent_training_run": (
            dict(parent_training_run) if parent_training_run else None
        ),
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
    critic_config = manifest.get("critic_config")
    if (
        critic_config is not None
        and critic_config != runner.trainer.critic_config.to_dict()
    ):
        raise ValueError("resume Critic architecture differs")
    requested = runner.config.to_dict()
    saved = dict(manifest["training_config"])
    saved.setdefault(
        "discovery_replay_fraction", requested["discovery_replay_fraction"]
    )
    saved.setdefault(
        "safety_replay_fraction", requested["safety_replay_fraction"]
    )
    saved.setdefault(
        "progress_replay_fraction", requested["progress_replay_fraction"]
    )
    if "augmentation_consistency_weight" not in saved:
        saved["augmentation_consistency_weight"] = saved.pop(
            "mirror_consistency_weight", 0.0
        )
    saved = _normalized_training_config(saved)
    for name in (
        "reflection_coupled_exploration_probability",
        "paired_gripper_exploration_probability",
        "global_random_burst_probability",
        "global_random_burst_steps",
        "actuator_dwell_probability",
        "actuator_dwell_steps",
        "actuator_initial_dwell_probability",
        "actuator_dwell_closed_probability",
        "policy_gripper_hold_steps",
        "frontier_reset_probability",
        "frontier_capacity_per_task",
        "frontier_signature_uniform_fraction",
        "frontier_max_entries_per_source_signature",
        "task_sampling_temperature",
        "task_sampling_maximum_probability",
    ):
        if name == "frontier_signature_uniform_fraction":
            legacy_default = 1.0
        elif name == "frontier_max_entries_per_source_signature":
            legacy_default = max(1, saved["frontier_capacity_per_task"] // 4)
        else:
            legacy_default = requested[name]
        saved.setdefault(name, legacy_default)
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


def fork_bimanual_training_run(
    path: Path,
    runner: BimanualTrainingRunner,
    *,
    reset_task_ids: Sequence[str] = (),
    discard_input_replay: bool = False,
    discard_replay: bool = False,
) -> dict[str, Any]:
    """Load an audited no-demonstration run with explicit exploration changes."""
    manifest = verify_bimanual_training_run(path)
    critic_config = manifest.get("critic_config")
    if critic_config != runner.trainer.critic_config.to_dict():
        raise ValueError("fork Critic architecture differs")
    saved = _normalized_training_config(manifest["training_config"])
    requested = runner.config.to_dict()
    changes = {
        name: {"parent": saved[name], "fork": requested[name]}
        for name in sorted(saved)
        if saved[name] != requested[name]
    }
    prohibited = sorted(set(changes) - FORKABLE_TRAINING_FIELDS)
    if prohibited:
        raise ValueError(
            "fork changes non-exploration training fields: "
            + ", ".join(prohibited)
        )
    input_shape_changes = sorted(set(changes) & ACTOR_INPUT_SHAPE_FIELDS)
    if input_shape_changes and not (discard_input_replay or discard_replay):
        raise ValueError(
            "fork changes Actor input shapes without --discard-input-replay: "
            + ", ".join(input_shape_changes)
        )
    if discard_input_replay and discard_replay:
        raise ValueError("replay discard modes are mutually exclusive")
    if (discard_input_replay or discard_replay) and reset_task_ids:
        raise ValueError(
            "replay discard and task-state reset require separate forks"
        )
    checkpoint_path = path / "training-checkpoint.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location=runner.trainer.device,
        weights_only=False,
    )
    runner.load_training_state(checkpoint)
    discarded_input_replay = _discard_actor_input_replay(
        runner,
        enabled=discard_input_replay,
        changed_fields=input_shape_changes,
    )
    discarded_replay = _discard_incompatible_replay(
        runner,
        enabled=discard_replay,
    )
    discarded_task_state = _discard_task_training_state(
        runner, reset_task_ids
    )
    return {
        "schema_version": "hwr.training-fork/v1",
        "parent_run_id": manifest["run_id"],
        "parent_source_commit": manifest["source_commit"],
        "parent_manifest_sha256": _sha256(path / "manifest.json"),
        "parent_checkpoint_sha256": _sha256(checkpoint_path),
        "fork_record_count": len(runner.records),
        "config_changes": changes,
        "discarded_actor_input_replay": discarded_input_replay,
        "discarded_replay": discarded_replay,
        "discarded_task_state": discarded_task_state,
        "replay_priority_migration": runner.replay.priority_migration_audit(),
        "inherited_action_labels": False,
        "inherited_expert_policies": False,
    }


def _discard_actor_input_replay(
    runner: BimanualTrainingRunner,
    *,
    enabled: bool,
    changed_fields: Sequence[str],
) -> dict[str, Any] | None:
    if not enabled:
        return None
    return {
        "schema_version": "hwr.fork-actor-input-replay-discard/v1",
        "reason": "actor_input_preprocessing_distribution_changed",
        "changed_fields": list(changed_fields),
        "replay": runner.replay.discard_tasks(runner.task_ids),
        "frontier": "inherited_physical_state_snapshots",
        "task_sampler": "inherited_outcome_history",
        "curriculum": "inherited_environment_outcomes",
        "shared_actor_critic_parameters": "inherited",
        "parent_episode_records": "retained_as_historical_lineage",
    }


def _discard_incompatible_replay(
    runner: BimanualTrainingRunner,
    *,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    return {
        "schema_version": "hwr.fork-replay-discard/v1",
        "reason": "replay_storage_semantics_changed",
        "replay": runner.replay.discard_tasks(runner.task_ids),
        "frontier": "inherited_but_runtime_reset_probability_controls_use",
        "task_sampler": "inherited_task_agnostic_outcome_history",
        "curriculum": "inherited_environment_outcomes",
        "shared_actor_critic_parameters": "inherited",
        "parent_episode_records": "retained_as_historical_lineage",
    }


def _discard_task_training_state(
    runner: BimanualTrainingRunner,
    task_ids: Sequence[str],
) -> dict[str, Any] | None:
    identities = tuple(dict.fromkeys(task_ids))
    if not identities:
        return None
    unknown = sorted(set(identities) - set(runner.task_ids))
    if unknown:
        raise ValueError(
            "fork cannot reset unknown tasks: " + ", ".join(unknown)
        )
    return {
        "schema_version": "hwr.fork-task-state-discard/v1",
        "task_ids": list(identities),
        "reason": "task_environment_or_asset_distribution_changed",
        "replay": runner.replay.discard_tasks(identities),
        "frontier": runner.frontier.discard_tasks(identities),
        "task_sampler": runner.task_sampler.discard_tasks(identities),
        "curriculum": runner.curriculum.discard_tasks(identities),
        "shared_actor_critic_parameters": "inherited",
        "parent_episode_records": "retained_as_historical_lineage",
    }


def _normalized_training_config(value: Mapping[str, Any]) -> dict[str, Any]:
    saved = dict(value)
    defaults = BimanualRLTrainingConfig().to_dict()
    for name, default in defaults.items():
        saved.setdefault(name, default)
    if "progress_replay_fraction" not in value:
        saved["progress_replay_fraction"] = 0.0
    if "frontier_signature_uniform_fraction" not in value:
        saved["frontier_signature_uniform_fraction"] = 1.0
    if "frontier_max_entries_per_source_signature" not in value:
        saved["frontier_max_entries_per_source_signature"] = max(
            1, int(saved["frontier_capacity_per_task"]) // 4
        )
    saved.pop("frontier_minimum_contact_stability_steps", None)
    saved.pop("frontier_reset_validation_steps", None)
    return saved
