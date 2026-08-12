from __future__ import annotations

import json

import pytest
import torch

from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
)
from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.policy.latent_value import LatentValueModel
from hwr.train.foundation_registry import (
    export_foundation_deployment,
    file_sha256,
    load_foundation_deployment,
    load_foundation_training_checkpoint,
    prune_versioned_artifacts,
    save_foundation_training_checkpoint,
)
from hwr.train.foundation_trainer import (
    FoundationTrainerConfig,
    FoundationWorldModelTrainer,
)
from hwr.train.imagination_rl import ImaginationRLConfig
from hwr.world_model import (
    ActionConditionedWorldModel,
    WorldModelConfig,
    WorldModelLoss,
    WorldModelLossConfig,
)


def _trainer() -> FoundationWorldModelTrainer:
    visual_config = VisualStudentConfig(
        image_size=32,
        visual_history=2,
        backbone_dimensions=(8, 12, 16, 24),
        backbone_depths=(1, 1, 1, 1),
        feature_dimension=8,
        state_queries=2,
        attention_heads=2,
        fusion_layers=1,
        temporal_layers=1,
        formal=False,
    )
    world_config = WorldModelConfig(
        visual_dimension=8,
        language_dimension=6,
        proprioception_dimension=5,
        action_dimension=16,
        observation_embedding_dimension=12,
        deterministic_dimension=10,
        stochastic_variables=3,
        stochastic_classes=4,
        hidden_dimension=16,
        prior_ensemble=2,
        reward_bins=11,
        formal=False,
    )
    student = VisualStudentModel(visual_config)
    objective = VisualFoundationObjectives(
        VisualObjectiveConfig(
            student_dimension=8, siglip_dimension=7, dinov2_dimension=5
        )
    )
    world = ActionConditionedWorldModel(world_config)
    actor_config = LatentActorConfig(
        world_config.feature_dimension,
        hidden_dimension=16,
        hidden_layers=2,
        formal=False,
    )
    actor = LatentActor(actor_config)
    value = LatentValueModel(
        world_config.feature_dimension, bins=11, hidden_dimension=16, hidden_layers=2
    )
    return FoundationWorldModelTrainer(
        student,
        objective,
        world,
        WorldModelLoss(world_config, WorldModelLossConfig()),
        actor,
        value,
        ImaginationRLConfig(horizon=3, value_bins=11, value_symlog_limit=5.0),
        FoundationTrainerConfig(),
    )


def test_training_checkpoint_restores_models_and_optimizer_metadata(tmp_path) -> None:
    trainer = _trainer()
    with torch.no_grad():
        trainer.actor.mean_head.bias.fill_(0.125)
    trainer.update_count = 9
    path = save_foundation_training_checkpoint(
        tmp_path / "training",
        trainer,
        source_commit="abc123",
        data_manifest_sha256="d" * 64,
    )
    with torch.no_grad():
        trainer.actor.mean_head.bias.zero_()
    trainer.update_count = 0

    manifest = load_foundation_training_checkpoint(path, trainer)

    torch.testing.assert_close(
        trainer.actor.mean_head.bias,
        torch.full_like(trainer.actor.mean_head.bias, 0.125),
    )
    assert trainer.update_count == 9
    assert manifest["lineage"]["expert_policies"] == []


def test_deployment_export_is_stripped_and_reloads_identically(tmp_path) -> None:
    trainer = _trainer()
    path = export_foundation_deployment(
        tmp_path / "deployment",
        trainer.visual_student,
        trainer.world_model,
        trainer.actor,
        LatentActionScaling(),
        source_commit="abc123",
        training_checkpoint_sha256="e" * 64,
        preprocessing={"fingerprint": "f" * 64},
        language_cache={"encoder_lock_sha256": "a" * 64, "dimension": 6},
    )

    loaded = load_foundation_deployment(path)

    assert loaded.manifest["deployable_only"] is True
    assert loaded.manifest["contains"] == [
        "visual_student", "state_filter", "actor"
    ]
    expected = trainer.actor.state_dict()
    actual = loaded.actor.state_dict()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
    deployed_names = {name for name, _ in loaded.state_filter.named_modules()}
    assert not any("reward" in name or "critic" in name for name in deployed_names)


def test_deployment_loader_rejects_artifact_hash_drift(tmp_path) -> None:
    trainer = _trainer()
    path = export_foundation_deployment(
        tmp_path / "deployment",
        trainer.visual_student,
        trainer.world_model,
        trainer.actor,
        LatentActionScaling(),
        source_commit="abc123",
        training_checkpoint_sha256="e" * 64,
        preprocessing={"fingerprint": "f" * 64},
        language_cache={"encoder_lock_sha256": "a" * 64},
    )
    artifact = path / "deployable-state.pt"
    artifact.write_bytes(artifact.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="hash verification"):
        load_foundation_deployment(path)


def test_manifest_hash_matches_deployment_artifact(tmp_path) -> None:
    trainer = _trainer()
    path = export_foundation_deployment(
        tmp_path / "deployment",
        trainer.visual_student,
        trainer.world_model,
        trainer.actor,
        LatentActionScaling(),
        source_commit="abc123",
        training_checkpoint_sha256="e" * 64,
        preprocessing={"fingerprint": "f" * 64},
        language_cache={"encoder_lock_sha256": "a" * 64},
    )
    manifest = json.loads((path / "manifest.json").read_text())

    assert manifest["artifact_sha256"] == file_sha256(
        path / manifest["artifact_file"]
    )


def test_versioned_artifact_retention_removes_only_old_update_directories(tmp_path) -> None:
    root = tmp_path / "checkpoints"
    for update in (1, 2, 10, 20):
        path = root / f"update-{update:09d}"
        path.mkdir(parents=True)
        (path / "state").write_text(str(update))
    unrelated = root / "manual-note"
    unrelated.mkdir()

    removed = prune_versioned_artifacts(root, retain=2)

    assert [path.name for path in removed] == ["update-000000001", "update-000000002"]
    assert sorted(path.name for path in root.iterdir()) == [
        "manual-note",
        "update-000000010",
        "update-000000020",
    ]
