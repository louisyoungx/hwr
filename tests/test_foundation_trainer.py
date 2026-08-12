from __future__ import annotations

import torch

from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
    VisualTeacherTargets,
)
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.policy.latent_value import LatentValueModel
from hwr.train.foundation_batch import FoundationTrainingBatch
from hwr.train.foundation_diagnostics import evaluate_foundation_action_causality
from hwr.train.foundation_trainer import (
    FoundationTrainerConfig,
    FoundationWorldModelTrainer,
)
from hwr.train.imagination_rl import ImaginationRLConfig
from hwr.world_model import (
    ActionCausalityCriteria,
    ActionConditionedWorldModel,
    WorldModelConfig,
    WorldModelLoss,
    WorldModelLossConfig,
)


def _visual_config() -> VisualStudentConfig:
    return VisualStudentConfig(
        image_size=32,
        visual_history=2,
        backbone_dimensions=(16, 24, 32, 48),
        backbone_depths=(1, 1, 1, 1),
        feature_dimension=16,
        state_queries=2,
        attention_heads=4,
        fusion_layers=1,
        temporal_layers=1,
        formal=False,
    )


def _world_config() -> WorldModelConfig:
    return WorldModelConfig(
        visual_dimension=16,
        language_dimension=6,
        proprioception_dimension=5,
        action_dimension=3,
        observation_embedding_dimension=16,
        deterministic_dimension=16,
        stochastic_variables=4,
        stochastic_classes=4,
        hidden_dimension=32,
        prior_ensemble=3,
        reward_bins=21,
        formal=False,
    )


def _batch(visual: VisualStudentConfig) -> FoundationTrainingBatch:
    sequences, observations = 1, 3
    flattened = sequences * observations
    history, size = visual.visual_history, visual.image_size
    student_inputs = {
        "rgb": torch.rand(flattened, history, 3, 3, size, size),
        "head_depth_m": torch.rand(flattened, history, 1, size, size) + 0.2,
        "head_depth_valid": torch.ones(flattened, history, 1, size, size, dtype=torch.bool),
        "camera_validity": torch.ones(flattened, history, 4, dtype=torch.bool),
        "intrinsics": torch.ones(flattened, history, 4, 4),
        "robot_from_camera": torch.eye(4).reshape(1, 1, 1, 4, 4).expand(flattened, history, 4, 4, 4).clone(),
        "repeated_frame": torch.zeros(flattened, history, dtype=torch.bool),
    }
    targets = VisualTeacherTargets(
        vision_language=torch.randn(flattened, history, 3, 3, 3, 12),
        vision_language_valid=torch.ones(
            flattened, history, 3, 3, 3, dtype=torch.bool
        ),
        dense_vision=torch.randn(flattened, history, 3, 4, 4, 10),
        dense_vision_valid=torch.ones(
            flattened, history, 3, 4, 4, dtype=torch.bool
        ),
        rgb=torch.rand(flattened, history, 3, 3, size, size),
        reconstruction_mask=torch.ones(flattened, history, 3, 1, size, size, dtype=torch.bool),
        head_depth_m=torch.ones(flattened, history, 1, size, size),
        head_depth_valid=torch.ones(flattened, history, 1, size, size, dtype=torch.bool),
        correspondences=torch.empty((0, 10), dtype=torch.long),
    )
    return FoundationTrainingBatch(
        student_inputs,
        targets,
        sequences,
        observations,
        torch.randn(sequences, 6),
        torch.randn(sequences, observations, 5),
        torch.randn(sequences, observations - 1, 3),
        torch.randn(sequences, observations - 1),
        torch.ones(sequences, observations - 1),
        torch.zeros(sequences, observations - 1),
    )


def _trainer() -> FoundationWorldModelTrainer:
    visual_config = _visual_config()
    world_config = _world_config()
    student = VisualStudentModel(visual_config)
    visual_objective = VisualFoundationObjectives(
        VisualObjectiveConfig(
            student_dimension=16,
            vision_language_dimension=12,
            dense_vision_dimension=10,
        )
    )
    world = ActionConditionedWorldModel(world_config)
    actor = LatentActor(
        LatentActorConfig(
            world_config.feature_dimension,
            action_dimension=3,
            hidden_dimension=32,
            hidden_layers=2,
            formal=False,
        )
    )
    value = LatentValueModel(
        world_config.feature_dimension, bins=21, hidden_dimension=32, hidden_layers=2
    )
    return FoundationWorldModelTrainer(
        student,
        visual_objective,
        world,
        WorldModelLoss(world_config, WorldModelLossConfig()),
        actor,
        value,
        ImaginationRLConfig(horizon=3, value_bins=21, value_symlog_limit=5.0),
        FoundationTrainerConfig(),
    )


def test_unified_trainer_updates_visual_world_actor_and_value_together() -> None:
    trainer = _trainer()
    visual_before = trainer.visual_student.rgb_backbone.stem[0].weight.detach().clone()
    world_before = trainer.world_model.visual_head[-1].weight.detach().clone()
    actor_before = trainer.actor.mean_head.weight.detach().clone()
    value_before = trainer.value.network[-1].weight.detach().clone()

    metrics = trainer.train_step(_batch(_visual_config()))

    assert torch.any(trainer.visual_student.rgb_backbone.stem[0].weight != visual_before)
    assert torch.any(trainer.world_model.visual_head[-1].weight != world_before)
    assert torch.any(trainer.actor.mean_head.weight != actor_before)
    assert torch.any(trainer.value.network[-1].weight != value_before)
    assert metrics["trainer/update_count"] == 1.0
    assert "visual/total" in metrics and "world/total" in metrics


def test_unified_trainer_optimizer_state_round_trip() -> None:
    first = _trainer()
    first.train_step(_batch(_visual_config()))
    state = first.optimizer_state_dict()
    second = _trainer()

    second.load_optimizer_state_dict(state)

    assert second.update_count == 1
    assert second.optimizer_state_dict()["update_count"] == 1


def test_foundation_diagnostic_uses_all_actual_outcome_targets() -> None:
    trainer = _trainer()
    batch = _batch(_visual_config())

    diagnostic = evaluate_foundation_action_causality(
        trainer, batch, ActionCausalityCriteria(1.05, 0.60)
    )

    assert diagnostic["action_source"] == "actual_executed_action"
    assert diagnostic["report"]["error_components"] == (
        "visual_latent",
        "proprioception",
        "reward",
        "continue",
        "safety",
    )
    assert diagnostic["assessment"]["horizon_count"] == 2
