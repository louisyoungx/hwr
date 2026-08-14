from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
    VisualTeacherTargets,
)
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_value import LatentValueModel
from hwr.train.foundation_batch import FoundationTrainingBatch
from hwr.train.foundation_diagnostics import (
    evaluate_foundation_action_causality,
    evaluate_foundation_action_causality_audit,
)
from hwr.train.foundation_trainer import (
    FoundationTrainerConfig,
    FoundationWorldModelTrainer,
)
from hwr.train.foundation_visual_update import _slice_targets
from hwr.train.imagination_rl import ImaginationRLConfig
from hwr.train.intrinsic_exploration import (
    IntrinsicExplorationConfig,
    _episodic_knn_novelty,
)
from hwr.world_model import (
    ActionCausalityCriteria,
    ActionConditionedWorldModel,
    CounterfactualCausalityReport,
    CounterfactualComponentReport,
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
        torch.randn(sequences, observations - 1, 3),
        torch.randn(sequences, observations - 1),
        torch.ones(sequences, observations - 1),
        torch.zeros(sequences, observations - 1),
        torch.zeros(sequences, observations - 1),
    )


def _trainer(
    *,
    visual_microbatch_observations: int = 4,
    visual_update_interval: int = 4,
) -> FoundationWorldModelTrainer:
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
        IntrinsicExplorationConfig(
            horizon=3, value_bins=21, value_symlog_limit=5.0
        ),
        LatentActionScaling(),
        FoundationTrainerConfig(
            visual_microbatch_observations=visual_microbatch_observations,
            visual_update_interval=visual_update_interval,
        ),
    )


def test_unified_trainer_updates_visual_world_actor_and_value_together() -> None:
    trainer = _trainer()
    visual_before = trainer.visual_student.rgb_backbone.stem[0].weight.detach().clone()
    camera_fusion_before = {
        name: value.detach().clone()
        for name, value in trainer.visual_student.camera_fusion.named_parameters()
    }
    temporal_fusion_before = {
        name: value.detach().clone()
        for name, value in trainer.visual_student.temporal_fusion.named_parameters()
    }
    output_norm_before = trainer.visual_student.output_norm.weight.detach().clone()
    world_before = trainer.world_model.visual_head[-1].weight.detach().clone()
    actor_before = trainer.actor.mean_head.weight.detach().clone()
    value_before = trainer.value.network[-1].weight.detach().clone()

    metrics = trainer.train_step(_batch(_visual_config()))

    assert torch.any(trainer.visual_student.rgb_backbone.stem[0].weight != visual_before)
    assert all(
        parameter.grad is not None
        for parameter in trainer.visual_student.camera_fusion.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in trainer.visual_student.temporal_fusion.parameters()
    )
    assert any(
        torch.any(value != camera_fusion_before[name])
        for name, value in trainer.visual_student.camera_fusion.named_parameters()
    )
    assert any(
        torch.any(value != temporal_fusion_before[name])
        for name, value in trainer.visual_student.temporal_fusion.named_parameters()
    )
    assert torch.any(trainer.visual_student.output_norm.weight != output_norm_before)
    assert torch.any(trainer.world_model.visual_head[-1].weight != world_before)
    assert torch.any(trainer.actor.mean_head.weight != actor_before)
    assert torch.any(trainer.value.network[-1].weight != value_before)
    assert metrics["trainer/update_count"] == 1.0
    assert "visual/total" in metrics and "world/total" in metrics
    assert metrics["trainer/visual_camera_fusion_gradient_norm"] > 0.0
    assert metrics["trainer/visual_temporal_fusion_gradient_norm"] > 0.0
    assert metrics["trainer/visual_output_norm_gradient_norm"] > 0.0


def test_unified_trainer_optimizer_state_round_trip() -> None:
    first = _trainer()
    first.train_step(_batch(_visual_config()))
    state = first.optimizer_state_dict()
    second = _trainer()

    second.load_optimizer_state_dict(state)

    assert second.update_count == 1
    assert second.optimizer_state_dict()["update_count"] == 1


def test_intrinsic_explorer_updates_without_environment_reward_actor() -> None:
    trainer = _trainer()
    task_actor = trainer.actor.mean_head.weight.detach().clone()
    explorer = trainer.exploration_actor.mean_head.weight.detach().clone()

    metrics = trainer.train_step(
        _batch(_visual_config()),
        train_task_actor=False,
        train_exploration_actor=True,
    )

    torch.testing.assert_close(trainer.actor.mean_head.weight, task_actor)
    assert torch.any(trainer.exploration_actor.mean_head.weight != explorer)
    assert metrics["trainer/task_actor_updated"] == 0.0
    assert metrics["trainer/exploration_actor_updated"] == 1.0
    assert "exploration/state_novelty" in metrics


def test_actor_warmup_does_not_change_audited_world_model() -> None:
    trainer = _trainer()
    world_before = {
        name: value.detach().clone()
        for name, value in trainer.world_model.state_dict().items()
    }
    actor_before = trainer.actor.mean_head.weight.detach().clone()

    metrics = trainer.actor_warmup_step(
        _batch(_visual_config()), train_task_actor=True
    )

    assert trainer.update_count == 0
    assert torch.any(trainer.actor.mean_head.weight != actor_before)
    assert all(
        torch.equal(value, world_before[name])
        for name, value in trainer.world_model.state_dict().items()
    )
    assert "imagination/actor" in metrics


def test_knn_novelty_penalizes_return_to_a_previously_seen_state() -> None:
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    following = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    novelty = _episodic_knn_novelty(features, following, neighbors=1)

    assert novelty[0, 0] == pytest.approx(1.0)
    assert novelty[0, 1] == pytest.approx(0.0)


def test_unified_trainer_bounds_visual_activation_microbatches() -> None:
    trainer = _trainer(visual_microbatch_observations=2)
    observed: list[int] = []
    handle = trainer.visual_student.register_forward_pre_hook(
        lambda module, arguments: observed.append(arguments[0]["rgb"].shape[0])
    )

    metrics = trainer.train_step(_batch(_visual_config()))
    handle.remove()

    assert observed == [2, 1]
    assert metrics["trainer/visual_microbatch_count"] == 2.0


def test_visual_microbatch_rebases_only_internal_correspondences() -> None:
    targets = _batch(_visual_config()).visual_targets
    correspondences = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        ]
    )

    sliced = _slice_targets(replace(targets, correspondences=correspondences), 1, 3)

    assert sliced.vision_language.shape[0] == 2
    assert sliced.correspondences.tolist() == [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]


def test_foundation_trainer_rejects_empty_visual_microbatches() -> None:
    with pytest.raises(ValueError, match="rates and limits must be positive"):
        FoundationTrainerConfig(visual_microbatch_observations=0)


def test_visual_student_updates_at_fixed_task_independent_interval() -> None:
    trainer = _trainer(visual_update_interval=2)
    batch = _batch(_visual_config())
    first = trainer.train_step(batch, train_task_actor=False)
    visual_after_first = (
        trainer.visual_student.rgb_backbone.stem[0].weight.detach().clone()
    )
    world_before = trainer.world_model.visual_head[-1].weight.detach().clone()

    second = trainer.train_step(
        replace(batch, visual_targets=None), train_task_actor=False
    )

    torch.testing.assert_close(
        trainer.visual_student.rgb_backbone.stem[0].weight, visual_after_first
    )
    assert torch.any(trainer.world_model.visual_head[-1].weight != world_before)
    assert first["trainer/visual_updated"] == 1.0
    assert second["trainer/visual_updated"] == 0.0
    assert "visual/total" not in second


def test_foundation_diagnostic_uses_all_actual_outcome_targets() -> None:
    trainer = _trainer()
    batch = _batch(_visual_config())

    diagnostic = evaluate_foundation_action_causality(
        trainer, batch, ActionCausalityCriteria(1.05, 0.60)
    )

    assert diagnostic["action_source"] == "actual_executed_action"
    assert diagnostic["safety_action_source"] == "actor_proposal"
    assert diagnostic["counterfactual_pairing"] == "proposal-executed-pair/v1"
    assert diagnostic["report"]["error_components"] == (
        "visual_latent",
        "proprioception",
        "reward",
        "continue",
        "safety",
    )
    assert diagnostic["assessment"]["horizon_count"] == 2
    assert set(diagnostic["assessment"]["components"]) == {
        "visual_latent",
        "proprioception",
        "reward",
        "continue",
        "safety",
    }
    assert diagnostic["counterfactual_transform"] == (
        "deterministic-global-derangement/v1"
    )


def test_foundation_causality_audit_requires_every_task_partition(
    monkeypatch,
) -> None:
    passing = _synthetic_causality_report(1.2)
    failing = _synthetic_causality_report(0.9)
    monkeypatch.setattr(
        "hwr.train.foundation_diagnostics._evaluate_batch_reports",
        lambda trainer, batch, shuffle_seeds: (
            (passing if shuffle_seeds[0] < 19 else failing,),
            (passing,),
        ),
    )
    batch = _batch(_visual_config())

    diagnostic = evaluate_foundation_action_causality_audit(
        _trainer(),
        {"task-a/v1": (batch, batch), "task-b/v1": (batch,)},
        ActionCausalityCriteria(1.05, 0.60),
        shuffle_seed=17,
    )

    assert diagnostic["window_count"] == 3
    assert diagnostic["batch_count"] == 3
    assert diagnostic["assessment"]["aggregate_passed"] is True
    assert diagnostic["assessment"]["all_partitions_passed"] is False
    assert diagnostic["assessment"]["passed"] is False
    assert diagnostic["one_step_action_utilization"]["assessment"]["passed"] is True


def _synthetic_causality_report(ratio: float) -> CounterfactualCausalityReport:
    names = ("visual_latent", "proprioception", "reward", "continue", "safety")
    components = {
        name: CounterfactualComponentReport(
            0.2, 0.2 * ratio, ratio, (0.2, 0.2), (0.2 * ratio,) * 2
        )
        for name in names
    }
    return CounterfactualCausalityReport(
        1.0,
        ratio,
        ratio,
        (1.0, 1.0),
        (ratio, ratio),
        (0.1, 0.1),
        components,
        names,
    )
