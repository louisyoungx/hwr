"""Construct the formal project-owned learning stack from versioned configs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
)
from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.policy.latent_value import LatentValueModel
from hwr.train.foundation_trainer import (
    FoundationTrainerConfig,
    FoundationWorldModelTrainer,
)
from hwr.train.imagination_rl import ImaginationRLConfig
from hwr.train.intrinsic_exploration import IntrinsicExplorationConfig
from hwr.world_model.config import WorldModelConfig
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.objectives import WorldModelLoss, WorldModelLossConfig


@dataclass(frozen=True)
class FoundationLearningStack:
    trainer: FoundationWorldModelTrainer
    action_scaling: LatentActionScaling


def _config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not str(value.pop("schema_version", "")).startswith("hwr."):
        raise ValueError(f"foundation config has no schema identity: {path}")
    return value


def build_foundation_learning_stack(
    config_root: Path, *, device: str = "cpu", seed: int
) -> FoundationLearningStack:
    if seed < 0:
        raise ValueError("foundation initialization seed cannot be negative")
    torch.manual_seed(seed)
    visual_values = _config(config_root / "visual-student-v1.json")
    visual_values["backbone_dimensions"] = tuple(visual_values["backbone_dimensions"])
    visual_values["backbone_depths"] = tuple(visual_values["backbone_depths"])
    visual_config = VisualStudentConfig(**visual_values)
    world_config = WorldModelConfig(**_config(config_root / "world-model-v1.json"))
    actor_config = LatentActorConfig(**_config(config_root / "latent-actor-v1.json"))
    visual_objective_config = VisualObjectiveConfig(
        **_config(config_root / "visual-objective-v1.json")
    )
    world_objective_config = WorldModelLossConfig(
        **_config(config_root / "world-objective-v1.json")
    )
    imagination_config = ImaginationRLConfig(
        **_config(config_root / "imagination-rl-v1.json")
    )
    intrinsic_config = IntrinsicExplorationConfig(
        **_config(config_root / "intrinsic-exploration-v1.json")
    )
    trainer_config = FoundationTrainerConfig(
        **_config(config_root / "unified-trainer-v1.json")
    )
    if actor_config.latent_dimension != world_config.feature_dimension:
        raise ValueError("formal Actor latent dimension differs from world model")
    scaling = LatentActionScaling(
        imagination_config.base_linear_scale,
        imagination_config.base_angular_scale,
        imagination_config.arm_velocity_scale,
    )
    expected_bounds = (
        (-scaling.base_linear, -scaling.base_angular, *(-scaling.arm_velocity,) * 12, 0.0, 0.0),
        (scaling.base_linear, scaling.base_angular, *(scaling.arm_velocity,) * 12, 1.0, 1.0),
    )
    if (
        world_config.action_minimum != expected_bounds[0]
        or world_config.action_maximum != expected_bounds[1]
    ):
        raise ValueError("world model action bounds differ from runtime scaling")
    student = VisualStudentModel(visual_config).to(device)
    visual_objective = VisualFoundationObjectives(visual_objective_config).to(device)
    world = ActionConditionedWorldModel(world_config).to(device)
    actor = LatentActor(actor_config).to(device)
    value = LatentValueModel(
        world_config.feature_dimension,
        bins=imagination_config.value_bins,
        hidden_dimension=actor_config.hidden_dimension,
        hidden_layers=actor_config.hidden_layers,
    ).to(device)
    trainer = FoundationWorldModelTrainer(
        student,
        visual_objective,
        world,
        WorldModelLoss(world_config, world_objective_config).to(device),
        actor,
        value,
        imagination_config,
        intrinsic_config,
        scaling,
        trainer_config,
    )
    return FoundationLearningStack(trainer, scaling)
