"""Unified representation, world-model, and imagination RL optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from hwr.perception.student import VisualStudentModel
from hwr.perception.student_objectives import VisualFoundationObjectives
from hwr.policy.latent_actor import LatentActor
from hwr.policy.latent_value import LatentValueModel
from hwr.train.foundation_batch import FoundationTrainingBatch
from hwr.train.foundation_visual_update import optimize_visual_student
from hwr.train.imagination_rl import (
    ImaginationActorCritic,
    ImaginationRLConfig,
    optimize_imagination_step,
)
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.objectives import (
    WorldModelLoss,
    WorldModelTargets,
)
from hwr.world_model.rssm import RSSMState


@dataclass(frozen=True)
class FoundationTrainerConfig:
    visual_learning_rate: float = 1.0e-4
    world_model_learning_rate: float = 1.0e-4
    actor_learning_rate: float = 3.0e-5
    value_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    maximum_gradient_norm: float = 100.0
    visual_microbatch_observations: int = 4

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if min(values) <= 0.0:
            raise ValueError("foundation trainer rates and limits must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FoundationWorldModelTrainer:
    """Update every learned component from the same autonomous sequence batch."""

    def __init__(
        self,
        visual_student: VisualStudentModel,
        visual_objective: VisualFoundationObjectives,
        world_model: ActionConditionedWorldModel,
        world_objective: WorldModelLoss,
        actor: LatentActor,
        value: LatentValueModel,
        imagination_config: ImaginationRLConfig,
        trainer_config: FoundationTrainerConfig,
    ) -> None:
        if visual_student.config.feature_dimension != world_model.config.visual_dimension:
            raise ValueError("visual student and world model dimensions differ")
        if actor.config.latent_dimension != world_model.config.feature_dimension:
            raise ValueError("latent Actor and world model dimensions differ")
        self.visual_student = visual_student
        self.visual_objective = visual_objective
        self.world_model = world_model
        self.world_objective = world_objective
        self.actor = actor
        self.value = value
        self.imagination = ImaginationActorCritic(
            world_model, actor, value, imagination_config
        )
        self.config = trainer_config
        visual_parameters = [
            *visual_student.parameters(),
            *visual_objective.parameters(),
        ]
        self.visual_optimizer = torch.optim.AdamW(
            visual_parameters,
            lr=trainer_config.visual_learning_rate,
            weight_decay=trainer_config.weight_decay,
        )
        self.world_optimizer = torch.optim.AdamW(
            world_model.parameters(),
            lr=trainer_config.world_model_learning_rate,
            weight_decay=trainer_config.weight_decay,
        )
        self.actor_optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=trainer_config.actor_learning_rate,
            weight_decay=trainer_config.weight_decay,
        )
        self.value_optimizer = torch.optim.AdamW(
            value.parameters(),
            lr=trainer_config.value_learning_rate,
            weight_decay=trainer_config.weight_decay,
        )
        self.update_count = 0

    def train_step(self, batch: FoundationTrainingBatch) -> dict[str, float]:
        self._check_batch_dimensions(batch)
        self.visual_student.train()
        self.world_model.train()
        visual_update = optimize_visual_student(
            self.visual_student,
            self.visual_objective,
            batch,
            self.visual_optimizer,
            microbatch_observations=self.config.visual_microbatch_observations,
            maximum_gradient_norm=self.config.maximum_gradient_norm,
        )
        visual_sequence = visual_update.pooled_state.reshape(
            batch.sequence_batch_size,
            batch.observation_count,
            self.world_model.config.visual_dimension,
        )
        world_output = self.world_model.observe(
            visual_sequence,
            batch.language_features,
            batch.proprioception,
            batch.actor_proposals,
            batch.executed_actions,
        )
        world_targets = WorldModelTargets(
            visual_sequence,
            batch.proprioception,
            batch.rewards,
            batch.continues,
            batch.safety_interventions,
        )
        world_losses = self.world_objective(world_output, world_targets)
        self.world_optimizer.zero_grad(set_to_none=True)
        world_losses["total"].backward()
        nn.utils.clip_grad_norm_(
            self.world_model.parameters(), self.config.maximum_gradient_norm
        )
        self.world_optimizer.step()

        initial = RSSMState(
            world_output.sequence.deterministic.detach().flatten(0, 1),
            world_output.sequence.stochastic.detach().flatten(0, 1),
        )
        imagination_metrics = optimize_imagination_step(
            self.imagination,
            initial,
            self.actor_optimizer,
            self.value_optimizer,
        )
        self.update_count += 1
        metrics = {
            **{f"visual/{name}": value for name, value in visual_update.losses.items()},
            **{f"world/{name}": float(value.detach()) for name, value in world_losses.items()},
            **{f"imagination/{name}": value for name, value in imagination_metrics.items()},
            "trainer/visual_microbatch_count": float(visual_update.microbatch_count),
            "trainer/update_count": float(self.update_count),
        }
        return metrics

    def _check_batch_dimensions(self, batch: FoundationTrainingBatch) -> None:
        config = self.world_model.config
        expected = {
            "language": config.language_dimension,
            "proprioception": config.proprioception_dimension,
            "action": config.action_dimension,
        }
        actual = {
            "language": batch.language_features.shape[-1],
            "proprioception": batch.proprioception.shape[-1],
            "action": batch.executed_actions.shape[-1],
        }
        if batch.actor_proposals.shape != batch.executed_actions.shape:
            raise ValueError("foundation proposal and executed action shapes differ")
        if expected != actual:
            raise ValueError(
                f"foundation trainer batch dimensions differ: {actual} != {expected}"
            )

    def optimizer_state_dict(self) -> dict[str, object]:
        return {
            "visual": self.visual_optimizer.state_dict(),
            "world_model": self.world_optimizer.state_dict(),
            "actor": self.actor_optimizer.state_dict(),
            "value": self.value_optimizer.state_dict(),
            "slow_value": self.imagination.slow_value.state_dict(),
            "update_count": self.update_count,
        }

    def load_optimizer_state_dict(self, state: dict[str, object]) -> None:
        self.visual_optimizer.load_state_dict(state["visual"])
        self.world_optimizer.load_state_dict(state["world_model"])
        self.actor_optimizer.load_state_dict(state["actor"])
        self.value_optimizer.load_state_dict(state["value"])
        self.imagination.slow_value.load_state_dict(state["slow_value"])
        self.update_count = int(state["update_count"])
