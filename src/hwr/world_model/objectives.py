"""World model reconstruction, dynamics, and outcome prediction objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hwr.world_model.config import WorldModelConfig
from hwr.world_model.distributions import two_hot_symlog
from hwr.world_model.model import WorldModelOutput


@dataclass(frozen=True)
class WorldModelLossConfig:
    visual_weight: float = 1.0
    proprioception_weight: float = 1.0
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    safety_weight: float = 1.0
    dynamics_weight: float = 0.5
    representation_weight: float = 0.1
    ensemble_weight: float = 0.5
    free_nats: float = 1.0

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if min(values) < 0.0:
            raise ValueError("world model loss weights cannot be negative")


@dataclass(frozen=True)
class WorldModelTargets:
    visual: torch.Tensor
    proprioception: torch.Tensor
    reward: torch.Tensor
    continues: torch.Tensor
    safety: torch.Tensor


class WorldModelLoss(nn.Module):
    def __init__(
        self, model_config: WorldModelConfig, loss_config: WorldModelLossConfig
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config

    def forward(
        self, output: WorldModelOutput, targets: WorldModelTargets
    ) -> dict[str, torch.Tensor]:
        self._check_shapes(output, targets)
        visual = nn.functional.mse_loss(output.visual_prediction, targets.visual)
        visual += (
            1.0
            - nn.functional.cosine_similarity(
                output.visual_prediction, targets.visual, dim=-1
            ).mean()
        )
        proprioception = nn.functional.mse_loss(
            output.proprioception_prediction, targets.proprioception
        )
        reward_target = two_hot_symlog(
            targets.reward,
            bins=self.model_config.reward_bins,
            limit=self.model_config.reward_symlog_limit,
        )
        reward = -(reward_target * output.reward_logits[:, 1:].log_softmax(dim=-1)).sum(
            dim=-1
        ).mean()
        continues = nn.functional.binary_cross_entropy_with_logits(
            output.continue_logits[:, 1:], targets.continues.float()
        )
        safety = nn.functional.binary_cross_entropy_with_logits(
            output.safety_logits[:, 1:], targets.safety.float()
        )
        dynamics, representation = self._balanced_kl(output)
        ensemble = self._ensemble_kl(output)
        losses = {
            "visual": visual,
            "proprioception": proprioception,
            "reward": reward,
            "continue": continues,
            "safety": safety,
            "dynamics": dynamics,
            "representation": representation,
            "ensemble": ensemble,
        }
        config = self.loss_config
        losses["total"] = (
            config.visual_weight * visual
            + config.proprioception_weight * proprioception
            + config.reward_weight * reward
            + config.continue_weight * continues
            + config.safety_weight * safety
            + config.dynamics_weight * dynamics
            + config.representation_weight * representation
            + config.ensemble_weight * ensemble
        )
        return losses

    def _balanced_kl(self, output: WorldModelOutput) -> tuple[torch.Tensor, torch.Tensor]:
        posterior_logits = output.sequence.posterior_logits[:, 1:]
        prior_logits = output.sequence.prior_logits[:, 1:]
        dynamics = _categorical_kl(posterior_logits.detach(), prior_logits)
        representation = _categorical_kl(posterior_logits, prior_logits.detach())
        free = self.loss_config.free_nats
        return dynamics.clamp_min(free).mean(), representation.clamp_min(free).mean()

    def _ensemble_kl(self, output: WorldModelOutput) -> torch.Tensor:
        posterior = output.sequence.posterior_logits[:, 1:].detach()
        ensemble = output.sequence.ensemble_prior_logits[:, 1:]
        posterior = posterior[:, :, None].expand_as(ensemble)
        return _categorical_kl(posterior, ensemble).mean()

    def _check_shapes(
        self, output: WorldModelOutput, targets: WorldModelTargets
    ) -> None:
        batch, observations = output.visual_prediction.shape[:2]
        expected = {
            "visual": tuple(output.visual_prediction.shape),
            "proprioception": tuple(output.proprioception_prediction.shape),
            "reward": (batch, observations - 1),
            "continues": (batch, observations - 1),
            "safety": (batch, observations - 1),
        }
        actual = {name: tuple(getattr(targets, name).shape) for name in expected}
        mismatches = {
            name: (actual[name], shape)
            for name, shape in expected.items()
            if actual[name] != shape
        }
        if mismatches:
            raise ValueError(f"world model target shapes are invalid: {mismatches}")


def _categorical_kl(posterior_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    posterior = posterior_logits.softmax(dim=-1)
    log_posterior = posterior_logits.log_softmax(dim=-1)
    log_prior = prior_logits.log_softmax(dim=-1)
    return (posterior * (log_posterior - log_prior)).sum(dim=-1).sum(dim=-1)
