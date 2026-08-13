"""Task-independent intrinsic RL in action-conditioned imagined trajectories."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch
from torch import nn

from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_actor import LatentActor
from hwr.policy.latent_value import LatentValueModel
from hwr.train.imagination import ImaginedTrajectory, imagine_trajectory
from hwr.train.imagination_rl import lambda_returns
from hwr.world_model.distributions import reward_expectation, two_hot_symlog
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMState


@dataclass(frozen=True)
class IntrinsicExplorationConfig:
    horizon: int = 15
    discount: float = 0.997
    lambda_return: float = 0.95
    uncertainty_weight: float = 1.0
    state_novelty_weight: float = 0.25
    safety_weight: float = 2.0
    motion_entropy_weight: float = 3.0e-3
    gripper_entropy_weight: float = 3.0e-4
    slow_value_rate: float = 0.01
    maximum_gradient_norm: float = 100.0
    value_bins: int = 255
    value_symlog_limit: float = 20.0

    def __post_init__(self) -> None:
        if self.horizon <= 1 or not 0.0 < self.discount <= 1.0:
            raise ValueError("intrinsic exploration horizon or discount is invalid")
        if not 0.0 <= self.lambda_return <= 1.0:
            raise ValueError("intrinsic exploration lambda-return is invalid")
        weights = (
            self.uncertainty_weight,
            self.state_novelty_weight,
            self.safety_weight,
            self.motion_entropy_weight,
            self.gripper_entropy_weight,
            self.slow_value_rate,
            self.maximum_gradient_norm,
        )
        if min(weights) < 0.0 or min(self.value_bins, self.value_symlog_limit) <= 0:
            raise ValueError("intrinsic exploration weights are invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class IntrinsicExplorationActorCritic(nn.Module):
    """A separate RL policy with no environment reward or task semantics."""

    def __init__(
        self,
        world_model: ActionConditionedWorldModel,
        actor: LatentActor,
        value: LatentValueModel,
        action_scaling: LatentActionScaling,
        config: IntrinsicExplorationConfig,
    ) -> None:
        super().__init__()
        if value.bins != config.value_bins:
            raise ValueError("intrinsic value bins differ from its config")
        self.world_model = world_model
        self.actor = actor
        self.value = value
        self.slow_value = copy.deepcopy(value).requires_grad_(False)
        self.action_scaling = action_scaling
        self.config = config

    def losses(
        self, initial: RSSMState
    ) -> tuple[dict[str, torch.Tensor], ImaginedTrajectory]:
        trajectory = imagine_trajectory(
            self.world_model,
            self.actor,
            initial,
            horizon=self.config.horizon,
            action_scaling=self.action_scaling,
        )
        novelty = _cosine_state_change(
            trajectory.features, trajectory.next_features
        )
        rewards = self.config.uncertainty_weight * trajectory.uncertainties
        rewards = rewards + self.config.state_novelty_weight * novelty
        rewards = rewards - self.config.safety_weight * trajectory.safety_probabilities
        rewards = rewards + self.config.motion_entropy_weight * trajectory.motion_entropies
        rewards = rewards + self.config.gripper_entropy_weight * trajectory.gripper_entropies
        with torch.no_grad():
            slow_values = reward_expectation(
                self.slow_value(trajectory.next_features),
                limit=self.config.value_symlog_limit,
            )
        returns = lambda_returns(
            rewards,
            self.config.discount * trajectory.continues,
            slow_values,
            lambda_=self.config.lambda_return,
        )
        value_logits = self.value(trajectory.features.detach())
        targets = two_hot_symlog(
            returns.detach(),
            bins=self.config.value_bins,
            limit=self.config.value_symlog_limit,
        )
        value_predictions = reward_expectation(
            value_logits, limit=self.config.value_symlog_limit
        )
        losses = {
            "actor": -returns.mean(),
            "value": -(targets * value_logits.log_softmax(dim=-1)).sum(dim=-1).mean(),
            "return": returns.mean(),
            "uncertainty": trajectory.uncertainties.mean(),
            "state_novelty": novelty.mean(),
            "safety": trajectory.safety_probabilities.mean(),
            "td_error": (value_predictions - returns.detach()).abs().mean(),
        }
        return losses, trajectory

    @torch.no_grad()
    def update_slow_value(self) -> None:
        for slow, current in zip(
            self.slow_value.parameters(), self.value.parameters(), strict=True
        ):
            slow.data.lerp_(current.data, self.config.slow_value_rate)


def optimize_intrinsic_exploration_step(
    algorithm: IntrinsicExplorationActorCritic,
    initial: RSSMState,
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    world_parameters = tuple(algorithm.world_model.parameters())
    gradient_states = tuple(value.requires_grad for value in world_parameters)
    try:
        for parameter in world_parameters:
            parameter.requires_grad_(False)
        actor_optimizer.zero_grad(set_to_none=True)
        value_optimizer.zero_grad(set_to_none=True)
        losses, _ = algorithm.losses(initial)
        losses["actor"].backward(retain_graph=True)
        actor_norm = nn.utils.clip_grad_norm_(
            algorithm.actor.parameters(), algorithm.config.maximum_gradient_norm
        )
        actor_optimizer.step()
        value_optimizer.zero_grad(set_to_none=True)
        losses["value"].backward()
        value_norm = nn.utils.clip_grad_norm_(
            algorithm.value.parameters(), algorithm.config.maximum_gradient_norm
        )
        value_optimizer.step()
        algorithm.update_slow_value()
    finally:
        for parameter, state in zip(world_parameters, gradient_states, strict=True):
            parameter.requires_grad_(state)
    metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
    metrics["actor_gradient_norm"] = float(actor_norm.detach().cpu())
    metrics["value_gradient_norm"] = float(value_norm.detach().cpu())
    return metrics


def _cosine_state_change(
    current: torch.Tensor, following: torch.Tensor
) -> torch.Tensor:
    left = nn.functional.normalize(current, dim=-1)
    right = nn.functional.normalize(following, dim=-1)
    return (1.0 - (left * right).sum(dim=-1)).clamp_min(0.0)
