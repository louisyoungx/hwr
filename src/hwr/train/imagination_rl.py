"""Generic Actor-Critic optimization over world-model imagined trajectories."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch
from torch import nn

from hwr.policy.latent_actor import LatentActor
from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_value import LatentValueModel
from hwr.train.imagination import ImaginedTrajectory, imagine_trajectory
from hwr.world_model.distributions import reward_expectation, two_hot_symlog
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMState


@dataclass(frozen=True)
class ImaginationRLConfig:
    horizon: int = 15
    discount: float = 0.997
    lambda_return: float = 0.95
    motion_entropy_weight: float = 2.0e-3
    gripper_entropy_weight: float = 2.0e-4
    uncertainty_weight: float = 0.1
    safety_weight: float = 2.0
    value_bins: int = 255
    value_symlog_limit: float = 20.0
    slow_value_rate: float = 0.01
    maximum_gradient_norm: float = 100.0
    base_linear_scale: float = 0.18
    base_angular_scale: float = 0.50
    arm_velocity_scale: float = 0.35

    def __post_init__(self) -> None:
        if self.horizon <= 1 or not 0.0 < self.discount <= 1.0:
            raise ValueError("imagination horizon or discount is invalid")
        if not 0.0 <= self.lambda_return <= 1.0:
            raise ValueError("imagination lambda-return is invalid")
        if min(
            self.motion_entropy_weight,
            self.gripper_entropy_weight,
            self.uncertainty_weight,
            self.safety_weight,
            self.slow_value_rate,
            self.maximum_gradient_norm,
            self.base_linear_scale,
            self.base_angular_scale,
            self.arm_velocity_scale,
        ) < 0.0:
            raise ValueError("imagination RL weights cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ImaginationActorCritic(nn.Module):
    def __init__(
        self,
        world_model: ActionConditionedWorldModel,
        actor: LatentActor,
        value: LatentValueModel,
        config: ImaginationRLConfig,
    ) -> None:
        super().__init__()
        if value.bins != config.value_bins:
            raise ValueError("latent value bins differ from imagination config")
        self.world_model = world_model
        self.actor = actor
        self.value = value
        self.slow_value = copy.deepcopy(value)
        self.slow_value.requires_grad_(False)
        self.config = config
        self.action_scaling = LatentActionScaling(
            config.base_linear_scale,
            config.base_angular_scale,
            config.arm_velocity_scale,
        )

    def losses(self, initial: RSSMState) -> tuple[dict[str, torch.Tensor], ImaginedTrajectory]:
        trajectory = imagine_trajectory(
            self.world_model,
            self.actor,
            initial,
            horizon=self.config.horizon,
            action_scaling=self.action_scaling,
        )
        with torch.no_grad():
            slow_logits = self.slow_value(trajectory.next_features)
            slow_values = reward_expectation(
                slow_logits, limit=self.config.value_symlog_limit
            )
        rewards = trajectory.rewards
        rewards = rewards + self.config.uncertainty_weight * trajectory.uncertainties
        rewards = rewards - self.config.safety_weight * trajectory.safety_probabilities
        discounts = self.config.discount * trajectory.continues
        returns = lambda_returns(
            rewards, discounts, slow_values, lambda_=self.config.lambda_return
        )
        actor_objective = returns
        actor_objective += self.config.motion_entropy_weight * trajectory.motion_entropies
        actor_objective += self.config.gripper_entropy_weight * trajectory.gripper_entropies
        actor_loss = -actor_objective.mean()
        value_logits = self.value(trajectory.features.detach())
        value_predictions = reward_expectation(
            value_logits, limit=self.config.value_symlog_limit
        )
        value_targets = two_hot_symlog(
            returns.detach(),
            bins=self.config.value_bins,
            limit=self.config.value_symlog_limit,
        )
        value_loss = -(value_targets * value_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
        losses = {
            "actor": actor_loss,
            "value": value_loss,
            "imagined_reward": trajectory.rewards.mean(),
            "imagined_return": returns.mean(),
            "imagined_safety": trajectory.safety_probabilities.mean(),
            "imagined_uncertainty": trajectory.uncertainties.mean(),
            "td_error": (value_predictions - returns.detach()).abs().mean(),
        }
        return losses, trajectory

    @torch.no_grad()
    def update_slow_value(self) -> None:
        rate = self.config.slow_value_rate
        for slow, current in zip(self.slow_value.parameters(), self.value.parameters()):
            slow.data.lerp_(current.data, rate)


def lambda_returns(
    reward: torch.Tensor,
    discount: torch.Tensor,
    bootstrap_values: torch.Tensor,
    *,
    lambda_: float,
) -> torch.Tensor:
    if reward.shape != discount.shape or reward.shape != bootstrap_values.shape:
        raise ValueError("lambda-return tensors must share a shape")
    if reward.ndim != 2 or not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda-return input is invalid")
    returns = torch.empty_like(reward)
    accumulator = bootstrap_values[:, -1]
    for index in range(reward.shape[1] - 1, -1, -1):
        next_value = (
            bootstrap_values[:, index + 1]
            if index + 1 < reward.shape[1]
            else bootstrap_values[:, index]
        )
        mixed = (1.0 - lambda_) * next_value + lambda_ * accumulator
        accumulator = reward[:, index] + discount[:, index] * mixed
        returns[:, index] = accumulator
    return returns


def optimize_imagination_step(
    algorithm: ImaginationActorCritic,
    initial: RSSMState,
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    world_parameters = tuple(algorithm.world_model.parameters())
    original_grad_states = tuple(
        parameter.requires_grad for parameter in world_parameters
    )
    try:
        for parameter in world_parameters:
            parameter.requires_grad_(False)
        actor_optimizer.zero_grad(set_to_none=True)
        value_optimizer.zero_grad(set_to_none=True)
        losses, _ = algorithm.losses(initial)
        losses["actor"].backward(retain_graph=True)
        nn.utils.clip_grad_norm_(
            algorithm.actor.parameters(), algorithm.config.maximum_gradient_norm
        )
        actor_optimizer.step()
        value_optimizer.zero_grad(set_to_none=True)
        losses["value"].backward()
        nn.utils.clip_grad_norm_(
            algorithm.value.parameters(), algorithm.config.maximum_gradient_norm
        )
        value_optimizer.step()
        algorithm.update_slow_value()
    finally:
        for parameter, requires_grad in zip(
            world_parameters, original_grad_states, strict=True
        ):
            parameter.requires_grad_(requires_grad)
    return {name: float(value.detach().cpu()) for name, value in losses.items()}
