"""Twin-critic off-policy updates with deployment-limited Actor observations."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from hwr.policy.privileged_critic import (
    PrivilegedCriticConfig,
    TwinPrivilegedCritic,
)
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS
from hwr.policy.vla_actions import VLAActionScaling, bounded_vla_actions
from hwr.policy.vla_model import VLAActorModel, VLAActorOutput


@dataclass(frozen=True)
class AsymmetricRLConfig:
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    discount: float = 0.99
    target_update_rate: float = 0.005
    behavior_regularization: float = 0.02
    policy_delay: int = 2
    actor_warmup_updates: int = 2000
    target_action_noise: float = 0.15
    target_noise_clip: float = 0.30
    action_magnitude_penalty: float = 0.08
    action_slew_penalty: float = 0.04
    base_linear_scale: float = 0.45
    base_angular_scale: float = 1.0
    arm_velocity_scale: float = 1.0

    def __post_init__(self) -> None:
        if min(
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.base_linear_scale,
            self.base_angular_scale,
            self.arm_velocity_scale,
        ) <= 0:
            raise ValueError("asymmetric RL learning rates must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("asymmetric RL discount must be in [0, 1]")
        if not 0.0 < self.target_update_rate <= 1.0:
            raise ValueError("target update rate must be in (0, 1]")
        if (
            self.behavior_regularization < 0.0
            or self.policy_delay <= 0
            or self.actor_warmup_updates < 0
        ):
            raise ValueError("asymmetric RL regularization or delay is invalid")
        regularizers = (
            self.target_action_noise,
            self.target_noise_clip,
            self.action_magnitude_penalty,
            self.action_slew_penalty,
        )
        if min(regularizers) < 0.0:
            raise ValueError("TD3 smoothing and action penalties cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def action_scaling(self) -> VLAActionScaling:
        return VLAActionScaling(
            self.base_linear_scale,
            self.base_angular_scale,
            self.arm_velocity_scale,
        )


@dataclass(frozen=True)
class AsymmetricRLBatch:
    actor_inputs: Mapping[str, torch.Tensor]
    next_actor_inputs: Mapping[str, torch.Tensor]
    privileged_state: torch.Tensor
    next_privileged_state: torch.Tensor
    action_chunks: torch.Tensor
    stop_decisions: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor
    actor_weights: torch.Tensor | None = None


def _action_scales(config: AsymmetricRLConfig, reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        (
            config.base_linear_scale,
            config.base_angular_scale,
            *(config.arm_velocity_scale,) * 12,
            1.0,
            1.0,
        ),
        dtype=reference.dtype,
        device=reference.device,
    )


def _smoothed_target_action(
    output: VLAActorOutput,
    config: AsymmetricRLConfig,
) -> torch.Tensor:
    bounded = bounded_vla_actions(output, config.action_scaling())
    scales = _action_scales(config, bounded)
    noise = torch.randn_like(bounded) * config.target_action_noise * scales
    noise = torch.clamp(
        noise,
        min=-config.target_noise_clip * scales,
        max=config.target_noise_clip * scales,
    )
    value = bounded + noise
    value[..., :14] = torch.clamp(value[..., :14], min=-scales[:14], max=scales[:14])
    value[..., 14:] = value[..., 14:].clamp(0.0, 1.0)
    stop = torch.zeros_like(output.stop_logits).unsqueeze(-1)
    return torch.cat((value, stop), dim=2).flatten(1)


def _executed_action_representation(batch: AsymmetricRLBatch) -> torch.Tensor:
    stop = batch.stop_decisions.unsqueeze(-1).to(batch.action_chunks.dtype)
    return torch.cat((batch.action_chunks, stop), dim=2).flatten(1)


def _soft_update(target: nn.Module, source: nn.Module, rate: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.lerp_(source_parameter, rate)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class AsymmetricActorCriticTrainer:
    """Update an Actor with privileged Q estimates without exposing truth inputs."""

    def __init__(
        self,
        actor: VLAActorModel,
        critic_config: PrivilegedCriticConfig,
        config: AsymmetricRLConfig,
        *,
        device: str = "cpu",
    ) -> None:
        if actor.config.action_chunk_size != critic_config.action_chunk_size:
            raise ValueError("Actor and privileged critic action chunks differ")
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device).eval()
        self.critic = TwinPrivilegedCritic(critic_config).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device).eval()
        self.critic_config = critic_config
        self.config = config
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.update_count = 0

    def update(self, batch: AsymmetricRLBatch) -> dict[str, float]:
        batch = self._to_device(batch)
        self._validate_batch(batch)
        critic_loss = self._update_critic(batch)
        actor_loss = torch.zeros((), device=self.device)
        actor_updated = (
            self.update_count >= self.config.actor_warmup_updates
            and (self.update_count + 1) % self.config.policy_delay == 0
        )
        if actor_updated:
            actor_loss = self._update_actor(batch)
            _soft_update(self.target_actor, self.actor, self.config.target_update_rate)
            _soft_update(self.target_critic, self.critic, self.config.target_update_rate)
        self.update_count += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_updated": float(actor_updated),
            "update": float(self.update_count),
        }

    def _update_critic(self, batch: AsymmetricRLBatch) -> torch.Tensor:
        with torch.no_grad():
            next_output = self.target_actor(batch.next_actor_inputs)
            next_action = _smoothed_target_action(next_output, self.config)
            target_q1, target_q2 = self.target_critic(
                batch.next_privileged_state, next_action
            )
            target_q = batch.rewards + (
                1.0 - batch.done
            ) * self.config.discount * torch.minimum(target_q1, target_q2)
        executed = _executed_action_representation(batch)
        q1, q2 = self.critic(batch.privileged_state, executed)
        loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        return loss

    def _update_actor(self, batch: AsymmetricRLBatch) -> torch.Tensor:
        _set_requires_grad(self.critic, False)
        output = self.actor(batch.actor_inputs)
        bounded = bounded_vla_actions(output, self.config.action_scaling())
        stop = torch.zeros_like(output.stop_logits).unsqueeze(-1)
        action = torch.cat((bounded, stop), dim=2).flatten(1)
        q1, _ = self.critic(batch.privileged_state, action)
        behavior = torch.zeros_like(q1)
        if self.config.behavior_regularization > 0.0:
            behavior = nn.functional.smooth_l1_loss(
                bounded, batch.action_chunks, reduction="none"
            ).mean(dim=(1, 2))
            behavior += nn.functional.binary_cross_entropy_with_logits(
                output.stop_logits,
                batch.stop_decisions.to(output.stop_logits.dtype),
                reduction="none",
            ).mean(dim=1)
        scales = _action_scales(self.config, bounded)
        magnitude = (bounded[..., :14] / scales[:14]).square().mean(dim=(1, 2))
        previous = batch.actor_inputs["action_history"][:, -1:, :]
        trajectory = torch.cat((previous, bounded), dim=1)
        slew = (
            (trajectory[:, 1:] - trajectory[:, :-1]) / scales
        ).square().mean(dim=(1, 2))
        weights = (
            batch.actor_weights
            if batch.actor_weights is not None
            else torch.ones_like(q1)
        ).to(q1.dtype)
        denominator = weights.sum().clamp_min(1.0)
        loss = (
            weights
            * (
                -q1
                + self.config.behavior_regularization * behavior
                + self.config.action_magnitude_penalty * magnitude
                + self.config.action_slew_penalty * slew
            )
        ).sum() / denominator
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        _set_requires_grad(self.critic, True)
        return loss

    def _validate_batch(self, batch: AsymmetricRLBatch) -> None:
        if frozenset(batch.actor_inputs) != VLA_POLICY_INPUT_FIELDS or frozenset(
            batch.next_actor_inputs
        ) != VLA_POLICY_INPUT_FIELDS:
            raise ValueError("asymmetric RL Actor received non-deployment fields")
        batch_size = batch.rewards.shape[0]
        expected_chunk = (
            batch_size,
            self.actor.config.action_chunk_size,
            self.actor.config.action_dim,
        )
        expected_stop = (batch_size, self.actor.config.action_chunk_size)
        expected_state = (batch_size, self.critic_config.privileged_state_dim)
        if tuple(batch.action_chunks.shape) != expected_chunk:
            raise ValueError("asymmetric RL action chunk shape is invalid")
        if tuple(batch.stop_decisions.shape) != expected_stop:
            raise ValueError("asymmetric RL stop decision shape is invalid")
        if tuple(batch.privileged_state.shape) != expected_state or tuple(
            batch.next_privileged_state.shape
        ) != expected_state:
            raise ValueError("asymmetric RL privileged state shape is invalid")
        if tuple(batch.rewards.shape) != (batch_size,) or tuple(batch.done.shape) != (
            batch_size,
        ):
            raise ValueError("asymmetric RL reward or done shape is invalid")
        if batch.actor_weights is not None and tuple(batch.actor_weights.shape) != (
            batch_size,
        ):
            raise ValueError("asymmetric RL Actor weights shape is invalid")

    def _to_device(self, batch: AsymmetricRLBatch) -> AsymmetricRLBatch:
        move = lambda values: {name: value.to(self.device) for name, value in values.items()}
        return AsymmetricRLBatch(
            actor_inputs=move(batch.actor_inputs),
            next_actor_inputs=move(batch.next_actor_inputs),
            privileged_state=batch.privileged_state.to(self.device),
            next_privileged_state=batch.next_privileged_state.to(self.device),
            action_chunks=batch.action_chunks.to(self.device),
            stop_decisions=batch.stop_decisions.to(self.device),
            rewards=batch.rewards.to(self.device),
            done=batch.done.to(self.device),
            actor_weights=(
                batch.actor_weights.to(self.device)
                if batch.actor_weights is not None
                else None
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        self.actor.load_state_dict(value["actor"])
        self.target_actor.load_state_dict(value["target_actor"])
        self.critic.load_state_dict(value["critic"])
        self.target_critic.load_state_dict(value["target_critic"])
        self.actor_optimizer.load_state_dict(value["actor_optimizer"])
        self.critic_optimizer.load_state_dict(value["critic_optimizer"])
        self.update_count = int(value["update_count"])
