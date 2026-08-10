"""Twin-critic off-policy updates with deployment-limited Actor observations."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
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
from hwr.train.stochastic_action import (
    SquashedGaussianAction,
    sample_squashed_gaussian_action,
)


@dataclass(frozen=True)
class AsymmetricRLConfig:
    actor_learning_rate: float = 3e-5
    critic_learning_rate: float = 3e-4
    discount: float = 0.99
    target_update_rate: float = 0.005
    behavior_regularization: float = 0.02
    policy_delay: int = 10
    actor_warmup_updates: int = 2000
    reward_scale: float = 0.25
    entropy_temperature: float = 0.02
    gripper_entropy_temperature: float = 0.002
    initial_motion_log_standard_deviation: float = -1.5
    minimum_motion_log_standard_deviation: float = -4.0
    maximum_motion_log_standard_deviation: float = -0.3
    initial_gripper_log_standard_deviation: float = -0.5
    minimum_gripper_log_standard_deviation: float = -3.0
    maximum_gripper_log_standard_deviation: float = 0.2
    action_magnitude_penalty: float = 0.08
    action_slew_penalty: float = 0.04
    gripper_head_learning_rate_scale: float = 4.0
    gripper_head_acceleration_updates: int = 5000
    safety_learning_rate: float = 3e-4
    safety_actor_penalty: float = 3.0
    conservative_critic_weight: float = 0.05
    conservative_action_samples: int = 4
    base_linear_scale: float = 0.18
    base_angular_scale: float = 0.50
    arm_velocity_scale: float = 0.35

    def __post_init__(self) -> None:
        if min(
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.base_linear_scale,
            self.base_angular_scale,
            self.arm_velocity_scale,
            self.safety_learning_rate,
            self.reward_scale,
            self.gripper_head_learning_rate_scale,
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
            or self.gripper_head_acceleration_updates < self.actor_warmup_updates
            or self.conservative_action_samples <= 0
        ):
            raise ValueError("asymmetric RL regularization or delay is invalid")
        regularizers = (
            self.entropy_temperature,
            self.gripper_entropy_temperature,
            self.action_magnitude_penalty,
            self.action_slew_penalty,
            self.safety_actor_penalty,
            self.conservative_critic_weight,
        )
        if min(regularizers) < 0.0:
            raise ValueError("entropy and action penalties cannot be negative")
        for name, log_std in (
            (
                "motion",
                (
                    self.minimum_motion_log_standard_deviation,
                    self.initial_motion_log_standard_deviation,
                    self.maximum_motion_log_standard_deviation,
                ),
            ),
            (
                "gripper",
                (
                    self.minimum_gripper_log_standard_deviation,
                    self.initial_gripper_log_standard_deviation,
                    self.maximum_gripper_log_standard_deviation,
                ),
            ),
        ):
            if not all(math.isfinite(value) for value in log_std) or not (
                log_std[0] < log_std[2]
                and log_std[0] <= log_std[1] <= log_std[2]
            ):
                raise ValueError(
                    f"stochastic Actor {name} log standard deviations are invalid"
                )

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
    proposed_action_chunks: torch.Tensor | None = None
    safety_costs: torch.Tensor | None = None
    bootstrap_discounts: torch.Tensor | None = None


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


def _executed_action_representation(batch: AsymmetricRLBatch) -> torch.Tensor:
    stop = batch.stop_decisions.unsqueeze(-1).to(batch.action_chunks.dtype)
    return torch.cat((batch.action_chunks, stop), dim=2).flatten(1)


def _proposed_action_representation(batch: AsymmetricRLBatch) -> torch.Tensor:
    chunks = batch.proposed_action_chunks
    if chunks is None:
        raise ValueError("safety critic requires proposed actions")
    stop = torch.zeros(
        (*chunks.shape[:2], 1), dtype=chunks.dtype, device=chunks.device
    )
    return torch.cat((chunks, stop), dim=2).flatten(1)


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
        self.safety_critic = TwinPrivilegedCritic(critic_config).to(self.device)
        self.critic_config = critic_config
        self.config = config
        self.actor_log_standard_deviation = nn.Parameter(
            torch.tensor(
                (
                    *(config.initial_motion_log_standard_deviation,) * 14,
                    *(config.initial_gripper_log_standard_deviation,) * 2,
                ),
                device=self.device,
            )[None, None].expand(
                1, actor.config.action_chunk_size, actor.config.action_dim
            ).clone()
        )
        self.update_count = 0
        self.actor_optimizer = self._actor_optimizer()
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.safety_optimizer = torch.optim.AdamW(
            self.safety_critic.parameters(), lr=config.safety_learning_rate
        )

    def update(self, batch: AsymmetricRLBatch) -> dict[str, float]:
        batch = self._to_device(batch)
        self._validate_batch(batch)
        critic_loss, conservative_loss = self._update_critic(batch)
        safety_loss = self._update_safety_critic(batch)
        actor_loss = torch.zeros((), device=self.device)
        actor_metrics = {
            "actor_reward_value": 0.0,
            "actor_safety_risk": 0.0,
            "reward_critic_disagreement": 0.0,
            "safety_critic_disagreement": 0.0,
            "actor_motion_mean_ratio": 0.0,
            "actor_motion_max_ratio": 0.0,
            "actor_entropy": 0.0,
            "actor_motion_log_standard_deviation": 0.0,
            "actor_gripper_log_standard_deviation": 0.0,
        }
        actor_updated = (
            self.update_count >= self.config.actor_warmup_updates
            and (self.update_count + 1) % self.config.policy_delay == 0
        )
        if actor_updated:
            actor_loss, actor_metrics = self._update_actor(batch)
            _soft_update(self.target_actor, self.actor, self.config.target_update_rate)
            _soft_update(self.target_critic, self.critic, self.config.target_update_rate)
        self.update_count += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "conservative_loss": float(conservative_loss.detach().cpu()),
            "safety_loss": float(safety_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_updated": float(actor_updated),
            "update": float(self.update_count),
            **actor_metrics,
        }

    def _actor_optimizer(self) -> torch.optim.AdamW:
        base = [*self.actor.parameters(), self.actor_log_standard_deviation]
        if not self.actor.config.separate_gripper_head:
            return torch.optim.AdamW(base, lr=self.config.actor_learning_rate)
        gripper = list(self.actor.gripper_head.parameters())
        gripper_ids = {id(parameter) for parameter in gripper}
        shared = [parameter for parameter in base if id(parameter) not in gripper_ids]
        return torch.optim.AdamW(
            (
                {
                    "params": shared,
                    "lr": self.config.actor_learning_rate,
                    "role": "base",
                },
                {
                    "params": gripper,
                    "lr": self._gripper_head_learning_rate(),
                    "role": "gripper",
                },
            )
        )

    def _gripper_head_learning_rate(self) -> float:
        scale = (
            self.config.gripper_head_learning_rate_scale
            if self.update_count >= self.config.gripper_head_acceleration_updates
            else 1.0
        )
        return self.config.actor_learning_rate * scale

    def _schedule_actor_learning_rates(self) -> None:
        for group in self.actor_optimizer.param_groups:
            if group.get("role") == "gripper":
                group["lr"] = self._gripper_head_learning_rate()

    def _update_critic(
        self, batch: AsymmetricRLBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_output = self.target_actor(batch.next_actor_inputs)
            next_sample = self._sample_action(next_output)
            next_action = self._critic_action(next_output, next_sample.values)
            target_q1, target_q2 = self.target_critic(
                batch.next_privileged_state, next_action
            )
            bootstrap = (
                batch.bootstrap_discounts
                if batch.bootstrap_discounts is not None
                else (1.0 - batch.done) * self.config.discount
            )
            target_q = (
                batch.rewards * self.config.reward_scale
                + bootstrap
                * (
                    torch.minimum(target_q1, target_q2)
                    - self._weighted_log_probability(next_sample)
                )
            )
        executed = _executed_action_representation(batch)
        q1, q2 = self.critic(batch.privileged_state, executed)
        temporal_difference = nn.functional.mse_loss(
            q1, target_q
        ) + nn.functional.mse_loss(q2, target_q)
        conservative = self._conservative_critic_loss(batch, q1, q2)
        loss = (
            temporal_difference
            + self.config.conservative_critic_weight * conservative
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        return loss, conservative

    def _conservative_critic_loss(
        self,
        batch: AsymmetricRLBatch,
        data_q1: torch.Tensor,
        data_q2: torch.Tensor,
    ) -> torch.Tensor:
        count = self.config.conservative_action_samples
        batch_size = batch.rewards.shape[0]
        state = batch.privileged_state[:, None, :].expand(
            -1, count, -1
        ).reshape(batch_size * count, -1)
        action = self._random_action_representations(batch_size, count)
        random_q1, random_q2 = self.critic(state, action)
        random_q1 = random_q1.reshape(batch_size, count)
        random_q2 = random_q2.reshape(batch_size, count)
        with torch.no_grad():
            output = self.actor(batch.actor_inputs)
            bounded = bounded_vla_actions(output, self.config.action_scaling())
            stop = torch.zeros_like(output.stop_logits).unsqueeze(-1)
            policy_action = torch.cat((bounded, stop), dim=2).flatten(1)
        policy_q1, policy_q2 = self.critic(
            batch.privileged_state, policy_action
        )
        candidates_q1 = torch.cat((random_q1, policy_q1[:, None]), dim=1)
        candidates_q2 = torch.cat((random_q2, policy_q2[:, None]), dim=1)
        normalizer = torch.log(
            torch.tensor(
                candidates_q1.shape[1],
                dtype=candidates_q1.dtype,
                device=candidates_q1.device,
            )
        )
        first = torch.logsumexp(candidates_q1, dim=1) - normalizer - data_q1
        second = torch.logsumexp(candidates_q2, dim=1) - normalizer - data_q2
        return torch.relu(first).mean() + torch.relu(second).mean()

    def _random_action_representations(
        self, batch_size: int, count: int
    ) -> torch.Tensor:
        shape = (
            batch_size,
            count,
            self.actor.config.action_chunk_size,
            self.actor.config.action_dim,
        )
        value = torch.rand(shape, device=self.device) * 2.0 - 1.0
        scales = _action_scales(self.config, value)
        value[..., :14] *= scales[:14]
        value[..., 14:] = (value[..., 14:] + 1.0) / 2.0
        stop = torch.zeros(
            (*shape[:-1], 1), dtype=value.dtype, device=value.device
        )
        return torch.cat((value, stop), dim=3).flatten(2).reshape(
            batch_size * count, -1
        )

    def _update_safety_critic(self, batch: AsymmetricRLBatch) -> torch.Tensor:
        if batch.proposed_action_chunks is None or batch.safety_costs is None:
            return torch.zeros((), device=self.device)
        proposed = _proposed_action_representation(batch)
        first, second = self.safety_critic(batch.privileged_state, proposed)
        target = batch.safety_costs.to(first.dtype)
        positives = target.sum()
        negatives = target.numel() - positives
        positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
        loss = nn.functional.binary_cross_entropy_with_logits(
            first, target, pos_weight=positive_weight
        )
        loss += nn.functional.binary_cross_entropy_with_logits(
            second, target, pos_weight=positive_weight
        )
        self.safety_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.safety_critic.parameters(), 1.0)
        self.safety_optimizer.step()
        return loss

    def _update_actor(
        self, batch: AsymmetricRLBatch
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self._schedule_actor_learning_rates()
        _set_requires_grad(self.critic, False)
        _set_requires_grad(self.safety_critic, False)
        output = self.actor(batch.actor_inputs)
        sample = self._sample_action(output)
        bounded = sample.values
        action = self._critic_action(output, bounded)
        q1, q2 = self.critic(batch.privileged_state, action)
        reward_value = torch.minimum(q1, q2)
        safety_risk = torch.zeros_like(q1)
        safety_disagreement = torch.zeros_like(q1)
        if batch.safety_costs is not None:
            safety_first, safety_second = self.safety_critic(
                batch.privileged_state, action
            )
            first_risk = torch.sigmoid(safety_first)
            second_risk = torch.sigmoid(safety_second)
            safety_risk = torch.maximum(first_risk, second_risk)
            safety_disagreement = (first_risk - second_risk).abs()
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
        normalized_motion = bounded[..., :14] / scales[:14]
        magnitude = normalized_motion.square().mean(dim=(1, 2))
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
                -reward_value
                + self._weighted_log_probability(sample)
                + self.config.behavior_regularization * behavior
                + self.config.action_magnitude_penalty * magnitude
                + self.config.action_slew_penalty * slew
                + self.config.safety_actor_penalty * safety_risk
            )
        ).sum() / denominator
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        _set_requires_grad(self.critic, True)
        _set_requires_grad(self.safety_critic, True)
        metrics = {
            "actor_reward_value": float(reward_value.mean().detach().cpu()),
            "actor_safety_risk": float(safety_risk.mean().detach().cpu()),
            "reward_critic_disagreement": float(
                (q1 - q2).abs().mean().detach().cpu()
            ),
            "safety_critic_disagreement": float(
                safety_disagreement.mean().detach().cpu()
            ),
            "actor_motion_mean_ratio": float(
                normalized_motion.abs().mean().detach().cpu()
            ),
            "actor_motion_max_ratio": float(
                normalized_motion.abs().max().detach().cpu()
            ),
            "actor_entropy": float(
                -sample.log_probability.mean().detach().cpu()
            ),
            "actor_motion_log_standard_deviation": float(
                self.actor_log_standard_deviation[..., :14].mean().detach().cpu()
            ),
            "actor_gripper_log_standard_deviation": float(
                self.actor_log_standard_deviation[..., 14:].mean().detach().cpu()
            ),
        }
        return loss, metrics

    def _weighted_log_probability(
        self, sample: SquashedGaussianAction
    ) -> torch.Tensor:
        return (
            self.config.entropy_temperature * sample.motion_log_probability
            + self.config.gripper_entropy_temperature
            * sample.gripper_log_probability
        )

    def sample_actor_action(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return a bounded deployable action sample without privileged inputs."""

        output = self.actor(inputs)
        return self._sample_action(output, deterministic=deterministic).values

    def _sample_action(
        self,
        output: VLAActorOutput,
        *,
        deterministic: bool = False,
    ) -> SquashedGaussianAction:
        return sample_squashed_gaussian_action(
            output,
            self.actor_log_standard_deviation,
            self.config.action_scaling(),
            motion_log_standard_deviation_bounds=(
                self.config.minimum_motion_log_standard_deviation,
                self.config.maximum_motion_log_standard_deviation,
            ),
            gripper_log_standard_deviation_bounds=(
                self.config.minimum_gripper_log_standard_deviation,
                self.config.maximum_gripper_log_standard_deviation,
            ),
            deterministic=deterministic,
        )

    @staticmethod
    def _critic_action(
        output: VLAActorOutput, bounded: torch.Tensor
    ) -> torch.Tensor:
        stop = torch.zeros_like(output.stop_logits).unsqueeze(-1)
        return torch.cat((bounded, stop), dim=2).flatten(1)

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
        if batch.bootstrap_discounts is not None and tuple(
            batch.bootstrap_discounts.shape
        ) != (batch_size,):
            raise ValueError("asymmetric RL bootstrap discounts shape is invalid")
        optional = (batch.proposed_action_chunks, batch.safety_costs)
        if (optional[0] is None) != (optional[1] is None):
            raise ValueError("safety proposal and cost must appear together")
        if optional[0] is not None:
            if tuple(optional[0].shape) != expected_chunk:
                raise ValueError("safety proposal action shape is invalid")
            if tuple(optional[1].shape) != (batch_size,):
                raise ValueError("safety cost shape is invalid")

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
            proposed_action_chunks=(
                batch.proposed_action_chunks.to(self.device)
                if batch.proposed_action_chunks is not None
                else None
            ),
            safety_costs=(
                batch.safety_costs.to(self.device)
                if batch.safety_costs is not None
                else None
            ),
            bootstrap_discounts=(
                batch.bootstrap_discounts.to(self.device)
                if batch.bootstrap_discounts is not None
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
            "safety_critic": self.safety_critic.state_dict(),
            "safety_optimizer": self.safety_optimizer.state_dict(),
            "actor_log_standard_deviation": (
                self.actor_log_standard_deviation.detach().cpu()
            ),
            "update_count": self.update_count,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        self.actor.load_state_dict(value["actor"])
        self.target_actor.load_state_dict(value["target_actor"])
        self.critic.load_state_dict(value["critic"])
        self.target_critic.load_state_dict(value["target_critic"])
        has_stochastic_state = "actor_log_standard_deviation" in value
        if has_stochastic_state:
            self.actor_log_standard_deviation.data.copy_(
                value["actor_log_standard_deviation"].to(self.device)
            )
            self.actor_optimizer.load_state_dict(value["actor_optimizer"])
        self.critic_optimizer.load_state_dict(value["critic_optimizer"])
        if "safety_critic" in value:
            self.safety_critic.load_state_dict(value["safety_critic"])
            self.safety_optimizer.load_state_dict(value["safety_optimizer"])
        self.update_count = int(value["update_count"])
