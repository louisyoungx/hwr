"""Task-agnostic stochastic Actor over learned world-model latent state."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn

from hwr.core.embodied import DUAL_ARM_ACTION_DIM


@dataclass(frozen=True)
class LatentActorConfig:
    latent_dimension: int
    action_dimension: int = DUAL_ARM_ACTION_DIM
    hidden_dimension: int = 512
    hidden_layers: int = 3
    minimum_log_standard_deviation: float = -5.0
    maximum_log_standard_deviation: float = 1.0
    formal: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.latent_dimension,
            self.action_dimension,
            self.hidden_dimension,
            self.hidden_layers,
        )
        if min(dimensions) <= 0:
            raise ValueError("latent Actor dimensions must be positive")
        if self.minimum_log_standard_deviation >= self.maximum_log_standard_deviation:
            raise ValueError("latent Actor standard deviation bounds are invalid")
        if self.formal and self.action_dimension != DUAL_ARM_ACTION_DIM:
            raise ValueError("formal latent Actor requires the canonical 16-D action")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LatentActorSample(NamedTuple):
    action: torch.Tensor
    log_probability: torch.Tensor
    motion_entropy: torch.Tensor
    gripper_entropy: torch.Tensor
    mean_action: torch.Tensor


class LatentActor(nn.Module):
    """One shared policy; task semantics can only arrive through latent state."""

    def __init__(self, config: LatentActorConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        input_dimension = config.latent_dimension
        for _ in range(config.hidden_layers):
            layers.extend(
                (
                    nn.Linear(input_dimension, config.hidden_dimension),
                    nn.LayerNorm(config.hidden_dimension),
                    nn.SiLU(),
                )
            )
            input_dimension = config.hidden_dimension
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(config.hidden_dimension, config.action_dimension)
        self.log_standard_deviation_head = nn.Linear(
            config.hidden_dimension, config.action_dimension
        )
        nn.init.uniform_(self.mean_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_standard_deviation_head.weight)
        with torch.no_grad():
            self.log_standard_deviation_head.bias[:14].fill_(
                _raw_log_standard_deviation(0.75)
            )
            self.log_standard_deviation_head.bias[14:].fill_(
                _raw_log_standard_deviation(11.0 / 12.0)
            )

    def distribution_parameters(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(latent)
        mean = self.mean_head(features)
        raw_log_standard_deviation = self.log_standard_deviation_head(features)
        span = (
            self.config.maximum_log_standard_deviation
            - self.config.minimum_log_standard_deviation
        )
        log_standard_deviation = self.config.minimum_log_standard_deviation
        log_standard_deviation += span * (
            torch.tanh(raw_log_standard_deviation) + 1.0
        ) / 2.0
        return mean, log_standard_deviation

    def sample(
        self, latent: torch.Tensor, *, deterministic: bool = False
    ) -> LatentActorSample:
        mean, log_standard_deviation = self.distribution_parameters(latent)
        standard_deviation = log_standard_deviation.exp()
        normal = torch.distributions.Normal(mean, standard_deviation)
        raw = mean if deterministic else normal.rsample()
        action = self._transform(raw)
        mean_action = self._transform(mean)
        component_log_probability = normal.log_prob(raw) - self._log_abs_jacobian(raw)
        motion_log_probability = component_log_probability[..., :14].sum(dim=-1)
        gripper_log_probability = component_log_probability[..., 14:].sum(dim=-1)
        return LatentActorSample(
            action=action,
            log_probability=motion_log_probability + gripper_log_probability,
            motion_entropy=-motion_log_probability,
            gripper_entropy=-gripper_log_probability,
            mean_action=mean_action,
        )

    def deterministic(self, latent: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(latent)
        return self._transform(mean)

    def _transform(self, raw: torch.Tensor) -> torch.Tensor:
        motion = torch.tanh(raw[..., :14])
        grippers = torch.sigmoid(raw[..., 14:])
        return torch.cat((motion, grippers), dim=-1)

    def _log_abs_jacobian(self, raw: torch.Tensor) -> torch.Tensor:
        motion = 2.0 * (math.log(2.0) - raw[..., :14] - nn.functional.softplus(-2.0 * raw[..., :14]))
        grippers = -nn.functional.softplus(-raw[..., 14:])
        grippers -= nn.functional.softplus(raw[..., 14:])
        return torch.cat((motion, grippers), dim=-1)


def _raw_log_standard_deviation(fraction: float) -> float:
    """Invert the configured tanh interpolation at a bound-relative fraction."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("initial Actor deviation fraction must be internal")
    return math.atanh(2.0 * fraction - 1.0)
