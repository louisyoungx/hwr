"""Reparameterized bounded actions for maximum-entropy continuous control."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from hwr.policy.vla_actions import VLAActionScaling
from hwr.policy.vla_model import VLAActorOutput


@dataclass(frozen=True)
class SquashedGaussianAction:
    """One differentiable action sample and its transform-corrected density."""

    values: torch.Tensor
    log_probability: torch.Tensor


def sample_squashed_gaussian_action(
    output: VLAActorOutput,
    log_standard_deviation: torch.Tensor,
    scaling: VLAActionScaling,
    *,
    minimum_log_standard_deviation: float,
    maximum_log_standard_deviation: float,
    deterministic: bool = False,
) -> SquashedGaussianAction:
    """Sample motion through tanh and grippers through sigmoid."""

    mean = output.action_chunks
    expected = (1, mean.shape[1], mean.shape[2])
    if tuple(log_standard_deviation.shape) != expected:
        raise ValueError("stochastic Actor log standard deviation shape differs")
    if not (
        math.isfinite(minimum_log_standard_deviation)
        and math.isfinite(maximum_log_standard_deviation)
        and minimum_log_standard_deviation < maximum_log_standard_deviation
    ):
        raise ValueError("stochastic Actor log standard deviation bounds are invalid")
    bounded_log_std = log_standard_deviation.clamp(
        minimum_log_standard_deviation,
        maximum_log_standard_deviation,
    )
    standard_deviation = bounded_log_std.exp()
    pre_transform = (
        mean
        if deterministic
        else mean + standard_deviation * torch.randn_like(mean)
    )
    motion = torch.tanh(pre_transform[..., :14])
    grippers = torch.sigmoid(pre_transform[..., 14:])
    scales = torch.tensor(
        (scaling.base_linear, scaling.base_angular, *(scaling.arm_velocity,) * 12),
        dtype=mean.dtype,
        device=mean.device,
    )
    values = torch.cat((motion * scales, grippers), dim=-1)
    gaussian_log_probability = -0.5 * (
        ((pre_transform - mean) / standard_deviation).square()
        + 2.0 * bounded_log_std
        + math.log(2.0 * math.pi)
    )
    motion_log_jacobian = torch.log1p(-motion.square() + 1.0e-6)
    gripper_log_jacobian = F.logsigmoid(pre_transform[..., 14:]) + F.logsigmoid(
        -pre_transform[..., 14:]
    )
    transform_log_jacobian = torch.cat(
        (motion_log_jacobian, gripper_log_jacobian), dim=-1
    )
    log_probability = (gaussian_log_probability - transform_log_jacobian).sum(
        dim=(1, 2)
    )
    return SquashedGaussianAction(values, log_probability)
