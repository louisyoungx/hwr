"""Shared mapping from VLA network heads to executable action units."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hwr.policy.vla_model import VLAActorOutput


@dataclass(frozen=True)
class VLAActionScaling:
    base_linear: float = 0.45
    base_angular: float = 1.0
    arm_velocity: float = 1.2

    def __post_init__(self) -> None:
        if min(self.base_linear, self.base_angular, self.arm_velocity) <= 0:
            raise ValueError("VLA action scales must be positive")


def bounded_vla_actions(
    output: VLAActorOutput, scaling: VLAActionScaling
) -> torch.Tensor:
    """Map unconstrained Actor outputs to canonical velocity and gripper units."""
    raw = output.action_chunks
    motion = torch.tanh(raw[..., :14])
    scales = torch.tensor(
        (scaling.base_linear, scaling.base_angular, *(scaling.arm_velocity,) * 12),
        dtype=raw.dtype,
        device=raw.device,
    )
    grippers = torch.sigmoid(raw[..., 14:16])
    return torch.cat((motion * scales, grippers), dim=-1)
