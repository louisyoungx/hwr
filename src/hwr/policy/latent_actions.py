"""Map normalized latent Actor outputs into canonical runtime action units."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from hwr.core.embodied import DUAL_ARM_ACTION_DIM


@dataclass(frozen=True)
class LatentActionScaling:
    base_linear: float = 0.18
    base_angular: float = 0.50
    arm_velocity: float = 0.35

    def __post_init__(self) -> None:
        if min(self.base_linear, self.base_angular, self.arm_velocity) <= 0.0:
            raise ValueError("latent action scales must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def scale_latent_action(action: torch.Tensor, scaling: LatentActionScaling) -> torch.Tensor:
    if action.shape[-1] != DUAL_ARM_ACTION_DIM:
        raise ValueError("latent action scaling requires the canonical 16-D action")
    scales = action.new_tensor(
        (scaling.base_linear, scaling.base_angular, *(scaling.arm_velocity,) * 12)
    )
    motion = action[..., :14].clamp(-1.0, 1.0) * scales
    grippers = action[..., 14:].clamp(0.0, 1.0)
    return torch.cat((motion, grippers), dim=-1)
