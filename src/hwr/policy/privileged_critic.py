"""Training-only privileged critic kept separate from the deployable Actor."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from hwr.core.embodied import DUAL_ARM_ACTION_DIM


@dataclass(frozen=True)
class PrivilegedCriticConfig:
    privileged_state_dim: int
    action_chunk_size: int
    hidden_dim: int = 256
    action_dim: int = DUAL_ARM_ACTION_DIM

    def __post_init__(self) -> None:
        if min(self.privileged_state_dim, self.action_chunk_size, self.hidden_dim) <= 0:
            raise ValueError("privileged critic dimensions must be positive")
        if self.action_dim != DUAL_ARM_ACTION_DIM:
            raise ValueError("critic action dimension differs from the dual-arm contract")

    @property
    def action_representation_dim(self) -> int:
        return self.action_chunk_size * (self.action_dim + 1)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _QNetwork(nn.Module):
    def __init__(self, config: PrivilegedCriticConfig) -> None:
        super().__init__()
        input_dim = config.privileged_state_dim + config.action_representation_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, action), dim=1)).squeeze(1)


class TwinPrivilegedCritic(nn.Module):
    """Twin Q estimators that exist only in the simulation training process."""

    def __init__(self, config: PrivilegedCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.q1 = _QNetwork(config)
        self.q2 = _QNetwork(config)

    def forward(
        self, privileged_state: torch.Tensor, action_representation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_state = (privileged_state.shape[0], self.config.privileged_state_dim)
        expected_action = (
            privileged_state.shape[0],
            self.config.action_representation_dim,
        )
        if tuple(privileged_state.shape) != expected_state:
            raise ValueError("privileged critic state shape is invalid")
        if tuple(action_representation.shape) != expected_action:
            raise ValueError("privileged critic action shape is invalid")
        return (
            self.q1(privileged_state, action_representation),
            self.q2(privileged_state, action_representation),
        )
