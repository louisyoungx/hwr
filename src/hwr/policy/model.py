"""Reference neural behavior cloning model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    observation_dim: int
    continuous_action_dim: int = 4
    hidden_dims: tuple[int, ...] = (128, 128)

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.continuous_action_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if not self.hidden_dims or min(self.hidden_dims) <= 0:
            raise ValueError("hidden dimensions must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BehaviorMLP(nn.Module):
    """Predict normalized continuous controls plus a gripper logit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        input_dim = config.observation_dim
        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, config.continuous_action_dim + 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)

