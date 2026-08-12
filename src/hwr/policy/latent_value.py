"""Distributional reward and safety value heads for imagined RL."""

from __future__ import annotations

from torch import nn


class LatentValueModel(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        *,
        bins: int = 255,
        hidden_dimension: int = 512,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        if min(latent_dimension, bins, hidden_dimension, hidden_layers) <= 0:
            raise ValueError("latent value dimensions must be positive")
        layers: list[nn.Module] = []
        input_dimension = latent_dimension
        for _ in range(hidden_layers):
            layers.extend(
                (
                    nn.Linear(input_dimension, hidden_dimension),
                    nn.LayerNorm(hidden_dimension),
                    nn.SiLU(),
                )
            )
            input_dimension = hidden_dimension
        layers.append(nn.Linear(input_dimension, bins))
        self.network = nn.Sequential(*layers)
        self.bins = bins

    def forward(self, latent):
        return self.network(latent)
