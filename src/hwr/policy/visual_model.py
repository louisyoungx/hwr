"""Compact multi-view visual policy network for local household training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn


@dataclass(frozen=True)
class VisualModelConfig:
    image_width: int
    image_height: int
    action_history: int
    instruction_count: int = 1
    phase_count: int = 1
    proprioception_dim: int = 24
    action_dim: int = 9
    visual_channels: tuple[int, ...] = (16, 32, 48)
    hidden_dim: int = 192
    phase_embedding_dim: int = 16

    def __post_init__(self) -> None:
        values = (
            self.image_width,
            self.image_height,
            self.action_history,
            self.instruction_count,
            self.phase_count,
            self.proprioception_dim,
            self.action_dim,
            self.hidden_dim,
            self.phase_embedding_dim,
            *self.visual_channels,
        )
        if min(values) <= 0 or self.action_dim != 9:
            raise ValueError("visual model dimensions are invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HouseholdVisualPolicyModel(nn.Module):
    """Fuse head RGB-D, wrist RGB, proprioception, instruction, and action history."""

    def __init__(self, config: VisualModelConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        channels = 7
        for output_channels in config.visual_channels:
            layers.extend(
                (
                    nn.Conv2d(channels, output_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(4, output_channels),
                    nn.SiLU(),
                )
            )
            channels = output_channels
        self.visual_encoder = nn.Sequential(*layers)
        feature_height = _stride_two_size(config.image_height, len(config.visual_channels))
        feature_width = _stride_two_size(config.image_width, len(config.visual_channels))
        visual_dim = channels * feature_height * feature_width
        self.instruction_embedding = nn.Embedding(config.instruction_count, 8)
        state_dim = config.proprioception_dim + config.action_history * 9 + 8
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + 128, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.phase_head = nn.Linear(config.hidden_dim, config.phase_count)
        self.phase_embedding = nn.Embedding(
            config.phase_count, config.phase_embedding_dim
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.phase_embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(
        self,
        head_rgb: torch.Tensor,
        head_depth: torch.Tensor,
        wrist_rgb: torch.Tensor,
        proprioception: torch.Tensor,
        action_history: torch.Tensor,
        instruction_id: torch.Tensor,
    ) -> "VisualModelOutput":
        visual = torch.cat((head_rgb, head_depth, wrist_rgb), dim=1)
        visual_features = self.visual_encoder(visual).flatten(1)
        instruction = self.instruction_embedding(instruction_id.flatten().long())
        state = torch.cat((proprioception, action_history.flatten(1), instruction), dim=1)
        state_features = self.state_encoder(state)
        fused = self.fusion(torch.cat((visual_features, state_features), dim=1))
        phase_embeddings = self.phase_embedding.weight.unsqueeze(0).expand(
            fused.shape[0], -1, -1
        )
        phase_features = fused.unsqueeze(1).expand(-1, self.config.phase_count, -1)
        actions = self.action_decoder(
            torch.cat((phase_features, phase_embeddings), dim=2)
        )
        return VisualModelOutput(actions=actions, phase_logits=self.phase_head(fused))


class VisualModelOutput(NamedTuple):
    actions: torch.Tensor
    phase_logits: torch.Tensor


def _stride_two_size(size: int, layers: int) -> int:
    for _ in range(layers):
        size = (size + 1) // 2
    return size
