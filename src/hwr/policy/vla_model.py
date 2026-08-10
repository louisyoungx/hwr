"""Compact Transformer Actor for local visual-language-action training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, NamedTuple

import torch
from torch import nn

from hwr.core.embodied import DUAL_ARM_ACTION_DIM
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS


@dataclass(frozen=True)
class VLAActorConfig:
    visual_history: int
    action_history: int
    proprioception_dim: int
    language_dim: int
    point_count: int
    action_chunk_size: int
    hidden_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.0
    action_dim: int = DUAL_ARM_ACTION_DIM
    action_head_init_scale: float = 1.0e-3
    isolated_gripper_head: bool = False

    def __post_init__(self) -> None:
        dimensions = (
            self.visual_history,
            self.action_history,
            self.proprioception_dim,
            self.language_dim,
            self.point_count,
            self.action_chunk_size,
            self.hidden_dim,
            self.attention_heads,
            self.transformer_layers,
        )
        if min(dimensions) <= 0 or self.action_dim != DUAL_ARM_ACTION_DIM:
            raise ValueError("VLA Actor dimensions are invalid")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("VLA hidden size must be divisible by attention heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("VLA dropout must be in [0, 1)")
        if not math.isfinite(self.action_head_init_scale) or not (
            0.0 < self.action_head_init_scale <= 1.0e-2
        ):
            raise ValueError("VLA action head initialization scale must be in (0, 0.01]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VLAActorOutput(NamedTuple):
    action_chunks: torch.Tensor
    stop_logits: torch.Tensor


class _ImageEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class VLAActorModel(nn.Module):
    """Fuse deployable tensors and directly predict a dual-arm action chunk."""

    def __init__(self, config: VLAActorConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.head_encoder = _ImageEncoder(5, hidden)
        self.wrist_encoder = _ImageEncoder(3, hidden)
        self.point_encoder = nn.Sequential(
            nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, hidden), nn.SiLU()
        )
        self.language_encoder = nn.Sequential(
            nn.Linear(config.language_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.proprioception_encoder = nn.Sequential(
            nn.Linear(config.proprioception_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.action_history_encoder = nn.Sequential(
            nn.Linear(config.action_history * config.action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.camera_quality_encoder = nn.Linear(3, hidden)
        self.token_type = nn.Parameter(torch.zeros(7, hidden))
        self.temporal_position = nn.Parameter(
            torch.zeros(config.visual_history, hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.attention_heads,
            dim_feedforward=hidden * 3,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden)
        if config.isolated_gripper_head:
            self.motion_head = nn.Linear(hidden, config.action_chunk_size * 14)
            self.gripper_head = nn.Linear(hidden, config.action_chunk_size * 2)
            self._initialize_action_head(self.motion_head)
            self._initialize_action_head(self.gripper_head)
        else:
            self.action_head = nn.Linear(
                hidden, config.action_chunk_size * config.action_dim
            )
            self._initialize_action_head(self.action_head)
        self.stop_head = nn.Linear(hidden, config.action_chunk_size)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> VLAActorOutput:
        if frozenset(inputs) != VLA_POLICY_INPUT_FIELDS:
            raise ValueError("VLA Actor received fields outside its deployment whitelist")
        self._check_shapes(inputs)
        head_tokens = self._head_tokens(inputs)
        wrist_tokens = self._wrist_tokens(inputs)
        point_tokens = self._point_tokens(inputs)
        global_tokens = self._global_tokens(inputs)
        tokens = torch.cat((global_tokens, head_tokens, wrist_tokens, point_tokens), dim=1)
        encoded = self.transformer(tokens)
        summary = self.output_norm(encoded[:, 0])
        actions = self._action_chunks(summary)
        return VLAActorOutput(actions, self.stop_head(summary))

    def _initialize_action_head(self, head: nn.Linear) -> None:
        nn.init.uniform_(
            head.weight,
            -self.config.action_head_init_scale,
            self.config.action_head_init_scale,
        )
        nn.init.zeros_(head.bias)

    def _action_chunks(self, summary: torch.Tensor) -> torch.Tensor:
        batch = summary.shape[0]
        chunks = self.config.action_chunk_size
        if not self.config.isolated_gripper_head:
            return self.action_head(summary).reshape(
                batch, chunks, self.config.action_dim
            )
        motion = self.motion_head(summary).reshape(batch, chunks, 14)
        grippers = self.gripper_head(summary.detach()).reshape(batch, chunks, 2)
        return torch.cat((motion, grippers), dim=-1)

    def _head_tokens(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rgb = inputs["head_rgb"]
        depth = inputs["head_depth"].unsqueeze(-1)
        valid = inputs["head_depth_valid"].unsqueeze(-1).to(rgb.dtype)
        value = torch.cat((rgb, depth, valid), dim=-1).permute(0, 1, 4, 2, 3)
        batch, history = value.shape[:2]
        tokens = self.head_encoder(value.reshape(batch * history, 5, *value.shape[-2:]))
        quality = inputs["camera_validity"][..., :2].all(dim=-1, keepdim=True)
        return self._temporal(tokens.reshape(batch, history, -1), quality, 3)

    def _wrist_tokens(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        streams = []
        for name, validity_index, token_type in (
            ("left_wrist_rgb", 2, 4),
            ("right_wrist_rgb", 3, 5),
        ):
            value = inputs[name].permute(0, 1, 4, 2, 3)
            batch, history = value.shape[:2]
            tokens = self.wrist_encoder(
                value.reshape(batch * history, 3, *value.shape[-2:])
            )
            quality = inputs["camera_validity"][..., validity_index : validity_index + 1]
            streams.append(
                self._temporal(tokens.reshape(batch, history, -1), quality, token_type)
            )
        return torch.cat(streams, dim=1)

    def _point_tokens(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        points = inputs["head_points"]
        valid = inputs["head_point_valid"].unsqueeze(-1).bool()
        features = self.point_encoder(points)
        minimum = torch.finfo(features.dtype).min
        pooled = features.masked_fill(~valid, minimum).amax(dim=2)
        any_valid = valid.any(dim=2)
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        return self._temporal(pooled, any_valid, 6)

    def _global_tokens(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        language = self.language_encoder(inputs["instruction_embedding"]) + self.token_type[0]
        proprioception = self.proprioception_encoder(inputs["proprioception"]) + self.token_type[1]
        history = self.action_history_encoder(inputs["action_history"].flatten(1))
        history = history + self.token_type[2]
        return torch.stack((language, proprioception, history), dim=1)

    def _temporal(
        self, tokens: torch.Tensor, validity: torch.Tensor, token_type: int
    ) -> torch.Tensor:
        quality = validity.to(tokens.dtype)
        tokens = tokens * quality
        tokens = tokens + self.camera_quality_encoder(
            quality.expand(-1, -1, 3)
        )
        return tokens + self.temporal_position[None] + self.token_type[token_type]

    def _check_shapes(self, inputs: Mapping[str, torch.Tensor]) -> None:
        config = self.config
        batch = inputs["head_rgb"].shape[0]
        shapes = {
            "head_rgb": (batch, config.visual_history, None, None, 3),
            "head_depth": (batch, config.visual_history, None, None),
            "head_depth_valid": (batch, config.visual_history, None, None),
            "head_points": (batch, config.visual_history, config.point_count, 6),
            "head_point_valid": (batch, config.visual_history, config.point_count),
            "left_wrist_rgb": (batch, config.visual_history, None, None, 3),
            "right_wrist_rgb": (batch, config.visual_history, None, None, 3),
            "camera_validity": (batch, config.visual_history, 4),
            "proprioception": (batch, config.proprioception_dim),
            "instruction_embedding": (batch, config.language_dim),
            "action_history": (batch, config.action_history, config.action_dim),
        }
        for name, expected in shapes.items():
            actual = inputs[name].shape
            if len(actual) != len(expected) or any(
                value is not None and actual[index] != value
                for index, value in enumerate(expected)
            ):
                raise ValueError(f"VLA Actor tensor {name} has shape {tuple(actual)}")
