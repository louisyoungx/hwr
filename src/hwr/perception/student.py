"""Deployable multi-camera visual student with spatial and temporal fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, NamedTuple

import torch
from torch import nn


VISUAL_STUDENT_INPUT_FIELDS = frozenset(
    {
        "rgb",
        "head_depth_m",
        "head_depth_valid",
        "camera_validity",
        "intrinsics",
        "robot_from_camera",
        "repeated_frame",
    }
)


@dataclass(frozen=True)
class VisualStudentConfig:
    image_size: int = 160
    visual_history: int = 4
    backbone_dimensions: tuple[int, int, int, int] = (96, 192, 384, 512)
    backbone_depths: tuple[int, int, int, int] = (3, 3, 9, 3)
    feature_dimension: int = 256
    state_queries: int = 16
    attention_heads: int = 8
    fusion_layers: int = 2
    temporal_layers: int = 2
    formal: bool = True

    def __post_init__(self) -> None:
        values = (
            self.image_size,
            self.visual_history,
            *self.backbone_dimensions,
            *self.backbone_depths,
            self.feature_dimension,
            self.state_queries,
            self.attention_heads,
            self.fusion_layers,
            self.temporal_layers,
        )
        if min(values) <= 0:
            raise ValueError("visual student dimensions must be positive")
        if len(self.backbone_dimensions) != 4 or len(self.backbone_depths) != 4:
            raise ValueError("visual student requires four backbone stages")
        if self.image_size % 32:
            raise ValueError("visual student image size must be divisible by 32")
        if self.feature_dimension % self.attention_heads:
            raise ValueError("visual student feature dimension must divide attention heads")
        if self.formal and (self.image_size < 160 or self.visual_history < 4):
            raise ValueError("formal visual student requires high resolution and history")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VisualStudentOutput(NamedTuple):
    spatial_features: torch.Tensor
    fused_tokens: torch.Tensor
    pooled_state: torch.Tensor
    depth_prediction_m: torch.Tensor
    rgb_reconstruction: torch.Tensor
    spatial_validity: torch.Tensor


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            dimension, dimension, kernel_size=7, padding=3, groups=dimension
        )
        self.norm = nn.LayerNorm(dimension, eps=1.0e-6)
        self.expand = nn.Linear(dimension, 4 * dimension)
        self.contract = nn.Linear(4 * dimension, dimension)
        self.scale = nn.Parameter(torch.full((dimension,), 1.0e-6))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value).permute(0, 2, 3, 1)
        value = self.contract(nn.functional.gelu(self.expand(self.norm(value))))
        value = (value * self.scale).permute(0, 3, 1, 2)
        return residual + value


class _ConvNeXtBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dimensions: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        feature_dimension: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dimensions[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dimensions[0]),
        )
        self.stages = nn.ModuleList(
            [nn.Sequential(*(_ConvNeXtBlock(dimensions[i]) for _ in range(depths[i]))) for i in range(4)]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, dimensions[i]),
                    nn.Conv2d(dimensions[i], dimensions[i + 1], kernel_size=2, stride=2),
                )
                for i in range(3)
            ]
        )
        self.local_projection = nn.Conv2d(dimensions[2], feature_dimension, 1)
        self.context_projection = nn.Conv2d(dimensions[3], feature_dimension, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        local = None
        for index, stage in enumerate(self.stages):
            value = stage(value)
            if index == 2:
                local = self.local_projection(value)
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        assert local is not None
        context = nn.functional.interpolate(
            self.context_projection(value),
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return local + context


class _CameraFusion(nn.Module):
    def __init__(self, config: VisualStudentConfig) -> None:
        super().__init__()
        dimension = config.feature_dimension
        self.calibration = nn.Sequential(
            nn.Linear(23, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.queries = nn.Parameter(torch.randn(config.state_queries, dimension) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            dimension, config.attention_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=config.attention_heads,
            dim_feedforward=dimension * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refine = nn.TransformerEncoder(
            layer, config.fusion_layers, enable_nested_tensor=False
        )

    def forward(
        self,
        spatial: torch.Tensor,
        camera_valid: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        batch_time, cameras, rows, columns, dimension = spatial.shape
        identity = torch.eye(cameras, device=spatial.device, dtype=spatial.dtype)
        identity = identity[None].expand(batch_time, -1, -1)
        calibration = torch.cat(
            (intrinsics, extrinsics.flatten(-2), identity), dim=-1
        )
        camera_embedding = self.calibration(calibration)
        tokens = spatial + camera_embedding[:, :, None, None]
        tokens = tokens.reshape(batch_time, cameras * rows * columns, dimension)
        missing = (~camera_valid).repeat_interleave(rows * columns, dim=1)
        all_missing = missing.all(dim=1)
        if bool(all_missing.any()):
            missing = missing.clone()
            missing[all_missing, 0] = False
            tokens = tokens.clone()
            tokens[all_missing, 0] = 0.0
        queries = self.queries[None].expand(batch_time, -1, -1)
        attended, _ = self.cross_attention(
            queries, tokens, tokens, key_padding_mask=missing, need_weights=False
        )
        return self.refine(self.cross_norm(queries + attended))


class VisualStudentModel(nn.Module):
    """Encode RGB-D history into dense camera features and fused latent state."""

    def __init__(self, config: VisualStudentConfig) -> None:
        super().__init__()
        self.config = config
        self.rgb_backbone = _ConvNeXtBackbone(
            3,
            config.backbone_dimensions,
            config.backbone_depths,
            config.feature_dimension,
        )
        depth_dimensions = tuple(max(16, value // 4) for value in config.backbone_dimensions)
        self.depth_backbone = _ConvNeXtBackbone(
            2, depth_dimensions, (1, 1, 2, 1), config.feature_dimension
        )
        self.head_rgbd_fusion = nn.Sequential(
            nn.Conv2d(2 * config.feature_dimension, config.feature_dimension, 1),
            nn.SiLU(),
        )
        self.camera_fusion = _CameraFusion(config)
        self.temporal_position = nn.Parameter(
            torch.randn(config.visual_history, config.state_queries, config.feature_dimension) * 0.01
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feature_dimension * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_fusion = nn.TransformerEncoder(
            temporal_layer, config.temporal_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(config.feature_dimension)
        self.depth_head = nn.Sequential(
            nn.Conv2d(config.feature_dimension, config.feature_dimension // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.feature_dimension // 2, 1, 1),
            nn.Softplus(),
        )
        self.rgb_head = nn.Sequential(
            nn.Conv2d(config.feature_dimension, config.feature_dimension // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.feature_dimension // 2, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> VisualStudentOutput:
        if frozenset(inputs) != VISUAL_STUDENT_INPUT_FIELDS:
            raise ValueError("visual student input fields violate the deployment contract")
        self._check_shapes(inputs)
        rgb = inputs["rgb"]
        batch, history, cameras = rgb.shape[:3]
        flattened_rgb = rgb.reshape(-1, 3, self.config.image_size, self.config.image_size)
        encoded_rgb = self.rgb_backbone(flattened_rgb)
        rows, columns = encoded_rgb.shape[-2:]
        spatial = encoded_rgb.reshape(
            batch, history, cameras, self.config.feature_dimension, rows, columns
        )
        depth_input = torch.cat(
            (inputs["head_depth_m"], inputs["head_depth_valid"].to(rgb.dtype)), dim=2
        )
        depth = self.depth_backbone(depth_input.reshape(-1, 2, *depth_input.shape[-2:]))
        head = torch.cat((spatial[:, :, 0].flatten(0, 1), depth), dim=1)
        spatial = spatial.clone()
        spatial[:, :, 0] = self.head_rgbd_fusion(head).reshape(
            batch, history, self.config.feature_dimension, rows, columns
        )
        rgb_validity = inputs["camera_validity"][..., (0, 2, 3)].bool()
        spatial = spatial * rgb_validity[..., None, None, None]
        spatial_channels_last = spatial.permute(0, 1, 2, 4, 5, 3)
        fused = self.camera_fusion(
            spatial_channels_last.flatten(0, 1),
            rgb_validity.flatten(0, 1),
            inputs["intrinsics"][..., (0, 2, 3), :].flatten(0, 1),
            inputs["robot_from_camera"][..., (0, 2, 3), :, :].flatten(0, 1),
        ).reshape(batch, history, self.config.state_queries, self.config.feature_dimension)
        repeated = inputs["repeated_frame"].to(fused.dtype)
        fused = fused * (1.0 - repeated[..., None, None])
        temporal = fused + self.temporal_position[None]
        temporal = self.temporal_fusion(temporal.flatten(1, 2))
        current = self.output_norm(temporal[:, -self.config.state_queries :])
        spatial_flat = spatial.flatten(0, 2)
        reconstruction = self.rgb_head(spatial_flat).reshape(
            batch, history, cameras, 3, rows, columns
        )
        depth_prediction = self.depth_head(spatial[:, :, 0].flatten(0, 1)).reshape(
            batch, history, rows, columns
        )
        patch_valid = rgb_validity[..., None, None].expand(-1, -1, -1, rows, columns)
        return VisualStudentOutput(
            spatial_channels_last,
            current,
            current.mean(dim=1),
            depth_prediction,
            reconstruction,
            patch_valid,
        )

    def _check_shapes(self, inputs: Mapping[str, torch.Tensor]) -> None:
        rgb = inputs["rgb"]
        if rgb.ndim != 6:
            raise ValueError("visual student RGB must be batch-history-camera-channel-height-width")
        batch = rgb.shape[0]
        expected = {
            "rgb": (batch, self.config.visual_history, 3, 3, self.config.image_size, self.config.image_size),
            "head_depth_m": (batch, self.config.visual_history, 1, self.config.image_size, self.config.image_size),
            "head_depth_valid": (batch, self.config.visual_history, 1, self.config.image_size, self.config.image_size),
            "camera_validity": (batch, self.config.visual_history, 4),
            "intrinsics": (batch, self.config.visual_history, 4, 4),
            "robot_from_camera": (batch, self.config.visual_history, 4, 4, 4),
            "repeated_frame": (batch, self.config.visual_history),
        }
        mismatches = {
            name: (tuple(inputs[name].shape), shape)
            for name, shape in expected.items()
            if tuple(inputs[name].shape) != shape
        }
        if mismatches:
            raise ValueError(f"visual student tensor shapes are invalid: {mismatches}")
