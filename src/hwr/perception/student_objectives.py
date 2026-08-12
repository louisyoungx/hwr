"""Action-free foundation distillation and geometric self-supervision."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from hwr.perception.student import VisualStudentOutput


@dataclass(frozen=True)
class VisualObjectiveConfig:
    student_dimension: int = 256
    siglip_dimension: int = 768
    dinov2_dimension: int = 384
    siglip_weight: float = 1.0
    dinov2_weight: float = 1.0
    depth_weight: float = 0.25
    reconstruction_weight: float = 0.25
    correspondence_weight: float = 0.5

    def __post_init__(self) -> None:
        dimensions = (
            self.student_dimension,
            self.siglip_dimension,
            self.dinov2_dimension,
        )
        weights = (
            self.siglip_weight,
            self.dinov2_weight,
            self.depth_weight,
            self.reconstruction_weight,
            self.correspondence_weight,
        )
        if min(dimensions) <= 0 or min(weights) < 0.0:
            raise ValueError("visual objective dimensions or weights are invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VisualTeacherTargets:
    siglip: torch.Tensor
    siglip_valid: torch.Tensor
    dinov2: torch.Tensor
    dinov2_valid: torch.Tensor
    rgb: torch.Tensor
    reconstruction_mask: torch.Tensor
    head_depth_m: torch.Tensor
    head_depth_valid: torch.Tensor
    correspondences: torch.Tensor


def _dense_cosine_loss(
    student: torch.Tensor,
    projection: nn.Linear,
    teacher: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    batch, history, cameras, rows, columns, dimension = student.shape
    projected = projection(student).reshape(
        batch * history * cameras, rows, columns, -1
    ).permute(0, 3, 1, 2)
    teacher_shape = teacher.shape
    projected = nn.functional.interpolate(
        projected,
        size=teacher_shape[-3:-1],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    projected = projected.reshape(*teacher_shape[:-1], teacher_shape[-1])
    cosine = nn.functional.cosine_similarity(projected, teacher, dim=-1)
    mask = valid.bool()
    if not bool(mask.any()):
        return student.sum() * 0.0
    return (1.0 - cosine[mask]).mean()


def _depth_loss(output: VisualStudentOutput, targets: VisualTeacherTargets) -> torch.Tensor:
    prediction = output.depth_prediction_m
    target = nn.functional.interpolate(
        targets.head_depth_m.flatten(0, 1),
        size=prediction.shape[-2:],
        mode="nearest",
    ).reshape_as(prediction)
    valid = nn.functional.interpolate(
        targets.head_depth_valid.float().flatten(0, 1),
        size=prediction.shape[-2:],
        mode="nearest",
    ).reshape_as(prediction).bool()
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    difference = torch.log(prediction[valid].clamp_min(1.0e-4))
    difference -= torch.log(target[valid].clamp_min(1.0e-4))
    return difference.square().mean() - 0.5 * difference.mean().square()


def _reconstruction_loss(
    output: VisualStudentOutput, targets: VisualTeacherTargets
) -> torch.Tensor:
    prediction = output.rgb_reconstruction.flatten(0, 2)
    target = nn.functional.interpolate(
        targets.rgb.flatten(0, 2),
        size=prediction.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).reshape_as(output.rgb_reconstruction)
    mask = nn.functional.interpolate(
        targets.reconstruction_mask.float().flatten(0, 2),
        size=prediction.shape[-2:],
        mode="nearest",
    ).reshape(output.rgb_reconstruction.shape[:3] + output.rgb_reconstruction.shape[-2:])
    mask = mask.bool() & output.spatial_validity
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    error = (output.rgb_reconstruction - target).abs().mean(dim=3)
    return error[mask].mean()


def _correspondence_loss(
    spatial: torch.Tensor, correspondences: torch.Tensor
) -> torch.Tensor:
    if correspondences.numel() == 0:
        return spatial.sum() * 0.0
    if correspondences.ndim != 2 or correspondences.shape[1] != 10:
        raise ValueError("visual correspondences must contain two five-axis indices")
    indices = correspondences.long()
    first = spatial[
        indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], indices[:, 4]
    ]
    second = spatial[
        indices[:, 5], indices[:, 6], indices[:, 7], indices[:, 8], indices[:, 9]
    ]
    return (1.0 - nn.functional.cosine_similarity(first, second, dim=-1)).mean()


class VisualFoundationObjectives(nn.Module):
    """Combine continuous teacher and geometry losses without action labels."""

    def __init__(self, config: VisualObjectiveConfig) -> None:
        super().__init__()
        self.config = config
        self.siglip_projection = nn.Linear(
            config.student_dimension, config.siglip_dimension
        )
        self.dinov2_projection = nn.Linear(
            config.student_dimension, config.dinov2_dimension
        )

    def forward(
        self, output: VisualStudentOutput, targets: VisualTeacherTargets
    ) -> dict[str, torch.Tensor]:
        losses = {
            "siglip": _dense_cosine_loss(
                output.spatial_features,
                self.siglip_projection,
                targets.siglip,
                targets.siglip_valid,
            ),
            "dinov2": _dense_cosine_loss(
                output.spatial_features,
                self.dinov2_projection,
                targets.dinov2,
                targets.dinov2_valid,
            ),
            "depth": _depth_loss(output, targets),
            "reconstruction": _reconstruction_loss(output, targets),
            "correspondence": _correspondence_loss(
                output.spatial_features, targets.correspondences
            ),
        }
        losses["total"] = (
            self.config.siglip_weight * losses["siglip"]
            + self.config.dinov2_weight * losses["dinov2"]
            + self.config.depth_weight * losses["depth"]
            + self.config.reconstruction_weight * losses["reconstruction"]
            + self.config.correspondence_weight * losses["correspondence"]
        )
        return losses
