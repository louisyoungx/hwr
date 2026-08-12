from __future__ import annotations

import torch

from hwr.perception.student import VisualStudentOutput
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
    VisualTeacherTargets,
)


def _fixture() -> tuple[VisualStudentOutput, VisualTeacherTargets]:
    spatial = torch.randn(1, 2, 3, 2, 2, 8, requires_grad=True)
    output = VisualStudentOutput(
        spatial_features=spatial,
        fused_tokens=torch.randn(1, 4, 8, requires_grad=True),
        pooled_state=torch.randn(1, 8, requires_grad=True),
        depth_prediction_m=torch.ones(1, 2, 2, 2, requires_grad=True),
        rgb_reconstruction=torch.rand(1, 2, 3, 3, 2, 2, requires_grad=True),
        spatial_validity=torch.ones(1, 2, 3, 2, 2, dtype=torch.bool),
    )
    targets = VisualTeacherTargets(
        siglip=torch.randn(1, 2, 3, 3, 3, 12),
        siglip_valid=torch.ones(1, 2, 3, 3, 3, dtype=torch.bool),
        dinov2=torch.randn(1, 2, 3, 4, 4, 10),
        dinov2_valid=torch.ones(1, 2, 3, 4, 4, dtype=torch.bool),
        rgb=torch.rand(1, 2, 3, 3, 8, 8),
        reconstruction_mask=torch.ones(1, 2, 3, 1, 8, 8, dtype=torch.bool),
        head_depth_m=torch.ones(1, 2, 1, 8, 8),
        head_depth_valid=torch.ones(1, 2, 1, 8, 8, dtype=torch.bool),
        correspondences=torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]),
    )
    return output, targets


def test_action_free_visual_objectives_are_finite_and_differentiable() -> None:
    output, targets = _fixture()
    objective = VisualFoundationObjectives(
        VisualObjectiveConfig(student_dimension=8, siglip_dimension=12, dinov2_dimension=10)
    )

    losses = objective(output, targets)
    losses["total"].backward()

    assert set(losses) == {
        "siglip", "dinov2", "depth", "reconstruction", "correspondence", "total"
    }
    assert all(torch.isfinite(value) for value in losses.values())
    assert output.spatial_features.grad is not None
    assert objective.siglip_projection.weight.grad is not None


def test_empty_masks_produce_zero_auxiliary_losses() -> None:
    output, targets = _fixture()
    targets = VisualTeacherTargets(
        **{
            **targets.__dict__,
            "siglip_valid": torch.zeros_like(targets.siglip_valid),
            "dinov2_valid": torch.zeros_like(targets.dinov2_valid),
            "reconstruction_mask": torch.zeros_like(targets.reconstruction_mask),
            "head_depth_valid": torch.zeros_like(targets.head_depth_valid),
            "correspondences": torch.empty((0, 10), dtype=torch.long),
        }
    )
    objective = VisualFoundationObjectives(
        VisualObjectiveConfig(student_dimension=8, siglip_dimension=12, dinov2_dimension=10)
    )

    losses = objective(output, targets)

    assert losses["siglip"] == 0.0
    assert losses["dinov2"] == 0.0
    assert losses["depth"] == 0.0
    assert losses["reconstruction"] == 0.0
    assert losses["correspondence"] == 0.0
