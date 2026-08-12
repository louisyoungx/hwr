from __future__ import annotations

import torch
import pytest

from hwr.perception.student import (
    VISUAL_STUDENT_INPUT_FIELDS,
    VisualStudentConfig,
    VisualStudentModel,
)


def _fixture_config() -> VisualStudentConfig:
    return VisualStudentConfig(
        image_size=32,
        visual_history=2,
        backbone_dimensions=(16, 24, 32, 48),
        backbone_depths=(1, 1, 1, 1),
        feature_dimension=32,
        state_queries=4,
        attention_heads=4,
        fusion_layers=1,
        temporal_layers=1,
        formal=False,
    )


def _inputs(config: VisualStudentConfig) -> dict[str, torch.Tensor]:
    batch, history, size = 1, config.visual_history, config.image_size
    identity = torch.eye(4).reshape(1, 1, 1, 4, 4).expand(batch, history, 4, 4, 4)
    return {
        "rgb": torch.rand(batch, history, 3, 3, size, size),
        "head_depth_m": torch.rand(batch, history, 1, size, size) + 0.2,
        "head_depth_valid": torch.ones(batch, history, 1, size, size, dtype=torch.bool),
        "camera_validity": torch.ones(batch, history, 4, dtype=torch.bool),
        "intrinsics": torch.ones(batch, history, 4, 4),
        "robot_from_camera": identity.clone(),
        "repeated_frame": torch.zeros(batch, history, dtype=torch.bool),
    }


def test_visual_student_preserves_spatial_features_and_fuses_history() -> None:
    config = _fixture_config()
    model = VisualStudentModel(config).eval()

    with torch.inference_mode():
        output = model(_inputs(config))

    assert output.spatial_features.shape == (1, 2, 3, 2, 2, 32)
    assert output.fused_tokens.shape == (1, 4, 32)
    assert output.pooled_state.shape == (1, 32)
    assert output.depth_prediction_m.shape == (1, 2, 2, 2)
    assert output.rgb_reconstruction.shape == (1, 2, 3, 3, 2, 2)
    assert output.spatial_validity.all()


def test_visual_student_masks_missing_camera_without_nan() -> None:
    config = _fixture_config()
    model = VisualStudentModel(config).eval()
    inputs = _inputs(config)
    inputs["camera_validity"].zero_()

    with torch.inference_mode():
        output = model(inputs)

    assert not output.spatial_validity.any()
    assert torch.isfinite(output.pooled_state).all()


def test_visual_student_rejects_field_or_shape_leaks() -> None:
    config = _fixture_config()
    model = VisualStudentModel(config)
    inputs = _inputs(config)
    inputs["reward"] = torch.zeros(1)
    with pytest.raises(ValueError, match="deployment contract"):
        model(inputs)
    inputs = _inputs(config)
    inputs["rgb"] = inputs["rgb"][:, :1]
    with pytest.raises(ValueError, match="shapes"):
        model(inputs)
    assert "reward" not in VISUAL_STUDENT_INPUT_FIELDS


def test_formal_visual_student_has_non_toy_capacity() -> None:
    model = VisualStudentModel(VisualStudentConfig())
    parameters = sum(value.numel() for value in model.parameters())

    assert 20_000_000 <= parameters <= 40_000_000
    assert model.config.image_size >= 160
    assert model.config.visual_history >= 4
