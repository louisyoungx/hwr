from __future__ import annotations

import numpy as np
import pytest

from hwr.core.embodied import (
    DualArmObservation,
    DualArmProprioception,
    FrameCameraCalibration,
    NaturalLanguageInstruction,
)
from hwr.core.types import CameraFrame
from hwr.perception import CameraCalibration, PinholeIntrinsics
from hwr.perception.high_resolution import (
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
    align_depth_to_rgb,
)


CAMERAS = ("head_rgb", "head_depth", "left_wrist_rgb", "right_wrist_rgb")


def _observation(size: int = 4, right_timestamp: int = 1_000) -> DualArmObservation:
    rgb = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    depth = np.linspace(0.05, 6.0, size * size, dtype=np.float32).reshape(size, size)
    proprioception = DualArmProprioception(
        (0.0,) * 6, (0.0,) * 6, (0.0,) * 6, (0.0,) * 6,
        0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0),
    )
    frames = []
    for camera_id in CAMERAS:
        is_depth = camera_id == "head_depth"
        value = depth if is_depth else rgb
        timestamp = right_timestamp if camera_id == "right_wrist_rgb" else 1_000
        frames.append(
            CameraFrame(
                camera_id, timestamp, 0, size, size,
                "depth32f" if is_depth else "rgb8", payload=value.tobytes(),
            )
        )
    return DualArmObservation(
        1_000, 4, "fixture/v1", NaturalLanguageInstruction("Carry a container with both hands"),
        proprioception, tuple(frames),
    )


def _preprocessor(
    maximum_time_skew_ns: int = 50, source_size: int = 4
) -> HighResolutionVisionPreprocessor:
    intrinsics = PinholeIntrinsics(
        source_size, source_size, 3.0, 3.0, 1.5, 1.5
    )
    identity = (
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
    calibrations = {
        name: CameraCalibration("high-resolution-fixture", name, intrinsics, identity)
        for name in CAMERAS
    }
    return HighResolutionVisionPreprocessor(
        HighResolutionVisionConfig(
            teacher_image_size=224,
            student_image_size=160,
            minimum_source_short_side=4,
            maximum_time_skew_ns=maximum_time_skew_ns,
        ),
        calibrations,
    )


def test_high_resolution_preprocessing_is_deterministic_and_spatial() -> None:
    preprocessor = _preprocessor()
    first = preprocessor.preprocess(_observation())
    second = preprocessor.preprocess(_observation())

    assert first.teacher_rgb.shape == (3, 224, 224, 3)
    assert first.student_rgb.shape == (3, 160, 160, 3)
    assert first.student_head_depth_m.shape == (160, 160)
    assert first.camera_validity.tolist() == [True, True, True, True]
    assert first.source_sha256 == second.source_sha256
    assert np.array_equal(first.teacher_rgb, second.teacher_rgb)
    assert not first.student_rgb.flags.writeable


def test_high_resolution_preprocessing_keeps_depth_independent_and_masks_stale_rgb() -> None:
    value = _preprocessor(maximum_time_skew_ns=10).preprocess(
        _observation(right_timestamp=2_000)
    )

    assert value.camera_validity.tolist() == [True, True, True, False]
    assert np.all(value.student_rgb[2] == 0.0)
    assert value.student_head_depth_valid.any()
    assert np.all(value.student_head_depth_m[~value.student_head_depth_valid] == 0.0)


def test_formal_high_resolution_config_rejects_toy_inputs() -> None:
    with pytest.raises(ValueError, match="toy"):
        HighResolutionVisionConfig(teacher_image_size=64)
    with pytest.raises(ValueError, match="toy"):
        HighResolutionVisionConfig(student_image_size=32)


def test_source_resolution_minimum_is_enforced() -> None:
    preprocessor = _preprocessor(source_size=3)
    with pytest.raises(ValueError, match="formal minimum"):
        preprocessor.preprocess(_observation(size=3))


def test_high_resolution_preprocessor_uses_per_frame_wrist_geometry() -> None:
    observation = _observation(size=160)
    dynamic = []
    for index, name in enumerate(CAMERAS):
        transform = np.eye(4, dtype=np.float64)
        transform[1, 3] = float(index)
        dynamic.append(
            FrameCameraCalibration(
                name,
                (120.0, 121.0, 70.0, 71.0),
                tuple(transform.reshape(-1)),
            )
        )
    observation = DualArmObservation(
        **{**observation.__dict__, "camera_calibrations": tuple(dynamic)}
    )

    result = _preprocessor(source_size=160).preprocess(observation)

    assert result.student_intrinsics[2].tolist() == [120.0, 121.0, 70.0, 71.0]
    assert result.robot_from_camera[2, 1, 3] == 2.0


def test_depth_is_projected_into_rgb_pixels_using_dynamic_baseline() -> None:
    depth = np.zeros((5, 5), np.float32)
    valid = np.zeros((5, 5), np.bool_)
    depth[2, 2] = 1.0
    valid[2, 2] = True
    intrinsics = np.asarray((10.0, 10.0, 2.0, 2.0), np.float32)
    robot_from_depth = np.eye(4, dtype=np.float32)
    robot_from_depth[0, 3] = 0.1

    aligned, aligned_valid = align_depth_to_rgb(
        depth,
        valid,
        intrinsics,
        intrinsics,
        robot_from_depth,
        np.eye(4, dtype=np.float32),
    )

    assert aligned_valid.sum() == 1
    assert aligned_valid[2, 3]
    assert aligned[2, 3] == 1.0
    assert not aligned_valid[2, 2]
