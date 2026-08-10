from __future__ import annotations

import numpy as np
import pytest

from hwr.core.types import CameraFrame, ObservationFrame
from hwr.perception import (
    CameraCalibration,
    DeterministicVisionPreprocessor,
    PinholeIntrinsics,
    VisionPreprocessConfig,
)


def _calibrations() -> dict[str, CameraCalibration]:
    intrinsics = PinholeIntrinsics(3, 2, 2.0, 2.0, 1.0, 0.5)
    transform = (
        1.0, 0.0, 0.0, 0.1,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.2,
        0.0, 0.0, 0.0, 1.0,
    )
    return {
        name: CameraCalibration("test-calibration", name, intrinsics, transform)
        for name in ("head_rgb", "head_depth", "wrist_rgb")
    }


def _observation(*, wrist_timestamp: int = 1_000, features=None) -> ObservationFrame:
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.asarray([[1.0, np.nan, 6.0], [0.05, 2.0, 3.0]], dtype=np.float32)
    cameras = (
        CameraFrame("head_rgb", 1_000, 0, 3, 2, "rgb8", payload=rgb.tobytes()),
        CameraFrame("head_depth", 1_000, 0, 3, 2, "depth32f", payload=depth.tobytes()),
        CameraFrame("wrist_rgb", wrist_timestamp, 0, 3, 2, "rgb8", payload=rgb.tobytes()),
    )
    return ObservationFrame(
        1_000,
        0,
        "housework/v1",
        "instruction_following",
        cameras=cameras,
        features={} if features is None else features,
    )


def _preprocessor(maximum_time_skew_ns: int = 50) -> DeterministicVisionPreprocessor:
    config = VisionPreprocessConfig(
        image_width=3,
        image_height=2,
        point_count=6,
        maximum_time_skew_ns=maximum_time_skew_ns,
    )
    return DeterministicVisionPreprocessor(config, _calibrations())


def test_rgbd_preprocessing_is_deterministic_finite_and_read_only() -> None:
    preprocessor = _preprocessor()

    first = preprocessor.preprocess(_observation())
    second = preprocessor.preprocess(_observation())

    assert np.array_equal(first.head_rgb, second.head_rgb)
    assert np.array_equal(first.head_depth, second.head_depth)
    assert np.array_equal(first.head_points, second.head_points)
    assert first.preprocess_fingerprint == second.preprocess_fingerprint
    assert first.head_depth_valid.sum() == 3
    assert np.isfinite(first.head_depth).all()
    assert np.all(first.head_points[~first.head_point_valid] == 0.0)
    assert not first.head_rgb.flags.writeable


def test_stale_camera_is_zeroed_and_exposed_through_validity_mask() -> None:
    value = _preprocessor(maximum_time_skew_ns=10).preprocess(
        _observation(wrist_timestamp=2_000)
    )

    assert value.camera_validity.tolist() == [True, True, False]
    assert np.all(value.wrist_rgb == 0.0)


def test_visual_preprocessing_rejects_privileged_features() -> None:
    with pytest.raises(ValueError, match="privileged"):
        _preprocessor().preprocess(_observation(features={"object_pose": (1.0,)}))
