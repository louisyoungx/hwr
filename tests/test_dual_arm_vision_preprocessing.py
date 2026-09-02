from __future__ import annotations

import numpy as np
import pytest

from hwr.core.embodied import (
    DualArmObservation,
    DualArmProprioception,
    NaturalLanguageInstruction,
)
from hwr.core.types import CameraFrame
from hwr.perception import (
    CameraCalibration,
    DualArmVisionPreprocessor,
    PinholeIntrinsics,
    VisionPreprocessConfig,
)


CAMERAS = ("head_rgb", "head_depth", "left_wrist_rgb", "right_wrist_rgb")


def _observation(*, right_timestamp: int = 1_000) -> DualArmObservation:
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.asarray([[1.0, np.nan, 6.0], [0.05, 2.0, 3.0]], dtype=np.float32)
    proprioception = DualArmProprioception(
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        0.0,
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0),
    )
    return DualArmObservation(
        timestamp_ns=1_000,
        sequence_id=0,
        task_id="carry_tray/v1",
        instruction=NaturalLanguageInstruction("Carry a tray steadily with both hands"),
        proprioception=proprioception,
        cameras=(
            CameraFrame("head_rgb", 1_000, 0, 3, 2, "rgb8", payload=rgb.tobytes()),
            CameraFrame(
                "head_depth", 1_000, 0, 3, 2, "depth32f", payload=depth.tobytes()
            ),
            CameraFrame(
                "left_wrist_rgb", 1_000, 0, 3, 2, "rgb8", payload=rgb.tobytes()
            ),
            CameraFrame(
                "right_wrist_rgb",
                right_timestamp,
                0,
                3,
                2,
                "rgb8",
                payload=rgb.tobytes(),
            ),
        ),
    )


def _preprocessor(maximum_time_skew_ns: int = 50) -> DualArmVisionPreprocessor:
    intrinsics = PinholeIntrinsics(3, 2, 2.0, 2.0, 1.0, 0.5)
    identity = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    calibrations = {
        name: CameraCalibration("dual-test", name, intrinsics, identity)
        for name in CAMERAS
    }
    return DualArmVisionPreprocessor(
        VisionPreprocessConfig(3, 2, 6, maximum_time_skew_ns=maximum_time_skew_ns),
        calibrations,
    )


def test_four_camera_preprocessing_is_deterministic_and_side_explicit() -> None:
    preprocessor = _preprocessor()

    first = preprocessor.preprocess(_observation())
    second = preprocessor.preprocess(_observation())

    assert np.array_equal(first.left_wrist_rgb, second.left_wrist_rgb)
    assert np.array_equal(first.right_wrist_rgb, second.right_wrist_rgb)
    assert first.camera_validity.tolist() == [True, True, True, True]
    assert first.frame_timestamps_ns.tolist() == [1_000] * 4
    assert first.head_depth_valid.sum() == 3
    assert not first.left_wrist_rgb.flags.writeable


def test_stale_right_wrist_is_zeroed_without_hiding_left_wrist() -> None:
    value = _preprocessor(maximum_time_skew_ns=10).preprocess(
        _observation(right_timestamp=2_000)
    )

    assert value.camera_validity.tolist() == [True, True, True, False]
    assert np.any(value.left_wrist_rgb != 0.0)
    assert np.all(value.right_wrist_rgb == 0.0)


def test_dual_preprocessor_rejects_legacy_observation_type() -> None:
    from hwr.core.types import ObservationFrame

    with pytest.raises(TypeError, match="DualArmObservation"):
        _preprocessor().preprocess(ObservationFrame(0, 0, "legacy", "legacy"))
