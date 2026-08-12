from __future__ import annotations

import numpy as np

from hwr.perception.geometric_correspondence import (
    batch_correspondence_indices,
    build_cross_camera_patch_correspondences,
)
from hwr.perception.high_resolution import HighResolutionVision


def _frame() -> HighResolutionVision:
    size = 160
    intrinsics = np.repeat(
        np.asarray([[100.0, 100.0, 79.5, 79.5]], dtype=np.float32), 4, axis=0
    )
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    return HighResolutionVision(
        teacher_rgb=np.zeros((3, 224, 224, 3), np.float32),
        student_rgb=np.zeros((3, size, size, 3), np.float32),
        student_head_depth_m=np.ones((size, size), np.float32),
        student_head_depth_valid=np.ones((size, size), np.bool_),
        camera_validity=np.ones(4, np.bool_),
        frame_timestamps_ns=np.arange(4, dtype=np.int64),
        student_intrinsics=intrinsics,
        robot_from_camera=transforms,
        preprocess_fingerprint="a" * 64,
        source_sha256="b" * 64,
    )


def test_cross_camera_correspondence_uses_geometry_and_camera_axes() -> None:
    pairs = build_cross_camera_patch_correspondences(
        _frame(), feature_grid_size=10, maximum_pairs=40
    )

    assert pairs.shape == (40, 6)
    assert set(pairs[:, 0]) == {0}
    assert set(pairs[:, 3]) == {1, 2}
    assert np.all((pairs[:, (1, 2, 4, 5)] >= 0) & (pairs[:, (1, 2, 4, 5)] < 10))
    batched = batch_correspondence_indices([[pairs]])
    assert batched.shape == (40, 10)
    assert batched[:, (0, 1, 5, 6)].eq(0).all()


def test_cross_camera_correspondence_respects_missing_depth_camera() -> None:
    frame = _frame()
    validity = frame.camera_validity.copy()
    validity[1] = False
    frame = HighResolutionVision(**{**frame.__dict__, "camera_validity": validity})

    pairs = build_cross_camera_patch_correspondences(frame, feature_grid_size=10)

    assert pairs.shape == (0, 6)


def test_correspondence_backprojects_aligned_depth_from_head_rgb_geometry() -> None:
    frame = _frame()
    transforms = frame.robot_from_camera.copy()
    transforms[1, 0, 3] = 4.0
    frame = HighResolutionVision(
        **{**frame.__dict__, "robot_from_camera": transforms}
    )

    pairs = build_cross_camera_patch_correspondences(
        frame, feature_grid_size=10, maximum_pairs=20
    )

    assert pairs.shape == (20, 6)
