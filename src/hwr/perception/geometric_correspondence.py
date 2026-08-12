"""Task-independent patch correspondences derived from RGB-D calibration."""

from __future__ import annotations

import numpy as np

from hwr.perception.high_resolution import HighResolutionVision


def build_cross_camera_patch_correspondences(
    frame: HighResolutionVision,
    *,
    feature_grid_size: int,
    maximum_pairs: int = 512,
) -> np.ndarray:
    """Project valid head depth into both wrist cameras without object semantics."""
    if feature_grid_size <= 0 or maximum_pairs <= 0:
        raise ValueError("correspondence grid and capacity must be positive")
    if not frame.camera_validity[0] or not frame.camera_validity[1]:
        return np.empty((0, 6), dtype=np.int64)
    rows, columns = np.nonzero(frame.student_head_depth_valid)
    if not len(rows):
        return np.empty((0, 6), dtype=np.int64)
    per_camera = max(1, maximum_pairs // 2)
    selected = np.linspace(0, len(rows) - 1, min(per_camera, len(rows))).round().astype(int)
    rows, columns = rows[selected], columns[selected]
    depth = frame.student_head_depth_m[rows, columns]
    intrinsics = frame.student_intrinsics[1]
    camera_points = np.stack(
        (
            (columns - intrinsics[2]) * depth / intrinsics[0],
            (rows - intrinsics[3]) * depth / intrinsics[1],
            depth,
            np.ones_like(depth),
        ),
        axis=1,
    )
    robot_points = camera_points @ frame.robot_from_camera[1].T
    pairs: list[np.ndarray] = []
    for spatial_camera, calibration_index in ((1, 2), (2, 3)):
        if not frame.camera_validity[calibration_index]:
            continue
        pairs.append(
            _project_pairs(
                robot_points,
                rows,
                columns,
                frame,
                spatial_camera,
                calibration_index,
                feature_grid_size,
            )
        )
    if not pairs:
        return np.empty((0, 6), dtype=np.int64)
    combined = np.concatenate(pairs)
    if not len(combined):
        return combined
    return np.unique(combined, axis=0)[:maximum_pairs]


def _project_pairs(
    robot_points: np.ndarray,
    source_rows: np.ndarray,
    source_columns: np.ndarray,
    frame: HighResolutionVision,
    spatial_camera: int,
    calibration_index: int,
    grid_size: int,
) -> np.ndarray:
    camera_from_robot = np.linalg.inv(frame.robot_from_camera[calibration_index])
    target = robot_points @ camera_from_robot.T
    positive = target[:, 2] > 1.0e-5
    intrinsics = frame.student_intrinsics[calibration_index]
    target_columns = intrinsics[0] * target[:, 0] / np.maximum(target[:, 2], 1.0e-5)
    target_columns += intrinsics[2]
    target_rows = intrinsics[1] * target[:, 1] / np.maximum(target[:, 2], 1.0e-5)
    target_rows += intrinsics[3]
    image_size = frame.student_rgb.shape[1]
    in_frame = (
        positive
        & (target_rows >= 0)
        & (target_rows < image_size)
        & (target_columns >= 0)
        & (target_columns < image_size)
    )
    source_grid_rows = np.floor(source_rows * grid_size / image_size).astype(np.int64)
    source_grid_columns = np.floor(source_columns * grid_size / image_size).astype(np.int64)
    target_grid_rows = np.floor(target_rows * grid_size / image_size).astype(np.int64)
    target_grid_columns = np.floor(target_columns * grid_size / image_size).astype(np.int64)
    return np.stack(
        (
            np.zeros_like(source_grid_rows),
            source_grid_rows,
            source_grid_columns,
            np.full_like(source_grid_rows, spatial_camera),
            target_grid_rows,
            target_grid_columns,
        ),
        axis=1,
    )[in_frame]


def batch_correspondence_indices(
    values: list[list[np.ndarray]], *, device: str = "cpu"
):
    """Add batch and history axes for the visual objective index format."""
    import torch

    rows: list[np.ndarray] = []
    for batch_index, history in enumerate(values):
        for history_index, pairs in enumerate(history):
            if pairs.ndim != 2 or pairs.shape[1] != 6:
                raise ValueError("patch correspondence arrays must have six columns")
            if not len(pairs):
                continue
            first = np.column_stack(
                (
                    np.full(len(pairs), batch_index),
                    np.full(len(pairs), history_index),
                    pairs[:, :3],
                )
            )
            second = np.column_stack(
                (
                    np.full(len(pairs), batch_index),
                    np.full(len(pairs), history_index),
                    pairs[:, 3:],
                )
            )
            rows.append(np.concatenate((first, second), axis=1))
    result = np.concatenate(rows) if rows else np.empty((0, 10), dtype=np.int64)
    return torch.from_numpy(result.astype(np.int64, copy=False)).to(device)
