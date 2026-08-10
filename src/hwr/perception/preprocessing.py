"""Deterministic RGB-D preprocessing shared by simulation and real adapters."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from hwr.core.types import CameraFrame, ObservationFrame
from hwr.core.embodied import DualArmObservation
from hwr.perception.contracts import (
    CAMERA_IDS,
    DUAL_ARM_CAMERA_IDS,
    CameraCalibration,
    DualArmProcessedVision,
    ProcessedVision,
    VisionPreprocessConfig,
)


CAMERA_ENCODINGS = {
    "head_rgb": "rgb8",
    "head_depth": "depth32f",
    "wrist_rgb": "rgb8",
}
DUAL_ARM_CAMERA_ENCODINGS = {
    "head_rgb": "rgb8",
    "head_depth": "depth32f",
    "left_wrist_rgb": "rgb8",
    "right_wrist_rgb": "rgb8",
}


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _resize_nearest(array: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = array.shape[:2]
    rows = np.linspace(0, source_height - 1, height).round().astype(np.int64)
    columns = np.linspace(0, source_width - 1, width).round().astype(np.int64)
    return np.ascontiguousarray(array[rows][:, columns])


def _decode(frame: CameraFrame) -> np.ndarray:
    if frame.payload is None:
        raise ValueError("camera frame has no in-memory payload")
    if frame.encoding == "rgb8":
        return np.frombuffer(frame.payload, dtype=np.uint8).reshape(
            frame.height, frame.width, 3
        )
    if frame.encoding == "depth32f":
        return np.frombuffer(frame.payload, dtype=np.float32).reshape(
            frame.height, frame.width
        )
    raise ValueError(f"unsupported camera encoding: {frame.encoding}")


class DeterministicVisionPreprocessor:
    """Turn raw whitelisted camera frames into finite deployment tensors."""

    def __init__(
        self,
        config: VisionPreprocessConfig,
        calibrations: Mapping[str, CameraCalibration],
    ) -> None:
        self.config = config
        self.calibrations = dict(calibrations)
        if set(self.calibrations) != set(CAMERA_IDS):
            raise ValueError("preprocessor requires calibration for every camera")
        if any(name != value.camera_id for name, value in self.calibrations.items()):
            raise ValueError("camera calibration keys and identities differ")
        self.fingerprint = config.fingerprint(self.calibrations)

    def preprocess(self, observation: ObservationFrame) -> ProcessedVision:
        if observation.task_stage != "instruction_following" or observation.features:
            raise ValueError("privileged task state cannot enter visual preprocessing")
        frames = self._validated_frames(observation)
        head_rgb, head_rgb_valid = self._rgb(
            frames.get("head_rgb"), observation.timestamp_ns
        )
        wrist_rgb, wrist_rgb_valid = self._rgb(
            frames.get("wrist_rgb"), observation.timestamp_ns
        )
        depth, depth_valid, head_depth_valid = self._depth(
            frames.get("head_depth"), observation.timestamp_ns
        )
        points, point_valid = self._point_cloud(
            depth, depth_valid, head_rgb, head_rgb_valid
        )
        validity = np.asarray(
            (head_rgb_valid, head_depth_valid, wrist_rgb_valid), dtype=np.bool_
        )
        timestamps = np.asarray(
            [frames[name].timestamp_ns if name in frames else -1 for name in CAMERA_IDS],
            dtype=np.int64,
        )
        return ProcessedVision(
            head_rgb=_readonly(head_rgb),
            head_depth=_readonly(depth),
            head_depth_valid=_readonly(depth_valid),
            head_points=_readonly(points),
            head_point_valid=_readonly(point_valid),
            wrist_rgb=_readonly(wrist_rgb),
            camera_validity=_readonly(validity),
            frame_timestamps_ns=_readonly(timestamps),
            preprocess_fingerprint=self.fingerprint,
        )

    def _validated_frames(self, observation: ObservationFrame) -> dict[str, CameraFrame]:
        frames = {frame.camera_id: frame for frame in observation.cameras}
        extras = set(frames) - set(CAMERA_IDS)
        if extras:
            raise ValueError(f"camera set contains non-deployment fields: {sorted(extras)}")
        for camera_id, frame in frames.items():
            expected = CAMERA_ENCODINGS[camera_id]
            if frame.encoding != expected:
                raise ValueError(f"camera {camera_id} encoding must be {expected}")
            intrinsics = self.calibrations[camera_id].intrinsics
            if (frame.width, frame.height) != (intrinsics.width, intrinsics.height):
                raise ValueError(f"camera {camera_id} dimensions differ from calibration")
        return frames

    def _is_current(self, frame: CameraFrame | None, timestamp_ns: int) -> bool:
        return bool(
            frame is not None
            and frame.payload is not None
            and abs(frame.timestamp_ns - timestamp_ns) <= self.config.maximum_time_skew_ns
        )

    def _rgb(
        self, frame: CameraFrame | None, timestamp_ns: int
    ) -> tuple[np.ndarray, bool]:
        shape = (self.config.image_height, self.config.image_width, 3)
        if not self._is_current(frame, timestamp_ns):
            return np.zeros(shape, dtype=np.float32), False
        assert frame is not None
        resized = _resize_nearest(
            _decode(frame), self.config.image_width, self.config.image_height
        ).astype(np.float32)
        normalized = resized / 255.0
        mean = np.asarray(self.config.rgb_mean, dtype=np.float32)
        standard_deviation = np.asarray(self.config.rgb_std, dtype=np.float32)
        return (normalized - mean) / standard_deviation, True

    def _depth(
        self, frame: CameraFrame | None, timestamp_ns: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        shape = (self.config.image_height, self.config.image_width)
        if not self._is_current(frame, timestamp_ns):
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.bool_), False
        assert frame is not None
        raw = _resize_nearest(
            _decode(frame), self.config.image_width, self.config.image_height
        ).astype(np.float32)
        valid = (
            np.isfinite(raw)
            & (raw >= self.config.minimum_depth_m)
            & (raw <= self.config.maximum_depth_m)
        )
        clean = np.where(valid, raw, 0.0).astype(np.float32)
        return clean, valid, True

    def _point_cloud(
        self,
        depth: np.ndarray,
        depth_valid: np.ndarray,
        normalized_rgb: np.ndarray,
        rgb_valid: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = depth.shape
        calibration = self.calibrations["head_depth"]
        intrinsics = calibration.intrinsics
        scale_x = width / intrinsics.width
        scale_y = height / intrinsics.height
        fx, fy = intrinsics.fx * scale_x, intrinsics.fy * scale_y
        cx, cy = intrinsics.cx * scale_x, intrinsics.cy * scale_y
        rows, columns = np.indices((height, width), dtype=np.float32)
        camera_points = np.stack(
            (
                (columns - cx) * depth / fx,
                (rows - cy) * depth / fy,
                depth,
                np.ones_like(depth),
            ),
            axis=-1,
        )
        transform = np.asarray(calibration.robot_from_camera, dtype=np.float32).reshape(4, 4)
        robot_points = camera_points.reshape(-1, 4) @ transform.T
        colors = normalized_rgb * np.asarray(self.config.rgb_std) + np.asarray(
            self.config.rgb_mean
        )
        if not rgb_valid:
            colors = np.zeros_like(colors)
        combined = np.concatenate((robot_points[:, :3], colors.reshape(-1, 3)), axis=1)
        indices = np.linspace(
            0, height * width - 1, self.config.point_count
        ).round().astype(np.int64)
        valid = depth_valid.reshape(-1)[indices]
        selected = combined[indices].astype(np.float32)
        selected[~valid] = 0.0
        return selected, valid


class DualArmVisionPreprocessor(DeterministicVisionPreprocessor):
    """Apply the same deterministic transform to all four deployable cameras."""

    def __init__(
        self,
        config: VisionPreprocessConfig,
        calibrations: Mapping[str, CameraCalibration],
    ) -> None:
        self.config = config
        self.calibrations = dict(calibrations)
        if set(self.calibrations) != set(DUAL_ARM_CAMERA_IDS):
            raise ValueError("dual-arm preprocessor requires four camera calibrations")
        if any(name != value.camera_id for name, value in self.calibrations.items()):
            raise ValueError("camera calibration keys and identities differ")
        self.fingerprint = config.fingerprint(self.calibrations)

    def preprocess(self, observation: DualArmObservation) -> DualArmProcessedVision:
        if not isinstance(observation, DualArmObservation):
            raise TypeError("dual-arm preprocessing requires DualArmObservation")
        frames = self._validated_dual_frames(observation)
        timestamp_ns = observation.timestamp_ns
        head_rgb, head_valid = self._rgb(frames.get("head_rgb"), timestamp_ns)
        left_wrist, left_valid = self._rgb(
            frames.get("left_wrist_rgb"), timestamp_ns
        )
        right_wrist, right_valid = self._rgb(
            frames.get("right_wrist_rgb"), timestamp_ns
        )
        depth, depth_valid, depth_frame_valid = self._depth(
            frames.get("head_depth"), timestamp_ns
        )
        points, point_valid = self._point_cloud(
            depth, depth_valid, head_rgb, head_valid
        )
        validity = np.asarray(
            (head_valid, depth_frame_valid, left_valid, right_valid),
            dtype=np.bool_,
        )
        timestamps = np.asarray(
            [
                frames[name].timestamp_ns if name in frames else -1
                for name in DUAL_ARM_CAMERA_IDS
            ],
            dtype=np.int64,
        )
        return DualArmProcessedVision(
            head_rgb=_readonly(head_rgb),
            head_depth=_readonly(depth),
            head_depth_valid=_readonly(depth_valid),
            head_points=_readonly(points),
            head_point_valid=_readonly(point_valid),
            left_wrist_rgb=_readonly(left_wrist),
            right_wrist_rgb=_readonly(right_wrist),
            camera_validity=_readonly(validity),
            frame_timestamps_ns=_readonly(timestamps),
            preprocess_fingerprint=self.fingerprint,
        )

    def _validated_dual_frames(
        self, observation: DualArmObservation
    ) -> dict[str, CameraFrame]:
        frames = {frame.camera_id: frame for frame in observation.cameras}
        extras = set(frames) - set(DUAL_ARM_CAMERA_IDS)
        if extras:
            raise ValueError(
                f"camera set contains non-deployment fields: {sorted(extras)}"
            )
        for camera_id, frame in frames.items():
            if frame.encoding != DUAL_ARM_CAMERA_ENCODINGS[camera_id]:
                raise ValueError(f"camera {camera_id} encoding is invalid")
            intrinsics = self.calibrations[camera_id].intrinsics
            if (frame.width, frame.height) != (intrinsics.width, intrinsics.height):
                raise ValueError(f"camera {camera_id} dimensions differ from calibration")
        return frames
