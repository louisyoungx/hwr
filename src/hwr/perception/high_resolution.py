"""Deterministic high-resolution inputs for foundation teachers and students."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from hwr.core.embodied import DualArmObservation
from hwr.core.types import CameraFrame
from hwr.perception.contracts import (
    DUAL_ARM_CAMERA_IDS,
    CameraCalibration,
)


HIGH_RESOLUTION_VISION_SCHEMA = "hwr.high-resolution-vision/v1"
RGB_CAMERA_IDS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
EXPECTED_ENCODINGS = {
    "head_rgb": "rgb8",
    "head_depth": "depth32f",
    "left_wrist_rgb": "rgb8",
    "right_wrist_rgb": "rgb8",
}


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _resize_bilinear(array: np.ndarray, size: int) -> np.ndarray:
    source_height, source_width = array.shape[:2]
    rows = np.linspace(0.0, source_height - 1, size, dtype=np.float32)
    columns = np.linspace(0.0, source_width - 1, size, dtype=np.float32)
    row0 = np.floor(rows).astype(np.int64)
    column0 = np.floor(columns).astype(np.int64)
    row1 = np.minimum(row0 + 1, source_height - 1)
    column1 = np.minimum(column0 + 1, source_width - 1)
    row_weight = (rows - row0)[:, None, None]
    column_weight = (columns - column0)[None, :, None]
    top = array[row0][:, column0] * (1.0 - column_weight)
    top += array[row0][:, column1] * column_weight
    bottom = array[row1][:, column0] * (1.0 - column_weight)
    bottom += array[row1][:, column1] * column_weight
    return (top * (1.0 - row_weight) + bottom * row_weight).astype(np.float32)


def _resize_nearest(array: np.ndarray, size: int) -> np.ndarray:
    rows = np.linspace(0, array.shape[0] - 1, size).round().astype(np.int64)
    columns = np.linspace(0, array.shape[1] - 1, size).round().astype(np.int64)
    return np.ascontiguousarray(array[rows][:, columns])


def _decode(frame: CameraFrame) -> np.ndarray:
    if frame.payload is None:
        raise ValueError("camera frame has no in-memory payload")
    dtype, shape = (
        (np.uint8, (frame.height, frame.width, 3))
        if frame.encoding == "rgb8"
        else (np.float32, (frame.height, frame.width))
    )
    return np.frombuffer(frame.payload, dtype=dtype).reshape(shape)


@dataclass(frozen=True)
class HighResolutionVisionConfig:
    teacher_image_size: int = 224
    student_image_size: int = 160
    minimum_source_short_side: int = 160
    minimum_depth_m: float = 0.10
    maximum_depth_m: float = 5.0
    maximum_time_skew_ns: int = 50_000_000
    schema_version: str = HIGH_RESOLUTION_VISION_SCHEMA

    def __post_init__(self) -> None:
        if min(
            self.teacher_image_size,
            self.student_image_size,
            self.minimum_source_short_side,
        ) <= 0:
            raise ValueError("high-resolution image sizes must be positive")
        if self.teacher_image_size < 224 or self.student_image_size < 160:
            raise ValueError("formal perception cannot use toy image resolutions")
        if not 0.0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("high-resolution depth range is invalid")
        if self.maximum_time_skew_ns < 0:
            raise ValueError("maximum time skew cannot be negative")


@dataclass(frozen=True)
class HighResolutionVision:
    teacher_rgb: np.ndarray
    student_rgb: np.ndarray
    student_head_depth_m: np.ndarray
    student_head_depth_valid: np.ndarray
    camera_validity: np.ndarray
    frame_timestamps_ns: np.ndarray
    student_intrinsics: np.ndarray
    robot_from_camera: np.ndarray
    preprocess_fingerprint: str
    source_sha256: str
    schema_version: str = HIGH_RESOLUTION_VISION_SCHEMA

    def __post_init__(self) -> None:
        teacher_size = self.teacher_rgb.shape[1]
        student_size = self.student_rgb.shape[1]
        expected = {
            "teacher_rgb": (len(RGB_CAMERA_IDS), teacher_size, teacher_size, 3),
            "student_rgb": (len(RGB_CAMERA_IDS), student_size, student_size, 3),
            "student_head_depth_m": (student_size, student_size),
            "student_head_depth_valid": (student_size, student_size),
            "camera_validity": (len(DUAL_ARM_CAMERA_IDS),),
            "frame_timestamps_ns": (len(DUAL_ARM_CAMERA_IDS),),
            "student_intrinsics": (len(DUAL_ARM_CAMERA_IDS), 4),
            "robot_from_camera": (len(DUAL_ARM_CAMERA_IDS), 4, 4),
        }
        mismatches = {
            name: (getattr(self, name).shape, shape)
            for name, shape in expected.items()
            if getattr(self, name).shape != shape
        }
        if mismatches:
            raise ValueError(f"high-resolution vision shapes are invalid: {mismatches}")
        for name in ("teacher_rgb", "student_rgb", "student_head_depth_m"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must contain finite values")
        for digest in (self.preprocess_fingerprint, self.source_sha256):
            if len(digest) != 64:
                raise ValueError("high-resolution vision requires SHA-256 identities")


class HighResolutionVisionPreprocessor:
    """Produce teacher and online-student views without semantic parsing."""

    def __init__(
        self,
        config: HighResolutionVisionConfig,
        calibrations: Mapping[str, CameraCalibration],
    ) -> None:
        self.config = config
        self.calibrations = dict(calibrations)
        if set(self.calibrations) != set(DUAL_ARM_CAMERA_IDS):
            raise ValueError("high-resolution preprocessor requires all camera calibrations")
        if any(key != value.camera_id for key, value in self.calibrations.items()):
            raise ValueError("camera calibration keys and identities differ")
        payload = {
            "config": asdict(config),
            "calibrations": {
                name: asdict(self.calibrations[name]) for name in sorted(self.calibrations)
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.fingerprint = hashlib.sha256(encoded).hexdigest()

    def preprocess(self, observation: DualArmObservation) -> HighResolutionVision:
        if not isinstance(observation, DualArmObservation):
            raise TypeError("high-resolution preprocessing requires DualArmObservation")
        frames = self._validated_frames(observation)
        teacher, student, rgb_validity = self._rgb_views(frames, observation.timestamp_ns)
        depth, depth_valid, depth_current = self._depth(frames, observation.timestamp_ns)
        validity = np.asarray(
            (rgb_validity[0], depth_current, rgb_validity[1], rgb_validity[2]),
            dtype=np.bool_,
        )
        timestamps = np.asarray(
            [frames[name].timestamp_ns if name in frames else -1 for name in DUAL_ARM_CAMERA_IDS],
            dtype=np.int64,
        )
        source_digest = self._source_digest(observation, frames)
        frame_calibrations = {
            value.camera_id: value for value in observation.camera_calibrations
        }
        return HighResolutionVision(
            teacher_rgb=_readonly(teacher),
            student_rgb=_readonly(student),
            student_head_depth_m=_readonly(depth),
            student_head_depth_valid=_readonly(depth_valid),
            camera_validity=_readonly(validity),
            frame_timestamps_ns=_readonly(timestamps),
            student_intrinsics=_readonly(
                self._scaled_intrinsics(frame_calibrations)
            ),
            robot_from_camera=_readonly(self._extrinsics(frame_calibrations)),
            preprocess_fingerprint=self.fingerprint,
            source_sha256=source_digest,
        )

    def _validated_frames(self, observation: DualArmObservation) -> dict[str, CameraFrame]:
        frames = {frame.camera_id: frame for frame in observation.cameras}
        extras = set(frames) - set(DUAL_ARM_CAMERA_IDS)
        if extras:
            raise ValueError(f"camera set contains non-deployment fields: {sorted(extras)}")
        for camera_id, frame in frames.items():
            calibration = self.calibrations[camera_id]
            if frame.encoding != EXPECTED_ENCODINGS[camera_id]:
                raise ValueError(f"camera {camera_id} encoding is invalid")
            if (frame.width, frame.height) != (
                calibration.intrinsics.width,
                calibration.intrinsics.height,
            ):
                raise ValueError(f"camera {camera_id} dimensions differ from calibration")
            if min(frame.width, frame.height) < self.config.minimum_source_short_side:
                raise ValueError(f"camera {camera_id} source resolution is below the formal minimum")
        return frames

    def _current(self, frame: CameraFrame | None, timestamp_ns: int) -> bool:
        return bool(
            frame is not None
            and frame.payload is not None
            and abs(frame.timestamp_ns - timestamp_ns) <= self.config.maximum_time_skew_ns
        )

    def _rgb_views(
        self, frames: Mapping[str, CameraFrame], timestamp_ns: int
    ) -> tuple[np.ndarray, np.ndarray, tuple[bool, ...]]:
        teacher: list[np.ndarray] = []
        student: list[np.ndarray] = []
        validity: list[bool] = []
        for camera_id in RGB_CAMERA_IDS:
            frame = frames.get(camera_id)
            current = self._current(frame, timestamp_ns)
            validity.append(current)
            if not current:
                teacher.append(np.zeros((self.config.teacher_image_size,) * 2 + (3,), np.float32))
                student.append(np.zeros((self.config.student_image_size,) * 2 + (3,), np.float32))
                continue
            assert frame is not None
            rgb = _decode(frame).astype(np.float32) / 255.0
            teacher.append(_resize_bilinear(rgb, self.config.teacher_image_size))
            student.append(_resize_bilinear(rgb, self.config.student_image_size))
        return np.stack(teacher), np.stack(student), tuple(validity)

    def _depth(
        self, frames: Mapping[str, CameraFrame], timestamp_ns: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        frame = frames.get("head_depth")
        shape = (self.config.student_image_size, self.config.student_image_size)
        if not self._current(frame, timestamp_ns):
            return np.zeros(shape, np.float32), np.zeros(shape, np.bool_), False
        assert frame is not None
        raw = _resize_nearest(_decode(frame), self.config.student_image_size).astype(np.float32)
        valid = (
            np.isfinite(raw)
            & (raw >= self.config.minimum_depth_m)
            & (raw <= self.config.maximum_depth_m)
        )
        return np.where(valid, raw, 0.0).astype(np.float32), valid, True

    def _scaled_intrinsics(self, frame_calibrations) -> np.ndarray:
        rows: list[tuple[float, ...]] = []
        for camera_id in DUAL_ARM_CAMERA_IDS:
            value = self.calibrations[camera_id].intrinsics
            scale_x = self.config.student_image_size / value.width
            scale_y = self.config.student_image_size / value.height
            raw = (
                frame_calibrations[camera_id].intrinsics
                if camera_id in frame_calibrations
                else (value.fx, value.fy, value.cx, value.cy)
            )
            rows.append(
                (raw[0] * scale_x, raw[1] * scale_y, raw[2] * scale_x, raw[3] * scale_y)
            )
        return np.asarray(rows, dtype=np.float32)

    def _extrinsics(self, frame_calibrations) -> np.ndarray:
        return np.asarray(
            [
                np.asarray(
                    frame_calibrations[name].robot_from_camera
                    if name in frame_calibrations
                    else self.calibrations[name].robot_from_camera
                ).reshape(4, 4)
                for name in DUAL_ARM_CAMERA_IDS
            ],
            dtype=np.float32,
        )

    def _source_digest(
        self, observation: DualArmObservation, frames: Mapping[str, CameraFrame]
    ) -> str:
        digest = hashlib.sha256()
        digest.update(str(observation.timestamp_ns).encode())
        digest.update(str(observation.sequence_id).encode())
        for camera_id in DUAL_ARM_CAMERA_IDS:
            frame = frames.get(camera_id)
            digest.update(camera_id.encode())
            digest.update(str(frame.timestamp_ns if frame else -1).encode())
            if frame is not None and frame.payload is not None:
                digest.update(frame.payload)
        return digest.hexdigest()
