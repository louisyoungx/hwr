"""Calibration and output contracts for deterministic visual preprocessing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


VISION_PREPROCESS_SCHEMA = "hwr.vision-preprocess/v1"
PROCESSED_VISION_SCHEMA = "hwr.processed-vision/v1"
CAMERA_IDS = ("head_rgb", "head_depth", "wrist_rgb")


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("intrinsic image dimensions must be positive")
        if not all(math.isfinite(value) for value in values) or min(self.fx, self.fy) <= 0:
            raise ValueError("pinhole intrinsics are invalid")


@dataclass(frozen=True)
class CameraCalibration:
    calibration_id: str
    camera_id: str
    intrinsics: PinholeIntrinsics
    robot_from_camera: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.calibration_id or self.camera_id not in CAMERA_IDS:
            raise ValueError("known camera and calibration identities are required")
        transform = _finite(self.robot_from_camera, "camera transform")
        if len(transform) != 16 or transform[12:] != (0.0, 0.0, 0.0, 1.0):
            raise ValueError("camera transform must be a homogeneous 4x4 matrix")
        object.__setattr__(self, "robot_from_camera", transform)


@dataclass(frozen=True)
class VisionPreprocessConfig:
    image_width: int
    image_height: int
    point_count: int
    minimum_depth_m: float = 0.10
    maximum_depth_m: float = 5.0
    maximum_time_skew_ns: int = 50_000_000
    rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    schema_version: str = VISION_PREPROCESS_SCHEMA

    def __post_init__(self) -> None:
        if min(self.image_width, self.image_height, self.point_count) <= 0:
            raise ValueError("preprocess tensor dimensions must be positive")
        if not 0.0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("depth range is invalid")
        if self.maximum_time_skew_ns < 0:
            raise ValueError("maximum time skew cannot be negative")
        means = _finite(self.rgb_mean, "RGB mean")
        standard_deviations = _finite(self.rgb_std, "RGB standard deviation")
        if len(means) != 3 or len(standard_deviations) != 3 or min(standard_deviations) <= 0:
            raise ValueError("RGB normalization requires three positive scales")
        object.__setattr__(self, "rgb_mean", means)
        object.__setattr__(self, "rgb_std", standard_deviations)

    def fingerprint(self, calibrations: Mapping[str, CameraCalibration]) -> str:
        payload = {
            "config": asdict(self),
            "calibrations": {
                name: asdict(calibrations[name]) for name in sorted(calibrations)
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProcessedVision:
    head_rgb: np.ndarray
    head_depth: np.ndarray
    head_depth_valid: np.ndarray
    head_points: np.ndarray
    head_point_valid: np.ndarray
    wrist_rgb: np.ndarray
    camera_validity: np.ndarray
    frame_timestamps_ns: np.ndarray
    preprocess_fingerprint: str
    schema_version: str = PROCESSED_VISION_SCHEMA

    def __post_init__(self) -> None:
        height, width = self.head_depth.shape
        expected = {
            "head_rgb": (height, width, 3),
            "head_depth_valid": (height, width),
            "head_points": (self.head_point_valid.shape[0], 6),
            "wrist_rgb": (height, width, 3),
            "camera_validity": (len(CAMERA_IDS),),
            "frame_timestamps_ns": (len(CAMERA_IDS),),
        }
        mismatches = {
            name: (getattr(self, name).shape, shape)
            for name, shape in expected.items()
            if getattr(self, name).shape != shape
        }
        if mismatches:
            raise ValueError(f"processed vision tensor shapes are invalid: {mismatches}")
        if len(self.preprocess_fingerprint) != 64:
            raise ValueError("processed vision requires a preprocess fingerprint")
        floating = (self.head_rgb, self.head_depth, self.head_points, self.wrist_rgb)
        if not all(np.isfinite(value).all() for value in floating):
            raise ValueError("processed vision tensors must be finite")
