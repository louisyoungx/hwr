"""Stateful assembly of deployable visual student history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from hwr.perception.high_resolution import HighResolutionVision
from hwr.perception.student import VISUAL_STUDENT_INPUT_FIELDS


@dataclass(frozen=True)
class VisualStudentInput:
    rgb: np.ndarray
    head_depth_m: np.ndarray
    head_depth_valid: np.ndarray
    camera_validity: np.ndarray
    intrinsics: np.ndarray
    robot_from_camera: np.ndarray
    repeated_frame: np.ndarray
    source_sha256: tuple[str, ...]
    preprocess_fingerprint: str

    def named_arrays(self) -> dict[str, np.ndarray]:
        values = {
            "rgb": self.rgb,
            "head_depth_m": self.head_depth_m,
            "head_depth_valid": self.head_depth_valid,
            "camera_validity": self.camera_validity,
            "intrinsics": self.intrinsics,
            "robot_from_camera": self.robot_from_camera,
            "repeated_frame": self.repeated_frame,
        }
        if frozenset(values) != VISUAL_STUDENT_INPUT_FIELDS:
            raise ValueError("visual student input arrays violate the field whitelist")
        return values


class VisualStudentInputAssembler:
    def __init__(self, *, visual_history: int, image_size: int) -> None:
        if visual_history <= 0 or image_size <= 0:
            raise ValueError("visual student history and image size must be positive")
        self.visual_history = visual_history
        self.image_size = image_size
        self._frames: deque[HighResolutionVision] = deque(maxlen=visual_history)
        self._fingerprint: str | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._fingerprint = None

    def build(self, frame: HighResolutionVision) -> VisualStudentInput:
        if frame.student_rgb.shape[1:3] != (self.image_size, self.image_size):
            raise ValueError("high-resolution frame differs from visual student image size")
        if self._fingerprint is None:
            self._fingerprint = frame.preprocess_fingerprint
        elif frame.preprocess_fingerprint != self._fingerprint:
            raise ValueError("visual preprocessing changed inside a student history")
        self._frames.append(frame)
        repeated = [True] * (self.visual_history - len(self._frames)) + [False] * len(
            self._frames
        )
        frames = [self._frames[0]] * (self.visual_history - len(self._frames)) + list(
            self._frames
        )
        rgb = np.stack([value.student_rgb.transpose(0, 3, 1, 2) for value in frames])
        depth = np.stack([value.student_head_depth_m[None] for value in frames])
        depth_valid = np.stack([value.student_head_depth_valid[None] for value in frames])
        return VisualStudentInput(
            rgb=_readonly(rgb, np.float32),
            head_depth_m=_readonly(depth, np.float32),
            head_depth_valid=_readonly(depth_valid, np.bool_),
            camera_validity=_readonly(
                np.stack([value.camera_validity for value in frames]), np.bool_
            ),
            intrinsics=_readonly(
                np.stack([value.student_intrinsics for value in frames]), np.float32
            ),
            robot_from_camera=_readonly(
                np.stack([value.robot_from_camera for value in frames]), np.float32
            ),
            repeated_frame=_readonly(np.asarray(repeated), np.bool_),
            source_sha256=tuple(value.source_sha256 for value in frames),
            preprocess_fingerprint=self._fingerprint,
        )


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def visual_student_tensors(
    value: VisualStudentInput, *, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(array.copy())[None].to(device)
        for name, array in value.named_arrays().items()
    }
