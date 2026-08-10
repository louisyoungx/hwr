"""Vendor-neutral contracts for language-conditioned dual-arm control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from hwr.core.types import CameraFrame, SafetyState


DUAL_ARM_ACTION_DIM = 16
DUAL_ARM_OBSERVATION_SCHEMA = "hwr.dual-arm-observation/v1"
DUAL_ARM_RUNTIME_ACTION_SCHEMA = "hwr.dual-arm-runtime-action/v1"


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class NaturalLanguageInstruction:
    """Unparsed instruction supplied by a person or a dataset."""

    text: str
    locale: str = "zh-CN"

    def __post_init__(self) -> None:
        text = " ".join(self.text.split())
        if not text:
            raise ValueError("instruction text is required")
        if not self.locale:
            raise ValueError("instruction locale is required")
        object.__setattr__(self, "text", text)


@dataclass(frozen=True)
class FrozenLanguageEmbedding:
    """Output of a locally versioned, frozen language encoder."""

    encoder_id: str
    weights_sha256: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.encoder_id:
            raise ValueError("language encoder id is required")
        digest = self.weights_sha256.lower()
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("language encoder weights require a SHA-256 digest")
        object.__setattr__(self, "weights_sha256", digest)
        object.__setattr__(self, "values", _finite(self.values, "language embedding"))


@dataclass(frozen=True)
class DualArmAction:
    """One simultaneous command; gripper zero is open and one is closed."""

    base_linear: float
    base_angular: float
    left_arm: tuple[float, ...]
    right_arm: tuple[float, ...]
    left_gripper: float
    right_gripper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_linear) or not math.isfinite(self.base_angular):
            raise ValueError("base commands must be finite")
        left = _finite(self.left_arm, "left arm command")
        right = _finite(self.right_arm, "right arm command")
        if len(left) != 6 or len(right) != 6:
            raise ValueError("dual-arm action requires six commands per arm")
        grippers = (float(self.left_gripper), float(self.right_gripper))
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in grippers):
            raise ValueError("gripper targets must be finite values between zero and one")
        object.__setattr__(self, "left_arm", left)
        object.__setattr__(self, "right_arm", right)
        object.__setattr__(self, "left_gripper", grippers[0])
        object.__setattr__(self, "right_gripper", grippers[1])

    def vector(self) -> tuple[float, ...]:
        return (
            self.base_linear,
            self.base_angular,
            *self.left_arm,
            *self.right_arm,
            self.left_gripper,
            self.right_gripper,
        )

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "DualArmAction":
        vector = tuple(float(value) for value in values)
        if len(vector) != DUAL_ARM_ACTION_DIM:
            raise ValueError(f"dual-arm action vector must have {DUAL_ARM_ACTION_DIM} values")
        return cls(
            base_linear=vector[0],
            base_angular=vector[1],
            left_arm=vector[2:8],
            right_arm=vector[8:14],
            left_gripper=vector[14],
            right_gripper=vector[15],
        )


@dataclass(frozen=True)
class ActionChunk:
    """A policy prediction containing several future executable actions."""

    actions: tuple[DualArmAction, ...]
    valid_steps: int

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("action chunk cannot be empty")
        if not 1 <= self.valid_steps <= len(actions):
            raise ValueError("valid action steps must fit inside the chunk")
        object.__setattr__(self, "actions", actions)

    def vectors(self) -> tuple[tuple[float, ...], ...]:
        return tuple(action.vector() for action in self.actions)


@dataclass(frozen=True)
class DualArmProprioception:
    """Deployable state; normalized gripper travel is open=0 and closed=1."""

    left_joint_position: tuple[float, ...]
    left_joint_velocity: tuple[float, ...]
    right_joint_position: tuple[float, ...]
    right_joint_velocity: tuple[float, ...]
    left_gripper_position: float
    right_gripper_position: float
    base_pose: tuple[float, float, float]
    base_twist: tuple[float, float]
    imu: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        names = (
            "left_joint_position",
            "left_joint_velocity",
            "right_joint_position",
            "right_joint_velocity",
        )
        for name in names:
            values = _finite(getattr(self, name), name)
            if len(values) != 6:
                raise ValueError(f"{name} must contain six values")
            object.__setattr__(self, name, values)
        grippers = (
            float(self.left_gripper_position),
            float(self.right_gripper_position),
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in grippers):
            raise ValueError("gripper positions must be finite values between zero and one")
        base_pose = _finite(self.base_pose, "base_pose")
        base_twist = _finite(self.base_twist, "base_twist")
        if len(base_pose) != 3 or len(base_twist) != 2:
            raise ValueError("base pose/twist dimensions must be three and two")
        object.__setattr__(self, "left_gripper_position", grippers[0])
        object.__setattr__(self, "right_gripper_position", grippers[1])
        object.__setattr__(self, "base_pose", base_pose)
        object.__setattr__(self, "base_twist", base_twist)
        object.__setattr__(self, "imu", tuple(float(value) for value in self.imu))
        if not all(math.isfinite(value) for value in self.imu):
            raise ValueError("imu must contain finite values")

    def vector(self) -> tuple[float, ...]:
        return (
            *self.left_joint_position,
            *self.left_joint_velocity,
            *self.right_joint_position,
            *self.right_joint_velocity,
            self.left_gripper_position,
            self.right_gripper_position,
            *self.base_pose,
            *self.base_twist,
            *self.imu,
        )


@dataclass(frozen=True)
class DualArmObservation:
    """Raw deployable observation; contains no reward, truth, stage, or goal token."""

    timestamp_ns: int
    sequence_id: int
    task_id: str
    instruction: NaturalLanguageInstruction
    proprioception: DualArmProprioception
    cameras: tuple[CameraFrame, ...]
    safety_state: SafetyState = SafetyState.OK
    quality: Mapping[str, float] = field(default_factory=dict)
    schema_version: str = DUAL_ARM_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or self.sequence_id < 0:
            raise ValueError("observation timestamp and sequence must be non-negative")
        if not self.task_id:
            raise ValueError("task id is required")
        cameras = tuple(self.cameras)
        camera_ids = tuple(camera.camera_id for camera in cameras)
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera ids must be unique")
        quality = {str(name): float(value) for name, value in self.quality.items()}
        if not all(math.isfinite(value) for value in quality.values()):
            raise ValueError("quality values must be finite")
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "quality", quality)

    def camera(self, camera_id: str) -> CameraFrame:
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        raise KeyError(camera_id)


@dataclass(frozen=True)
class DualArmActionFrame:
    """Time-bounded runtime envelope around one canonical 16-D action."""

    created_at_ns: int
    valid_from_ns: int
    valid_until_ns: int
    source: str
    action: DualArmAction
    confidence: float = 1.0
    policy_version: str | None = None
    schema_version: str = DUAL_ARM_RUNTIME_ACTION_SCHEMA

    def __post_init__(self) -> None:
        if min(self.created_at_ns, self.valid_from_ns, self.valid_until_ns) < 0:
            raise ValueError("action timestamps must be non-negative")
        if self.valid_until_ns < self.valid_from_ns:
            raise ValueError("action validity range is inverted")
        if not self.source:
            raise ValueError("action source is required")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        object.__setattr__(self, "confidence", confidence)
