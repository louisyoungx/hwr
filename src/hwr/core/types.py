"""Versioned, vendor-neutral runtime and dataset schemas."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence


OBSERVATION_SCHEMA = "hwr.observation/v1"
ACTION_SCHEMA = "hwr.action/v1"
EPISODE_SCHEMA = "hwr.episode/v1"


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


class SafetyState(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    timestamp_ns: int
    frame_index: int
    width: int
    height: int
    encoding: str = "rgb8"
    uri: str | None = None

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id is required")
        if self.timestamp_ns < 0 or self.frame_index < 0:
            raise ValueError("camera timestamp and frame index must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")


@dataclass(frozen=True)
class ObservationFrame:
    timestamp_ns: int
    sequence_id: int
    task_id: str
    task_stage: str
    joint_position: tuple[float, ...] = ()
    joint_velocity: tuple[float, ...] = ()
    gripper_position: float = 0.0
    base_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_twist: tuple[float, float] = (0.0, 0.0)
    imu: tuple[float, ...] = ()
    cameras: tuple[CameraFrame, ...] = ()
    features: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    safety_state: SafetyState = SafetyState.OK
    quality: Mapping[str, float] = field(default_factory=dict)
    schema_version: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or self.sequence_id < 0:
            raise ValueError("observation timestamp and sequence must be non-negative")
        if not self.task_id:
            raise ValueError("task_id is required")
        object.__setattr__(self, "joint_position", _finite(self.joint_position, "joint_position"))
        object.__setattr__(self, "joint_velocity", _finite(self.joint_velocity, "joint_velocity"))
        object.__setattr__(self, "base_pose", _finite(self.base_pose, "base_pose"))
        object.__setattr__(self, "base_twist", _finite(self.base_twist, "base_twist"))
        object.__setattr__(self, "imu", _finite(self.imu, "imu"))
        if len(self.base_pose) != 3 or len(self.base_twist) != 2:
            raise ValueError("base_pose must have 3 values and base_twist must have 2")
        if self.joint_velocity and len(self.joint_velocity) != len(self.joint_position):
            raise ValueError("joint_position and joint_velocity lengths must match")
        if not math.isfinite(self.gripper_position):
            raise ValueError("gripper_position must be finite")
        clean_features = {
            str(name): _finite(values, f"features.{name}") for name, values in self.features.items()
        }
        clean_quality = {str(name): float(value) for name, value in self.quality.items()}
        if not all(math.isfinite(value) for value in clean_quality.values()):
            raise ValueError("quality values must be finite")
        object.__setattr__(self, "features", clean_features)
        object.__setattr__(self, "quality", clean_quality)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["safety_state"] = self.safety_state.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationFrame":
        data = dict(value)
        data["cameras"] = tuple(CameraFrame(**camera) for camera in data.get("cameras", ()))
        data["safety_state"] = SafetyState(data.get("safety_state", SafetyState.OK))
        for name in ("joint_position", "joint_velocity", "base_pose", "base_twist", "imu"):
            if name in data:
                data[name] = tuple(data[name])
        data["features"] = {
            name: tuple(values) for name, values in data.get("features", {}).items()
        }
        return cls(**data)


@dataclass(frozen=True)
class ActionFrame:
    created_at_ns: int
    valid_from_ns: int
    valid_until_ns: int
    source: str
    base_linear: float = 0.0
    base_angular: float = 0.0
    arm_command: tuple[float, ...] = ()
    gripper_target: float = 0.0
    confidence: float = 1.0
    policy_version: str | None = None
    schema_version: str = ACTION_SCHEMA

    def __post_init__(self) -> None:
        if min(self.created_at_ns, self.valid_from_ns, self.valid_until_ns) < 0:
            raise ValueError("action timestamps must be non-negative")
        if self.valid_until_ns < self.valid_from_ns:
            raise ValueError("action validity range is inverted")
        if not self.source:
            raise ValueError("action source is required")
        object.__setattr__(self, "arm_command", _finite(self.arm_command, "arm_command"))
        scalars = (self.base_linear, self.base_angular, self.gripper_target, self.confidence)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("action scalars must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionFrame":
        data = dict(value)
        data["arm_command"] = tuple(data.get("arm_command", ()))
        return cls(**data)


@dataclass(frozen=True)
class EpisodeEvent:
    timestamp_ns: int
    event_type: str
    source: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("event timestamp must be non-negative")
        if not self.event_type or not self.source:
            raise ValueError("event type and source are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeEvent":
        return cls(**dict(value))


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    task_id: str
    robot_spec_version: str
    task_spec_version: str
    source_type: str
    seed: int
    started_at_ns: int
    calibration_id: str | None = None
    schema_version: str = EPISODE_SCHEMA

    def __post_init__(self) -> None:
        if not self.episode_id or not self.task_id:
            raise ValueError("episode_id and task_id are required")
        if self.started_at_ns < 0:
            raise ValueError("episode start time must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    reason: str
    ended_at_ns: int
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ended_at_ns < 0:
            raise ValueError("episode end time must be non-negative")
        clean_metrics = {str(name): float(value) for name, value in self.metrics.items()}
        if not all(math.isfinite(value) for value in clean_metrics.values()):
            raise ValueError("metrics must be finite")
        object.__setattr__(self, "metrics", clean_metrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepRecord:
    observation: ObservationFrame
    proposed_action: ActionFrame
    applied_action: ActionFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "proposed_action": self.proposed_action.to_dict(),
            "applied_action": self.applied_action.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StepRecord":
        return cls(
            observation=ObservationFrame.from_dict(value["observation"]),
            proposed_action=ActionFrame.from_dict(value["proposed_action"]),
            applied_action=ActionFrame.from_dict(value["applied_action"]),
        )

