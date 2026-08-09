"""Mutable state held internally by the reference simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimObjectState:
    object_id: str
    label: str
    x: float
    y: float
    radius: float
    mass: float
    target_zone_id: str
    placed: bool = False


@dataclass
class SimRobotState:
    x: float
    y: float
    heading: float
    arm_x: float = 0.15
    arm_y: float = 0.0
    gripper: float = 0.0
    carrying_object_id: str | None = None


@dataclass(frozen=True)
class ObjectSnapshot:
    """Immutable object state exposed to diagnostics and render adapters."""

    object_id: str
    label: str
    x: float
    y: float
    radius: float
    target_zone_id: str
    placed: bool


@dataclass(frozen=True)
class RobotSnapshot:
    """Immutable mobile manipulator state for one simulation instant."""

    x: float
    y: float
    heading: float
    arm_x: float
    arm_y: float
    end_effector_x: float
    end_effector_y: float
    gripper: float
    carrying_object_id: str | None


@dataclass(frozen=True)
class SimulationSnapshot:
    """Read-only value snapshot; consumers cannot mutate the live simulator."""

    sequence_id: int
    timestamp_ns: int
    task_stage: str
    steps: int
    collisions: int
    grasps: int
    robot: RobotSnapshot
    objects: tuple[ObjectSnapshot, ...]
