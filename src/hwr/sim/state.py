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

