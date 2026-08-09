"""Versioned specifications for the reference household simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("bounds must have positive area")


@dataclass(frozen=True)
class ObstacleSpec:
    obstacle_id: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        if not self.obstacle_id:
            raise ValueError("obstacle_id is required")
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("obstacle must have positive area")


@dataclass(frozen=True)
class ZoneSpec:
    zone_id: str
    center_x: float
    center_y: float
    radius: float
    accepts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.zone_id or not self.accepts:
            raise ValueError("zone id and accepted labels are required")
        if self.radius <= 0:
            raise ValueError("zone radius must be positive")


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    label: str
    center_x: float
    center_y: float
    radius: float
    mass: float
    target_zone_id: str
    spawn_jitter: float = 0.0

    def __post_init__(self) -> None:
        if not self.object_id or not self.label or not self.target_zone_id:
            raise ValueError("object id, label, and target zone are required")
        if self.radius <= 0 or self.mass <= 0 or self.spawn_jitter < 0:
            raise ValueError("object dimensions and mass must be positive")


@dataclass(frozen=True)
class RobotSpec:
    spec_id: str = "mobile_manipulator_2d/v1"
    base_radius: float = 0.18
    max_linear_speed: float = 0.45
    max_angular_speed: float = 1.2
    arm_reach: float = 0.65
    arm_speed: float = 0.8
    gripper_radius: float = 0.08
    payload: float = 0.5
    control_hz: float = 10.0
    lidar_range: float = 2.5
    lidar_rays: int = 8

    def __post_init__(self) -> None:
        numeric = (
            self.base_radius,
            self.max_linear_speed,
            self.max_angular_speed,
            self.arm_reach,
            self.arm_speed,
            self.gripper_radius,
            self.payload,
            self.control_hz,
            self.lidar_range,
        )
        if not self.spec_id or min(numeric) <= 0 or self.lidar_rays <= 0:
            raise ValueError("robot specification values must be positive")


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    bounds: Bounds
    robot_start: tuple[float, float, float]
    objects: tuple[ObjectSpec, ...]
    zones: tuple[ZoneSpec, ...]
    obstacles: tuple[ObstacleSpec, ...] = ()
    start_jitter: float = 0.0

    def __post_init__(self) -> None:
        if not self.scene_id or len(self.robot_start) != 3:
            raise ValueError("scene id and robot start pose are required")
        if not self.objects or not self.zones:
            raise ValueError("scene requires at least one object and zone")
        zone_ids = {zone.zone_id for zone in self.zones}
        if len(zone_ids) != len(self.zones):
            raise ValueError("zone ids must be unique")
        object_ids = {item.object_id for item in self.objects}
        if len(object_ids) != len(self.objects):
            raise ValueError("object ids must be unique")
        if any(item.target_zone_id not in zone_ids for item in self.objects):
            raise ValueError("every object target must reference a scene zone")
        if self.start_jitter < 0:
            raise ValueError("start jitter must be non-negative")


@dataclass(frozen=True)
class HouseholdTaskSpec:
    task_id: str
    scene: SceneSpec
    max_steps: int = 400
    collision_terminates: bool = False
    success_reward: float = 10.0
    grasp_reward: float = 1.0
    collision_penalty: float = 1.0
    spec_version: str = "hwr.task/v1"

    def __post_init__(self) -> None:
        if not self.task_id or self.max_steps <= 0:
            raise ValueError("task id and positive max_steps are required")

