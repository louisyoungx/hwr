"""Collision-aware waypoint tables for the privileged formal-scene expert."""

from __future__ import annotations

import math


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def object_approach_yaw(task_id: str, object_id: str | None) -> float:
    if task_id.startswith("tidy_living"):
        return math.pi if object_id == "football" else 0.0
    if task_id.startswith("store_kitchen") and object_id == "cleaner_pink":
        return math.pi
    return math.pi / 2


def target_approach_yaw(task_id: str) -> float:
    if task_id.startswith("store_kitchen"):
        return math.pi / 2
    if task_id.startswith("tidy_living"):
        return 0.80
    return math.pi / 2


def gripper_rotation(yaw: float) -> tuple[tuple[float, float, float], ...]:
    return (
        (math.cos(yaw), -math.sin(yaw), 0.0),
        (math.sin(yaw), math.cos(yaw), 0.0),
        (0.0, 0.0, 1.0),
    )


def top_down_gripper_rotation(
    yaw: float,
) -> tuple[tuple[float, float, float], ...]:
    """Point the finger length downward while rotating its opening in the floor plane."""
    return (
        (0.0, -math.sin(yaw), math.cos(yaw)),
        (0.0, math.cos(yaw), math.sin(yaw)),
        (-1.0, 0.0, 0.0),
    )


def top_down_site_compensation(
    yaw: float, finger_pad_offset_m: float = 0.04
) -> tuple[float, float, float]:
    """Move the site so the physical finger-pad midpoint reaches the target."""
    return (
        finger_pad_offset_m * math.cos(yaw),
        finger_pad_offset_m * math.sin(yaw),
        0.0,
    )


def object_orientation_weight(
    task_id: str, stage_kind: str, object_id: str | None = None
) -> float:
    if task_id.startswith("tidy_living"):
        return {
            "arm_object_clearance": 0.0,
            "arm_object_above": 0.15 if object_id == "football" else 0.50,
            "arm_object_descend": 0.05,
            "arm_object_lift": 0.10,
        }.get(stage_kind, 0.0)
    return 0.35 if "object" in stage_kind else 0.0


def navigation_tolerances(task_id: str, stage_kind: str) -> tuple[float, float]:
    if task_id.startswith("store_kitchen"):
        if stage_kind == "nav_drawer":
            return 0.04, 0.04
        if stage_kind == "nav_object":
            return 0.025, 0.04
        if stage_kind == "nav_target":
            return 0.10, 0.10
    return 0.12, 0.10


def navigation_linear_speed(task_id: str, stage_kind: str, distance: float) -> float:
    speed = min(0.42, 1.1 * distance)
    if task_id.startswith("store_kitchen") and stage_kind == "nav_object":
        speed = max(0.08, speed)
    return speed


def drawer_base_commands(
    base_pose: tuple[float, float, float], target_y: float = 1.36
) -> tuple[float, float]:
    """Drive a differential base back to the drawer manipulation pose."""
    x, y, yaw = base_pose
    dx, dy = 1.54 - x, target_y - y
    distance = math.hypot(dx, dy)
    if distance <= 0.018:
        return 0.0, _clip(3.0 * wrap_angle(math.pi / 2 - yaw), -0.85, 0.85)
    heading = math.atan2(dy, dx)
    heading_error = wrap_angle(heading - yaw)
    direction = 1.0
    if abs(heading_error) > math.pi / 2:
        direction = -1.0
        heading_error = wrap_angle(heading + math.pi - yaw)
    alignment = max(0.0, 1.0 - abs(heading_error) / 0.75)
    speed = max(0.08, min(0.30, 1.5 * distance))
    linear = direction * speed * alignment
    angular = _clip(2.5 * heading_error, -0.85, 0.85)
    return linear, angular


def drawer_base_is_aligned(
    base_pose: tuple[float, float, float], target_y: float = 1.36
) -> bool:
    x, y, yaw = base_pose
    return math.hypot(1.54 - x, target_y - y) <= 0.025 and abs(
        wrap_angle(math.pi / 2 - yaw)
    ) <= 0.04


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def formal_waypoints(
    task_id: str,
    stage_kind: str,
    object_id: str | None,
) -> list[tuple[float, float]]:
    """Return world-frame waypoints without depending on engine state."""
    if task_id.startswith("tidy_living"):
        if stage_kind == "nav_object" and object_id == "duck":
            return [(0.70, -1.35)]
        if stage_kind == "nav_object":
            return [(1.60, 0.20), (0.65, -0.90)]
        if stage_kind == "nav_target" and object_id == "duck":
            return [(1.05, 0.30), (1.28, 0.92)]
        if stage_kind == "nav_target":
            return [(0.45, -0.75), (1.00, 0.20), (1.28, 0.92)]
        return []
    if task_id.startswith("clear_dining"):
        if stage_kind == "nav_object":
            return (
                [(2.85, 1.10), (2.85, -0.35), (1.75, -0.35)]
                if object_id == "plate"
                else [(0.30, -0.95)]
            )
        corridor_y = -0.65 if object_id == "plate" else -0.35
        return [
            (1.80, corridor_y),
            (2.84, corridor_y),
            (2.84, 0.80 if object_id == "plate" else 1.20),
        ]
    if stage_kind == "nav_drawer":
        return [(1.55, -0.70), (1.55, 0.65)]
    if stage_kind == "nav_object":
        if object_id == "cleaner_pink":
            return [(1.35, -0.20), (1.35, 0.45)]
        # Clear the island's south-east corner before turning west.
        return [(1.55, -0.45), (-0.42, -0.45)]
    start_x = 1.35 if object_id == "cleaner_pink" else -0.42
    return [(start_x, -0.45), (1.90, -0.45), (1.90, 0.65)]
