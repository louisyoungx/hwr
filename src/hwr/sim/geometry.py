"""Small deterministic 2D geometry helpers."""

from __future__ import annotations

import math

from hwr.sim.specs import Bounds, ObstacleSpec


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotate_to_world(local_x: float, local_y: float, heading: float) -> tuple[float, float]:
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return cosine * local_x - sine * local_y, sine * local_x + cosine * local_y


def rotate_to_local(world_x: float, world_y: float, heading: float) -> tuple[float, float]:
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return cosine * world_x + sine * world_y, -sine * world_x + cosine * world_y


def circle_hits_bounds(x: float, y: float, radius: float, bounds: Bounds) -> bool:
    return (
        x - radius < bounds.min_x
        or x + radius > bounds.max_x
        or y - radius < bounds.min_y
        or y + radius > bounds.max_y
    )


def circle_hits_rect(x: float, y: float, radius: float, obstacle: ObstacleSpec) -> bool:
    nearest_x = clamp(x, obstacle.min_x, obstacle.max_x)
    nearest_y = clamp(y, obstacle.min_y, obstacle.max_y)
    return math.hypot(x - nearest_x, y - nearest_y) < radius


def ray_rect_distance(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    obstacle: ObstacleSpec,
) -> float | None:
    t_min = -math.inf
    t_max = math.inf
    for origin, direction, lower, upper in (
        (origin_x, direction_x, obstacle.min_x, obstacle.max_x),
        (origin_y, direction_y, obstacle.min_y, obstacle.max_y),
    ):
        if abs(direction) < 1e-12:
            if origin < lower or origin > upper:
                return None
            continue
        first = (lower - origin) / direction
        second = (upper - origin) / direction
        if first > second:
            first, second = second, first
        t_min = max(t_min, first)
        t_max = min(t_max, second)
        if t_min > t_max:
            return None
    if t_max < 0:
        return None
    return max(0.0, t_min)


def ray_bounds_distance(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    bounds: Bounds,
) -> float:
    distances: list[float] = []
    if abs(direction_x) > 1e-12:
        for boundary in (bounds.min_x, bounds.max_x):
            distance = (boundary - origin_x) / direction_x
            y = origin_y + distance * direction_y
            if distance >= 0 and bounds.min_y <= y <= bounds.max_y:
                distances.append(distance)
    if abs(direction_y) > 1e-12:
        for boundary in (bounds.min_y, bounds.max_y):
            distance = (boundary - origin_y) / direction_y
            x = origin_x + distance * direction_x
            if distance >= 0 and bounds.min_x <= x <= bounds.max_x:
                distances.append(distance)
    return min(distances) if distances else math.inf


def range_scan(
    x: float,
    y: float,
    heading: float,
    bounds: Bounds,
    obstacles: tuple[ObstacleSpec, ...],
    ray_count: int,
    max_range: float,
) -> tuple[float, ...]:
    distances: list[float] = []
    for ray_index in range(ray_count):
        angle = heading + 2.0 * math.pi * ray_index / ray_count
        direction_x = math.cos(angle)
        direction_y = math.sin(angle)
        distance = ray_bounds_distance(x, y, direction_x, direction_y, bounds)
        for obstacle in obstacles:
            hit = ray_rect_distance(x, y, direction_x, direction_y, obstacle)
            if hit is not None:
                distance = min(distance, hit)
        distances.append(min(distance, max_range) / max_range)
    return tuple(distances)

