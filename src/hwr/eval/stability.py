"""Engine-independent physical placement and stability acceptance criteria."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetVolume:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    def __post_init__(self) -> None:
        if self.min_x >= self.max_x or self.min_y >= self.max_y or self.min_z >= self.max_z:
            raise ValueError("target volume must have positive extent")

    def contains(self, position: tuple[float, float, float]) -> bool:
        x, y, z = position
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )


@dataclass(frozen=True)
class StabilityConfig:
    control_hz: float
    hold_seconds: float = 2.0
    max_linear_speed: float = 0.03
    max_angular_speed: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.control_hz,
            self.hold_seconds,
            self.max_linear_speed,
            self.max_angular_speed,
        )
        if min(values) <= 0:
            raise ValueError("stability configuration values must be positive")


class StablePlacementCriterion:
    """Require an object to remain inside a volume and nearly still for a full window."""

    def __init__(self, target: TargetVolume, config: StabilityConfig) -> None:
        self.target = target
        self.config = config
        self.required_steps = math.ceil(config.hold_seconds * config.control_hz)
        self.stable_steps = 0

    def reset(self) -> None:
        self.stable_steps = 0

    def update(
        self,
        *,
        position: tuple[float, float, float],
        linear_velocity: tuple[float, float, float],
        angular_velocity: tuple[float, float, float],
    ) -> bool:
        linear_speed = math.sqrt(sum(value * value for value in linear_velocity))
        angular_speed = math.sqrt(sum(value * value for value in angular_velocity))
        stable = (
            self.target.contains(position)
            and linear_speed <= self.config.max_linear_speed
            and angular_speed <= self.config.max_angular_speed
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        return self.stable_steps >= self.required_steps
