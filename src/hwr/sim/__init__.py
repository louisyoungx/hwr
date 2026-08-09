"""Vendor-neutral simulation interfaces and reference backends."""

from hwr.sim.environment import Household2DEnv
from hwr.sim.specs import (
    Bounds,
    HouseholdTaskSpec,
    ObjectSpec,
    ObstacleSpec,
    RobotSpec,
    SceneSpec,
    ZoneSpec,
)

__all__ = [
    "Bounds",
    "Household2DEnv",
    "HouseholdTaskSpec",
    "ObjectSpec",
    "ObstacleSpec",
    "RobotSpec",
    "SceneSpec",
    "ZoneSpec",
]

