from __future__ import annotations

import pytest

from hwr.sim.geometry import range_scan, rotate_to_local, rotate_to_world
from hwr.sim.specs import Bounds, ObstacleSpec


def test_rotation_round_trip() -> None:
    world = rotate_to_world(0.4, -0.2, 0.7)
    restored = rotate_to_local(*world, 0.7)
    assert restored == pytest.approx((0.4, -0.2))


def test_range_scan_detects_room_and_obstacle() -> None:
    scan = range_scan(
        1.0,
        1.0,
        0.0,
        Bounds(0.0, 4.0, 0.0, 3.0),
        (ObstacleSpec("wall", 2.0, 2.2, 0.5, 1.5),),
        ray_count=4,
        max_range=4.0,
    )
    assert scan[0] == 0.25
    assert scan[2] == 0.25
