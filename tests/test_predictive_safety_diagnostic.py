from __future__ import annotations

import math

from hwr.adapters.mujoco.predictive_safety_diagnostic import (
    RawContactPoint,
    contact_point_witness,
)


def _point(
    index: int,
    force: float,
    geoms: tuple[int, int] = (1, 2),
) -> RawContactPoint:
    return RawContactPoint(
        contact_index=index,
        normal_force=force,
        geom_ids=geoms,
        geom_names=tuple(f"geom_{value}" for value in geoms),
        body_ids=geoms,
        body_names=tuple(f"body_{value}" for value in geoms),
    )


def test_witness_uses_maximum_contact_point_not_pair_sum() -> None:
    witness = contact_point_witness(
        (_point(0, 130.0), _point(1, 130.0)),
        robot_geom_ids=frozenset((1,)),
        allowed_environment_geom_ids=frozenset(),
        threshold=220.0,
    )

    assert witness["maximum_forbidden_contact_point_force"] == 130.0
    assert witness["witness_violation"] is False
    assert witness["forbidden_contact_point_count"] == 2


def test_witness_matches_inclusive_threshold_and_tie_break() -> None:
    witness = contact_point_witness(
        (
            _point(4, 220.0, (1, 5)),
            _point(3, -220.0, (1, 4)),
            _point(2, 219.999, (1, 3)),
        ),
        robot_geom_ids=frozenset((1,)),
        allowed_environment_geom_ids=frozenset(),
        threshold=220.0,
    )

    assert witness["witness_violation"] is True
    assert witness["display_maximum"]["contact_index"] == 3
    assert witness["display_maximum"]["normal_force"] == 220.0


def test_witness_excludes_allowed_and_non_cross_pairs() -> None:
    witness = contact_point_witness(
        (
            _point(0, 500.0, (1, 2)),
            _point(1, 500.0, (1, 1)),
            _point(2, 500.0, (3, 4)),
        ),
        robot_geom_ids=frozenset((1,)),
        allowed_environment_geom_ids=frozenset((2,)),
        threshold=220.0,
    )

    assert witness["forbidden_contact_point_count"] == 0
    assert witness["witness_violation"] is False


def test_witness_fails_closed_on_nonfinite_force() -> None:
    witness = contact_point_witness(
        (_point(0, math.nan),),
        robot_geom_ids=frozenset((1,)),
        allowed_environment_geom_ids=frozenset(),
        threshold=220.0,
    )

    assert witness["valid"] is False
    assert witness["invalid_force_count"] == 1
    assert witness["witness_violation"] is False
