from __future__ import annotations

import pytest

from hwr.core.embodied import DualArmAction, DualArmActionFrame
from hwr.safety import DualArmSafetySupervisor, SafetyLimits


def _frame(*, valid_until_ns: int = 20) -> DualArmActionFrame:
    return DualArmActionFrame(
        created_at_ns=10,
        valid_from_ns=10,
        valid_until_ns=valid_until_ns,
        source="actor",
        action=DualArmAction(
            1.0,
            -2.0,
            (2.0, -2.0, 0.5, 0.0, 1.5, -1.5),
            (-2.0, 2.0, -0.5, 0.0, -1.5, 1.5),
            0.0,
            1.0,
        ),
    )


def test_dual_arm_safety_clamps_both_sides_symmetrically() -> None:
    supervisor = DualArmSafetySupervisor(SafetyLimits())

    filtered, events = supervisor.filter(
        _frame(), now_ns=15, hold_grippers=(0.2, 0.8)
    )

    assert filtered.action.base_linear == 0.3
    assert filtered.action.base_angular == -0.8
    assert filtered.action.left_arm == pytest.approx(
        (1.0, -1.0, 0.5, 0.0, 1.0, -1.0)
    )
    assert filtered.action.right_arm == pytest.approx(
        (-1.0, 1.0, -0.5, 0.0, -1.0, 1.0)
    )
    assert events[0].event_type == "action_clamped"


def test_dual_arm_safety_stops_all_motion_and_holds_both_grippers() -> None:
    supervisor = DualArmSafetySupervisor(SafetyLimits())

    filtered, events = supervisor.filter(
        _frame(valid_until_ns=11), now_ns=21, hold_grippers=(0.2, 0.8)
    )

    assert filtered.source == "safety"
    assert filtered.action.base_linear == 0.0
    assert filtered.action.left_arm == (0.0,) * 6
    assert filtered.action.right_arm == (0.0,) * 6
    assert (filtered.action.left_gripper, filtered.action.right_gripper) == (0.2, 0.8)
    assert events[0].details["reason"] == "outside_validity_window"
