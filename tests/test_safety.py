from __future__ import annotations

from hwr.core.types import ActionFrame
from hwr.safety import SafetyLimits, SafetySupervisor


def _action(**overrides: float) -> ActionFrame:
    values = {
        "created_at_ns": 10,
        "valid_from_ns": 10,
        "valid_until_ns": 20,
        "source": "policy",
        "base_linear": 1.0,
        "base_angular": -2.0,
        "arm_command": (2.0, -2.0),
        "gripper_target": 2.0,
    }
    values.update(overrides)
    return ActionFrame(**values)


def test_safety_supervisor_clamps_actions() -> None:
    supervisor = SafetySupervisor(SafetyLimits(), arm_dof=2)

    filtered, events = supervisor.filter(_action(), now_ns=15)

    assert filtered.base_linear == 0.3
    assert filtered.base_angular == -0.8
    assert filtered.arm_command == (1.0, -1.0)
    assert filtered.gripper_target == 1.0
    assert events[0].event_type == "action_clamped"


def test_safety_supervisor_stops_expired_action() -> None:
    supervisor = SafetySupervisor(SafetyLimits(), arm_dof=2)

    filtered, events = supervisor.filter(_action(), now_ns=21)

    assert filtered.source == "safety"
    assert filtered.base_linear == 0.0
    assert filtered.arm_command == (0.0, 0.0)
    assert events[0].details["reason"] == "outside_validity_window"

