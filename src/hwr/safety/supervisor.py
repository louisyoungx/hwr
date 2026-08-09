"""Safety supervisor that filters all policy and teleoperation actions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from hwr.core.types import ActionFrame, EpisodeEvent


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class SafetyLimits:
    max_base_linear: float = 0.3
    max_base_angular: float = 0.8
    max_arm_command: float = 1.0
    min_gripper: float = 0.0
    max_gripper: float = 1.0

    def __post_init__(self) -> None:
        if min(self.max_base_linear, self.max_base_angular, self.max_arm_command) <= 0:
            raise ValueError("safety motion limits must be positive")
        if self.min_gripper >= self.max_gripper:
            raise ValueError("gripper limits are inverted")


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits, *, arm_dof: int) -> None:
        if arm_dof < 0:
            raise ValueError("arm_dof must be non-negative")
        self.limits = limits
        self.arm_dof = arm_dof

    def filter(self, action: ActionFrame, *, now_ns: int) -> tuple[ActionFrame, tuple[EpisodeEvent, ...]]:
        if now_ns < action.valid_from_ns or now_ns > action.valid_until_ns:
            stopped = replace(
                action,
                base_linear=0.0,
                base_angular=0.0,
                arm_command=(0.0,) * self.arm_dof,
                source="safety",
                confidence=1.0,
            )
            event = EpisodeEvent(
                timestamp_ns=now_ns,
                event_type="action_rejected",
                source="safety",
                details={"reason": "outside_validity_window"},
            )
            return stopped, (event,)

        if len(action.arm_command) != self.arm_dof:
            raise ValueError(f"arm command has {len(action.arm_command)} values; expected {self.arm_dof}")

        filtered = replace(
            action,
            base_linear=_clamp(action.base_linear, self.limits.max_base_linear),
            base_angular=_clamp(action.base_angular, self.limits.max_base_angular),
            arm_command=tuple(
                _clamp(value, self.limits.max_arm_command) for value in action.arm_command
            ),
            gripper_target=max(
                self.limits.min_gripper,
                min(self.limits.max_gripper, action.gripper_target),
            ),
        )
        if filtered == action:
            return filtered, ()
        event = EpisodeEvent(
            timestamp_ns=now_ns,
            event_type="action_clamped",
            source="safety",
            details={"original_source": action.source},
        )
        return filtered, (event,)

