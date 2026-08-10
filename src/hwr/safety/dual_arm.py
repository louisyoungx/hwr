"""Runtime-independent safety filtering for canonical dual-arm actions."""

from __future__ import annotations

from dataclasses import replace

from hwr.core.embodied import DualArmAction, DualArmActionFrame
from hwr.core.types import EpisodeEvent
from hwr.safety.supervisor import SafetyLimits


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class DualArmSafetySupervisor:
    """Clamp both arms symmetrically without selecting a preferred side."""

    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    def filter(
        self,
        frame: DualArmActionFrame,
        *,
        now_ns: int,
        hold_grippers: tuple[float, float],
    ) -> tuple[DualArmActionFrame, tuple[EpisodeEvent, ...]]:
        if len(hold_grippers) != 2 or not all(
            0.0 <= value <= 1.0 for value in hold_grippers
        ):
            raise ValueError("hold grippers must contain two normalized values")
        if now_ns < frame.valid_from_ns or now_ns > frame.valid_until_ns:
            stopped = replace(
                frame,
                action=DualArmAction(
                    0.0,
                    0.0,
                    (0.0,) * 6,
                    (0.0,) * 6,
                    hold_grippers[0],
                    hold_grippers[1],
                ),
                source="safety",
                confidence=1.0,
            )
            return stopped, (
                EpisodeEvent(
                    timestamp_ns=now_ns,
                    event_type="action_rejected",
                    source="safety",
                    details={"reason": "outside_validity_window"},
                ),
            )

        command = frame.action
        filtered_action = DualArmAction(
            base_linear=_clamp(command.base_linear, self.limits.max_base_linear),
            base_angular=_clamp(command.base_angular, self.limits.max_base_angular),
            left_arm=tuple(
                _clamp(value, self.limits.max_arm_command)
                for value in command.left_arm
            ),
            right_arm=tuple(
                _clamp(value, self.limits.max_arm_command)
                for value in command.right_arm
            ),
            left_gripper=max(
                self.limits.min_gripper,
                min(self.limits.max_gripper, command.left_gripper),
            ),
            right_gripper=max(
                self.limits.min_gripper,
                min(self.limits.max_gripper, command.right_gripper),
            ),
        )
        if filtered_action == command:
            return frame, ()
        filtered = replace(frame, action=filtered_action)
        return filtered, (
            EpisodeEvent(
                timestamp_ns=now_ns,
                event_type="action_clamped",
                source="safety",
                details={"original_source": frame.source},
            ),
        )
