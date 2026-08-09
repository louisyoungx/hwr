"""Observation-only reference expert for mobile pick-and-place data generation."""

from __future__ import annotations

import math

from hwr.core.types import ActionFrame, ObservationFrame
from hwr.sim.geometry import clamp
from hwr.sim.specs import RobotSpec


class PickPlaceExpert:
    def __init__(self, robot_spec: RobotSpec, *, source: str = "rule_expert") -> None:
        self.robot_spec = robot_spec
        self.source = source

    def action(self, observation: ObservationFrame) -> ActionFrame:
        carrying = observation.features["carrying"][0] > 0.5
        target = (
            observation.features["target_zone_relative"]
            if carrying
            else observation.features["target_object_relative"]
        )
        arm_x, arm_y = observation.joint_position
        distance = math.hypot(*target)
        work_distance = self.robot_spec.arm_reach * 0.72
        base_linear = 0.0
        base_angular = 0.0
        arm_command = (0.0, 0.0)
        gripper = 1.0 if carrying else 0.0

        if distance > work_distance + 0.03:
            angle_error = math.atan2(target[1], target[0])
            base_angular = clamp(2.0 * angle_error, -1.0, 1.0)
            if abs(angle_error) < 0.45:
                base_linear = clamp(0.8 * (distance - work_distance), 0.0, 0.35)
            arm_command = (-1.5 * (arm_x - 0.15), -1.5 * arm_y)
        else:
            error_x = target[0] - arm_x
            error_y = target[1] - arm_y
            arm_command = (
                clamp(3.0 * error_x, -self.robot_spec.arm_speed, self.robot_spec.arm_speed),
                clamp(3.0 * error_y, -self.robot_spec.arm_speed, self.robot_spec.arm_speed),
            )
            endpoint_error = math.hypot(error_x, error_y)
            if endpoint_error <= self.robot_spec.gripper_radius * 0.65:
                gripper = 0.0 if carrying else 1.0

        now_ns = observation.timestamp_ns
        period_ns = round(1_000_000_000 / self.robot_spec.control_hz)
        return ActionFrame(
            created_at_ns=now_ns,
            valid_from_ns=now_ns,
            valid_until_ns=now_ns + period_ns,
            source=self.source,
            base_linear=base_linear,
            base_angular=base_angular,
            arm_command=arm_command,
            gripper_target=gripper,
        )
