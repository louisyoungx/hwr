"""Joint motion-planning teacher for the living-room basket task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.bimanual_teacher import BASKET_TASK_ID
from hwr.adapters.mujoco.joint_basket_acquire import acquire_feedback
from hwr.adapters.mujoco.joint_basket_planner import (
    GRASP_GRIPPER,
    JointGraspPlan,
    plan_joint_grasp,
)
from hwr.core.embodied import DualArmAction, DualArmObservation


JOINT_TEACHER_SOURCE = "r0020_joint_basket_motion_teacher"


@dataclass(frozen=True)
class JointTeacherOutput:
    action: DualArmAction
    stage: str


@dataclass(frozen=True)
class BaseMotion:
    kind: Literal["turn", "drive"]
    target: tuple[float, ...]
    reverse: bool = False


class JointBasketMotionTeacher:
    """Privileged full-task teacher with joint planning and feedback tracking."""

    implemented_task_phases = frozenset(
        {
            "approach",
            "acquire",
            "secure",
            "lift",
            "target_transport",
            "place",
            "release",
            "stabilize",
        }
    )

    def __init__(self, backend: MujocoBimanualTaskBackend, *, seed: int) -> None:
        if backend.task.task_id != BASKET_TASK_ID:
            raise ValueError("R0020 teacher only supports the living-room basket task")
        self.backend = backend
        self.seed = int(seed)
        self.stage = "approach"
        self.stage_step = 0
        self.failure_stage: str | None = None
        self._grasp_plan: JointGraspPlan | None = None
        self._waypoint_index = 0
        self._secure_contact_steps = 0
        self._lost_contact_steps = 0
        self._payload_start_position: np.ndarray | None = None
        self._payload_start_rotation: np.ndarray | None = None
        self._base_start_yaw: float | None = None
        self._payload_from_site: tuple[
            tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
        ] | None = None
        self._carry_offset: np.ndarray | None = None
        self._lift_height: float | None = None
        self._route: tuple[BaseMotion, ...] = ()
        self._route_index = 0
        self._placed_rotation: np.ndarray | None = None
        self._retract_targets: tuple[np.ndarray, np.ndarray] | None = None
        self._approach_settle_steps = 0
        self._grasp_closing = False
        self._geometry_acquire_started = False

    @property
    def grasp_plan(self) -> JointGraspPlan | None:
        return self._grasp_plan

    def action(self, observation: DualArmObservation) -> JointTeacherOutput:
        if self._grasp_plan is None:
            self._grasp_plan = plan_joint_grasp(self.backend, seed=self.seed)
        audit = self.backend.task_audit()
        metrics = audit["metrics"]
        if self.stage == "approach":
            return self._approach(observation)
        if self.stage == "acquire":
            return self._acquire(observation, audit)
        if self.stage == "secure":
            return self._secure(observation, audit)
        if self.stage == "lift":
            return self._lift(observation, metrics)
        if self.stage == "target_transport":
            return self._transport(observation, metrics)
        if self.stage == "place":
            return self._place(observation, metrics)
        if self.stage == "release":
            return self._release(observation, metrics)
        if self.stage == "stabilize":
            return self._stabilize(observation)
        return JointTeacherOutput(self._hold(observation), self.stage)

    def _approach(self, observation: DualArmObservation) -> JointTeacherOutput:
        assert self._grasp_plan is not None
        error = self._grasp_plan.base_x - observation.proprioception.base_pose[0]
        stopped = (
            abs(error) <= 0.012
            and abs(observation.proprioception.base_twist[0]) < 0.025
        )
        self._approach_settle_steps = (
            self._approach_settle_steps + 1 if stopped else 0
        )
        if self._approach_settle_steps >= 12:
            revised = plan_joint_grasp(self.backend, seed=self.seed)
            revised_error = (
                revised.base_x - observation.proprioception.base_pose[0]
            )
            self._grasp_plan = revised
            self._approach_settle_steps = 0
            if abs(revised_error) > 0.006:
                self.stage_step += 1
                return JointTeacherOutput(
                    _action(
                        base_linear=float(
                            np.clip(1.2 * revised_error, -0.07, 0.07)
                        )
                    ),
                    self.stage,
                )
            self._advance("acquire")
            return self._acquire(observation, self.backend.task_audit())
        if self.stage_step >= 160:
            self._fail("approach_timeout")
            return JointTeacherOutput(self._hold(observation), self.stage)
        self.stage_step += 1
        return JointTeacherOutput(
            _action(base_linear=float(np.clip(1.2 * error, -0.07, 0.07))),
            self.stage,
        )

    def _acquire(
        self,
        observation: DualArmObservation,
        audit: dict[str, object],
    ) -> JointTeacherOutput:
        assert self._grasp_plan is not None
        waypoints = self._grasp_plan.waypoints
        target = waypoints[self._waypoint_index]
        error = self._joint_error(target, observation)
        geometry_entry_index = 3
        if error < 0.035 and self._waypoint_index < geometry_entry_index:
            self._waypoint_index += 1
            target = waypoints[self._waypoint_index]
            error = self._joint_error(target, observation)
        concurrent = int(audit["concurrent_steps"])
        if concurrent >= self.backend.task.concurrent_steps:
            self._capture_grasp_transform(observation)
            self._advance("secure")
            return self._secure(observation, audit)
        if self.stage_step >= 480:
            self._fail("acquire_timeout")
            return JointTeacherOutput(self._hold(observation), self.stage)
        self.stage_step += 1
        if self._waypoint_index >= geometry_entry_index:
            self._geometry_acquire_started = True
            feedback = acquire_feedback(
                self.backend,
                target_rotations=(
                    self._grasp_plan.left_site_rotation,
                    self._grasp_plan.right_site_rotation,
                ),
                target_handle_from_midpoints=(
                    self._grasp_plan.left_handle_from_pad_midpoint,
                    self._grasp_plan.right_handle_from_pad_midpoint,
                ),
                current_grippers=(
                    observation.proprioception.left_gripper_position,
                    observation.proprioception.right_gripper_position,
                ),
            )
            self._grasp_closing = max(feedback.grippers) > 0.0
            return JointTeacherOutput(
                self._site_tracking_action(
                    feedback.targets,
                    gripper=feedback.grippers,
                ),
                self.stage,
            )
        return JointTeacherOutput(
            self._joint_tracking_action(target, 0.0),
            self.stage,
        )

    def _secure(
        self,
        observation: DualArmObservation,
        audit: dict[str, object],
    ) -> JointTeacherOutput:
        assert self._grasp_plan is not None
        metrics = audit["metrics"]
        bilateral = bool(metrics["left_contact"] and metrics["right_contact"])
        self._secure_contact_steps = self._secure_contact_steps + 1 if bilateral else 0
        if self._secure_contact_steps >= 16:
            self._capture_grasp_transform(observation)
            self._prepare_lift()
            self._advance("lift")
            return self._lift(observation, metrics)
        if self.stage_step >= 100:
            self._fail("secure_timeout")
            return JointTeacherOutput(self._hold(observation), self.stage)
        self.stage_step += 1
        target = (
            self._grasp_plan.left_joint_target,
            self._grasp_plan.right_joint_target,
        )
        return JointTeacherOutput(
            self._joint_tracking_action(target, GRASP_GRIPPER),
            self.stage,
        )

    def _lift(
        self,
        observation: DualArmObservation,
        metrics: dict[str, float],
    ) -> JointTeacherOutput:
        assert self._payload_start_position is not None
        assert self._payload_start_rotation is not None
        assert self._lift_height is not None
        payload_goal = self._payload_start_position.copy()
        payload_goal[2] = self._lift_height
        action = self._payload_tracking_action(
            payload_goal,
            self._payload_start_rotation,
            gripper=GRASP_GRIPPER,
        )
        if (
            abs(
                float(
                    self.backend.data.xpos[
                        self.backend.task_ids.payload_body
                    ][2]
                )
                - self._lift_height
            )
            <= 0.025
            and float(metrics["payload_linear_speed"]) <= 0.08
        ):
            self._prepare_transport(observation)
            self._advance("target_transport")
            return self._transport(observation, metrics)
        if self._contact_failed(metrics) or self.stage_step >= 180:
            self._fail("lift_contact_lost" if self._lost_contact_steps else "lift_timeout")
            return JointTeacherOutput(self._hold(observation), self.stage)
        self.stage_step += 1
        return JointTeacherOutput(action, self.stage)

    def _transport(
        self,
        observation: DualArmObservation,
        metrics: dict[str, float],
    ) -> JointTeacherOutput:
        if self._contact_failed(metrics):
            self._fail("target_transport_contact_lost")
            return JointTeacherOutput(self._hold(observation), self.stage)
        if self._route_index >= len(self._route):
            self._placed_rotation = self._desired_payload_rotation(observation)
            self._advance("place")
            return self._place(observation, metrics)
        motion = self._route[self._route_index]
        base_linear, base_angular, complete = self._base_motion_action(
            motion, observation
        )
        if complete:
            self._route_index += 1
            return self._transport(observation, metrics)
        payload_goal = self._carried_payload_position(observation)
        action = self._payload_tracking_action(
            payload_goal,
            self._desired_payload_rotation(observation),
            gripper=GRASP_GRIPPER,
            base_linear=base_linear,
            base_angular=base_angular,
        )
        if self.stage_step >= 560:
            self._fail("target_transport_timeout")
            return JointTeacherOutput(self._hold(observation), self.stage)
        self.stage_step += 1
        return JointTeacherOutput(action, self.stage)

    def _place(
        self,
        observation: DualArmObservation,
        metrics: dict[str, float],
    ) -> JointTeacherOutput:
        target = self.backend.data.site_xpos[self.backend.task_ids.target_site].copy()
        rotation = (
            self._placed_rotation
            if self._placed_rotation is not None
            else self._desired_payload_rotation(observation)
        )
        action = self._payload_tracking_action(
            target,
            rotation,
            gripper=GRASP_GRIPPER,
        )
        placed = (
            bool(metrics["support_contact"])
            and float(metrics["target_distance"]) <= 0.09
        )
        if placed:
            self._advance("release")
            return self._release(observation, metrics)
        if self._contact_failed(metrics) or self.stage_step >= 180:
            self._fail("place_contact_lost" if self._lost_contact_steps else "place_timeout")
            return JointTeacherOutput(self._hold(observation), self.stage)
        self.stage_step += 1
        return JointTeacherOutput(action, self.stage)

    def _release(
        self,
        observation: DualArmObservation,
        metrics: dict[str, float],
    ) -> JointTeacherOutput:
        target = self.backend.data.site_xpos[self.backend.task_ids.target_site].copy()
        rotation = (
            self._placed_rotation
            if self._placed_rotation is not None
            else self._desired_payload_rotation(observation)
        )
        action = self._payload_tracking_action(target, rotation, gripper=0.0)
        released = (
            observation.proprioception.left_gripper_position <= 0.16
            and observation.proprioception.right_gripper_position <= 0.16
            and not bool(metrics["left_contact"] or metrics["right_contact"])
        )
        if released:
            self._prepare_retract()
            self._advance("stabilize")
            return self._stabilize(observation)
        if self.stage_step >= 80:
            self._fail("release_timeout")
            return JointTeacherOutput(self._hold(observation), self.stage)
        self.stage_step += 1
        return JointTeacherOutput(action, self.stage)

    def _stabilize(self, observation: DualArmObservation) -> JointTeacherOutput:
        assert self._retract_targets is not None
        if self.stage_step >= 120:
            self._fail("stabilize_timeout")
            return JointTeacherOutput(self._hold(observation), self.stage)
        self.stage_step += 1
        return JointTeacherOutput(
            self._site_tracking_action(self._retract_targets, gripper=0.0),
            self.stage,
        )

    def _capture_grasp_transform(self, observation: DualArmObservation) -> None:
        data = self.backend.data
        ids = self.backend.task_ids
        payload_position = data.xpos[ids.payload_body].copy()
        payload_rotation = data.xmat[ids.payload_body].reshape(3, 3).copy()
        transforms = []
        for site in (ids.left_grasp_site, ids.right_grasp_site):
            site_position = data.site_xpos[site].copy()
            site_rotation = data.site_xmat[site].reshape(3, 3).copy()
            transforms.append(
                (
                    payload_rotation.T @ (site_position - payload_position),
                    payload_rotation.T @ site_rotation,
                )
            )
        self._payload_from_site = (transforms[0], transforms[1])
        self._payload_start_position = payload_position
        self._payload_start_rotation = payload_rotation
        self._base_start_yaw = observation.proprioception.base_pose[2]

    def _prepare_lift(self) -> None:
        assert self._payload_start_position is not None
        target_support = self.backend.task_ids.target_support_geom
        support_top = float(
            self.backend.data.geom_xpos[target_support][2]
            + self.backend.model.geom_size[target_support][2]
        )
        payload = self.backend.task_ids.payload_reference_geom
        payload_half_height = float(self.backend.model.geom_size[payload][2])
        self._lift_height = max(
            float(self._payload_start_position[2]) + 0.18,
            support_top + payload_half_height + 0.11,
        )

    def _prepare_transport(self, observation: DualArmObservation) -> None:
        payload = self.backend.data.xpos[self.backend.task_ids.payload_body].copy()
        base = np.asarray((*observation.proprioception.base_pose[:2], payload[2]))
        yaw = observation.proprioception.base_pose[2]
        self._carry_offset = _yaw_rotation(-yaw) @ (payload - base)
        target = self.backend.data.site_xpos[self.backend.task_ids.target_site].copy()
        final_yaw = math.pi / 2
        final_offset = _yaw_rotation(final_yaw) @ self._carry_offset
        support = self.backend.task_ids.target_support_geom
        support_center = self.backend.data.geom_xpos[support]
        support_half_y = float(self.backend.model.geom_size[support][1])
        chassis = int(
            mujoco.mj_name2id(
                self.backend.model, mujoco.mjtObj.mjOBJ_GEOM, "chassis_collision"
            )
        )
        chassis_size = self.backend.model.geom_size[chassis]
        turn_radius = float(np.linalg.norm(chassis_size[:2]) + 0.045)
        corridor_y = float(support_center[1] - support_half_y - turn_radius - 0.08)
        start_support = int(
            mujoco.mj_name2id(
                self.backend.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "basket_start_support",
            )
        )
        start_min_x = float(
            self.backend.data.geom_xpos[start_support][0]
            - self.backend.model.geom_size[start_support][0]
        )
        retreat_x = min(
            observation.proprioception.base_pose[0],
            start_min_x - turn_radius - 0.04,
        )
        final_base_x = float(target[0] - final_offset[0])
        safe_final_y = float(support_center[1] - support_half_y - 0.38)
        final_base_y = min(float(target[1] - final_offset[1]), safe_final_y)
        self._route = (
            BaseMotion("drive", (retreat_x, observation.proprioception.base_pose[1]), True),
            BaseMotion("turn", (-math.pi / 2,)),
            BaseMotion("drive", (retreat_x, corridor_y)),
            BaseMotion("turn", (0.0,)),
            BaseMotion("drive", (final_base_x, corridor_y)),
            BaseMotion("turn", (final_yaw,)),
            BaseMotion("drive", (final_base_x, final_base_y)),
        )
        self._route_index = 0

    def _carried_payload_position(
        self, observation: DualArmObservation
    ) -> np.ndarray:
        assert self._carry_offset is not None
        assert self._lift_height is not None
        x, y, yaw = observation.proprioception.base_pose
        offset = _yaw_rotation(yaw) @ self._carry_offset
        return np.asarray((x + offset[0], y + offset[1], self._lift_height))

    def _desired_payload_rotation(
        self, observation: DualArmObservation
    ) -> np.ndarray:
        assert self._payload_start_rotation is not None
        assert self._base_start_yaw is not None
        delta = observation.proprioception.base_pose[2] - self._base_start_yaw
        return _yaw_rotation(delta) @ self._payload_start_rotation

    def _base_motion_action(
        self,
        motion: BaseMotion,
        observation: DualArmObservation,
    ) -> tuple[float, float, bool]:
        x, y, yaw = observation.proprioception.base_pose
        linear_speed, angular_speed = observation.proprioception.base_twist
        if motion.kind == "turn":
            error = _wrap_angle(motion.target[0] - yaw)
            complete = abs(error) <= 0.025 and abs(angular_speed) <= 0.045
            return 0.0, float(np.clip(1.4 * error, -0.22, 0.22)), complete
        delta = np.asarray(motion.target) - np.asarray((x, y))
        distance = float(np.linalg.norm(delta))
        if distance <= 0.025 and abs(linear_speed) <= 0.025:
            return 0.0, 0.0, True
        desired_heading = math.atan2(float(delta[1]), float(delta[0]))
        direction = -1.0 if motion.reverse else 1.0
        if motion.reverse:
            desired_heading = _wrap_angle(desired_heading + math.pi)
        heading_error = _wrap_angle(desired_heading - yaw)
        linear = direction * min(0.075, max(0.018, 0.7 * distance))
        linear *= max(0.0, math.cos(heading_error))
        angular = float(np.clip(1.2 * heading_error, -0.18, 0.18))
        return linear, angular, False

    def _payload_tracking_action(
        self,
        payload_position: np.ndarray,
        payload_rotation: np.ndarray,
        *,
        gripper: float | tuple[float, float],
        base_linear: float = 0.0,
        base_angular: float = 0.0,
    ) -> DualArmAction:
        assert self._payload_from_site is not None
        targets = tuple(
            (
                payload_position + payload_rotation @ relative_position,
                payload_rotation @ relative_rotation,
            )
            for relative_position, relative_rotation in self._payload_from_site
        )
        return self._site_tracking_action(
            targets,
            gripper=gripper,
            base_linear=base_linear,
            base_angular=base_angular,
        )

    def _site_tracking_action(
        self,
        targets: tuple[
            tuple[np.ndarray, np.ndarray] | np.ndarray,
            tuple[np.ndarray, np.ndarray] | np.ndarray,
        ],
        *,
        gripper: float,
        base_linear: float = 0.0,
        base_angular: float = 0.0,
    ) -> DualArmAction:
        data = self.backend.data
        ids = self.backend.task_ids
        base_rotation = data.xmat[self.backend.bundle.ids.base_body].reshape(3, 3)
        commands = []
        for target, site in zip(
            targets, (ids.left_grasp_site, ids.right_grasp_site), strict=True
        ):
            if isinstance(target, tuple):
                target_position, target_rotation = target
                rotation_error = _rotation_error(
                    target_rotation,
                    data.site_xmat[site].reshape(3, 3),
                )
            else:
                target_position = target
                rotation_error = np.zeros(3)
            linear = _clip_norm(
                2.4 * (target_position - data.site_xpos[site]),
                0.095,
            )
            angular = _clip_norm(1.6 * rotation_error, 0.30)
            command = np.concatenate(
                (
                    base_rotation.T @ linear
                    / self.backend.config.max_tool_linear_velocity,
                    base_rotation.T @ angular
                    / self.backend.config.max_tool_angular_velocity,
                )
            )
            maximum = float(np.abs(command).max())
            if maximum > 0.35:
                command *= 0.35 / maximum
            commands.append(tuple(float(value) for value in command))
        return _action(
            base_linear=base_linear,
            base_angular=base_angular,
            left_arm=commands[0],
            right_arm=commands[1],
            gripper=gripper,
        )

    def _joint_tracking_action(
        self,
        target: tuple[np.ndarray, np.ndarray],
        gripper: float,
    ) -> DualArmAction:
        commands = []
        base_rotation = self.backend.data.xmat[
            self.backend.bundle.ids.base_body
        ].reshape(3, 3)
        for desired, joint_ids, site in (
            (
                target[0],
                self.backend.bundle.ids.secondary_arm_joints,
                self.backend.task_ids.left_grasp_site,
            ),
            (
                target[1],
                self.backend.bundle.ids.arm_joints,
                self.backend.task_ids.right_grasp_site,
            ),
        ):
            current = np.asarray(
                [
                    self.backend.data.qpos[self.backend.model.jnt_qposadr[joint]]
                    for joint in joint_ids
                ]
            )
            desired_velocity = _clip_norm(1.8 * (desired - current), 0.34)
            jacobian_position = np.zeros((3, self.backend.model.nv))
            jacobian_rotation = np.zeros((3, self.backend.model.nv))
            mujoco.mj_jacSite(
                self.backend.model,
                self.backend.data,
                jacobian_position,
                jacobian_rotation,
                site,
            )
            dofs = [self.backend.model.jnt_dofadr[joint] for joint in joint_ids]
            jacobian = np.vstack(
                (jacobian_position[:, dofs], jacobian_rotation[:, dofs])
            )
            inverse = jacobian.T @ np.linalg.inv(
                jacobian @ jacobian.T
                + np.eye(6) * self.backend.config.ik_damping**2
            )
            twist = np.linalg.lstsq(inverse, desired_velocity, rcond=1.0e-5)[0]
            command = np.concatenate(
                (
                    base_rotation.T @ twist[:3]
                    / self.backend.config.max_tool_linear_velocity,
                    base_rotation.T @ twist[3:]
                    / self.backend.config.max_tool_angular_velocity,
                )
            )
            maximum = float(np.abs(command).max())
            if maximum > 0.35:
                command *= 0.35 / maximum
            commands.append(tuple(float(value) for value in command))
        return _action(
            left_arm=commands[0],
            right_arm=commands[1],
            gripper=gripper,
        )

    def _joint_error(
        self,
        target: tuple[np.ndarray, np.ndarray],
        observation: DualArmObservation,
    ) -> float:
        current = np.asarray(
            (
                *observation.proprioception.left_joint_position,
                *observation.proprioception.right_joint_position,
            )
        )
        desired = np.concatenate(target)
        return float(np.max(np.abs(desired - current)))

    def _contact_failed(self, metrics: dict[str, float]) -> bool:
        bilateral = bool(metrics["left_contact"] and metrics["right_contact"])
        self._lost_contact_steps = 0 if bilateral else self._lost_contact_steps + 1
        return self._lost_contact_steps >= 8

    def _prepare_retract(self) -> None:
        data = self.backend.data
        ids = self.backend.task_ids
        self._retract_targets = (
            data.site_xpos[ids.left_grasp_site] + np.asarray((0.0, 0.10, 0.05)),
            data.site_xpos[ids.right_grasp_site] + np.asarray((0.0, -0.10, 0.05)),
        )

    def _advance(self, stage: str) -> None:
        self.stage = stage
        self.stage_step = 0
        self._lost_contact_steps = 0

    def _fail(self, failure_stage: str) -> None:
        self.failure_stage = failure_stage
        self.stage = "failed_hold"

    @staticmethod
    def _hold(observation: DualArmObservation) -> DualArmAction:
        return _action(
            gripper=(
                observation.proprioception.left_gripper_position,
                observation.proprioception.right_gripper_position,
            )
        )


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = target @ current.T
    return 0.5 * np.asarray(
        (
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        )
    )


def _yaw_rotation(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or norm <= maximum:
        return vector
    return vector * maximum / norm


def _action(
    *,
    base_linear: float = 0.0,
    base_angular: float = 0.0,
    left_arm: tuple[float, ...] = (0.0,) * 6,
    right_arm: tuple[float, ...] = (0.0,) * 6,
    gripper: float | tuple[float, float] = 0.0,
) -> DualArmAction:
    grippers = (gripper, gripper) if isinstance(gripper, float) else gripper
    return DualArmAction(
        base_linear,
        base_angular,
        left_arm,
        right_arm,
        float(grippers[0]),
        float(grippers[1]),
    )
