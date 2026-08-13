"""Privileged finite-state expert used only to generate formal 3D demonstrations."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from hwr.adapters.mujoco.expert import PrivilegedCartesianExpert
from hwr.adapters.mujoco.formal_expert_types import ExpertStage, FormalExpertOutput
from hwr.adapters.mujoco.formal_postures import (
    ARM_DRAWER_GRASP,
    ARM_READY_DRAWER,
    ARM_READY_KITCHEN,
    ARM_READY_TABLE,
    ARM_STOW,
    DRAWER_GRIPPER_ROTATION,
    drawer_posture,
)
from hwr.adapters.mujoco.formal_routes import (
    drawer_base_commands,
    drawer_base_is_aligned,
    formal_waypoints,
    gripper_rotation,
    navigation_linear_speed,
    navigation_tolerances,
    object_approach_yaw,
    object_orientation_weight,
    target_approach_yaw,
    top_down_gripper_rotation,
    top_down_site_compensation,
    wrap_angle,
)
from hwr.adapters.mujoco.formal_state import FormalExpertStateMixin
from hwr.adapters.mujoco.household_backend import MujocoHouseholdBackend
from hwr.adapters.mujoco.names import ARM_HOME
from hwr.core.types import ActionFrame, ObservationFrame


class PrivilegedHouseholdExpert(FormalExpertStateMixin):
    """Read engine truth for labels; never use this object during learned-policy evaluation."""

    def __init__(self, backend: MujocoHouseholdBackend) -> None:
        if backend._task_id is None:  # noqa: SLF001 - adapter-private expert
            raise RuntimeError("reset the backend before constructing its expert")
        self.backend = backend
        self.task = backend.task
        self.cartesian = PrivilegedCartesianExpert(backend, source="privileged_3d_expert")
        self.stages = self._build_stages()
        self.stage_index = 0
        self.stage_step = 0
        self.stage_target: tuple[float, float, float] | None = None
        self.nav_targets: list[tuple[float, float, float | None]] = []
        self.nav_aligning = False
        self.holding_object: str | None = None
        self.drawer_holding = False
        self.drawer_bilateral_steps = 0
        self.object_bilateral_steps = 0
        self.drawer_contact_start_substeps = 0
        self.drawer_pull_origin: tuple[float, float, float] | None = None
        self.object_contact_start_substeps = 0
        self.holding_contact_loss_steps = 0
        self.failed = False
        self._enter_stage()

    @property
    def stage(self) -> ExpertStage:
        return self.stages[min(self.stage_index, len(self.stages) - 1)]

    @property
    def done(self) -> bool:
        return self.stage_index >= len(self.stages)

    @property
    def phase_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    def action(self, observation: ObservationFrame) -> FormalExpertOutput:
        if self.done:
            return FormalExpertOutput(self._stop(observation), "done", self.stage_step)
        if self._holding_contact_timed_out():
            self.failed = True
            return FormalExpertOutput(
                self._stop(observation), self.stage.name, self.stage_step
            )
        stage = self.stage
        current_index = self.stage_index
        current_step = self.stage_step
        if stage.kind.startswith("stow_for_"):
            action = self._stow_phase_action(observation)
        elif stage.kind.startswith("nav_"):
            action = self._navigation_action(observation)
        elif stage.kind == "transport_arm":
            action = self._transport_action(observation)
        elif stage.kind.startswith("arm_"):
            action = self._arm_action(observation)
        elif stage.kind == "unstow_arm":
            action = self._unstow_action(observation)
        elif stage.kind == "grip_object":
            action = self._hold_action(observation, self._grip_fraction(stage.object_id))
            contact_substeps = (
                self.backend._bilateral_contact_steps[stage.object_id]  # noqa: SLF001
                - self.object_contact_start_substeps
            )
            if contact_substeps >= 150:
                self.holding_object = stage.object_id
                self.holding_contact_loss_steps = 0
                self._advance()
            elif self.stage_step >= 219:
                self.failed = True
                self._advance()
        elif stage.kind == "release_object":
            action = self._hold_action(observation, 0.0)
            if self.stage_step >= 49:
                self.holding_object = None
                if self._object_inside_target(stage.object_id):
                    self._advance()
                elif self.stage_step >= 119:
                    self.failed = True
                    self._advance()
        elif stage.kind == "grip_drawer":
            error = np.asarray(ARM_DRAWER_GRASP) - np.asarray(
                observation.joint_position
            )
            arm_command = tuple(
                float(value) for value in np.clip(2.0 * error, -0.5, 0.5)
            )
            linear, angular = drawer_base_commands(observation.base_pose, 1.30)
            action = self._action(
                observation,
                linear=linear,
                angular=angular,
                gripper=1.0,
                arm_command=arm_command,
            )
            contact_substeps = (
                self.backend._drawer_bilateral_contact_steps  # noqa: SLF001
                - self.drawer_contact_start_substeps
            )
            if contact_substeps >= 150:
                self.drawer_holding = True
                self._advance()
            elif self.stage_step >= 279:
                self.failed = True
                self._advance()
        elif stage.kind == "pull_drawer":
            action = self._pull_drawer_action(observation)
        elif stage.kind == "back_away_drawer":
            action = self._back_away_drawer_action(observation)
        elif stage.kind == "release_drawer":
            action = self._hold_action(observation, 0.0)
            if self.stage_step >= 39:
                self.drawer_holding = False
                self._advance()
        else:
            action = self._stop(observation)
            if self.backend.result() is not None or self.stage_step >= 119:
                self._advance()
        output = FormalExpertOutput(action, stage.name, current_step)
        if self.stage_index == current_index:
            self.stage_step += 1
        return output

    def _build_stages(self) -> tuple[ExpertStage, ...]:
        stages: list[ExpertStage] = []
        if self.task.articulation is not None:
            stages.extend(
                ExpertStage(name, kind)
                for name, kind in (
                    ("stow_for_drawer", "stow_for_nav_drawer"),
                    ("navigate_to_drawer", "nav_drawer"),
                    ("unfold_arm_for_drawer", "unstow_arm"),
                    ("approach_drawer_handle", "arm_drawer_above"),
                    ("prealign_over_drawer_handle", "arm_drawer_prealign"),
                    ("descend_to_drawer_handle", "arm_drawer_descend"),
                    ("close_on_drawer_handle", "grip_drawer"),
                    ("contact_pull_drawer", "pull_drawer"),
                    ("release_drawer_handle", "release_drawer"),
                    ("retract_from_drawer", "arm_drawer_retract"),
                    ("back_away_from_open_drawer", "back_away_drawer"),
                )
            )
        object_kinds = [
            "stow_for_nav_object",
            "nav_object",
            "unstow_arm",
            "arm_object_above",
            "arm_object_descend",
            "grip_object",
            "arm_object_lift",
            "transport_arm",
            "nav_target",
            "arm_target_above",
            "arm_target_lower",
            "release_object",
            "arm_target_retract",
        ]
        if self.task.task_id.startswith("tidy_living"):
            object_kinds.insert(3, "arm_object_clearance")
        if self.task.task_id.startswith("store_kitchen"):
            object_kinds.insert(8, "arm_target_raise")
        for obj in self.task.objects:
            stages.extend(
                ExpertStage(f"{kind}_{obj.object_id}", kind, obj.object_id)
                for kind in object_kinds
            )
        stages.append(ExpertStage("wait_for_two_second_stability", "wait"))
        return tuple(stages)

    def _enter_stage(self) -> None:
        if self.done:
            return
        self.stage_step = 0
        self.stage_target = None
        self.nav_targets = []
        self.nav_aligning = False
        if self.stage.kind == "grip_drawer":
            self.drawer_bilateral_steps = 0
            self.drawer_contact_start_substeps = (
                self.backend._drawer_bilateral_contact_steps  # noqa: SLF001
            )
        elif self.stage.kind == "grip_object":
            self.object_bilateral_steps = 0
            self.object_contact_start_substeps = (
                self.backend._bilateral_contact_steps[self.stage.object_id]  # noqa: SLF001
            )
        elif self.stage.kind == "pull_drawer":
            self.drawer_pull_origin = self.cartesian.site_position()
        stage = self.stage
        if stage.kind.startswith("nav_"):
            self.nav_targets = self._navigation_targets(stage)
        elif stage.kind.startswith("arm_"):
            self.cartesian.reset_orientation_target()
            if "drawer" in stage.kind:
                self.cartesian.set_orientation_target(np.asarray(DRAWER_GRIPPER_ROTATION))
            elif "object" in stage.kind:
                yaw = object_approach_yaw(self.task.task_id, stage.object_id)
                rotation = (
                    top_down_gripper_rotation(yaw)
                    if self.task.task_id.startswith("tidy_living")
                    else gripper_rotation(yaw)
                )
                self.cartesian.set_orientation_target(np.asarray(rotation))
            self.stage_target = self._cartesian_target(stage)

    def _advance(self) -> None:
        self.stage_index += 1
        self._enter_stage()

    def _navigation_targets(self, stage: ExpertStage) -> list[tuple[float, float, float | None]]:
        waypoints = formal_waypoints(self.task.task_id, stage.kind, stage.object_id)
        if stage.kind == "nav_drawer":
            target = self._drawer_handle_position()
            yaw, standoff = math.pi / 2, 0.61
        elif stage.kind == "nav_object":
            target = self._object_position(stage.object_id)
            yaw = object_approach_yaw(self.task.task_id, stage.object_id)
            standoff = self._object_spec(stage.object_id).standoff_m
            if (
                self.task.task_id.startswith("clear_dining")
                and stage.object_id != "plate"
            ):
                standoff -= 0.12
            elif self.task.task_id.startswith("tidy_living"):
                standoff = min(standoff, 0.58)
            elif (
                self.task.task_id.startswith("store_kitchen")
                and stage.object_id == "cleaner_yellow"
            ):
                standoff += 0.15
        else:
            target = self._target_position(stage.object_id)
            yaw = target_approach_yaw(self.task.task_id)
            standoff = (
                0.65
                if self.task.task_id.startswith("store_kitchen")
                else self._object_spec(stage.object_id).standoff_m
            )
            if self.task.task_id.startswith("clear_dining"):
                standoff -= 0.22
        lateral_offset = 0.0
        if self.task.task_id.startswith("store_kitchen"):
            if stage.kind == "nav_drawer":
                lateral_offset = 0.30
            elif stage.kind == "nav_target":
                lateral_offset = 0.20
        shoulder_id = mujoco.mj_name2id(
            self.backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_shoulder_pan_link",
        )
        shoulder_x, shoulder_y = (
            float(value) for value in self.backend.model.body_pos[shoulder_id, :2]
        )
        mount_x = shoulder_x * math.cos(yaw) - shoulder_y * math.sin(yaw)
        mount_y = shoulder_x * math.sin(yaw) + shoulder_y * math.cos(yaw)
        goal = (
            target[0] - standoff * math.cos(yaw) - mount_x + lateral_offset,
            target[1] - standoff * math.sin(yaw) - mount_y,
            yaw,
        )
        targets = [(*point, None) for point in waypoints]
        if (
            self.task.task_id.startswith("tidy_living")
            and stage.kind == "nav_target"
            and stage.object_id == "football"
            and targets
        ):
            base = self.backend.data.xpos[self.backend.bundle.ids.base_body]
            heading = math.atan2(targets[0][1] - base[1], targets[0][0] - base[0])
            targets.insert(0, (float(base[0]), float(base[1]), heading))
        if (
            self.task.task_id.startswith("store_kitchen")
            and stage.kind == "nav_drawer"
        ):
            targets.append((goal[0], 0.65, yaw))
        elif (
            self.task.task_id.startswith("store_kitchen")
            and stage.kind == "nav_target"
        ):
            targets.append((goal[0], 0.65, yaw))
        if (
            self.task.task_id.startswith("clear_dining")
            and stage.kind == "nav_target"
            and stage.object_id == "plate"
        ):
            base = self.backend.data.xpos[self.backend.bundle.ids.base_body]
            targets.insert(
                0,
                (
                    float(base[0]) - 0.30 * math.cos(yaw),
                    float(base[1]) - 0.30 * math.sin(yaw),
                    yaw,
                ),
            )
            targets.append(
                (
                    goal[0] - 0.35 * math.cos(yaw),
                    goal[1] - 0.35 * math.sin(yaw),
                    yaw,
                )
            )
        if (
            self.task.task_id.startswith("clear_dining")
            and stage.kind == "nav_object"
            and stage.object_id == "plate"
        ):
            targets.append(
                (
                    goal[0] - 0.30 * math.cos(yaw),
                    goal[1] - 0.30 * math.sin(yaw),
                    yaw,
                )
            )
        return targets + [goal]

    def _navigation_action(self, observation: ObservationFrame) -> ActionFrame:
        if not self.nav_targets:
            self._advance()
            return self._stop(observation)
        navigation_timeout = (
            1299
            if self.holding_object is not None
            or (
                self.task.task_id.startswith("store_kitchen")
                and self.stage.kind in {"nav_drawer", "nav_object", "nav_target"}
            )
            else (1199 if self.task.task_id.startswith("clear_dining") else 899)
        )
        if self.stage_step >= navigation_timeout:
            self.failed = True
            self._advance()
            return self._stop(observation)
        target_x, target_y, final_yaw = self.nav_targets[0]
        x, y, yaw = observation.base_pose
        distance = math.hypot(target_x - x, target_y - y)
        final_tolerance, yaw_tolerance = navigation_tolerances(
            self.task.task_id, self.stage.kind
        )
        if self.nav_aligning and distance > 1.5 * final_tolerance:
            self.nav_aligning = False
        if final_yaw is None and distance <= 0.13:
            self.nav_targets.pop(0)
            return self._stop(observation)
        if final_yaw is not None and (self.nav_aligning or distance <= final_tolerance):
            self.nav_aligning = True
            yaw_error = wrap_angle(final_yaw - yaw)
            if abs(yaw_error) <= yaw_tolerance:
                self.nav_targets.pop(0)
                self.nav_aligning = False
                if not self.nav_targets:
                    self._advance()
                return self._stop(observation)
            if abs(yaw_error) > yaw_tolerance:
                return self._action(
                    observation,
                    linear=0.0,
                    angular=float(np.clip(1.8 * yaw_error, -0.75, 0.75)),
                    gripper=self._gripper(),
                )
        heading = math.atan2(target_y - y, target_x - x) if distance > 0.05 else final_yaw or yaw
        heading_error = wrap_angle(heading - yaw)
        direction = 1.0
        if abs(heading_error) > math.pi / 2:
            direction = -1.0
            heading = wrap_angle(heading + math.pi)
            heading_error = wrap_angle(heading - yaw)
        yaw_error = wrap_angle(
            (
                final_yaw
                if distance < final_tolerance and final_yaw is not None
                else heading
            )
            - yaw
        )
        alignment = max(0.0, 1.0 - abs(heading_error) / 0.75)
        linear = direction * navigation_linear_speed(
            self.task.task_id, self.stage.kind, distance
        ) * alignment
        angular = float(np.clip(1.8 * yaw_error, -0.85, 0.85))
        if self.holding_object is not None:
            linear_limit, angular_limit = (
                (0.12, 0.15)
                if self.holding_object == "football"
                else (0.24, 0.40)
            )
            linear = float(np.clip(linear, -linear_limit, linear_limit))
            angular = float(np.clip(angular, -angular_limit, angular_limit))
        return self._action(observation, linear=linear, angular=angular, gripper=self._gripper())

    def _stow_phase_action(self, observation: ObservationFrame) -> ActionFrame:
        action = self._stow_action(observation)
        if action is not None:
            return action
        self._advance()
        return self._stop(observation)

    def _unstow_action(self, observation: ObservationFrame) -> ActionFrame:
        error = np.asarray(self._operation_ready()) - np.asarray(observation.joint_position)
        drawer_stage = self.stage.object_id is None and self.task.articulation is not None
        base_aligned = not drawer_stage or drawer_base_is_aligned(
            observation.base_pose, 1.30
        )
        if (
            float(np.max(np.abs(error))) <= 0.04
            and self._base_is_settled(observation)
            and base_aligned
        ):
            self._advance()
            return self._stop(observation)
        arm_command = (
            tuple(float(value) for value in np.clip(2.0 * error, -0.5, 0.5))
            if base_aligned
            else self._hold_arm_command(observation)
        )
        linear, angular = (
            drawer_base_commands(observation.base_pose, 1.30)
            if drawer_stage
            else (0.0, 0.0)
        )
        return self._action(
            observation,
            linear=linear,
            angular=angular,
            gripper=self._gripper(),
            arm_command=arm_command,
        )

    def _transport_action(self, observation: ObservationFrame) -> ActionFrame:
        error = np.asarray(ARM_READY_KITCHEN) - np.asarray(observation.joint_position)
        if float(np.max(np.abs(error))) <= 0.04 and self._base_is_settled(observation):
            self._advance()
            return self._stop(observation)
        if self.stage_step >= 179:
            self.failed = True
            self._advance()
            return self._stop(observation)
        arm_command = tuple(float(value) for value in np.clip(2.0 * error, -0.5, 0.5))
        return self._action(
            observation,
            linear=0.0,
            angular=0.0,
            gripper=self._gripper(),
            arm_command=arm_command,
        )

    def _operation_ready(self) -> tuple[float, ...]:
        if self.task.task_id.startswith("clear_dining"):
            return ARM_READY_TABLE
        if self.task.task_id.startswith("store_kitchen"):
            return ARM_READY_KITCHEN if self.stage.object_id is not None else ARM_READY_DRAWER
        return ARM_HOME

    def _base_is_settled(self, observation: ObservationFrame) -> bool:
        rotation = self.backend.data.xmat[self.backend.bundle.ids.base_body].reshape(3, 3)
        upright = float(rotation[2, 2]) >= 0.995
        return upright and max(abs(value) for value in observation.base_twist) <= 0.02

    def _stow_action(self, observation: ObservationFrame) -> ActionFrame | None:
        if self.holding_object is not None or self.drawer_holding:
            return None
        target = (
            ARM_READY_KITCHEN
            if self.task.task_id.startswith("store_kitchen")
            and self.stage.kind in {"nav_object", "stow_for_nav_object"}
            else ARM_STOW
        )
        error = np.asarray(target) - np.asarray(observation.joint_position)
        if float(np.max(np.abs(error))) <= 0.05:
            return None
        arm_command = tuple(float(value) for value in np.clip(2.0 * error, -1.0, 1.0))
        return self._action(
            observation,
            linear=0.0,
            angular=0.0,
            gripper=self._gripper(),
            arm_command=arm_command,
        )

    def _arm_action(self, observation: ObservationFrame) -> ActionFrame:
        if self.stage_target is None:
            raise RuntimeError("arm stage has no target")
        stage = self.stage
        if stage.kind.startswith("arm_drawer_") and stage.kind != "arm_drawer_retract":
            return self._drawer_joint_action(observation, drawer_posture(stage.kind))
        if (
            self.task.task_id.startswith("store_kitchen")
            and stage.kind == "arm_object_descend"
        ):
            self.stage_target = self._cartesian_target(stage)
        grip = (
            (0.0 if stage.object_id == "cleaner_pink" else 0.55)
            if self.task.task_id.startswith("store_kitchen")
            and stage.kind == "arm_object_descend"
            else self._gripper()
        )
        action = self.cartesian.action(
            observation,
            target_position=self.stage_target,
            gripper_target=grip,
            orientation_weight=(
                0.50
                if "drawer" in stage.kind
                else object_orientation_weight(
                    self.task.task_id, stage.kind, stage.object_id
                )
            ),
        )
        if stage.kind == "arm_object_clearance":
            position_error = float(
                np.linalg.norm(
                    np.asarray(self.stage_target)
                    - np.asarray(self.cartesian.site_position())
                )
            )
            if self.stage_step >= 25 and position_error < 0.05:
                self._advance()
                return self._stop(observation)
            if self.stage_step >= 179:
                self.failed = True
                self._advance()
                return self._stop(observation)
            return action
        if (
            stage.kind in {"arm_target_above", "arm_target_lower"}
            and self.stage_step >= 4
            and self._object_inside_target(stage.object_id)
        ):
            self._advance()
            return self._stop(observation)
        error = np.linalg.norm(np.asarray(self.stage_target) - np.asarray(self.cartesian.site_position()))
        relaxed_target = self.task.task_id.startswith("store_kitchen") and stage.kind.startswith(
            "arm_target"
        )
        relaxed_dining_target = self.task.task_id.startswith(
            "clear_dining"
        ) and stage.kind != "arm_object_descend"
        tolerance = 0.07 if relaxed_dining_target or relaxed_target else 0.035
        if (
            self.task.task_id.startswith("clear_dining")
            and stage.kind == "arm_object_descend"
            and stage.object_id == "plate"
        ):
            tolerance = 0.012
        if stage.kind == "arm_drawer_descend":
            tolerance = 0.012
        failure_error = (
            0.12 if relaxed_dining_target or relaxed_target else 0.08
        )
        if (self.stage_step >= 25 and error < tolerance) or self.stage_step >= 179:
            physically_validated_later = (
                stage.kind.startswith("arm_target")
                or stage.kind.startswith("arm_object")
                or stage.kind.startswith("arm_drawer")
            )
            if (
                self.stage_step >= 179
                and error >= failure_error
                and not physically_validated_later
            ):
                self.failed = True
            self._advance()
            return self._stop(observation)
        return action

    def _drawer_joint_action(
        self, observation: ObservationFrame, target: tuple[float, ...]
    ) -> ActionFrame:
        error = np.asarray(target) - np.asarray(observation.joint_position)
        if (
            float(np.max(np.abs(error))) <= 0.035
            and drawer_base_is_aligned(observation.base_pose, 1.30)
        ) or self.stage_step >= 399:
            self._advance()
            return self._stop(observation)
        arm_command = (
            tuple(float(value) for value in np.clip(2.0 * error, -0.5, 0.5))
            if drawer_base_is_aligned(observation.base_pose, 1.30)
            else self._hold_arm_command(observation)
        )
        linear, angular = drawer_base_commands(observation.base_pose, 1.30)
        return self._action(
            observation,
            linear=linear,
            angular=angular,
            gripper=self._gripper(),
            arm_command=arm_command,
        )

    def _object_inside_target(self, object_id: str | None) -> bool:
        if object_id is None:
            return False
        sample = self.backend._placement_sample(object_id)  # noqa: SLF001 - label-only expert
        return sample.target.contains(sample.position)

    def _holding_contact_timed_out(self) -> bool:
        if self.stage.kind == "release_object":
            self.holding_contact_loss_steps = 0
            return False
        if self.holding_object is None:
            self.holding_contact_loss_steps = 0
            return False
        monitor = self.backend._contact_monitors[self.holding_object]  # noqa: SLF001
        if monitor.sample(self.backend.data).bilateral:
            self.holding_contact_loss_steps = 0
            return False
        if self._held_object_tracks_gripper(self.holding_object):
            # A rigid object can alternate between finger contacts while it is
            # still physically trapped in the gripper.  Initial acquisition is
            # gated by bilateral contact; during transport, reject a real drop
            # by checking separation instead of requiring both pads at every
            # 20 Hz observation boundary.
            self.holding_contact_loss_steps = 0
            return False
        self.holding_contact_loss_steps += 1
        return self.holding_contact_loss_steps >= 20

    def _held_object_tracks_gripper(self, object_id: str) -> bool:
        position = np.asarray(self._object_position(object_id), dtype=np.float64)
        position[2] += self._object_spec(object_id).grasp_site_z_offset
        gripper = np.asarray(self.cartesian.site_position(), dtype=np.float64)
        return float(np.linalg.norm(position - gripper)) <= 0.18

    def _pull_drawer_action(self, observation: ObservationFrame) -> ActionFrame:
        joint_id = self.backend.household_ids.articulation_joint
        if joint_id is None:
            raise RuntimeError("drawer stage has no joint")
        position = float(self.backend.data.qpos[self.backend.model.jnt_qposadr[joint_id]])
        if position >= 0.38:
            self._advance()
            return self._stop(observation, gripper=1.0)
        if self.stage_step >= 549:
            self.failed = True
            self._advance()
            return self._stop(observation, gripper=1.0)
        progress = min(0.42, 0.0008 * self.stage_step)
        error = np.asarray(ARM_DRAWER_GRASP) - np.asarray(
            observation.joint_position
        )
        arm_command = tuple(
            float(value) for value in np.clip(2.0 * error, -0.5, 0.5)
        )
        linear, angular = drawer_base_commands(
            observation.base_pose, 1.30 - progress
        )
        return self._action(
            observation,
            linear=linear,
            angular=angular,
            gripper=1.0,
            arm_command=arm_command,
        )

    def _back_away_drawer_action(self, observation: ObservationFrame) -> ActionFrame:
        if observation.base_pose[1] <= 0.40:
            self._advance()
            return self._stop(observation, gripper=0.0)
        if self.stage_step >= 159:
            self.failed = True
            self._advance()
            return self._stop(observation, gripper=0.0)
        return self._action(
            observation,
            linear=-0.12,
            angular=0.0,
            gripper=0.0,
            arm_command=self._hold_arm_command(observation),
        )

    def _cartesian_target(self, stage: ExpertStage) -> tuple[float, float, float]:
        if "drawer" in stage.kind:
            handle = self._drawer_handle_position()
            if stage.kind.endswith("above"):
                return (handle[0], handle[1], handle[2] + 0.23)
            if stage.kind.endswith("descend"):
                return self._drawer_grasp_target()
            site = self.cartesian.site_position()
            return (site[0], site[1] - 0.05, site[2] + 0.24)
        spec = self._object_spec(stage.object_id)
        if "object" in stage.kind:
            if stage.kind == "arm_object_clearance":
                site = self.cartesian.site_position()
                return (site[0], site[1], site[2] + 0.25)
            position = self._object_position(stage.object_id)
            grasp = (position[0], position[1], position[2] + spec.grasp_site_z_offset)
            if self.task.task_id.startswith("tidy_living"):
                yaw = object_approach_yaw(self.task.task_id, stage.object_id)
                compensation = top_down_site_compensation(yaw)
                grasp = tuple(
                    grasp[index] + compensation[index] for index in range(3)
                )
            if stage.kind.endswith("above"):
                clearance = (
                    0.20
                    if self.task.task_id.startswith("store_kitchen")
                    else (0.36 if self.task.task_id.startswith("tidy_living") else 0.24)
                )
                return (grasp[0], grasp[1], grasp[2] + clearance)
            if stage.kind.endswith("descend"):
                return grasp
            site = self.cartesian.site_position()
            if self.task.task_id.startswith("store_kitchen"):
                return (site[0], site[1], site[2] + 0.06)
            lift = 0.20 if self.task.task_id.startswith("clear_dining") else 0.35
            return (site[0] - 0.05, site[1], site[2] + lift)
        target = self._target_position(stage.object_id)
        if self.task.task_id.startswith(("clear_dining", "tidy_living")):
            placement = self.backend._placement_sample(stage.object_id).position  # noqa: SLF001
            site = self.cartesian.site_position()
            grasp_offset = tuple(site[index] - placement[index] for index in range(3))
            lateral_bias = 0.05 if self.task.task_id.startswith("clear_dining") else 0.0
            lower = (
                target[0] + grasp_offset[0],
                target[1] + grasp_offset[1] + lateral_bias,
                target[2] + grasp_offset[2],
            )
        else:
            lower = (target[0], target[1], target[2] + spec.grasp_site_z_offset)
        if stage.kind.endswith("raise"):
            site = self.cartesian.site_position()
            clearance = 0.35 if self.task.task_id.startswith("store_kitchen") else 0.15
            return (site[0], site[1], lower[2] + clearance)
        if stage.kind.endswith("above"):
            clearance = 0.35 if self.task.task_id.startswith("store_kitchen") else 0.25
            return (lower[0], lower[1], lower[2] + clearance)
        if stage.kind.endswith("lower"):
            if self.task.task_id.startswith("tidy_living"):
                return (lower[0], lower[1], lower[2] + 0.22)
            if self.task.task_id.startswith("store_kitchen"):
                # Open above the drawer front and let gravity complete the placement.
                # Driving the fingers down to the target centre would collide with
                # a realistic-height drawer front while the bottle is still held.
                return (lower[0], lower[1], lower[2] + 0.18)
            return lower
        site = self.cartesian.site_position()
        return (site[0], site[1], site[2] + 0.24)

    def _action(
        self,
        observation: ObservationFrame,
        *,
        linear: float,
        angular: float,
        gripper: float,
        arm_command: tuple[float, ...] = (0.0,) * 6,
    ) -> ActionFrame:
        period = round(1_000_000_000 / self.task.control_hz)
        return ActionFrame(
            observation.timestamp_ns,
            observation.timestamp_ns,
            observation.timestamp_ns + period,
            "privileged_3d_expert",
            base_linear=linear,
            base_angular=angular,
            arm_command=arm_command,
            gripper_target=gripper,
        )
