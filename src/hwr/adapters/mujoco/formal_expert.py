"""Privileged finite-state expert used only to generate formal 3D demonstrations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from hwr.adapters.mujoco.expert import PrivilegedCartesianExpert
from hwr.adapters.mujoco.household_backend import MujocoHouseholdBackend
from hwr.adapters.mujoco.names import ARM_HOME
from hwr.core.types import ActionFrame, ObservationFrame


ARM_STOW = (0.0, -0.90, -1.00, 0.0, 1.70, 0.0)
ARM_READY_TABLE = (0.0, -1.30, 0.70, 0.0, 0.60, 0.0)
ARM_READY_DRAWER = (0.0, -1.50, 0.50, 0.0, 1.00, 0.0)
ARM_READY_KITCHEN = (0.0, -1.60, 0.0, 0.0, 1.60, 0.0)


@dataclass(frozen=True)
class ExpertStage:
    name: str
    kind: str
    object_id: str | None = None


@dataclass(frozen=True)
class FormalExpertOutput:
    action: ActionFrame
    stage: str
    stage_step: int
    privileged_label: bool = True


class PrivilegedHouseholdExpert:
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
        if stage.kind.startswith("nav_"):
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
            action = self.cartesian.action(
                observation,
                target_position=self._drawer_grasp_target(),
                gripper_target=1.0,
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
                    ("navigate_to_drawer", "nav_drawer"),
                    ("unfold_arm_for_drawer", "unstow_arm"),
                    ("approach_drawer_handle", "arm_drawer_above"),
                    ("descend_to_drawer_handle", "arm_drawer_descend"),
                    ("close_on_drawer_handle", "grip_drawer"),
                    ("contact_pull_drawer", "pull_drawer"),
                    ("release_drawer_handle", "release_drawer"),
                    ("retract_from_drawer", "arm_drawer_retract"),
                    ("back_away_from_open_drawer", "back_away_drawer"),
                )
            )
        object_kinds = [
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
        stage = self.stage
        if stage.kind.startswith("nav_"):
            self.nav_targets = self._navigation_targets(stage)
        elif stage.kind.startswith("arm_"):
            self.cartesian.reset_orientation_target()
            self.stage_target = self._cartesian_target(stage)

    def _advance(self) -> None:
        self.stage_index += 1
        self._enter_stage()

    def _navigation_targets(self, stage: ExpertStage) -> list[tuple[float, float, float | None]]:
        waypoints = self._waypoints(stage)
        if stage.kind == "nav_drawer":
            target = self._drawer_handle_position()
            yaw, standoff = math.pi / 2, 0.55
        elif stage.kind == "nav_object":
            target = self._object_position(stage.object_id)
            yaw = self._object_approach_yaw(stage.object_id)
            standoff = self._object_spec(stage.object_id).standoff_m
            if self.task.task_id.startswith("clear_dining"):
                standoff -= 0.12
            elif (
                self.task.task_id.startswith("store_kitchen")
                and stage.object_id == "cleaner_yellow"
            ):
                standoff += 0.15
        else:
            target = self._target_position(stage.object_id)
            yaw = self._target_approach_yaw()
            standoff = (
                0.65
                if self.task.task_id.startswith("store_kitchen")
                else self._object_spec(stage.object_id).standoff_m
            )
        lateral_offset = (
            0.20
            if self.task.task_id.startswith("store_kitchen")
            and stage.kind == "nav_target"
            else 0.0
        )
        goal = (
            target[0] - standoff * math.cos(yaw) + lateral_offset,
            target[1] - standoff * math.sin(yaw),
            yaw,
        )
        return [(*point, None) for point in waypoints] + [goal]

    def _navigation_action(self, observation: ObservationFrame) -> ActionFrame:
        if not self.nav_targets:
            self._advance()
            return self._stop(observation)
        stow_action = self._stow_action(observation)
        if stow_action is not None:
            return stow_action
        navigation_timeout = (
            1299
            if self.task.task_id.startswith("store_kitchen")
            and self.stage.kind in {"nav_object", "nav_target"}
            else (1199 if self.task.task_id.startswith("clear_dining") else 899)
        )
        if self.stage_step >= navigation_timeout:
            self.failed = True
            self._advance()
            return self._stop(observation)
        target_x, target_y, final_yaw = self.nav_targets[0]
        x, y, yaw = observation.base_pose
        distance = math.hypot(target_x - x, target_y - y)
        final_tolerance = (
            (0.05 if self.stage.kind == "nav_object" else 0.10)
            if self.task.task_id.startswith("store_kitchen")
            and self.stage.kind in {"nav_object", "nav_target"}
            else 0.12
        )
        if self.nav_aligning and distance > 1.5 * final_tolerance:
            self.nav_aligning = False
        if final_yaw is None and distance <= 0.13:
            self.nav_targets.pop(0)
            return self._stop(observation)
        if final_yaw is not None and (self.nav_aligning or distance <= final_tolerance):
            self.nav_aligning = True
            yaw_error = _wrap(final_yaw - yaw)
            if abs(yaw_error) <= 0.10:
                self.nav_targets.pop(0)
                self.nav_aligning = False
                if not self.nav_targets:
                    self._advance()
                return self._stop(observation)
            if abs(yaw_error) > 0.10:
                return self._action(
                    observation,
                    linear=0.0,
                    angular=float(np.clip(1.8 * yaw_error, -0.75, 0.75)),
                    gripper=self._gripper(),
                )
        heading = math.atan2(target_y - y, target_x - x) if distance > 0.05 else final_yaw or yaw
        heading_error = _wrap(heading - yaw)
        direction = 1.0
        if abs(heading_error) > math.pi / 2:
            direction = -1.0
            heading = _wrap(heading + math.pi)
            heading_error = _wrap(heading - yaw)
        yaw_error = _wrap(
            (
                final_yaw
                if distance < final_tolerance and final_yaw is not None
                else heading
            )
            - yaw
        )
        alignment = max(0.0, 1.0 - abs(heading_error) / 0.75)
        linear = direction * min(0.42, 1.1 * distance) * alignment
        angular = float(np.clip(1.8 * yaw_error, -0.85, 0.85))
        return self._action(observation, linear=linear, angular=angular, gripper=self._gripper())

    def _unstow_action(self, observation: ObservationFrame) -> ActionFrame:
        error = np.asarray(self._operation_ready()) - np.asarray(observation.joint_position)
        if float(np.max(np.abs(error))) <= 0.04 and self._base_is_settled(observation):
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
            and self.stage.kind == "nav_object"
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
        )
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
        if self.stage_step >= 499:
            self.failed = True
            self._advance()
            return self._stop(observation, gripper=1.0)
        return self._action(observation, linear=-0.02, angular=0.0, gripper=1.0)

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
            position = self._object_position(stage.object_id)
            grasp = (position[0], position[1], position[2] + spec.grasp_site_z_offset)
            if stage.kind.endswith("above"):
                clearance = 0.20 if self.task.task_id.startswith("store_kitchen") else 0.24
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
            lateral_bias = -0.07 if self.task.task_id.startswith("clear_dining") else 0.0
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

    def _waypoints(self, stage: ExpertStage) -> list[tuple[float, float]]:
        task = self.task.task_id
        object_id = stage.object_id
        if task.startswith("tidy_living"):
            if stage.kind == "nav_object" and object_id == "duck":
                return [(0.70, -1.35)]
            if stage.kind == "nav_object":
                return [(1.60, 0.20), (0.65, -0.90)]
            return []
        if task.startswith("clear_dining"):
            if stage.kind == "nav_object":
                return (
                    [(2.85, 1.10), (2.85, -0.35), (1.75, -0.35)]
                    if object_id == "plate"
                    else [(0.30, -0.95)]
                )
            return [(1.80, -0.35), (2.75, -0.35), (2.85, 0.30)]
        if stage.kind == "nav_drawer":
            return [(1.55, -0.70), (1.55, 1.10), (1.25, 1.10)]
        if stage.kind == "nav_object":
            if object_id == "cleaner_pink":
                return [(1.35, -0.20), (1.35, 0.45)]
            # Clear the island's south-east corner before turning west.  A
            # closer waypoint lets the differential drive cut the corner and
            # puts its front-right wheel into the island under some seeds.
            return [(1.55, -0.45), (-0.42, -0.45)]
        start_x = 1.35 if object_id == "cleaner_pink" else -0.42
        return [(start_x, -0.45), (1.90, -0.45), (1.90, 1.20)]

    def _object_approach_yaw(self, object_id: str | None) -> float:
        if self.task.task_id.startswith("tidy_living"):
            return math.pi if object_id == "football" else 0.0
        if self.task.task_id.startswith("store_kitchen") and object_id == "cleaner_pink":
            return math.pi
        return math.pi / 2

    def _target_approach_yaw(self) -> float:
        if self.task.task_id.startswith("store_kitchen"):
            return math.pi / 2
        if self.task.task_id.startswith("tidy_living"):
            return 0.80
        return math.pi / 2

    def _object_position(self, object_id: str | None) -> tuple[float, float, float]:
        if object_id is None:
            raise ValueError("object ID is required")
        body_id = self.backend.household_ids.object_bodies[object_id]
        return tuple(float(value) for value in self.backend.data.xpos[body_id])

    def _target_position(self, object_id: str | None) -> tuple[float, float, float]:
        if object_id is None:
            raise ValueError("object ID is required")
        site_id = self.backend.household_ids.target_sites[object_id]
        return tuple(float(value) for value in self.backend.data.site_xpos[site_id])

    def _drawer_handle_position(self) -> tuple[float, float, float]:
        binding = self.backend.binding.articulation
        if binding is None:
            raise RuntimeError("task has no drawer binding")
        geom_id = mujoco.mj_name2id(
            self.backend.model, mujoco.mjtObj.mjOBJ_GEOM, binding.handle_geom
        )
        return tuple(float(value) for value in self.backend.data.geom_xpos[geom_id])

    def _drawer_grasp_target(self) -> tuple[float, float, float]:
        handle = self._drawer_handle_position()
        return (handle[0] - 0.020, handle[1] - 0.040, handle[2] + 0.03)

    def _object_spec(self, object_id: str | None):
        return next(obj for obj in self.task.objects if obj.object_id == object_id)

    def _grip_fraction(self, object_id: str | None) -> float:
        return self._object_spec(object_id).grip_fraction

    def _gripper(self) -> float:
        if self.drawer_holding:
            return 1.0
        return self._grip_fraction(self.holding_object) if self.holding_object else 0.0

    def _hold_action(self, observation: ObservationFrame, gripper: float) -> ActionFrame:
        return self._action(
            observation,
            linear=0.0,
            angular=0.0,
            gripper=gripper,
            arm_command=self._hold_arm_command(observation),
        )

    def _hold_arm_command(self, observation: ObservationFrame) -> tuple[float, ...]:
        target_error = (
            np.asarray(observation.joint_position)
            - self.backend._arm_targets  # noqa: SLF001
        )
        return tuple(
            float(value) for value in np.clip(20.0 * target_error, -1.0, 1.0)
        )

    def _stop(self, observation: ObservationFrame, gripper: float | None = None) -> ActionFrame:
        return self._hold_action(observation, self._gripper() if gripper is None else gripper)

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


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
