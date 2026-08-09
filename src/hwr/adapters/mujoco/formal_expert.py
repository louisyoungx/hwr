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
        self.failed = False
        self._enter_stage()

    @property
    def stage(self) -> ExpertStage:
        return self.stages[min(self.stage_index, len(self.stages) - 1)]

    @property
    def done(self) -> bool:
        return self.stage_index >= len(self.stages)

    def action(self, observation: ObservationFrame) -> FormalExpertOutput:
        if self.done:
            return FormalExpertOutput(self._stop(observation), "done", self.stage_step)
        stage = self.stage
        current_index = self.stage_index
        current_step = self.stage_step
        if stage.kind.startswith("nav_"):
            action = self._navigation_action(observation)
        elif stage.kind.startswith("arm_"):
            action = self._arm_action(observation)
        elif stage.kind == "unstow_arm":
            action = self._unstow_action(observation)
        elif stage.kind == "grip_object":
            action = self._hold_action(observation, self._grip_fraction(stage.object_id))
            if self.stage_step >= 69:
                self.holding_object = stage.object_id
                self._advance()
        elif stage.kind == "release_object":
            action = self._hold_action(observation, 0.0)
            if self.stage_step >= 49:
                self.holding_object = None
                self._advance()
        elif stage.kind == "grip_drawer":
            action = self._hold_action(observation, 0.95)
            if self.stage_step >= 69:
                self.drawer_holding = True
                self._advance()
        elif stage.kind == "pull_drawer":
            action = self._pull_drawer_action(observation)
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
                )
            )
        for obj in self.task.objects:
            stages.extend(
                ExpertStage(f"{kind}_{obj.object_id}", kind, obj.object_id)
                for kind in (
                    "nav_object",
                    "unstow_arm",
                    "arm_object_above",
                    "arm_object_descend",
                    "grip_object",
                    "arm_object_lift",
                    "nav_target",
                    "arm_target_above",
                    "arm_target_lower",
                    "release_object",
                    "arm_target_retract",
                )
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
        else:
            target = self._target_position(stage.object_id)
            yaw = self._target_approach_yaw()
            standoff = self._object_spec(stage.object_id).standoff_m
        goal = (
            target[0] - standoff * math.cos(yaw),
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
        if self.stage_step >= 899:
            self.failed = True
            self._advance()
            return self._stop(observation)
        target_x, target_y, final_yaw = self.nav_targets[0]
        x, y, yaw = observation.base_pose
        distance = math.hypot(target_x - x, target_y - y)
        if final_yaw is None and distance <= 0.13:
            self.nav_targets.pop(0)
            return self._stop(observation)
        if final_yaw is not None and (self.nav_aligning or distance <= 0.12):
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
        yaw_error = _wrap((final_yaw if distance < 0.10 and final_yaw is not None else heading) - yaw)
        alignment = max(0.0, 1.0 - abs(heading_error) / 0.75)
        linear = direction * min(0.42, 1.1 * distance) * alignment
        angular = float(np.clip(1.8 * yaw_error, -0.85, 0.85))
        return self._action(observation, linear=linear, angular=angular, gripper=self._gripper())

    def _unstow_action(self, observation: ObservationFrame) -> ActionFrame:
        error = np.asarray(ARM_HOME) - np.asarray(observation.joint_position)
        if float(np.max(np.abs(error))) <= 0.05:
            self._advance()
            return self._stop(observation)
        arm_command = tuple(float(value) for value in np.clip(2.0 * error, -1.0, 1.0))
        return self._action(
            observation,
            linear=0.0,
            angular=0.0,
            gripper=self._gripper(),
            arm_command=arm_command,
        )

    def _stow_action(self, observation: ObservationFrame) -> ActionFrame | None:
        if self.holding_object is not None or self.drawer_holding:
            return None
        error = np.asarray(ARM_STOW) - np.asarray(observation.joint_position)
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
        grip = self._gripper()
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
            return action
        error = np.linalg.norm(np.asarray(self.stage_target) - np.asarray(self.cartesian.site_position()))
        if (self.stage_step >= 25 and error < 0.035) or self.stage_step >= 179:
            if self.stage_step >= 179 and error >= 0.08:
                self.failed = True
            self._advance()
        return action

    def _object_inside_target(self, object_id: str | None) -> bool:
        if object_id is None:
            return False
        sample = self.backend._placement_sample(object_id)  # noqa: SLF001 - label-only expert
        return sample.target.contains(sample.position)

    def _pull_drawer_action(self, observation: ObservationFrame) -> ActionFrame:
        joint_id = self.backend.household_ids.articulation_joint
        if joint_id is None:
            raise RuntimeError("drawer stage has no joint")
        position = float(self.backend.data.qpos[self.backend.model.jnt_qposadr[joint_id]])
        if position >= 0.32:
            self._advance()
            return self._stop(observation, gripper=0.95)
        if self.stage_step >= 139:
            self.failed = True
            self._advance()
        return self._action(observation, linear=-0.10, angular=0.0, gripper=0.95)

    def _cartesian_target(self, stage: ExpertStage) -> tuple[float, float, float]:
        if "drawer" in stage.kind:
            handle = self._drawer_handle_position()
            if stage.kind.endswith("above"):
                return (handle[0], handle[1], handle[2] + 0.23)
            if stage.kind.endswith("descend"):
                return (handle[0], handle[1], handle[2] + 0.03)
            site = self.cartesian.site_position()
            return (site[0], site[1] - 0.05, site[2] + 0.24)
        spec = self._object_spec(stage.object_id)
        if "object" in stage.kind:
            position = self._object_position(stage.object_id)
            grasp = (position[0], position[1], position[2] + spec.grasp_site_z_offset)
            if stage.kind.endswith("above"):
                return (grasp[0], grasp[1], grasp[2] + 0.24)
            if stage.kind.endswith("descend"):
                return grasp
            site = self.cartesian.site_position()
            return (site[0] - 0.05, site[1], site[2] + 0.35)
        target = self._target_position(stage.object_id)
        lower = (target[0], target[1], target[2] + spec.grasp_site_z_offset)
        if stage.kind.endswith("above"):
            return (lower[0], lower[1], lower[2] + 0.25)
        if stage.kind.endswith("lower"):
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
                return [(2.75, 0.35), (1.75, -0.35)] if object_id == "plate" else [(0.30, -0.95)]
            return [(1.80, -0.35), (2.80, 0.45)]
        if stage.kind == "nav_drawer":
            return [(1.70, -0.70), (1.70, 1.35)]
        if stage.kind == "nav_object":
            return [(1.70, 1.35), (1.65, -0.10), (object_id == "cleaner_pink" and 0.18 or -0.10, -0.05)]
        return [(-0.45, 1.45), (0.25, 1.90)]

    def _object_approach_yaw(self, object_id: str | None) -> float:
        if self.task.task_id.startswith("tidy_living"):
            return math.pi if object_id == "football" else 0.0
        return math.pi / 2

    def _target_approach_yaw(self) -> float:
        if self.task.task_id.startswith("store_kitchen"):
            return 0.0
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

    def _object_spec(self, object_id: str | None):
        return next(obj for obj in self.task.objects if obj.object_id == object_id)

    def _grip_fraction(self, object_id: str | None) -> float:
        return self._object_spec(object_id).grip_fraction

    def _gripper(self) -> float:
        if self.drawer_holding:
            return 0.95
        return self._grip_fraction(self.holding_object) if self.holding_object else 0.0

    def _hold_action(self, observation: ObservationFrame, gripper: float) -> ActionFrame:
        return self._action(observation, linear=0.0, angular=0.0, gripper=gripper)

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
