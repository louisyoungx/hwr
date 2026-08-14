"""Canonical 16-D dual-arm MuJoCo runtime with deployable observations only."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from hwr.adapters.mujoco.model import MujocoModelBundle
from hwr.adapters.mujoco.names import (
    ARM_HOME,
    FINGER_TRAVEL,
    SECONDARY_ARM_HOME,
    TRACK_WIDTH,
    WHEEL_RADIUS,
)
from hwr.adapters.mujoco.rendering import MujocoCameraRenderer
from hwr.core.embodied import (
    DualArmAction,
    DualArmActionFrame,
    DualArmObservation,
    DualArmProprioception,
    FrameCameraCalibration,
    NaturalLanguageInstruction,
)
from hwr.core.runtime import RuntimeStepOutcome
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.core.types import EpisodeEvent, EpisodeResult, SafetyState
from hwr.safety import DualArmSafetySupervisor, SafetyLimits


@dataclass(frozen=True)
class MujocoDualArmConfig:
    model_path: Path
    task_id: str = "mujoco_dual_arm_smoke/v1"
    instruction_text: str = "同时控制左右机械臂完成任务"
    instruction_locale: str = "zh-CN"
    control_hz: float = 20.0
    max_steps: int = 200
    camera_width: int = 160
    camera_height: int = 120
    max_base_linear: float = 0.45
    max_base_angular: float = 1.0
    max_arm_velocity: float = 1.2
    max_tool_linear_velocity: float = 0.30
    max_tool_angular_velocity: float = 1.0
    ik_damping: float = 0.08
    max_arm_servo_error: float = 0.30
    left_arm_home: tuple[float, ...] = SECONDARY_ARM_HOME
    right_arm_home: tuple[float, ...] = ARM_HOME
    primary_object_joint_name: str | None = "smoke_object_free"
    primary_object_reset_z: float = 0.805

    def __post_init__(self) -> None:
        numeric = (
            self.control_hz,
            self.max_steps,
            self.camera_width,
            self.camera_height,
            self.max_base_linear,
            self.max_base_angular,
            self.max_arm_velocity,
            self.max_tool_linear_velocity,
            self.max_tool_angular_velocity,
            self.ik_damping,
            self.max_arm_servo_error,
            self.primary_object_reset_z,
        )
        if not self.task_id or not self.instruction_text or min(numeric) <= 0:
            raise ValueError("dual-arm MuJoCo configuration values must be positive")
        if len(self.left_arm_home) != 6 or len(self.right_arm_home) != 6:
            raise ValueError("dual-arm home posture requires six values per side")
        if not all(
            math.isfinite(value)
            for value in (*self.left_arm_home, *self.right_arm_home)
        ):
            raise ValueError("dual-arm home posture must be finite")


class MujocoDualArmBackend:
    """Owns physics while exposing only the project dual-arm runtime contract."""

    def __init__(self, config: MujocoDualArmConfig) -> None:
        self.config = config
        self.bundle = MujocoModelBundle.load(
            config.model_path,
            object_joint_name=config.primary_object_joint_name,
        )
        self.model = self.bundle.model
        self.data = mujoco.MjData(self.model)
        self.renderer = MujocoCameraRenderer(
            self.model,
            width=config.camera_width,
            height=config.camera_height,
        )
        self.safety = DualArmSafetySupervisor(
            SafetyLimits(
                max_base_linear=config.max_base_linear,
                max_base_angular=config.max_base_angular,
                max_arm_command=1.0,
            )
        )
        self._substeps = _control_substeps(config.control_hz, self.model.opt.timestep)
        self._instruction = NaturalLanguageInstruction(
            config.instruction_text,
            config.instruction_locale,
        )
        self._rng = random.Random()
        self._task_id: str | None = None
        self._sequence = 0
        self._steps = 0
        self._result: EpisodeResult | None = None
        self._left_targets = np.asarray(config.left_arm_home, dtype=np.float64)
        self._right_targets = np.asarray(config.right_arm_home, dtype=np.float64)
        self._left_tool_site = self._site_id("left_grasp_center")
        self._right_tool_site = self._site_id("right_grasp_center")

    def reset(
        self,
        *,
        seed: int,
        task_id: str,
        initial_state: PhysicalStateSnapshot | None = None,
    ) -> DualArmObservation:
        if task_id != self.config.task_id:
            raise ValueError(f"backend provides {self.config.task_id}, not {task_id}")
        self._rng.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._task_id = task_id
        self._sequence = 0
        self._steps = 0
        self._result = None
        self._reset_base()
        self._reset_arms()
        self._reset_object()
        self._write_controls(_zero_action())
        mujoco.mj_forward(self.model, self.data)
        if initial_state is not None:
            self._reset_state_snapshot(initial_state)
        return self._observation()

    def observe(self) -> DualArmObservation:
        self._require_active()
        return self._observation()

    def capture_state_snapshot(self) -> PhysicalStateSnapshot:
        """Capture physical continuation state without reward or task stage."""
        self._require_active()
        specification = _snapshot_state_specification()
        runtime_state = np.empty(
            mujoco.mj_stateSize(self.model, specification), dtype=np.float64
        )
        mujoco.mj_getState(self.model, self.data, runtime_state, specification)
        return PhysicalStateSnapshot(
            task_id=self._task_id or self.config.task_id,
            backend_fingerprint=self._state_snapshot_fingerprint(),
            generalized_positions=tuple(float(value) for value in self.data.qpos),
            generalized_velocities=tuple(float(value) for value in self.data.qvel),
            generalized_accelerations=tuple(float(value) for value in self.data.qacc),
            actuator_controls=tuple(float(value) for value in self.data.ctrl),
            solver_state=tuple(float(value) for value in self.data.qacc_warmstart),
            runtime_state=tuple(float(value) for value in runtime_state),
        )

    def _reset_state_snapshot(self, snapshot: PhysicalStateSnapshot) -> None:
        """Apply a visited configuration only inside the Episode reset boundary."""
        if snapshot.task_id != self._task_id:
            raise ValueError("snapshot task identity differs from active backend")
        if snapshot.backend_fingerprint != self._state_snapshot_fingerprint():
            raise ValueError("snapshot backend fingerprint differs")
        if len(snapshot.generalized_positions) != self.model.nq:
            raise ValueError("snapshot generalized-position dimension differs")
        specification = _snapshot_state_specification()
        if snapshot.runtime_state:
            expected = mujoco.mj_stateSize(self.model, specification)
            if len(snapshot.runtime_state) != expected:
                raise ValueError("snapshot runtime-state dimension differs")
            mujoco.mj_setState(
                self.model,
                self.data,
                np.asarray(snapshot.runtime_state, dtype=np.float64),
                specification,
            )
        else:
            self.data.qpos[:] = snapshot.generalized_positions
            if snapshot.generalized_velocities:
                if len(snapshot.generalized_velocities) != self.model.nv:
                    raise ValueError("snapshot generalized-velocity dimension differs")
                self.data.qvel[:] = snapshot.generalized_velocities
            else:
                self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            if snapshot.solver_state:
                if len(snapshot.solver_state) != self.model.nv:
                    raise ValueError("snapshot solver-state dimension differs")
                self.data.qacc_warmstart[:] = snapshot.solver_state
            else:
                self.data.qacc_warmstart[:] = 0.0
            if self.model.na:
                self.data.act[:] = 0.0
            if snapshot.actuator_controls:
                if len(snapshot.actuator_controls) != self.model.nu:
                    raise ValueError("snapshot actuator-control dimension differs")
                self.data.ctrl[:] = snapshot.actuator_controls
            else:
                self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        if snapshot.actuator_controls:
            self._synchronize_arm_targets_to_controls()
        else:
            self._synchronize_arm_targets_to_measured()
            self._write_controls(_hold_action(self._gripper_positions()))
        mujoco.mj_forward(self.model, self.data)
        if snapshot.generalized_accelerations:
            if len(snapshot.generalized_accelerations) != self.model.nv:
                raise ValueError("snapshot generalized-acceleration dimension differs")
            self.data.qacc[:] = snapshot.generalized_accelerations

    def apply(self, frame: DualArmActionFrame) -> RuntimeStepOutcome:
        self._require_active()
        if self._result is not None:
            raise RuntimeError("episode is complete; call reset")
        if not isinstance(frame, DualArmActionFrame):
            raise TypeError("canonical runtime requires DualArmActionFrame")
        timestamp_ns = self._timestamp_ns()
        hold_grippers = self._gripper_positions()
        applied, events = self.safety.filter(
            frame,
            now_ns=timestamp_ns,
            hold_grippers=hold_grippers,
        )
        applied, predictive_events = self._predictive_filter(
            applied, hold_grippers
        )
        events = (*events, *predictive_events)
        self._write_controls(applied.action)
        if not predictive_events:
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, self.data)
                self._after_physics_substep()
            mujoco.mj_forward(self.model, self.data)
        self._steps += 1
        self._sequence += 1
        self._after_control_step(applied)
        task_result = self._task_result_after_step()
        if task_result is not None:
            self._result = task_result
        terminated = task_result is not None
        truncated = not terminated and self._steps >= self.config.max_steps
        if truncated:
            self._result = self._timeout_result()
        safety_intervened = applied.action != frame.action or bool(events)
        return RuntimeStepOutcome(
            observation=self._observation(),
            reward=self._step_reward(task_result),
            terminated=terminated,
            truncated=truncated,
            events=events,
            info={
                "applied_action": applied,
                "physics_contacts": int(self.data.ncon),
                "physics_advanced": not predictive_events,
                "safety_intervened": safety_intervened,
            },
        )

    def _after_physics_substep(self) -> None:
        """Adapter hook for physical evidence that may last one substep."""

    def _after_control_step(self, action: DualArmActionFrame) -> None:
        """Adapter hook for task-independent state accounting."""

    def _task_result_after_step(self) -> EpisodeResult | None:
        return None

    def _step_reward(self, result: EpisodeResult | None) -> float:
        return float(result is not None and result.success)

    def _timeout_result(self) -> EpisodeResult:
        return EpisodeResult(
            success=False,
            reason="episode_timeout",
            ended_at_ns=self._timestamp_ns(),
            metrics={"steps": self._steps, "contacts": self.data.ncon},
        )

    def _predictive_filter(
        self,
        frame: DualArmActionFrame,
        hold_grippers: tuple[float, float],
    ) -> tuple[DualArmActionFrame, tuple[EpisodeEvent, ...]]:
        if not self._predictive_safety_enabled():
            return frame, ()
        actual_data = self.data
        trial_data = mujoco.MjData(self.model)
        mujoco.mj_copyData(trial_data, self.model, actual_data)
        left_targets = self._left_targets.copy()
        right_targets = self._right_targets.copy()
        unsafe = False
        try:
            self.data = trial_data
            self._write_controls(frame.action)
            horizon = self._predictive_horizon_control_steps() * self._substeps
            for substep in range(horizon):
                mujoco.mj_step(self.model, self.data)
                control_boundary = (substep + 1) % self._substeps == 0
                if control_boundary and self._predictive_safety_violation():
                    unsafe = True
                    break
        finally:
            self.data = actual_data
            self._left_targets = left_targets
            self._right_targets = right_targets
        if not unsafe:
            return frame, ()
        self._synchronize_arm_targets_to_measured()
        stopped = replace(
            frame,
            action=_hold_action(hold_grippers),
            source="safety",
            confidence=1.0,
        )
        return stopped, (
            EpisodeEvent(
                timestamp_ns=self._timestamp_ns(),
                event_type="action_rejected",
                source="safety",
                details={"reason": "predicted_severe_collision"},
            ),
        )

    def _predictive_safety_enabled(self) -> bool:
        return False

    def _predictive_safety_violation(self) -> bool:
        return False

    def _predictive_horizon_control_steps(self) -> int:
        return 1

    def _synchronize_arm_targets_to_measured(self) -> None:
        for targets, joint_ids in (
            (self._left_targets, self.bundle.ids.secondary_arm_joints),
            (self._right_targets, self.bundle.ids.arm_joints),
        ):
            for index, joint_id in enumerate(joint_ids):
                targets[index] = self.data.qpos[self.model.jnt_qposadr[joint_id]]

    def _synchronize_arm_targets_to_controls(self) -> None:
        for targets, actuator_ids in (
            (self._left_targets, self.bundle.ids.secondary_arm_actuators),
            (self._right_targets, self.bundle.ids.arm_actuators),
        ):
            targets[:] = self.data.ctrl[list(actuator_ids)]

    def _state_snapshot_fingerprint(self) -> str:
        return (
            f"mujoco-qpos/v1:nq={self.model.nq}:nv={self.model.nv}:"
            f"nu={self.model.nu}"
        )

    def result(self) -> EpisodeResult | None:
        return self._result

    def close(self) -> None:
        self.renderer.close()
        self._task_id = None

    def render_evidence_rgb(self, camera_name: str = "third_person") -> bytes:
        self._require_active()
        return self.renderer.rgb(
            self.data,
            camera_name,
            timestamp_ns=self._timestamp_ns(),
            frame_index=self._sequence,
        ).payload or b""

    def _reset_base(self) -> None:
        address = self.model.jnt_qposadr[self.bundle.ids.base_joint]
        self.data.qpos[address : address + 7] = (0.0, 0.0, 0.22, 1.0, 0.0, 0.0, 0.0)
        dof_address = self.model.jnt_dofadr[self.bundle.ids.base_joint]
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _reset_arms(self) -> None:
        self._left_targets = np.asarray(self.config.left_arm_home, dtype=np.float64)
        self._right_targets = np.asarray(self.config.right_arm_home, dtype=np.float64)
        for targets, joint_ids in (
            (self._left_targets, self.bundle.ids.secondary_arm_joints),
            (self._right_targets, self.bundle.ids.arm_joints),
        ):
            for target, joint_id in zip(targets, joint_ids, strict=True):
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = target
        for joint_id in (
            *self.bundle.ids.secondary_finger_joints,
            *self.bundle.ids.finger_joints,
            *self.bundle.ids.wheel_joints,
        ):
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = 0.0

    def _reset_object(self) -> None:
        joint_id = self.bundle.ids.object_joint
        if joint_id is None:
            return
        address = self.model.jnt_qposadr[joint_id]
        yaw = self._rng.uniform(-math.pi, math.pi)
        self.data.qpos[address : address + 7] = (
            0.72 + self._rng.uniform(-0.04, 0.04),
            self._rng.uniform(-0.18, -0.02),
            self.config.primary_object_reset_z,
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        )
        dof_address = self.model.jnt_dofadr[joint_id]
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _write_controls(self, action: DualArmAction) -> None:
        left_wheel = (
            action.base_linear - action.base_angular * TRACK_WIDTH / 2
        ) / WHEEL_RADIUS
        right_wheel = (
            action.base_linear + action.base_angular * TRACK_WIDTH / 2
        ) / WHEEL_RADIUS
        self.data.ctrl[list(self.bundle.ids.wheel_actuators)] = (
            left_wheel,
            left_wheel,
            right_wheel,
            right_wheel,
        )
        control_dt = 1.0 / self.config.control_hz
        self._advance_tool_targets(
            self._left_targets,
            action.left_arm,
            self.bundle.ids.secondary_arm_joints,
            self._left_tool_site,
            control_dt,
        )
        self._advance_tool_targets(
            self._right_targets,
            action.right_arm,
            self.bundle.ids.arm_joints,
            self._right_tool_site,
            control_dt,
        )
        self.data.ctrl[list(self.bundle.ids.secondary_arm_actuators)] = self._left_targets
        self.data.ctrl[list(self.bundle.ids.arm_actuators)] = self._right_targets
        self.data.ctrl[list(self.bundle.ids.secondary_finger_actuators)] = (
            action.left_gripper * FINGER_TRAVEL
        )
        self.data.ctrl[list(self.bundle.ids.finger_actuators)] = (
            action.right_gripper * FINGER_TRAVEL
        )

    def _advance_tool_targets(
        self,
        targets: np.ndarray,
        command: tuple[float, ...],
        joint_ids: tuple[int, ...],
        tool_site: int,
        control_dt: float,
    ) -> None:
        targets += self._resolved_joint_velocity(
            command, joint_ids, tool_site
        ) * control_dt
        for index, joint_id in enumerate(joint_ids):
            low, high = self.model.jnt_range[joint_id]
            actual = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            servo_low = max(float(low), actual - self.config.max_arm_servo_error)
            servo_high = min(float(high), actual + self.config.max_arm_servo_error)
            targets[index] = np.clip(targets[index], servo_low, servo_high)

    def _resolved_joint_velocity(
        self,
        command: tuple[float, ...],
        joint_ids: tuple[int, ...],
        tool_site: int,
    ) -> np.ndarray:
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            tool_site,
        )
        dofs = [int(self.model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
        jacobian = np.vstack(
            (jacobian_position[:, dofs], jacobian_rotation[:, dofs])
        )
        base_rotation = self.data.xmat[self.bundle.ids.base_body].reshape(3, 3)
        value = np.asarray(command, dtype=np.float64)
        twist = np.concatenate(
            (
                base_rotation @ value[:3] * self.config.max_tool_linear_velocity,
                base_rotation @ value[3:] * self.config.max_tool_angular_velocity,
            )
        )
        regularized = jacobian @ jacobian.T
        regularized += np.eye(6) * self.config.ik_damping**2
        joint_velocity = jacobian.T @ np.linalg.solve(regularized, twist)
        return np.clip(
            joint_velocity,
            -self.config.max_arm_velocity,
            self.config.max_arm_velocity,
        )

    def _site_id(self, name: str) -> int:
        site = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name))
        if site < 0:
            raise ValueError(f"dual-arm model is missing tool site {name}")
        return site

    def _observation(self) -> DualArmObservation:
        timestamp_ns = self._timestamp_ns()
        base_pose, base_twist = self._base_state()
        left_gripper, right_gripper = self._gripper_positions()
        proprioception = DualArmProprioception(
            left_joint_position=self._joint_values(
                self.bundle.ids.secondary_arm_joints, velocity=False
            ),
            left_joint_velocity=self._joint_values(
                self.bundle.ids.secondary_arm_joints, velocity=True
            ),
            right_joint_position=self._joint_values(
                self.bundle.ids.arm_joints, velocity=False
            ),
            right_joint_velocity=self._joint_values(
                self.bundle.ids.arm_joints, velocity=True
            ),
            left_gripper_position=left_gripper,
            right_gripper_position=right_gripper,
            base_pose=base_pose,
            base_twist=base_twist,
            imu=tuple(float(value) for value in self.data.sensordata),
        )
        cameras = (
            self.renderer.rgb(
                self.data,
                "head_rgb",
                timestamp_ns=timestamp_ns,
                frame_index=self._sequence,
            ),
            self.renderer.depth(
                self.data,
                "head_depth",
                timestamp_ns=timestamp_ns,
                frame_index=self._sequence,
            ),
            self.renderer.rgb(
                self.data,
                "left_wrist_rgb",
                timestamp_ns=timestamp_ns,
                frame_index=self._sequence,
            ),
            self.renderer.rgb(
                self.data,
                "wrist_rgb",
                timestamp_ns=timestamp_ns,
                frame_index=self._sequence,
                camera_id="right_wrist_rgb",
            ),
        )
        return DualArmObservation(
            timestamp_ns=timestamp_ns,
            sequence_id=self._sequence,
            task_id=self._task_id or self.config.task_id,
            instruction=self._instruction,
            proprioception=proprioception,
            cameras=cameras,
            camera_calibrations=self._camera_calibrations(cameras),
            safety_state=SafetyState.OK,
            quality={
                "simulation": 1.0,
                "head_rgb": 1.0,
                "head_depth": 1.0,
                "left_wrist_rgb": 1.0,
                "right_wrist_rgb": 1.0,
            },
        )

    def _camera_calibrations(
        self, cameras
    ) -> tuple[FrameCameraCalibration, ...]:
        base_rotation = self.data.xmat[self.bundle.ids.base_body].reshape(3, 3)
        base_position = self.data.xpos[self.bundle.ids.base_body]
        world_from_robot = np.eye(4, dtype=np.float64)
        world_from_robot[:3, :3] = base_rotation
        world_from_robot[:3, 3] = base_position
        robot_from_world = np.linalg.inv(world_from_robot)
        values = []
        camera_names = {
            "head_rgb": "head_rgb",
            "head_depth": "head_depth",
            "left_wrist_rgb": "left_wrist_rgb",
            "right_wrist_rgb": "wrist_rgb",
        }
        for frame in cameras:
            camera_id = int(
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_names[frame.camera_id]
                )
            )
            rotation = self.data.cam_xmat[camera_id].reshape(3, 3)
            world_from_camera = np.eye(4, dtype=np.float64)
            world_from_camera[:3, :3] = np.column_stack(
                (rotation[:, 0], -rotation[:, 1], -rotation[:, 2])
            )
            world_from_camera[:3, 3] = self.data.cam_xpos[camera_id]
            robot_from_camera = robot_from_world @ world_from_camera
            focal = frame.height / (
                2.0 * math.tan(math.radians(float(self.model.cam_fovy[camera_id])) / 2.0)
            )
            values.append(
                FrameCameraCalibration(
                    frame.camera_id,
                    (focal, focal, (frame.width - 1) / 2, (frame.height - 1) / 2),
                    tuple(float(item) for item in robot_from_camera.reshape(-1)),
                )
            )
        return tuple(values)

    def _joint_values(
        self, joint_ids: tuple[int, ...], *, velocity: bool
    ) -> tuple[float, ...]:
        addresses = self.model.jnt_dofadr if velocity else self.model.jnt_qposadr
        values = self.data.qvel if velocity else self.data.qpos
        return tuple(float(values[addresses[joint_id]]) for joint_id in joint_ids)

    def _gripper_positions(self) -> tuple[float, float]:
        def normalized(joint_ids: tuple[int, ...]) -> float:
            total = sum(
                float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
                for joint_id in joint_ids
            )
            return float(np.clip(total / (len(joint_ids) * FINGER_TRAVEL), 0.0, 1.0))

        return (
            normalized(self.bundle.ids.secondary_finger_joints),
            normalized(self.bundle.ids.finger_joints),
        )

    def _base_state(self) -> tuple[tuple[float, float, float], tuple[float, float]]:
        qpos_address = self.model.jnt_qposadr[self.bundle.ids.base_joint]
        x, y = (float(value) for value in self.data.qpos[qpos_address : qpos_address + 2])
        quaternion = self.data.qpos[qpos_address + 3 : qpos_address + 7]
        yaw = _quaternion_yaw(quaternion)
        dof_address = self.model.jnt_dofadr[self.bundle.ids.base_joint]
        world_linear = self.data.qvel[dof_address : dof_address + 3]
        rotation = self.data.xmat[self.bundle.ids.base_body].reshape(3, 3)
        local_linear = rotation.T @ world_linear
        return (x, y, yaw), (
            float(local_linear[0]),
            float(self.data.qvel[dof_address + 5]),
        )

    def _timestamp_ns(self) -> int:
        return round(float(self.data.time) * 1_000_000_000)

    def _require_active(self) -> None:
        if self._task_id is None:
            raise RuntimeError("backend is not reset or has been closed")


def _control_substeps(control_hz: float, timestep: float) -> int:
    substeps = round(1.0 / (control_hz * timestep))
    if substeps <= 0 or not math.isclose(
        substeps * timestep,
        1.0 / control_hz,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("control period must be an integer number of physics steps")
    return substeps


def _snapshot_state_specification() -> mujoco.mjtState:
    return mujoco.mjtState(
        int(mujoco.mjtState.mjSTATE_INTEGRATION)
        | int(mujoco.mjtState.mjSTATE_CTRL)
    )


def _zero_action() -> DualArmAction:
    return DualArmAction(0.0, 0.0, (0.0,) * 6, (0.0,) * 6, 0.0, 0.0)


def _hold_action(grippers: tuple[float, float]) -> DualArmAction:
    return DualArmAction(
        0.0,
        0.0,
        (0.0,) * 6,
        (0.0,) * 6,
        grippers[0],
        grippers[1],
    )


def _quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
