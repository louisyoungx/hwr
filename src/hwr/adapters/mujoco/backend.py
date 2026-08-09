"""MuJoCo three-dimensional backend implementing project runtime contracts."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from hwr.adapters.mujoco.model import MujocoModelBundle
from hwr.adapters.mujoco.names import ARM_HOME, FINGER_TRAVEL, TRACK_WIDTH, WHEEL_RADIUS
from hwr.adapters.mujoco.rendering import MujocoCameraRenderer
from hwr.core.runtime import StepOutcome
from hwr.core.types import ActionFrame, EpisodeResult, ObservationFrame, SafetyState
from hwr.safety import SafetyLimits, SafetySupervisor


@dataclass(frozen=True)
class Mujoco3DConfig:
    model_path: Path
    task_id: str = "mujoco_mobile_manipulator_smoke/v1"
    control_hz: float = 20.0
    max_steps: int = 200
    camera_width: int = 160
    camera_height: int = 120
    max_base_linear: float = 0.45
    max_base_angular: float = 1.0
    max_arm_velocity: float = 1.2

    def __post_init__(self) -> None:
        numeric = (
            self.control_hz,
            self.max_steps,
            self.camera_width,
            self.camera_height,
            self.max_base_linear,
            self.max_base_angular,
            self.max_arm_velocity,
        )
        if not self.task_id or min(numeric) <= 0:
            raise ValueError("MuJoCo backend configuration values must be positive")


class Mujoco3DBackend:
    """Reference 3D backend; task-specific success logic is added above this base."""

    def __init__(self, config: Mujoco3DConfig) -> None:
        self.config = config
        self.bundle = MujocoModelBundle.load(config.model_path)
        self.model = self.bundle.model
        self.data = mujoco.MjData(self.model)
        self.renderer = MujocoCameraRenderer(
            self.model,
            width=config.camera_width,
            height=config.camera_height,
        )
        self.safety = SafetySupervisor(
            SafetyLimits(
                max_base_linear=config.max_base_linear,
                max_base_angular=config.max_base_angular,
                max_arm_command=config.max_arm_velocity,
            ),
            arm_dof=6,
        )
        substeps = round(1.0 / (config.control_hz * self.model.opt.timestep))
        if substeps <= 0 or not math.isclose(
            substeps * self.model.opt.timestep,
            1.0 / config.control_hz,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("control period must be an integer number of physics steps")
        self._substeps = substeps
        self._rng = random.Random()
        self._sequence = 0
        self._steps = 0
        self._task_id: str | None = None
        self._result: EpisodeResult | None = None
        self._arm_targets = np.asarray(ARM_HOME, dtype=np.float64)

    def reset(self, *, seed: int, task_id: str) -> ObservationFrame:
        if task_id != self.config.task_id:
            raise ValueError(f"backend provides {self.config.task_id}, not {task_id}")
        self._rng.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._sequence = 0
        self._steps = 0
        self._task_id = task_id
        self._result = None
        self._reset_base()
        self._reset_arm()
        self._reset_object()
        self._set_controls(ActionFrame(0, 0, 1, "reset", arm_command=(0.0,) * 6))
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def observe(self) -> ObservationFrame:
        self._require_active()
        return self._observation()

    def apply(self, action: ActionFrame) -> StepOutcome:
        self._require_active()
        if self._result is not None:
            raise RuntimeError("episode is complete; call reset")
        timestamp_ns = self._timestamp_ns()
        applied, events = self.safety.filter(action, now_ns=timestamp_ns)
        self._set_controls(applied)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        self._sequence += 1
        truncated = self._steps >= self.config.max_steps
        if truncated:
            self._result = EpisodeResult(
                success=False,
                reason="smoke_timeout",
                ended_at_ns=self._timestamp_ns(),
                metrics={"steps": self._steps, "contacts": self.data.ncon},
            )
        return StepOutcome(
            observation=self._observation(),
            truncated=truncated,
            events=events,
            info={"applied_action": applied, "physics_contacts": int(self.data.ncon)},
        )

    def result(self) -> EpisodeResult | None:
        return self._result

    def close(self) -> None:
        self.renderer.close()
        self._task_id = None

    def render_evidence_rgb(self, camera_name: str = "third_person") -> bytes:
        """Capture evidence separately so policy observations never include this view."""
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

    def _reset_arm(self) -> None:
        self._arm_targets = np.asarray(ARM_HOME, dtype=np.float64)
        for target, joint_id in zip(self._arm_targets, self.bundle.ids.arm_joints, strict=True):
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = target
        for joint_id in (*self.bundle.ids.wheel_joints, *self.bundle.ids.finger_joints):
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = 0.0

    def _reset_object(self) -> None:
        address = self.model.jnt_qposadr[self.bundle.ids.object_joint]
        object_x = 0.88 + self._rng.uniform(-0.04, 0.04)
        object_y = self._rng.uniform(-0.12, 0.12)
        yaw = self._rng.uniform(-math.pi, math.pi)
        self.data.qpos[address : address + 7] = (
            object_x,
            object_y,
            0.09,
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        )
        dof_address = self.model.jnt_dofadr[self.bundle.ids.object_joint]
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _set_controls(self, action: ActionFrame) -> None:
        left = (action.base_linear - action.base_angular * TRACK_WIDTH / 2) / WHEEL_RADIUS
        right = (action.base_linear + action.base_angular * TRACK_WIDTH / 2) / WHEEL_RADIUS
        wheel_values = (left, left, right, right)
        self.data.ctrl[list(self.bundle.ids.wheel_actuators)] = wheel_values
        control_dt = 1.0 / self.config.control_hz
        self._arm_targets += np.asarray(action.arm_command, dtype=np.float64) * control_dt
        for index, joint_id in enumerate(self.bundle.ids.arm_joints):
            low, high = self.model.jnt_range[joint_id]
            self._arm_targets[index] = np.clip(self._arm_targets[index], low, high)
        self.data.ctrl[list(self.bundle.ids.arm_actuators)] = self._arm_targets
        finger_target = action.gripper_target * FINGER_TRAVEL
        self.data.ctrl[list(self.bundle.ids.finger_actuators)] = finger_target

    def _observation(self) -> ObservationFrame:
        timestamp_ns = self._timestamp_ns()
        arm_positions = tuple(
            float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            for joint_id in self.bundle.ids.arm_joints
        )
        arm_velocities = tuple(
            float(self.data.qvel[self.model.jnt_dofadr[joint_id]])
            for joint_id in self.bundle.ids.arm_joints
        )
        finger_position = sum(
            float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            for joint_id in self.bundle.ids.finger_joints
        ) / (len(self.bundle.ids.finger_joints) * FINGER_TRAVEL)
        base_pose, base_twist = self._base_state()
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
                "wrist_rgb",
                timestamp_ns=timestamp_ns,
                frame_index=self._sequence,
            ),
        )
        return ObservationFrame(
            timestamp_ns=timestamp_ns,
            sequence_id=self._sequence,
            task_id=self._task_id or self.config.task_id,
            task_stage="instruction_following",
            joint_position=arm_positions,
            joint_velocity=arm_velocities,
            gripper_position=finger_position,
            base_pose=base_pose,
            base_twist=base_twist,
            imu=tuple(float(value) for value in self.data.sensordata),
            cameras=cameras,
            features={},
            safety_state=SafetyState.OK,
            quality={"simulation": 1.0, "rgb": 1.0, "depth": 1.0},
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
        yaw_rate = float(self.data.qvel[dof_address + 5])
        return (x, y, yaw), (float(local_linear[0]), yaw_rate)

    def _timestamp_ns(self) -> int:
        return round(float(self.data.time) * 1_000_000_000)

    def _require_active(self) -> None:
        if self._task_id is None:
            raise RuntimeError("backend is not reset or has been closed")


def _quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
