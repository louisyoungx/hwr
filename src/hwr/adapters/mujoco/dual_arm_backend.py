"""Canonical 16-D dual-arm MuJoCo runtime with deployable observations only."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
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
    NaturalLanguageInstruction,
)
from hwr.core.runtime import RuntimeStepOutcome
from hwr.core.types import EpisodeResult, SafetyState
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
                max_arm_command=config.max_arm_velocity,
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

    def reset(self, *, seed: int, task_id: str) -> DualArmObservation:
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
        return self._observation()

    def observe(self) -> DualArmObservation:
        self._require_active()
        return self._observation()

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
        self._write_controls(applied.action)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        self._sequence += 1
        truncated = self._steps >= self.config.max_steps
        if truncated:
            self._result = EpisodeResult(
                success=False,
                reason="episode_timeout",
                ended_at_ns=self._timestamp_ns(),
                metrics={"steps": self._steps, "contacts": self.data.ncon},
            )
        return RuntimeStepOutcome(
            observation=self._observation(),
            reward=0.0,
            terminated=False,
            truncated=truncated,
            events=events,
            info={
                "applied_action": applied,
                "physics_contacts": int(self.data.ncon),
            },
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
        self._advance_targets(
            self._left_targets,
            action.left_arm,
            self.bundle.ids.secondary_arm_joints,
            control_dt,
        )
        self._advance_targets(
            self._right_targets,
            action.right_arm,
            self.bundle.ids.arm_joints,
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

    def _advance_targets(
        self,
        targets: np.ndarray,
        command: tuple[float, ...],
        joint_ids: tuple[int, ...],
        control_dt: float,
    ) -> None:
        targets += np.asarray(command, dtype=np.float64) * control_dt
        for index, joint_id in enumerate(joint_ids):
            low, high = self.model.jnt_range[joint_id]
            actual = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            servo_low = max(float(low), actual - self.config.max_arm_servo_error)
            servo_high = min(float(high), actual + self.config.max_arm_servo_error)
            targets[index] = np.clip(targets[index], servo_low, servo_high)

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
            safety_state=SafetyState.OK,
            quality={
                "simulation": 1.0,
                "head_rgb": 1.0,
                "head_depth": 1.0,
                "left_wrist_rgb": 1.0,
                "right_wrist_rgb": 1.0,
            },
        )

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


def _zero_action() -> DualArmAction:
    return DualArmAction(0.0, 0.0, (0.0,) * 6, (0.0,) * 6, 0.0, 0.0)


def _quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
