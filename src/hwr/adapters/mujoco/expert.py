"""Privileged Cartesian controller used only to generate 3D demonstrations."""

from __future__ import annotations

import mujoco
import numpy as np

from hwr.adapters.mujoco.backend import Mujoco3DBackend
from hwr.adapters.mujoco.names import ARM_HOME
from hwr.core.types import ActionFrame, ObservationFrame


class PrivilegedCartesianExpert:
    """Use engine state for labels; this object must never be used as an evaluation policy."""

    def __init__(
        self,
        backend: Mujoco3DBackend,
        *,
        site_name: str = "right_grasp_center",
        source: str = "privileged_3d_expert",
    ) -> None:
        self.backend = backend
        self.model = backend.model
        self.data = backend.data
        self.source = source
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"model is missing site: {site_name}")
        self.arm_dofs = np.asarray(
            [self.model.jnt_dofadr[joint_id] for joint_id in backend.bundle.ids.arm_joints]
        )
        self.target_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def reset_orientation_target(self) -> None:
        self.target_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def set_orientation_target(self, rotation: np.ndarray) -> None:
        value = np.asarray(rotation, dtype=np.float64)
        if value.shape != (3, 3):
            raise ValueError("orientation target must be a 3x3 rotation matrix")
        self.target_rotation = value.copy()

    def action(
        self,
        observation: ObservationFrame,
        *,
        target_position: tuple[float, float, float],
        gripper_target: float,
        orientation_weight: float = 0.0,
    ) -> ActionFrame:
        translation = np.zeros((3, self.model.nv), dtype=np.float64)
        rotation = (
            np.zeros((3, self.model.nv), dtype=np.float64)
            if orientation_weight > 0.0
            else None
        )
        mujoco.mj_jacSite(self.model, self.data, translation, rotation, self.site_id)
        position_error = np.asarray(target_position) - self.data.site_xpos[self.site_id]
        jacobian = translation[:, self.arm_dofs]
        desired_twist = 4.0 * position_error
        if rotation is not None:
            current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            rotation_error = 0.5 * sum(
                (
                    np.cross(current_rotation[:, axis], self.target_rotation[:, axis])
                    for axis in range(3)
                ),
                start=np.zeros(3),
            )
            jacobian = np.vstack(
                (jacobian, orientation_weight * rotation[:, self.arm_dofs])
            )
            desired_twist = np.concatenate(
                (desired_twist, orientation_weight * 2.0 * rotation_error)
            )
        damping = 0.005 * np.eye(jacobian.shape[0])
        inverse = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping,
            np.eye(jacobian.shape[0]),
        )
        joint_velocity = inverse @ desired_twist
        positions = np.asarray(observation.joint_position, dtype=np.float64)
        posture_velocity = 0.05 * (np.asarray(ARM_HOME) - positions)
        nullspace = np.eye(len(self.arm_dofs)) - inverse @ jacobian
        joint_velocity += nullspace @ posture_velocity
        joint_velocity = np.clip(joint_velocity, -1.0, 1.0)
        period_ns = round(1_000_000_000 / self.backend.config.control_hz)
        return ActionFrame(
            created_at_ns=observation.timestamp_ns,
            valid_from_ns=observation.timestamp_ns,
            valid_until_ns=observation.timestamp_ns + period_ns,
            source=self.source,
            arm_command=tuple(float(value) for value in joint_velocity),
            gripper_target=gripper_target,
        )

    def site_position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.data.site_xpos[self.site_id])
