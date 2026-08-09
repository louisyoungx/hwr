"""Inspect compiled models instead of trusting declarations in MJCF text."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from hwr.adapters.mujoco.names import ARM_JOINTS, FINGER_JOINTS, POLICY_CAMERAS, WHEEL_JOINTS


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> set[str]:
    return {
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, object_type, index)) is not None
    }


@dataclass(frozen=True)
class RobotModelReport:
    wheel_joint_count: int
    arm_joint_count: int
    finger_joint_count: int
    policy_cameras: tuple[str, ...]
    dynamic_body_count: int
    invalid_dynamic_bodies: tuple[str, ...]
    equality_constraint_count: int
    timestep: float
    gravity: tuple[float, float, float]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_robot_model(model: mujoco.MjModel) -> RobotModelReport:
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    camera_names = _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
    wheel_count = sum(name in joint_names for name in WHEEL_JOINTS)
    arm_count = sum(name in joint_names for name in ARM_JOINTS)
    finger_count = sum(name in joint_names for name in FINGER_JOINTS)
    invalid_bodies: list[str] = []
    dynamic_count = 0
    for body_id in range(1, model.nbody):
        if model.body_jntnum[body_id] <= 0:
            continue
        dynamic_count += 1
        mass_valid = model.body_mass[body_id] > 0
        inertia_valid = bool(np.all(model.body_inertia[body_id] > 0))
        if not mass_valid or not inertia_valid:
            invalid_bodies.append(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or str(body_id)
            )
    errors: list[str] = []
    if wheel_count != 4:
        errors.append(f"expected 4 wheel joints, found {wheel_count}")
    if arm_count != 6:
        errors.append(f"expected 6 arm joints, found {arm_count}")
    if finger_count != 2:
        errors.append(f"expected 2 finger joints, found {finger_count}")
    missing_cameras = tuple(name for name in POLICY_CAMERAS if name not in camera_names)
    if missing_cameras:
        errors.append(f"missing policy cameras: {', '.join(missing_cameras)}")
    if invalid_bodies:
        errors.append(f"invalid mass or inertia: {', '.join(invalid_bodies)}")
    if model.neq:
        errors.append("formal robot smoke model must not contain equality constraints")
    if model.opt.gravity[2] >= 0:
        errors.append("gravity must point downward")
    return RobotModelReport(
        wheel_joint_count=wheel_count,
        arm_joint_count=arm_count,
        finger_joint_count=finger_count,
        policy_cameras=tuple(name for name in POLICY_CAMERAS if name in camera_names),
        dynamic_body_count=dynamic_count,
        invalid_dynamic_bodies=tuple(invalid_bodies),
        equality_constraint_count=model.neq,
        timestep=float(model.opt.timestep),
        gravity=tuple(float(value) for value in model.opt.gravity),
        errors=tuple(errors),
    )
