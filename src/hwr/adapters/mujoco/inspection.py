"""Inspect compiled models instead of trusting declarations in MJCF text."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from hwr.adapters.mujoco.names import (
    ALL_ARM_JOINTS,
    ALL_FINGER_JOINTS,
    FINGER_JOINTS,
    POLICY_CAMERAS,
    SECONDARY_FINGER_JOINTS,
    WHEEL_JOINTS,
)


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> set[str]:
    return {
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, object_type, index)) is not None
    }


@dataclass(frozen=True)
class RobotModelReport:
    wheel_joint_count: int
    manipulator_count: int
    arm_joint_count: int
    finger_joint_count: int
    central_body_present: bool
    top_camera_present: bool
    overall_height_m: float
    arm_mount_y_m: tuple[float, float]
    finger_joint_travel_m: tuple[float, ...]
    gripper_open_gap_m: float
    gripper_closed_gap_m: float
    gripper_open_gaps_m: tuple[float, ...]
    gripper_closed_gaps_m: tuple[float, ...]
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
    arm_count = sum(name in joint_names for name in ALL_ARM_JOINTS)
    finger_count = sum(name in joint_names for name in ALL_FINGER_JOINTS)
    finger_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ALL_FINGER_JOINTS
    ]
    finger_travel = tuple(
        float(np.ptp(model.jnt_range[joint_id])) for joint_id in finger_ids if joint_id >= 0
    )
    gripper_gaps = tuple(
        _gripper_gaps(model, prefix, joints)
        for prefix, joints in (
            ("right_gripper", FINGER_JOINTS),
            ("left_gripper", SECONDARY_FINGER_JOINTS),
        )
    )
    open_gaps = tuple(value[0] for value in gripper_gaps)
    closed_gaps = tuple(value[1] for value in gripper_gaps)
    open_gap = open_gaps[0]
    closed_gap = closed_gaps[0]
    right_shoulder = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pan_link"
    )
    left_shoulder = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pan_link"
    )
    arm_mount_y = tuple(
        float(model.body_pos[body_id, 1])
        for body_id in (right_shoulder, left_shoulder)
        if body_id >= 0
    )
    central_body_present = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_box_collision") >= 0
    )
    top_camera_present = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top_camera_head") >= 0
    )
    overall_height = _overall_height(model)
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
    if arm_count != 12:
        errors.append(f"expected 12 arm joints across two manipulators, found {arm_count}")
    if finger_count != 4:
        errors.append(f"expected 4 finger joints across two grippers, found {finger_count}")
    if len(arm_mount_y) != 2 or not arm_mount_y[0] < 0 < arm_mount_y[1]:
        errors.append("mechanical arms must be mounted on opposite box sides")
    if not central_body_present:
        errors.append("central body box is missing")
    if not top_camera_present:
        errors.append("top camera head is missing")
    if not np.isclose(overall_height, 1.60, atol=0.005):
        errors.append(f"overall robot height must be 1.60 m, found {overall_height:.3f} m")
    if any(value < 0.18 for value in open_gaps) or any(
        not 0.02 <= value <= 0.07 for value in closed_gaps
    ):
        errors.append(
            f"gripper gaps are not household-object capable: open={open_gaps}, closed={closed_gaps}"
        )
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
        manipulator_count=arm_count // 6,
        arm_joint_count=arm_count,
        finger_joint_count=finger_count,
        central_body_present=central_body_present,
        top_camera_present=top_camera_present,
        overall_height_m=overall_height,
        arm_mount_y_m=arm_mount_y,
        finger_joint_travel_m=finger_travel,
        gripper_open_gap_m=open_gap,
        gripper_closed_gap_m=closed_gap,
        gripper_open_gaps_m=open_gaps,
        gripper_closed_gaps_m=closed_gaps,
        policy_cameras=tuple(name for name in POLICY_CAMERAS if name in camera_names),
        dynamic_body_count=dynamic_count,
        invalid_dynamic_bodies=tuple(invalid_bodies),
        equality_constraint_count=model.neq,
        timestep=float(model.opt.timestep),
        gravity=tuple(float(value) for value in model.opt.gravity),
        errors=tuple(errors),
    )


def _overall_height(model: mujoco.MjModel) -> float:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base")
    camera_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "top_camera_head"
    )
    housing_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "camera_head_housing"
    )
    if min(base_id, camera_body_id, housing_id) < 0:
        return 0.0
    return float(
        model.body_pos[base_id, 2]
        + model.body_pos[camera_body_id, 2]
        + model.geom_pos[housing_id, 2]
        + model.geom_size[housing_id, 2]
    )


def _gripper_gaps(
    model: mujoco.MjModel,
    prefix: str,
    finger_joints: tuple[str, ...],
) -> tuple[float, float]:
    left_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_left_finger"
    )
    right_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_right_finger"
    )
    left_pad = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_left_pad"
    )
    right_pad = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_right_pad"
    )
    ids = (left_body, right_body, left_pad, right_pad)
    if min(ids) < 0:
        return (0.0, 0.0)
    center_distance = abs(
        float(model.body_pos[left_body, 1] - model.body_pos[right_body, 1])
    )
    open_gap = center_distance - float(
        model.geom_size[left_pad, 1] + model.geom_size[right_pad, 1]
    )
    travel = sum(
        float(np.ptp(model.jnt_range[joint_id]))
        for name in finger_joints
        if (joint_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) >= 0
    )
    return (open_gap, open_gap - travel)
