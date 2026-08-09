"""Load MuJoCo models and resolve named engine entities once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from hwr.adapters.mujoco.names import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    FINGER_ACTUATORS,
    FINGER_JOINTS,
    WHEEL_ACTUATORS,
    WHEEL_JOINTS,
)


def _ids(model: mujoco.MjModel, object_type: mujoco.mjtObj, names: tuple[str, ...]) -> tuple[int, ...]:
    resolved = tuple(mujoco.mj_name2id(model, object_type, name) for name in names)
    missing = [name for name, entity_id in zip(names, resolved, strict=True) if entity_id < 0]
    if missing:
        raise ValueError(f"model is missing required entities: {', '.join(missing)}")
    return resolved


@dataclass(frozen=True)
class MujocoEntityIds:
    base_joint: int
    base_body: int
    object_joint: int | None
    wheel_joints: tuple[int, ...]
    wheel_actuators: tuple[int, ...]
    arm_joints: tuple[int, ...]
    arm_actuators: tuple[int, ...]
    finger_joints: tuple[int, ...]
    finger_actuators: tuple[int, ...]

    @classmethod
    def resolve(
        cls, model: mujoco.MjModel, object_joint_name: str | None
    ) -> "MujocoEntityIds":
        return cls(
            base_joint=_ids(model, mujoco.mjtObj.mjOBJ_JOINT, ("base_free",))[0],
            base_body=_ids(model, mujoco.mjtObj.mjOBJ_BODY, ("robot_base",))[0],
            object_joint=(
                _ids(model, mujoco.mjtObj.mjOBJ_JOINT, (object_joint_name,))[0]
                if object_joint_name is not None
                else None
            ),
            wheel_joints=_ids(model, mujoco.mjtObj.mjOBJ_JOINT, WHEEL_JOINTS),
            wheel_actuators=_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEEL_ACTUATORS),
            arm_joints=_ids(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS),
            arm_actuators=_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_ACTUATORS),
            finger_joints=_ids(model, mujoco.mjtObj.mjOBJ_JOINT, FINGER_JOINTS),
            finger_actuators=_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, FINGER_ACTUATORS),
        )


@dataclass(frozen=True)
class MujocoModelBundle:
    model_path: Path
    model: mujoco.MjModel
    ids: MujocoEntityIds

    @classmethod
    def load(
        cls,
        model_path: Path,
        object_joint_name: str | None = "smoke_object_free",
    ) -> "MujocoModelBundle":
        resolved = model_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"MuJoCo model is unavailable: {resolved}")
        model = mujoco.MjModel.from_xml_path(str(resolved))
        return cls(resolved, model, MujocoEntityIds.resolve(model, object_joint_name))
