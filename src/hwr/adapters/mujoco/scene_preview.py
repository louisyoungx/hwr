"""Read-only visual review helpers for compiled formal scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from hwr.adapters.mujoco.names import (
    ARM_ACTUATORS,
    ARM_HOME,
    ARM_JOINTS,
    SECONDARY_ARM_ACTUATORS,
    SECONDARY_ARM_HOME,
    SECONDARY_ARM_JOINTS,
)


@dataclass(frozen=True)
class ScenePreview:
    third_person_rgb: bytes
    head_rgb: bytes
    width: int
    height: int
    simulation_time: float
    contacts: int


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    entity_id = int(mujoco.mj_name2id(model, kind, name))
    if entity_id < 0:
        raise ValueError(f"model is missing {name}")
    return entity_id


def _reset_preview_robot(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name, home in zip(ARM_JOINTS, ARM_HOME, strict=True):
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = home
    for name, home in zip(ARM_ACTUATORS, ARM_HOME, strict=True):
        actuator_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[actuator_id] = home
    for name, home in zip(SECONDARY_ARM_JOINTS, SECONDARY_ARM_HOME, strict=True):
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = home
    for name, home in zip(SECONDARY_ARM_ACTUATORS, SECONDARY_ARM_HOME, strict=True):
        actuator_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[actuator_id] = home


def _render(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> bytes:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    return np.ascontiguousarray(renderer.render(), dtype=np.uint8).tobytes()


def render_scene_preview(
    model_path: Path,
    *,
    width: int,
    height: int,
    settle_seconds: float = 0.5,
) -> ScenePreview:
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    _reset_preview_robot(model, data)
    mujoco.mj_forward(model, data)
    steps = round(settle_seconds / float(model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        third_person = _render(renderer, data, "third_person")
        head_rgb = _render(renderer, data, "head_rgb")
    finally:
        renderer.close()
    return ScenePreview(
        third_person_rgb=third_person,
        head_rgb=head_rgb,
        width=width,
        height=height,
        simulation_time=float(data.time),
        contacts=int(data.ncon),
    )
