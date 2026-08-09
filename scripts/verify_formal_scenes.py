#!/usr/bin/env python3
"""Compile and inspect all formal household scenes from the engine model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs/scenes/formal_3d_v1.json"


def _name_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(model, kind, name))


def _missing(model: mujoco.MjModel, kind: mujoco.mjtObj, names: list[str]) -> list[str]:
    return [name for name in names if _name_id(model, kind, name) < 0]


def _actuated_joint_ids(model: mujoco.MjModel) -> set[int]:
    return {
        int(model.actuator_trnid[index, 0])
        for index in range(model.nu)
        if int(model.actuator_trnid[index, 0]) >= 0
    }


def _settled_object_speeds(
    model: mujoco.MjModel, joint_names: list[str], seconds: float = 2.0
) -> dict[str, float]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(round(seconds / float(model.opt.timestep))):
        mujoco.mj_step(model, data)
    speeds: dict[str, float] = {}
    for name in joint_names:
        joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        dof_address = int(model.jnt_dofadr[joint_id])
        speeds[name] = float(np.linalg.norm(data.qvel[dof_address : dof_address + 6]))
        qpos_address = int(model.jnt_qposadr[joint_id])
        if float(data.qpos[qpos_address + 2]) < -0.01:
            speeds[name] = float("inf")
    return speeds


def _scene_report(entry: dict[str, Any]) -> dict[str, Any]:
    model_path = ROOT / entry["model"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    errors: list[str] = []
    expected_cameras = ["head_rgb", "head_depth", "wrist_rgb", "third_person", "room_overview"]
    checks = (
        (mujoco.mjtObj.mjOBJ_BODY, entry["furniture_bodies"], "furniture bodies"),
        (mujoco.mjtObj.mjOBJ_GEOM, entry["textured_visual_geoms"], "visual geoms"),
        (mujoco.mjtObj.mjOBJ_GEOM, entry["required_collision_geoms"], "collision geoms"),
        (mujoco.mjtObj.mjOBJ_JOINT, entry["manipulated_free_joints"], "object joints"),
        (mujoco.mjtObj.mjOBJ_SITE, entry["target_sites"], "target sites"),
        (mujoco.mjtObj.mjOBJ_CAMERA, expected_cameras, "cameras"),
    )
    for kind, names, label in checks:
        absent = _missing(model, kind, names)
        if absent:
            errors.append(f"missing {label}: {absent}")
    if not np.allclose(model.opt.gravity, (0.0, 0.0, -9.81)):
        errors.append(f"gravity is {model.opt.gravity.tolist()}")
    if model.neq:
        errors.append(f"equality constraints are forbidden, found {model.neq}")
    if model.nmesh < 3 or model.ntex < 5 or model.nlight < 2:
        errors.append("scene lacks mesh, texture, or lighting complexity")
    for name in entry["textured_visual_geoms"]:
        geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0 and int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            errors.append(f"{name} is not a mesh geom")
        if geom_id >= 0 and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            errors.append(f"{name} must not double as collision geometry")
        if geom_id >= 0:
            material_id = int(model.geom_matid[geom_id])
            has_texture = material_id >= 0 and bool(np.any(model.mat_texid[material_id] >= 0))
            if not has_texture:
                errors.append(f"{name} is not bound to a texture")
    for name in entry["manipulated_free_joints"]:
        joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0 and int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            errors.append(f"{name} must be a physical free joint")
    for joint_name, geom_name in entry["object_collision_geoms"].items():
        joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if joint_id < 0 or geom_id < 0:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        if int(model.geom_bodyid[geom_id]) != body_id:
            errors.append(f"{geom_name} is not attached to {joint_name}'s body")
        if not model.geom_contype[geom_id] or not model.geom_conaffinity[geom_id]:
            errors.append(f"{geom_name} has collision disabled")
        if model.body_mass[body_id] <= 0 or not np.all(model.body_inertia[body_id] > 0):
            errors.append(f"{joint_name}'s body has invalid mass or inertia")
    drawer_name = entry.get("articulated_furniture_joint")
    drawer_unactuated = None
    if drawer_name:
        joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, drawer_name)
        drawer_unactuated = joint_id >= 0 and joint_id not in _actuated_joint_ids(model)
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            errors.append("articulated furniture joint is not a slide joint")
        elif not drawer_unactuated:
            errors.append("articulated furniture has an actuator; contact-only interaction required")
        elif float(np.ptp(model.jnt_range[joint_id])) < 0.3:
            errors.append("articulated furniture travel is less than 0.3 m")
        handle_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, entry["articulated_handle_geom"])
        if handle_id < 0 or int(model.geom_bodyid[handle_id]) != int(model.jnt_bodyid[joint_id]):
            errors.append("drawer handle is not physically attached to the sliding body")
    settled_speeds = _settled_object_speeds(model, entry["manipulated_free_joints"])
    unstable = {name: speed for name, speed in settled_speeds.items() if speed >= 0.02}
    if unstable:
        errors.append(f"objects are not stable after 2 seconds: {unstable}")
    return {
        "scene_id": entry["scene_id"],
        "task_id": entry["task_id"],
        "model": entry["model"],
        "valid": not errors,
        "errors": errors,
        "compiled": {
            "bodies": int(model.nbody),
            "geoms": int(model.ngeom),
            "meshes": int(model.nmesh),
            "textures": int(model.ntex),
            "lights": int(model.nlight),
            "cameras": int(model.ncam),
            "joints": int(model.njnt),
            "actuators": int(model.nu),
            "equality_constraints": int(model.neq),
        },
        "furniture_count": len(entry["furniture_bodies"]),
        "manipulated_object_count": len(entry["manipulated_free_joints"]),
        "separate_visual_collision": not any(
            "double as collision" in error for error in errors
        ),
        "articulated_furniture_unactuated": drawer_unactuated,
        "reset_object_speeds_after_2s": settled_speeds,
    }


def verify(catalog_path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    reports = [_scene_report(entry) for entry in catalog["scenes"]]
    errors = [
        f"{report['scene_id']}: {error}"
        for report in reports
        for error in report["errors"]
    ]
    result = {
        "schema_version": "hwr.formal-scene-verification/v1",
        "valid": not errors and len(reports) == 3,
        "scene_count": len(reports),
        "reports": reports,
        "errors": errors,
    }
    if not result["valid"]:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    print(json.dumps(verify(args.catalog.resolve()), indent=2))


if __name__ == "__main__":
    main()
