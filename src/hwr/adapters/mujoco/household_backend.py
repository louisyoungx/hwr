"""Formal multi-object household runtime behind the project RuntimeBackend contract."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np

from hwr.adapters.mujoco.backend import Mujoco3DBackend, Mujoco3DConfig
from hwr.adapters.mujoco.bindings import MujocoTaskBinding
from hwr.adapters.mujoco.contact import GraspContactMonitor
from hwr.core.types import ActionFrame, CameraFrame, EpisodeResult, ObservationFrame
from hwr.eval import (
    MultiObjectStabilityCriterion,
    PlacementSample,
    StabilityConfig,
    TargetVolume,
)
from hwr.scenarios.formal3d import Formal3DTaskSpec


@dataclass(frozen=True)
class HouseholdEntityIds:
    object_bodies: dict[str, int]
    object_joints: dict[str, int]
    object_geoms: dict[str, int]
    target_sites: dict[str, int]
    articulation_joint: int | None
    robot_geoms: frozenset[int]
    allowed_contact_geoms: frozenset[int]


@dataclass(frozen=True)
class PhysicalDefaults:
    body_mass: dict[str, float]
    body_inertia: dict[str, np.ndarray]
    geom_friction: dict[str, np.ndarray]
    light_diffuse: np.ndarray
    material_rgba: np.ndarray


def _entity_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    entity_id = int(mujoco.mj_name2id(model, kind, name))
    if entity_id < 0:
        raise ValueError(f"formal scene is missing {name}")
    return entity_id


class MujocoHouseholdBackend(Mujoco3DBackend):
    """Task runtime that exposes pixels/proprioception and keeps truth adapter-private."""

    def __init__(
        self,
        task: Formal3DTaskSpec,
        binding: MujocoTaskBinding,
        *,
        camera_width: int = 128,
        camera_height: int = 96,
        severe_force_threshold: float = 220.0,
    ) -> None:
        if task.task_id != binding.task_id:
            raise ValueError("task and MuJoCo binding IDs differ")
        if {obj.object_id for obj in task.objects} != set(binding.objects):
            raise ValueError("task objects and MuJoCo object bindings differ")
        self.task = task
        self.binding = binding
        self.severe_force_threshold = severe_force_threshold
        super().__init__(
            Mujoco3DConfig(
                model_path=binding.model_path,
                task_id=task.task_id,
                control_hz=task.control_hz,
                max_steps=task.max_steps,
                camera_width=camera_width,
                camera_height=camera_height,
                primary_object_joint_name=None,
            )
        )
        self.household_ids = self._resolve_household_ids()
        self._defaults = self._capture_defaults()
        stability = StabilityConfig(
            control_hz=task.control_hz,
            hold_seconds=task.hold_seconds,
        )
        self._placement = MultiObjectStabilityCriterion(
            tuple(obj.object_id for obj in task.objects), stability
        )
        self._contact_monitors = {
            object_id: GraspContactMonitor(self.model, object_geom=value.collision_geom)
            for object_id, value in binding.objects.items()
        }
        self._drawer_contact_monitor = (
            GraspContactMonitor(self.model, object_geom=binding.articulation.handle_geom)
            if binding.articulation is not None
            else None
        )
        self._episode_seed = 0
        self._randomization: dict[str, Any] = {}
        self._severe_collision_count = 0
        self._maximum_forbidden_force = 0.0
        self._maximum_forbidden_pair: tuple[str, str] | None = None
        self._bilateral_contact_steps = {object_id: 0 for object_id in binding.objects}
        self._drawer_bilateral_contact_steps = 0

    def reset(self, *, seed: int, task_id: str) -> ObservationFrame:
        self._episode_seed = seed
        self._placement.reset()
        self._severe_collision_count = 0
        self._maximum_forbidden_force = 0.0
        self._maximum_forbidden_pair = None
        self._bilateral_contact_steps = {object_id: 0 for object_id in self.binding.objects}
        self._drawer_bilateral_contact_steps = 0
        self._prepare_model_randomization(seed)
        return super().reset(seed=seed, task_id=task_id)

    def _resolve_household_ids(self) -> HouseholdEntityIds:
        bodies: dict[str, int] = {}
        joints: dict[str, int] = {}
        geoms: dict[str, int] = {}
        sites: dict[str, int] = {}
        for object_id, value in self.binding.objects.items():
            bodies[object_id] = _entity_id(self.model, mujoco.mjtObj.mjOBJ_BODY, value.body)
            joints[object_id] = _entity_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, value.joint)
            geoms[object_id] = _entity_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, value.collision_geom)
            sites[object_id] = _entity_id(self.model, mujoco.mjtObj.mjOBJ_SITE, value.target_site)
        articulation_joint = (
            _entity_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.binding.articulation.joint)
            if self.binding.articulation
            else None
        )
        robot_root = self.bundle.ids.base_body
        robot_geoms = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.body_rootid[self.model.geom_bodyid[geom_id]]) == robot_root
        )
        allowed = frozenset(
            _entity_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.binding.allowed_robot_contact_geoms
        )
        return HouseholdEntityIds(
            bodies, joints, geoms, sites, articulation_joint, robot_geoms, allowed
        )

    def _capture_defaults(self) -> PhysicalDefaults:
        return PhysicalDefaults(
            body_mass={
                object_id: float(self.model.body_mass[body_id])
                for object_id, body_id in self.household_ids.object_bodies.items()
            },
            body_inertia={
                object_id: self.model.body_inertia[body_id].copy()
                for object_id, body_id in self.household_ids.object_bodies.items()
            },
            geom_friction={
                object_id: self.model.geom_friction[geom_id].copy()
                for object_id, geom_id in self.household_ids.object_geoms.items()
            },
            light_diffuse=self.model.light_diffuse.copy(),
            material_rgba=self.model.mat_rgba.copy(),
        )

    def _reset_base(self) -> None:
        x, y, z, yaw = self.task.initial_base.sample(self._rng)
        joint_id = self.bundle.ids.base_joint
        qpos = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qpos : qpos + 7] = (
            x,
            y,
            z,
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        )
        dof = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dof : dof + 6] = 0.0

    def _reset_object(self) -> None:
        for object_spec in self.task.objects:
            object_id = object_spec.object_id
            joint_id = self.household_ids.object_joints[object_id]
            x, y, z, yaw = object_spec.reset.sample(self._rng)
            qpos = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos : qpos + 7] = (
                x,
                y,
                z,
                math.cos(yaw / 2),
                0.0,
                0.0,
                math.sin(yaw / 2),
            )
            dof = int(self.model.jnt_dofadr[joint_id])
            self.data.qvel[dof : dof + 6] = 0.0
            self._randomization["objects"][object_id]["pose"] = [x, y, z, yaw]
        self._reset_articulation()

    def _prepare_model_randomization(self, seed: int) -> None:
        rng = random.Random(seed ^ 0x5A17C0DE)
        spec = self.task.randomization
        values: dict[str, Any] = {"objects": {}}
        for object_spec in self.task.objects:
            object_id = object_spec.object_id
            mass_scale = spec.mass_scale.sample(rng)
            friction_scale = spec.friction_scale.sample(rng)
            body_id = self.household_ids.object_bodies[object_id]
            geom_id = self.household_ids.object_geoms[object_id]
            self.model.body_mass[body_id] = self._defaults.body_mass[object_id] * mass_scale
            self.model.body_inertia[body_id] = self._defaults.body_inertia[object_id] * mass_scale
            self.model.geom_friction[geom_id] = (
                self._defaults.geom_friction[object_id] * friction_scale
            )
            values["objects"][object_id] = {
                "mass_scale": mass_scale,
                "friction_scale": friction_scale,
            }
        values.update(self._reset_visual_randomization(rng))
        self._randomization = values
        mujoco.mj_setConst(self.model, self.data)

    def _reset_articulation(self) -> None:
        joint_id = self.household_ids.articulation_joint
        if joint_id is None:
            return
        qpos = int(self.model.jnt_qposadr[joint_id])
        dof = int(self.model.jnt_dofadr[joint_id])
        self.data.qpos[qpos] = 0.0
        self.data.qvel[dof] = 0.0

    def _reset_visual_randomization(self, rng: random.Random) -> dict[str, float]:
        spec = self.task.randomization
        light_scale = spec.light_scale.sample(rng)
        material_tint = spec.material_tint.sample(rng)
        self.model.light_diffuse[:] = np.clip(
            self._defaults.light_diffuse * light_scale, 0.0, 1.0
        )
        self.model.mat_rgba[:] = self._defaults.material_rgba
        self.model.mat_rgba[:, :3] = np.clip(
            self._defaults.material_rgba[:, :3] * material_tint, 0.0, 1.0
        )
        return {
            "light_scale": light_scale,
            "material_tint": material_tint,
            "rgb_noise_std": spec.rgb_noise_std.sample(rng),
            "depth_dropout": spec.depth_dropout.sample(rng),
        }

    def _after_physics_substep(self) -> None:
        for object_id, monitor in self._contact_monitors.items():
            self._bilateral_contact_steps[object_id] += int(monitor.sample(self.data).bilateral)
        if self._drawer_contact_monitor is not None:
            self._drawer_bilateral_contact_steps += int(
                self._drawer_contact_monitor.sample(self.data).bilateral
            )
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = (int(contact.geom1), int(contact.geom2))
            robot_first = pair[0] in self.household_ids.robot_geoms
            robot_second = pair[1] in self.household_ids.robot_geoms
            if robot_first == robot_second:
                continue
            other = pair[1] if robot_first else pair[0]
            if other in self.household_ids.allowed_contact_geoms:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = abs(float(force[0]))
            if normal_force > self._maximum_forbidden_force:
                self._maximum_forbidden_force = normal_force
                self._maximum_forbidden_pair = tuple(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    or f"geom_{geom_id}"
                    for geom_id in pair
                )
            if normal_force >= self.severe_force_threshold:
                self._severe_collision_count += 1

    def _task_result_after_step(self) -> EpisodeResult | None:
        articulation_ok = self._articulation_satisfied()
        if not articulation_ok:
            self._placement.reset()
        samples = {
            object_id: self._placement_sample(object_id)
            for object_id in self.household_ids.object_bodies
        }
        stable = self._placement.update(samples) if articulation_ok else False
        if stable and self._severe_collision_count == 0:
            return EpisodeResult(
                success=True,
                reason="all_objects_physically_stable",
                ended_at_ns=self._timestamp_ns(),
                metrics=self._metrics(),
            )
        return None

    def _articulation_satisfied(self) -> bool:
        requirement = self.task.articulation
        joint_id = self.household_ids.articulation_joint
        if requirement is None:
            return True
        if joint_id is None:
            return False
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]]) >= requirement.minimum_position

    def _placement_sample(self, object_id: str) -> PlacementSample:
        geom_id = self.household_ids.object_geoms[object_id]
        joint_id = self.household_ids.object_joints[object_id]
        dof = int(self.model.jnt_dofadr[joint_id])
        position = tuple(float(value) for value in self.data.geom_xpos[geom_id])
        velocity = self.data.qvel[dof : dof + 6]
        site_id = self.household_ids.target_sites[object_id]
        center = self.data.site_xpos[site_id]
        size = self.model.site_size[site_id]
        target = TargetVolume(
            float(center[0] - size[0]),
            float(center[0] + size[0]),
            float(center[1] - size[1]),
            float(center[1] + size[1]),
            float(center[2] - size[2]),
            float(center[2] + size[2]),
        )
        return PlacementSample(
            position=position,
            linear_velocity=tuple(float(value) for value in velocity[:3]),
            angular_velocity=tuple(float(value) for value in velocity[3:6]),
            target=target,
        )

    def _metrics(self) -> dict[str, float]:
        articulation = 1.0 if self._articulation_satisfied() else 0.0
        return {
            "steps": float(self._steps),
            "stable_steps": float(self._placement.stable_steps),
            "severe_collisions": float(self._severe_collision_count),
            "maximum_forbidden_force": self._maximum_forbidden_force,
            "articulation_satisfied": articulation,
            "drawer_bilateral_contact_steps": float(self._drawer_bilateral_contact_steps),
            **{
                f"bilateral_contact_steps.{name}": float(value)
                for name, value in self._bilateral_contact_steps.items()
            },
        }

    def _timeout_result(self) -> EpisodeResult:
        return EpisodeResult(
            success=False,
            reason="formal_task_timeout",
            ended_at_ns=self._timestamp_ns(),
            metrics=self._metrics(),
        )

    def _observation(self) -> ObservationFrame:
        observation = super()._observation()
        if not self._randomization:
            return observation
        cameras = tuple(self._sensor_noise(frame) for frame in observation.cameras)
        return replace(observation, cameras=cameras, features={}, task_stage="instruction_following")

    def _sensor_noise(self, frame: CameraFrame) -> CameraFrame:
        if frame.payload is None:
            return frame
        camera_index = {"head_rgb": 1, "head_depth": 2, "wrist_rgb": 3}[frame.camera_id]
        rng = np.random.default_rng(
            np.random.SeedSequence([self._episode_seed, self._sequence, camera_index])
        )
        if frame.encoding == "rgb8":
            pixels = np.frombuffer(frame.payload, dtype=np.uint8).astype(np.float32)
            std = float(self._randomization["rgb_noise_std"])
            payload = np.clip(pixels + rng.normal(0.0, std, pixels.shape), 0, 255).astype(np.uint8)
        else:
            payload = np.frombuffer(frame.payload, dtype=np.float32).copy()
            dropout = float(self._randomization["depth_dropout"])
            payload[rng.random(payload.shape) < dropout] = 0.0
        return replace(frame, payload=np.ascontiguousarray(payload).tobytes())

    def audit_snapshot(self) -> dict[str, Any]:
        objects: dict[str, Any] = {}
        for object_id in self.household_ids.object_bodies:
            sample = self._placement_sample(object_id)
            objects[object_id] = {
                "position": list(sample.position),
                "target": [
                    sample.target.min_x,
                    sample.target.max_x,
                    sample.target.min_y,
                    sample.target.max_y,
                    sample.target.min_z,
                    sample.target.max_z,
                ],
                "inside_target": sample.target.contains(sample.position),
                "bilateral_contact_steps": self._bilateral_contact_steps[object_id],
            }
        return {
            "seed": self._episode_seed,
            "randomization": self._randomization,
            "objects": objects,
            "articulation_satisfied": self._articulation_satisfied(),
            "drawer_bilateral_contact_steps": self._drawer_bilateral_contact_steps,
            "severe_collision_count": self._severe_collision_count,
            "maximum_forbidden_force": self._maximum_forbidden_force,
            "maximum_forbidden_pair": self._maximum_forbidden_pair,
            "stable_steps": self._placement.stable_steps,
        }
