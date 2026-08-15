"""Canonical dual-arm runtime for the formal multi-object household scenes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np

from hwr.adapters.mujoco.bindings import MujocoTaskBinding
from hwr.adapters.mujoco.contact import GraspContactMonitor
from hwr.adapters.mujoco.dual_arm_backend import (
    MujocoDualArmBackend,
    MujocoDualArmConfig,
)
from hwr.core.embodied import (
    DUAL_ARM_ACTION_MAXIMUM,
    DUAL_ARM_ACTION_MINIMUM,
    DualArmAction,
    DualArmActionFrame,
    DualArmObservation,
    NaturalLanguageInstruction,
)
from hwr.core.runtime import LegalEnvironmentTransform, RuntimeStepOutcome
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.core.types import CameraFrame, EpisodeResult
from hwr.eval import (
    MultiObjectStabilityCriterion,
    PlacementSample,
    StabilityConfig,
    TargetVolume,
)
from hwr.scenarios.formal3d import Formal3DTaskSpec


@dataclass(frozen=True)
class FormalHouseholdEntityIds:
    object_bodies: dict[str, int]
    object_joints: dict[str, int]
    object_geoms: dict[str, int]
    target_sites: dict[str, int]
    articulation_joint: int | None
    robot_geoms: frozenset[int]
    allowed_contact_geoms: frozenset[int]


@dataclass(frozen=True)
class FormalHouseholdDefaults:
    body_mass: dict[str, float]
    body_inertia: dict[str, np.ndarray]
    geom_friction: dict[str, np.ndarray]
    light_diffuse: np.ndarray
    material_rgba: np.ndarray
    camera_position: dict[str, np.ndarray]
    camera_quaternion: dict[str, np.ndarray]
    camera_fovy: dict[str, float]


_ACTOR_CAMERA_NAMES = {
    "head_rgb": "head_rgb",
    "head_depth": "head_depth",
    "left_wrist_rgb": "left_wrist_rgb",
    "right_wrist_rgb": "wrist_rgb",
}


def _entity_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise ValueError(f"formal dual-arm scene is missing {name}")
    return value


class MujocoFormalHouseholdDualArmBackend(MujocoDualArmBackend):
    """Train and evaluate formal household tasks through the 16-D contract."""

    def __init__(
        self,
        task: Formal3DTaskSpec,
        binding: MujocoTaskBinding,
        *,
        camera_width: int = 256,
        camera_height: int = 192,
        evaluation_profile: bool = False,
        severe_force_threshold: float = 220.0,
    ) -> None:
        if task.task_id != binding.task_id:
            raise ValueError("formal dual-arm task and binding IDs differ")
        if {item.object_id for item in task.objects} != set(binding.objects):
            raise ValueError("formal dual-arm task objects differ from bindings")
        self.task = task
        self.binding = binding
        self.evaluation_profile = bool(evaluation_profile)
        self.severe_force_threshold = float(severe_force_threshold)
        super().__init__(
            MujocoDualArmConfig(
                model_path=binding.model_path,
                task_id=task.task_id,
                instruction_text=task.instruction,
                control_hz=task.control_hz,
                max_steps=task.max_steps,
                camera_width=camera_width,
                camera_height=camera_height,
                primary_object_joint_name=None,
            )
        )
        self.household_ids = self._resolve_household_ids()
        self._defaults = self._capture_defaults()
        self._placement = MultiObjectStabilityCriterion(
            tuple(item.object_id for item in task.objects),
            StabilityConfig(task.control_hz, task.hold_seconds),
        )
        self._left_monitors = self._grasp_monitors(left=True)
        self._right_monitors = self._grasp_monitors(left=False)
        self._episode_seed = 0
        self._randomization: dict[str, Any] = {}
        self._severe_collision_count = 0
        self._maximum_forbidden_force = 0.0
        self._maximum_forbidden_pair: tuple[str, str] | None = None
        self._left_contact_steps = 0
        self._right_contact_steps = 0
        self._simultaneous_contact_steps = 0
        self._concurrent_steps = 0
        self._maximum_concurrent_steps = 0
        self._initial_target_distance = 0.0
        self._maximum_controlled_target_progress = 0.0
        self._maximum_controlled_articulation_progress = 0.0
        self._previous_potential = 0.0
        self._step_left_contact = False
        self._step_right_contact = False
        self._action_queue: list[DualArmAction] = []
        self._observation_queue: list[DualArmObservation] = []

    def reset(
        self,
        *,
        seed: int,
        task_id: str,
        initial_state: PhysicalStateSnapshot | None = None,
    ) -> DualArmObservation:
        self._episode_seed = seed
        self._reset_evidence()
        self._prepare_model_randomization(seed)
        self._instruction = NaturalLanguageInstruction(
            self.task.instruction_for_seed(seed, evaluation=self.evaluation_profile)
        )
        observation = super().reset(
            seed=seed, task_id=task_id, initial_state=initial_state
        )
        self._placement.reset()
        self._initial_target_distance = max(self._target_distance(), 1.0e-6)
        self._previous_potential = self._task_potential()
        return self._delay_observation(observation)

    def apply(self, frame: DualArmActionFrame) -> RuntimeStepOutcome:
        self._step_left_contact = False
        self._step_right_contact = False
        plant_frame = replace(
            frame,
            action=self._delayed_scaled_action(frame.action),
            source=f"plant:{frame.source}",
        )
        outcome = super().apply(plant_frame)
        info = {
            **outcome.info,
            "plant_action_latency_steps": self._randomization["action_latency_steps"],
            "plant_actuator_scale": self._randomization["actuator_scale"],
        }
        return replace(
            outcome,
            observation=self._delay_observation(outcome.observation),
            info=info,
        )

    def legal_environment_transforms(self) -> tuple[LegalEnvironmentTransform, ...]:
        return ()

    def _reset_evidence(self) -> None:
        self._placement.reset()
        self._severe_collision_count = 0
        self._maximum_forbidden_force = 0.0
        self._maximum_forbidden_pair = None
        self._left_contact_steps = 0
        self._right_contact_steps = 0
        self._simultaneous_contact_steps = 0
        self._concurrent_steps = 0
        self._maximum_concurrent_steps = 0
        self._maximum_controlled_target_progress = 0.0
        self._maximum_controlled_articulation_progress = 0.0
        self._action_queue = []
        self._observation_queue = []

    def _resolve_household_ids(self) -> FormalHouseholdEntityIds:
        bodies = {
            name: _entity_id(self.model, mujoco.mjtObj.mjOBJ_BODY, value.body)
            for name, value in self.binding.objects.items()
        }
        joints = {
            name: _entity_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, value.joint)
            for name, value in self.binding.objects.items()
        }
        geoms = {
            name: _entity_id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, value.collision_geom
            )
            for name, value in self.binding.objects.items()
        }
        sites = {
            name: _entity_id(self.model, mujoco.mjtObj.mjOBJ_SITE, value.target_site)
            for name, value in self.binding.objects.items()
        }
        articulation_joint = (
            _entity_id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                self.binding.articulation.joint,
            )
            if self.binding.articulation
            else None
        )
        robot_root = self.bundle.ids.base_body
        robot_geoms = frozenset(
            index
            for index in range(self.model.ngeom)
            if int(self.model.body_rootid[self.model.geom_bodyid[index]]) == robot_root
        )
        allowed = frozenset(
            _entity_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.binding.allowed_robot_contact_geoms
        )
        return FormalHouseholdEntityIds(
            bodies, joints, geoms, sites, articulation_joint, robot_geoms, allowed
        )

    def _capture_defaults(self) -> FormalHouseholdDefaults:
        ids = self.household_ids
        return FormalHouseholdDefaults(
            {
                name: float(self.model.body_mass[value])
                for name, value in ids.object_bodies.items()
            },
            {
                name: self.model.body_inertia[value].copy()
                for name, value in ids.object_bodies.items()
            },
            {
                name: self.model.geom_friction[value].copy()
                for name, value in ids.object_geoms.items()
            },
            self.model.light_diffuse.copy(),
            self.model.mat_rgba.copy(),
            {
                actor_name: self.model.cam_pos[self._camera_id(model_name)].copy()
                for actor_name, model_name in _ACTOR_CAMERA_NAMES.items()
            },
            {
                actor_name: self.model.cam_quat[self._camera_id(model_name)].copy()
                for actor_name, model_name in _ACTOR_CAMERA_NAMES.items()
            },
            {
                actor_name: float(self.model.cam_fovy[self._camera_id(model_name)])
                for actor_name, model_name in _ACTOR_CAMERA_NAMES.items()
            },
        )

    def _camera_id(self, name: str) -> int:
        return _entity_id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    def _grasp_monitors(self, *, left: bool) -> dict[str, GraspContactMonitor]:
        prefix = "left" if left else "right"
        return {
            name: GraspContactMonitor(
                self.model,
                object_geom=value.collision_geom,
                left_pad=f"{prefix}_gripper_left_pad",
                right_pad=f"{prefix}_gripper_right_pad",
            )
            for name, value in self.binding.objects.items()
        }

    def _reset_base(self) -> None:
        x, y, z, yaw = self.task.initial_base.sample(self._rng)
        joint = self.bundle.ids.base_joint
        qpos = int(self.model.jnt_qposadr[joint])
        self.data.qpos[qpos : qpos + 7] = (
            x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
        )
        dof = int(self.model.jnt_dofadr[joint])
        self.data.qvel[dof : dof + 6] = 0.0

    def _reset_object(self) -> None:
        for item in self.task.objects:
            joint = self.household_ids.object_joints[item.object_id]
            x, y, z, yaw = item.reset.sample(self._rng)
            qpos = int(self.model.jnt_qposadr[joint])
            self.data.qpos[qpos : qpos + 7] = (
                x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
            )
            dof = int(self.model.jnt_dofadr[joint])
            self.data.qvel[dof : dof + 6] = 0.0
            self._randomization["objects"][item.object_id]["pose"] = [x, y, z, yaw]
        joint = self.household_ids.articulation_joint
        if joint is not None:
            self.data.qpos[self.model.jnt_qposadr[joint]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[joint]] = 0.0

    def _prepare_model_randomization(self, seed: int) -> None:
        rng = random.Random(seed ^ 0x5A17C0DE)
        spec = (
            self.task.evaluation_randomization
            if self.evaluation_profile
            else self.task.randomization
        )
        values: dict[str, Any] = {
            "profile": "evaluation" if self.evaluation_profile else "train",
            "objects": {},
        }
        for item in self.task.objects:
            name = item.object_id
            mass = spec.mass_scale.sample(rng)
            friction = spec.friction_scale.sample(rng)
            body = self.household_ids.object_bodies[name]
            geom = self.household_ids.object_geoms[name]
            self.model.body_mass[body] = self._defaults.body_mass[name] * mass
            self.model.body_inertia[body] = self._defaults.body_inertia[name] * mass
            self.model.geom_friction[geom] = self._defaults.geom_friction[name] * friction
            values["objects"][name] = {"mass_scale": mass, "friction_scale": friction}
        light = spec.light_scale.sample(rng)
        tint = spec.material_tint.sample(rng)
        self.model.light_diffuse[:] = np.clip(
            self._defaults.light_diffuse * light, 0.0, 1.0
        )
        self.model.mat_rgba[:] = self._defaults.material_rgba
        self.model.mat_rgba[:, :3] = np.clip(
            self._defaults.material_rgba[:, :3] * tint, 0.0, 1.0
        )
        focal_scale = spec.focal_scale.sample(rng)
        values.update(
            {
                "light_scale": light,
                "material_tint": tint,
                "rgb_noise_std": spec.rgb_noise_std.sample(rng),
                "depth_dropout": spec.depth_dropout.sample(rng),
                "depth_noise_std_m": spec.depth_noise_std_m.sample(rng),
                "focal_scale": focal_scale,
                "actuator_scale": spec.actuator_scale.sample(rng),
                "action_latency_steps": int(
                    round(spec.action_latency_steps.sample(rng))
                ),
                "observation_latency_steps": int(
                    round(spec.observation_latency_steps.sample(rng))
                ),
                "camera_perturbations": self._randomize_cameras(
                    rng, spec, focal_scale
                ),
            }
        )
        self._randomization = values
        mujoco.mj_setConst(self.model, self.data)

    def _randomize_cameras(
        self, rng: random.Random, spec, focal_scale: float
    ) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for actor_name, model_name in _ACTOR_CAMERA_NAMES.items():
            translation = [
                math.copysign(spec.camera_translation_m.sample(rng), rng.uniform(-1, 1))
                for _ in range(3)
            ]
            yaw = math.copysign(spec.camera_rotation_radians.sample(rng), rng.uniform(-1, 1))
            camera_id = self._camera_id(model_name)
            self.model.cam_pos[camera_id] = (
                self._defaults.camera_position[actor_name] + translation
            )
            yaw_quaternion = np.asarray(
                (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
            )
            self.model.cam_quat[camera_id] = _quaternion_product(
                self._defaults.camera_quaternion[actor_name], yaw_quaternion
            )
            default_fovy = math.radians(self._defaults.camera_fovy[actor_name])
            self.model.cam_fovy[camera_id] = math.degrees(
                2.0 * math.atan(math.tan(default_fovy / 2.0) / focal_scale)
            )
            result[actor_name] = [*translation, yaw]
        return result

    def _delayed_scaled_action(self, action: DualArmAction) -> DualArmAction:
        scale = float(self._randomization["actuator_scale"])
        values = np.asarray(action.vector(), np.float64)
        values[:14] *= scale
        values = np.clip(values, DUAL_ARM_ACTION_MINIMUM, DUAL_ARM_ACTION_MAXIMUM)
        scaled = DualArmAction.from_vector(values)
        delay = int(self._randomization["action_latency_steps"])
        self._action_queue.append(scaled)
        if len(self._action_queue) <= delay:
            left, right = self._gripper_positions()
            return DualArmAction(0.0, 0.0, (0.0,) * 6, (0.0,) * 6, left, right)
        return self._action_queue.pop(0)

    def _delay_observation(self, observation: DualArmObservation) -> DualArmObservation:
        delay = int(self._randomization.get("observation_latency_steps", 0))
        self._observation_queue.append(observation)
        if len(self._observation_queue) <= delay:
            return self._observation_queue[0]
        return self._observation_queue.pop(0)

    def _after_physics_substep(self) -> None:
        left = any(monitor.sample(self.data).bilateral for monitor in self._left_monitors.values())
        right = any(monitor.sample(self.data).bilateral for monitor in self._right_monitors.values())
        self._step_left_contact |= left
        self._step_right_contact |= right
        self._scan_forbidden_contacts()

    def _after_control_step(self, action: DualArmActionFrame) -> None:
        del action
        self._left_contact_steps += int(self._step_left_contact)
        self._right_contact_steps += int(self._step_right_contact)
        simultaneous = self._step_left_contact and self._step_right_contact
        self._simultaneous_contact_steps += int(simultaneous)
        self._concurrent_steps = self._concurrent_steps + 1 if simultaneous else 0
        self._maximum_concurrent_steps = max(
            self._maximum_concurrent_steps, self._concurrent_steps
        )
        if self._step_left_contact or self._step_right_contact:
            progress = max(0.0, self._initial_target_distance - self._target_distance())
            self._maximum_controlled_target_progress = max(
                self._maximum_controlled_target_progress, progress
            )
            self._maximum_controlled_articulation_progress = max(
                self._maximum_controlled_articulation_progress,
                self._articulation_position(),
            )

    def _scan_forbidden_contacts(self) -> None:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            robot_first = pair[0] in self.household_ids.robot_geoms
            robot_second = pair[1] in self.household_ids.robot_geoms
            if robot_first == robot_second:
                continue
            other = pair[1] if robot_first else pair[0]
            if other in self.household_ids.allowed_contact_geoms:
                continue
            force = np.zeros(6, np.float64)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal = abs(float(force[0]))
            if normal > self._maximum_forbidden_force:
                self._maximum_forbidden_force = normal
                self._maximum_forbidden_pair = tuple(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, value)
                    or f"geom_{value}"
                    for value in pair
                )
            self._severe_collision_count += int(normal >= self.severe_force_threshold)

    def _predictive_safety_enabled(self) -> bool:
        return True

    def _predictive_safety_violation(self) -> bool:
        return self._current_maximum_forbidden_force() >= self.severe_force_threshold

    def _predictive_horizon_control_steps(self) -> int:
        return 2

    def _current_maximum_forbidden_force(self) -> float:
        maximum = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            robot_first = first in self.household_ids.robot_geoms
            robot_second = second in self.household_ids.robot_geoms
            if robot_first == robot_second:
                continue
            other = second if robot_first else first
            if other in self.household_ids.allowed_contact_geoms:
                continue
            force = np.zeros(6, np.float64)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            maximum = max(maximum, abs(float(force[0])))
        return maximum

    def _task_result_after_step(self) -> EpisodeResult | None:
        if self._severe_collision_count > 0:
            return EpisodeResult(
                False,
                "severe_collision",
                self._timestamp_ns(),
                self._metrics(),
            )
        articulation_ok = self._articulation_satisfied()
        if not articulation_ok:
            self._placement.reset()
        samples = {
            name: self._placement_sample(name)
            for name in self.household_ids.object_bodies
        }
        stable = self._placement.update(samples) if articulation_ok else False
        required = math.ceil(
            self.task.minimum_each_arm_contact_seconds * self.task.control_hz
        )
        bimanual = (
            self._left_contact_steps >= required
            and self._right_contact_steps >= required
            and self._maximum_concurrent_steps >= required
        )
        if stable and bimanual and self._severe_collision_count == 0:
            return EpisodeResult(
                True,
                "formal_household_bimanual_success",
                self._timestamp_ns(),
                self._metrics(),
            )
        return None

    def _step_reward(self, result: EpisodeResult | None) -> float:
        potential = self._task_potential()
        improvement = potential - self._previous_potential
        self._previous_potential = potential
        success = float(result is not None and result.success)
        return float(np.clip(improvement, -0.05, 0.05) + success)

    def _timeout_result(self) -> EpisodeResult:
        return EpisodeResult(
            False, "formal_household_timeout", self._timestamp_ns(), self._metrics()
        )

    def _task_potential(self) -> float:
        target = 1.0 - min(1.0, self._target_distance() / self._initial_target_distance)
        requirement = self.task.articulation
        articulation = (
            min(1.0, self._articulation_position() / requirement.minimum_position)
            if requirement is not None
            else 1.0
        )
        return 0.8 * target + 0.2 * articulation

    def _target_distance(self) -> float:
        return sum(self._distance_to_target(name) for name in self.household_ids.object_bodies)

    def _distance_to_target(self, object_id: str) -> float:
        position = self.data.geom_xpos[self.household_ids.object_geoms[object_id]]
        site = self.household_ids.target_sites[object_id]
        center = self.data.site_xpos[site]
        size = self.model.site_size[site]
        delta = np.maximum(np.abs(position - center) - size, 0.0)
        return float(np.linalg.norm(delta))

    def _placement_sample(self, object_id: str) -> PlacementSample:
        geom = self.household_ids.object_geoms[object_id]
        joint = self.household_ids.object_joints[object_id]
        dof = int(self.model.jnt_dofadr[joint])
        position = tuple(float(value) for value in self.data.geom_xpos[geom])
        velocity = self.data.qvel[dof : dof + 6]
        site = self.household_ids.target_sites[object_id]
        center = self.data.site_xpos[site]
        size = self.model.site_size[site]
        target = TargetVolume(
            float(center[0] - size[0]), float(center[0] + size[0]),
            float(center[1] - size[1]), float(center[1] + size[1]),
            float(center[2] - size[2]), float(center[2] + size[2]),
        )
        return PlacementSample(
            position,
            tuple(float(value) for value in velocity[:3]),
            tuple(float(value) for value in velocity[3:6]),
            target,
        )

    def _articulation_position(self) -> float:
        joint = self.household_ids.articulation_joint
        return (
            max(0.0, float(self.data.qpos[self.model.jnt_qposadr[joint]]))
            if joint is not None
            else 0.0
        )

    def _articulation_satisfied(self) -> bool:
        requirement = self.task.articulation
        return requirement is None or self._articulation_position() >= requirement.minimum_position

    def _metrics(self) -> dict[str, float]:
        return {
            "steps": float(self._steps),
            "stable_steps": float(self._placement.stable_steps),
            "maximum_concurrent_steps": float(self._maximum_concurrent_steps),
            "left_contact_steps": float(self._left_contact_steps),
            "right_contact_steps": float(self._right_contact_steps),
            "simultaneous_contact_steps": float(self._simultaneous_contact_steps),
            "maximum_controlled_target_progress": self._maximum_controlled_target_progress,
            "maximum_controlled_articulation_progress": self._maximum_controlled_articulation_progress,
            "severe_collisions": float(self._severe_collision_count),
            "maximum_forbidden_force": self._maximum_forbidden_force,
            "articulation_satisfied": float(self._articulation_satisfied()),
        }

    def task_audit(self) -> dict[str, object]:
        return {
            "task_id": self.task.task_id,
            "randomization": self._randomization,
            "instruction_split": "evaluation" if self.evaluation_profile else "train",
            "instruction": self._instruction.text,
            "stable_steps": self._placement.stable_steps,
            "maximum_concurrent_steps": self._maximum_concurrent_steps,
            "left_contact_steps": self._left_contact_steps,
            "right_contact_steps": self._right_contact_steps,
            "simultaneous_contact_steps": self._simultaneous_contact_steps,
            "severe_collision_count": self._severe_collision_count,
            "maximum_forbidden_force": self._maximum_forbidden_force,
            "maximum_forbidden_pair": self._maximum_forbidden_pair,
            "metrics": self._metrics(),
        }

    def _observation(self) -> DualArmObservation:
        observation = super()._observation()
        if not self._randomization or not self._camera_rendering_enabled:
            return observation
        cameras = tuple(self._sensor_noise(frame) for frame in observation.cameras)
        return replace(observation, cameras=cameras)

    def _sensor_noise(self, frame: CameraFrame) -> CameraFrame:
        if frame.payload is None:
            return frame
        camera_index = {
            "head_rgb": 1,
            "head_depth": 2,
            "left_wrist_rgb": 3,
            "right_wrist_rgb": 4,
        }[frame.camera_id]
        rng = np.random.default_rng(
            np.random.SeedSequence([self._episode_seed, self._sequence, camera_index])
        )
        if frame.encoding == "rgb8":
            pixels = np.frombuffer(frame.payload, np.uint8).astype(np.float32)
            noise = rng.normal(0.0, self._randomization["rgb_noise_std"], pixels.shape)
            payload = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        else:
            payload = np.frombuffer(frame.payload, np.float32).copy()
            payload += rng.normal(
                0.0, self._randomization["depth_noise_std_m"], payload.shape
            ).astype(np.float32)
            dropout = self._randomization["depth_dropout"]
            payload[rng.random(payload.shape) < dropout] = 0.0
        return replace(frame, payload=np.ascontiguousarray(payload).tobytes())



def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    value = np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )
    return value / np.linalg.norm(value)
