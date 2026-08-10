"""Programmatic bimanual household tasks on the canonical MuJoCo runtime."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import mujoco
import numpy as np

from hwr.adapters.mujoco.bimanual_bindings import BimanualMujocoBinding
from hwr.adapters.mujoco.dual_arm_backend import (
    MujocoDualArmBackend,
    MujocoDualArmConfig,
)
from hwr.core.embodied import DualArmActionFrame, DualArmObservation
from hwr.core.runtime import RuntimeStepOutcome
from hwr.core.types import EpisodeEvent, EpisodeResult
from hwr.tasks import (
    BimanualTaskSample,
    BimanualTaskSpec,
    BimanualTaskTracker,
    PrivilegedTaskState,
    TaskUpdate,
)


BIMANUAL_READY_HOME = (0.0, 0.80, -0.40, 0.0, -0.40, 0.0)


def _entity_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    entity_id = int(mujoco.mj_name2id(model, kind, name))
    if entity_id < 0:
        raise ValueError(f"bimanual scene is missing {name}")
    return entity_id


@dataclass(frozen=True)
class BimanualEntityIds:
    payload_body: int
    payload_joint: int
    payload_geoms: frozenset[int]
    payload_reference_geom: int
    left_interaction_geom: int
    right_interaction_geom: int
    target_site: int
    target_support_geom: int
    articulation_joint: int | None
    left_pads: frozenset[int]
    right_pads: frozenset[int]
    left_grasp_site: int
    right_grasp_site: int
    robot_geoms: frozenset[int]
    allowed_robot_contacts: frozenset[int]


@dataclass(frozen=True)
class BimanualPhysicalDefaults:
    payload_mass: float
    payload_inertia: np.ndarray
    payload_friction: dict[int, np.ndarray]
    light_diffuse: np.ndarray
    material_rgba: np.ndarray


class MujocoBimanualTaskBackend(MujocoDualArmBackend):
    """Adds procedural truth, reward, and result tracking outside Actor observations."""

    def __init__(
        self,
        task: BimanualTaskSpec,
        binding: BimanualMujocoBinding,
        *,
        camera_width: int = 128,
        camera_height: int = 96,
        severe_force_threshold: float = 220.0,
    ) -> None:
        if task.task_id != binding.task_id:
            raise ValueError("bimanual task and MuJoCo binding IDs differ")
        self.task = task
        self.binding = binding
        self.severe_force_threshold = float(severe_force_threshold)
        self._curriculum_level = 1.0
        super().__init__(
            MujocoDualArmConfig(
                model_path=binding.model_path,
                task_id=task.task_id,
                instruction_text=task.instruction,
                control_hz=task.control_hz,
                max_steps=task.max_steps,
                camera_width=camera_width,
                camera_height=camera_height,
                primary_object_joint_name=binding.payload_joint,
                left_arm_home=BIMANUAL_READY_HOME,
                right_arm_home=BIMANUAL_READY_HOME,
            )
        )
        self.task_ids = self._resolve_task_ids()
        self._defaults = self._capture_physical_defaults()
        self.tracker = BimanualTaskTracker(task)
        self._last_sample: BimanualTaskSample | None = None
        self._last_update: TaskUpdate | None = None
        self._severe_collision_count = 0
        self._maximum_forbidden_force = 0.0
        self._episode_randomization: dict[str, float] = {}

    def reset(self, *, seed: int, task_id: str) -> DualArmObservation:
        self._severe_collision_count = 0
        self._maximum_forbidden_force = 0.0
        self._last_update = None
        self._randomize_model(seed)
        observation = super().reset(seed=seed, task_id=task_id)
        self._last_sample = self._task_sample()
        self.tracker.reset(self._last_sample)
        return observation

    def set_curriculum_level(self, level: float) -> None:
        if not 0.0 <= level <= 1.0:
            raise ValueError("curriculum level must be in [0, 1]")
        self._curriculum_level = float(level)

    def apply(self, frame: DualArmActionFrame) -> RuntimeStepOutcome:
        outcome = super().apply(frame)
        self._scan_forbidden_contacts()
        sample = self._task_sample()
        update = self.tracker.update(sample)
        self._last_sample = sample
        self._last_update = update
        events = [*outcome.events]
        if update.terminated:
            reason = "bimanual_task_success" if update.success else "severe_collision"
            self._result = EpisodeResult(
                success=update.success,
                reason=reason,
                ended_at_ns=self._timestamp_ns(),
                metrics={**update.metrics, "steps": float(self._steps)},
            )
            events.append(
                EpisodeEvent(
                    timestamp_ns=self._timestamp_ns(),
                    event_type=reason,
                    source="bimanual_task",
                    details={"success": update.success},
                )
            )
        elif outcome.truncated:
            self._result = EpisodeResult(
                success=False,
                reason="bimanual_task_timeout",
                ended_at_ns=self._timestamp_ns(),
                metrics={**update.metrics, "steps": float(self._steps)},
            )
        return RuntimeStepOutcome(
            observation=outcome.observation,
            reward=update.reward,
            terminated=update.terminated,
            truncated=outcome.truncated and not update.terminated,
            events=tuple(events),
            info=outcome.info,
        )

    def privileged_training_state(self) -> PrivilegedTaskState:
        self._require_active()
        sample = self._last_sample or self._task_sample()
        update = self._last_update
        desired = update.desired_goal if update else self.tracker.desired_goal()
        metrics = update.metrics if update else self._initial_metrics(sample)
        base_pose, base_twist = self._base_state()
        critic_state = (
            *sample.achieved_goal(),
            *desired,
            sample.left_reach_distance,
            sample.right_reach_distance,
            *(target - actual for target, actual in zip(
                sample.target_position, sample.payload_position, strict=True
            )),
            *self._joint_values(self.bundle.ids.secondary_arm_joints, velocity=False),
            *self._joint_values(self.bundle.ids.arm_joints, velocity=False),
            *self._joint_values(self.bundle.ids.secondary_arm_joints, velocity=True),
            *self._joint_values(self.bundle.ids.arm_joints, velocity=True),
            *self._gripper_positions(),
            *base_pose,
            *base_twist,
        )
        return PrivilegedTaskState(
            critic_state=critic_state,
            achieved_goal=sample.achieved_goal(),
            desired_goal=desired,
            metrics=metrics,
        )

    def task_audit(self) -> dict[str, object]:
        state = self.privileged_training_state()
        return {
            "task_id": self.task.task_id,
            "objective": self.task.objective,
            "curriculum_level": self._curriculum_level,
            "randomization": self._episode_randomization,
            "stable_steps": self.tracker.stable_steps,
            "concurrent_steps": self.tracker.concurrent_steps,
            "maximum_concurrent_steps": self.tracker.maximum_concurrent_steps,
            "severe_collision_count": self._severe_collision_count,
            "maximum_forbidden_force": self._maximum_forbidden_force,
            "metrics": dict(state.metrics),
        }

    def _reset_base(self) -> None:
        pose = self.task.initial_base
        x, y, z, yaw = (
            self._sample_range(value)
            for value in (pose.x, pose.y, pose.z, pose.yaw)
        )
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
        pose = self.task.payload_reset
        x, y, z, yaw = (
            self._sample_range(value)
            for value in (pose.x, pose.y, pose.z, pose.yaw)
        )
        joint_id = self.task_ids.payload_joint
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
        if self.task_ids.articulation_joint is not None:
            joint = self.task_ids.articulation_joint
            self.data.qpos[self.model.jnt_qposadr[joint]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[joint]] = 0.0

    def _resolve_task_ids(self) -> BimanualEntityIds:
        model = self.model
        geom = mujoco.mjtObj.mjOBJ_GEOM
        left_pads = frozenset(
            _entity_id(model, geom, name)
            for name in ("left_gripper_left_pad", "left_gripper_right_pad")
        )
        right_pads = frozenset(
            _entity_id(model, geom, name)
            for name in ("right_gripper_left_pad", "right_gripper_right_pad")
        )
        robot_root = self.bundle.ids.base_body
        robot_geoms = frozenset(
            index
            for index in range(model.ngeom)
            if int(model.body_rootid[model.geom_bodyid[index]]) == robot_root
        )
        articulation = (
            _entity_id(model, mujoco.mjtObj.mjOBJ_JOINT, self.binding.articulation_joint)
            if self.binding.articulation_joint
            else None
        )
        return BimanualEntityIds(
            payload_body=_entity_id(model, mujoco.mjtObj.mjOBJ_BODY, self.binding.payload_body),
            payload_joint=_entity_id(model, mujoco.mjtObj.mjOBJ_JOINT, self.binding.payload_joint),
            payload_geoms=frozenset(
                _entity_id(model, geom, name) for name in self.binding.payload_geoms
            ),
            payload_reference_geom=_entity_id(model, geom, self.binding.payload_reference_geom),
            left_interaction_geom=_entity_id(model, geom, self.binding.left_interaction_geom),
            right_interaction_geom=_entity_id(model, geom, self.binding.right_interaction_geom),
            target_site=_entity_id(model, mujoco.mjtObj.mjOBJ_SITE, self.binding.target_site),
            target_support_geom=_entity_id(model, geom, self.binding.target_support_geom),
            articulation_joint=articulation,
            left_pads=left_pads,
            right_pads=right_pads,
            left_grasp_site=_entity_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_center"),
            right_grasp_site=_entity_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_center"),
            robot_geoms=robot_geoms,
            allowed_robot_contacts=frozenset(
                _entity_id(model, geom, name)
                for name in self.binding.allowed_robot_contact_geoms
            ),
        )

    def _capture_physical_defaults(self) -> BimanualPhysicalDefaults:
        body = self.task_ids.payload_body
        return BimanualPhysicalDefaults(
            payload_mass=float(self.model.body_mass[body]),
            payload_inertia=self.model.body_inertia[body].copy(),
            payload_friction={
                geom: self.model.geom_friction[geom].copy()
                for geom in self.task_ids.payload_geoms
            },
            light_diffuse=self.model.light_diffuse.copy(),
            material_rgba=self.model.mat_rgba.copy(),
        )

    def _randomize_model(self, seed: int) -> None:
        rng = random.Random(seed ^ 0xB14A2A1)
        randomization = self.task.randomization
        mass = self._scaled_random(rng, randomization.mass_scale)
        friction = self._scaled_random(rng, randomization.friction_scale)
        light = self._scaled_random(rng, randomization.light_scale)
        material = self._scaled_random(rng, randomization.material_scale)
        body = self.task_ids.payload_body
        self.model.body_mass[body] = self._defaults.payload_mass * mass
        self.model.body_inertia[body] = self._defaults.payload_inertia * mass
        for geom, default in self._defaults.payload_friction.items():
            self.model.geom_friction[geom] = default * friction
        self.model.light_diffuse[:] = np.clip(self._defaults.light_diffuse * light, 0, 1)
        self.model.mat_rgba[:] = self._defaults.material_rgba
        self.model.mat_rgba[:, :3] = np.clip(
            self._defaults.material_rgba[:, :3] * material, 0, 1
        )
        self._episode_randomization = {
            "mass_scale": mass,
            "friction_scale": friction,
            "light_scale": light,
            "material_scale": material,
        }
        mujoco.mj_setConst(self.model, self.data)

    def _sample_range(self, value) -> float:
        midpoint = (value.low + value.high) / 2
        sampled = self._rng.uniform(value.low, value.high)
        return midpoint + (sampled - midpoint) * self._curriculum_level

    def _scaled_random(self, rng: random.Random, value) -> float:
        sampled = rng.uniform(value.low, value.high)
        return 1.0 + (sampled - 1.0) * self._curriculum_level

    def _task_sample(self) -> BimanualTaskSample:
        ids = self.task_ids
        position = tuple(float(value) for value in self.data.geom_xpos[ids.payload_reference_geom])
        target = tuple(float(value) for value in self.data.site_xpos[ids.target_site])
        joint_dof = int(self.model.jnt_dofadr[ids.payload_joint])
        velocity = self.data.qvel[joint_dof : joint_dof + 6]
        rotation = self.data.xmat[ids.payload_body].reshape(3, 3)
        tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
        articulation_position, articulation_speed = self._articulation_state()
        return BimanualTaskSample(
            payload_position=position,
            target_position=target,
            payload_tilt_radians=tilt,
            payload_linear_speed=float(np.linalg.norm(velocity[:3])),
            payload_angular_speed=float(np.linalg.norm(velocity[3:])),
            left_reach_distance=self._site_geom_distance(
                ids.left_grasp_site, ids.left_interaction_geom
            ),
            right_reach_distance=self._site_geom_distance(
                ids.right_grasp_site, ids.right_interaction_geom
            ),
            left_contact=self._bilateral_contact(ids.left_pads, ids.left_interaction_geom),
            right_contact=self._bilateral_contact(ids.right_pads, ids.right_interaction_geom),
            support_contact=self._support_contact(),
            inside_target=self._inside_target(position),
            articulation_position=articulation_position,
            articulation_speed=articulation_speed,
            severe_collision_count=self._severe_collision_count,
        )

    def _contact_pairs(self) -> set[frozenset[int]]:
        return {
            frozenset((int(contact.geom1), int(contact.geom2)))
            for contact in self.data.contact[: self.data.ncon]
        }

    def _bilateral_contact(self, pads: frozenset[int], other: int) -> bool:
        pairs = self._contact_pairs()
        return all(frozenset((pad, other)) in pairs for pad in pads)

    def _support_contact(self) -> bool:
        pairs = self._contact_pairs()
        return any(
            frozenset((geom, self.task_ids.target_support_geom)) in pairs
            for geom in self.task_ids.payload_geoms
        )

    def _inside_target(self, position: tuple[float, ...]) -> bool:
        site = self.task_ids.target_site
        center = self.data.site_xpos[site]
        size = self.model.site_size[site]
        return all(
            abs(value - float(center[index])) <= float(size[index])
            for index, value in enumerate(position)
        )

    def _articulation_state(self) -> tuple[float, float]:
        joint = self.task_ids.articulation_joint
        if joint is None:
            return 0.0, 0.0
        return (
            float(self.data.qpos[self.model.jnt_qposadr[joint]]),
            float(self.data.qvel[self.model.jnt_dofadr[joint]]),
        )

    def _site_geom_distance(self, site: int, geom: int) -> float:
        return float(np.linalg.norm(self.data.site_xpos[site] - self.data.geom_xpos[geom]))

    def _scan_forbidden_contacts(self) -> None:
        ids = self.task_ids
        severe_this_step = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            robot_first = first in ids.robot_geoms
            robot_second = second in ids.robot_geoms
            if robot_first == robot_second:
                continue
            other = second if robot_first else first
            if other in ids.allowed_robot_contacts:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal = abs(float(force[0]))
            self._maximum_forbidden_force = max(self._maximum_forbidden_force, normal)
            severe_this_step = severe_this_step or normal >= self.severe_force_threshold
        self._severe_collision_count += int(severe_this_step)

    def _initial_metrics(self, sample: BimanualTaskSample) -> dict[str, float]:
        return {
            "target_distance": sample.target_distance,
            "stable_steps": 0.0,
            "maximum_concurrent_steps": 0.0,
            "severe_collisions": float(self._severe_collision_count),
        }
