"""Continuous goals and automatic criteria for physically bimanual chores."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


Objective = Literal["carry_payload", "hold_drawer_place"]
BIMANUAL_GOAL_DIM = 12


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class ScalarRange:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high) or self.low > self.high:
            raise ValueError("scalar range is invalid")


@dataclass(frozen=True)
class PoseRange:
    x: ScalarRange
    y: ScalarRange
    z: ScalarRange
    yaw: ScalarRange


@dataclass(frozen=True)
class SuccessCriteria:
    hold_seconds: float
    minimum_concurrent_seconds: float
    maximum_target_distance: float
    maximum_tilt_radians: float
    maximum_linear_speed: float
    maximum_angular_speed: float
    minimum_articulation_position: float = 0.0
    maximum_articulation_speed: float = 0.08

    def __post_init__(self) -> None:
        values = (
            self.hold_seconds,
            self.minimum_concurrent_seconds,
            self.maximum_target_distance,
            self.maximum_tilt_radians,
            self.maximum_linear_speed,
            self.maximum_angular_speed,
            self.maximum_articulation_speed,
        )
        if min(values) <= 0.0 or self.minimum_articulation_position < 0.0:
            raise ValueError("success criteria values must be positive")


@dataclass(frozen=True)
class RewardWeights:
    target_distance: float = 4.0
    left_reach: float = 1.5
    right_reach: float = 1.5
    worst_side_reach: float = 6.0
    tilt: float = 1.0
    articulation: float = 2.0
    contact: float = 1.0
    near_handle_closure: float = 1.5
    joint_grasp_readiness: float = 3.0
    bilateral_contact: float = 4.0
    support: float = 0.5
    progress_scale: float = 8.0
    step_cost: float = 0.002
    success: float = 50.0
    severe_collision: float = 50.0

    def __post_init__(self) -> None:
        if min(self.__dict__.values()) < 0.0 or self.success <= 0.0:
            raise ValueError("reward weights must be non-negative with positive success")


@dataclass(frozen=True)
class DomainRandomization:
    mass_scale: ScalarRange
    friction_scale: ScalarRange
    light_scale: ScalarRange
    material_scale: ScalarRange


@dataclass(frozen=True)
class BimanualTaskSpec:
    task_id: str
    scene_id: str
    instruction: str
    objective: Objective
    control_hz: float
    max_steps: int
    initial_base: PoseRange
    payload_reset: PoseRange
    target_position: tuple[float, float, float]
    criteria: SuccessCriteria
    reward: RewardWeights
    randomization: DomainRandomization
    schema_version: str = "hwr.bimanual-task/v1"

    def __post_init__(self) -> None:
        if not self.task_id or not self.scene_id or not " ".join(self.instruction.split()):
            raise ValueError("task, scene, and raw instruction are required")
        if self.objective not in ("carry_payload", "hold_drawer_place"):
            raise ValueError("unsupported bimanual objective")
        if self.control_hz <= 0.0 or self.max_steps <= 0:
            raise ValueError("task timing must be positive")
        target = _finite(self.target_position, "target position")
        if len(target) != 3:
            raise ValueError("target position requires three values")
        object.__setattr__(self, "instruction", " ".join(self.instruction.split()))
        object.__setattr__(self, "target_position", target)

    @property
    def hold_steps(self) -> int:
        return math.ceil(self.criteria.hold_seconds * self.control_hz)

    @property
    def concurrent_steps(self) -> int:
        return math.ceil(self.criteria.minimum_concurrent_seconds * self.control_hz)


@dataclass(frozen=True)
class BimanualTaskSample:
    payload_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    payload_tilt_radians: float
    payload_linear_speed: float
    payload_angular_speed: float
    left_reach_distance: float
    right_reach_distance: float
    left_contact: bool
    right_contact: bool
    support_contact: bool
    inside_target: bool
    articulation_position: float = 0.0
    articulation_speed: float = 0.0
    severe_collision_count: int = 0
    left_gripper_position: float = 1.0
    right_gripper_position: float = 1.0

    def __post_init__(self) -> None:
        for name in ("payload_position", "target_position"):
            values = _finite(getattr(self, name), name)
            if len(values) != 3:
                raise ValueError(f"{name} requires three values")
            object.__setattr__(self, name, values)
        scalars = (
            self.payload_tilt_radians,
            self.payload_linear_speed,
            self.payload_angular_speed,
            self.left_reach_distance,
            self.right_reach_distance,
            self.articulation_position,
            self.articulation_speed,
            self.left_gripper_position,
            self.right_gripper_position,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("task sample scalars must be finite")
        if min(scalars[:5]) < 0.0 or self.severe_collision_count < 0:
            raise ValueError("task distances, speeds, and collisions cannot be negative")
        if not all(0.0 <= value <= 1.0 for value in scalars[-2:]):
            raise ValueError("task gripper positions must be normalized")

    @property
    def target_distance(self) -> float:
        return math.dist(self.payload_position, self.target_position)

    def achieved_goal(self) -> tuple[float, ...]:
        return (
            *self.payload_position,
            self.payload_tilt_radians,
            self.payload_linear_speed,
            self.payload_angular_speed,
            self.articulation_position,
            float(self.left_contact),
            float(self.right_contact),
            float(self.support_contact),
            float(self.inside_target),
            float(self.severe_collision_count),
        )


@dataclass(frozen=True)
class TaskUpdate:
    reward: float
    success: bool
    terminated: bool
    stable_steps: int
    concurrent_steps: int
    maximum_concurrent_steps: int
    achieved_goal: tuple[float, ...]
    desired_goal: tuple[float, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class PrivilegedTaskState:
    """Training-only continuous state; never part of a deployable observation."""

    critic_state: tuple[float, ...]
    achieved_goal: tuple[float, ...]
    desired_goal: tuple[float, ...]
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in ("critic_state", "achieved_goal", "desired_goal"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if len(self.achieved_goal) != len(self.desired_goal):
            raise ValueError("achieved and desired goal dimensions differ")


class BimanualTaskTracker:
    """Stateful result/reward tracker with no prescribed action sequence."""

    def __init__(self, spec: BimanualTaskSpec) -> None:
        self.spec = spec
        self._previous_potential: float | None = None
        self.stable_steps = 0
        self.concurrent_steps = 0
        self.maximum_concurrent_steps = 0
        self.left_contact_steps = 0
        self.right_contact_steps = 0
        self.simultaneous_contact_steps = 0

    def reset(self, initial: BimanualTaskSample) -> None:
        self._previous_potential = self._potential(initial)
        self.stable_steps = 0
        self.concurrent_steps = 0
        self.maximum_concurrent_steps = 0
        self.left_contact_steps = 0
        self.right_contact_steps = 0
        self.simultaneous_contact_steps = 0

    def desired_goal(self) -> tuple[float, ...]:
        return self._desired_goal()

    def update(self, sample: BimanualTaskSample) -> TaskUpdate:
        if self._previous_potential is None:
            raise RuntimeError("task tracker must be reset before update")
        concurrent = sample.left_contact and sample.right_contact
        self.left_contact_steps += int(sample.left_contact)
        self.right_contact_steps += int(sample.right_contact)
        self.simultaneous_contact_steps += int(concurrent)
        self.concurrent_steps = self.concurrent_steps + 1 if concurrent else 0
        self.maximum_concurrent_steps = max(
            self.maximum_concurrent_steps, self.concurrent_steps
        )
        stable_now = self._stable(sample)
        self.stable_steps = self.stable_steps + 1 if stable_now else 0
        potential = self._potential(sample)
        reward = (
            (potential - self._previous_potential) * self.spec.reward.progress_scale
            - self.spec.reward.step_cost
        )
        self._previous_potential = potential
        severe = sample.severe_collision_count > 0
        bimanual = self.maximum_concurrent_steps >= self.spec.concurrent_steps
        success = self.stable_steps >= self.spec.hold_steps and bimanual and not severe
        if severe:
            reward = min(
                reward - self.spec.reward.severe_collision,
                -self.spec.reward.severe_collision,
            )
        if success:
            reward += self.spec.reward.success
        return TaskUpdate(
            reward=reward,
            success=success,
            terminated=success or severe,
            stable_steps=self.stable_steps,
            concurrent_steps=self.concurrent_steps,
            maximum_concurrent_steps=self.maximum_concurrent_steps,
            achieved_goal=sample.achieved_goal(),
            desired_goal=self._desired_goal(),
            metrics=self._metrics(sample),
        )

    def _stable(self, sample: BimanualTaskSample) -> bool:
        criteria = self.spec.criteria
        common = (
            sample.target_distance <= criteria.maximum_target_distance
            and sample.payload_tilt_radians <= criteria.maximum_tilt_radians
            and sample.payload_linear_speed <= criteria.maximum_linear_speed
            and sample.payload_angular_speed <= criteria.maximum_angular_speed
            and sample.support_contact
        )
        if self.spec.objective == "carry_payload":
            return common
        return (
            common
            and sample.inside_target
            and sample.left_contact
            and sample.articulation_position >= criteria.minimum_articulation_position
            and abs(sample.articulation_speed) <= criteria.maximum_articulation_speed
        )

    def _potential(self, sample: BimanualTaskSample) -> float:
        weights = self.spec.reward
        value = -weights.target_distance * sample.target_distance
        value -= weights.left_reach * sample.left_reach_distance
        value -= weights.right_reach * sample.right_reach_distance
        value -= weights.worst_side_reach * max(
            sample.left_reach_distance, sample.right_reach_distance
        )
        value -= weights.tilt * sample.payload_tilt_radians
        value += weights.contact * (sample.left_contact + sample.right_contact)
        left_ready = math.exp(-sample.left_reach_distance / 0.08) * (
            sample.left_gripper_position
        )
        right_ready = math.exp(-sample.right_reach_distance / 0.08) * (
            sample.right_gripper_position
        )
        value += weights.near_handle_closure * (left_ready + right_ready)
        value += weights.joint_grasp_readiness * left_ready * right_ready
        value += weights.bilateral_contact * (
            sample.left_contact and sample.right_contact
        )
        value += weights.support * sample.support_contact
        if self.spec.objective == "hold_drawer_place":
            required = self.spec.criteria.minimum_articulation_position
            value += weights.articulation * min(sample.articulation_position, required)
            value += weights.support * sample.inside_target
        return float(value)

    def _desired_goal(self) -> tuple[float, ...]:
        criteria = self.spec.criteria
        return (
            *self.spec.target_position,
            criteria.maximum_tilt_radians,
            criteria.maximum_linear_speed,
            criteria.maximum_angular_speed,
            criteria.minimum_articulation_position,
            1.0,
            1.0,
            1.0,
            1.0 if self.spec.objective == "hold_drawer_place" else 0.0,
            0.0,
        )

    def _metrics(self, sample: BimanualTaskSample) -> dict[str, float]:
        return {
            "target_distance": sample.target_distance,
            "payload_tilt_radians": sample.payload_tilt_radians,
            "payload_linear_speed": sample.payload_linear_speed,
            "payload_angular_speed": sample.payload_angular_speed,
            "articulation_position": sample.articulation_position,
            "stable_steps": float(self.stable_steps),
            "maximum_concurrent_steps": float(self.maximum_concurrent_steps),
            "left_contact": float(sample.left_contact),
            "right_contact": float(sample.right_contact),
            "left_reach_distance": sample.left_reach_distance,
            "right_reach_distance": sample.right_reach_distance,
            "left_gripper_position": sample.left_gripper_position,
            "right_gripper_position": sample.right_gripper_position,
            "support_contact": float(sample.support_contact),
            "inside_target": float(sample.inside_target),
            "severe_collisions": float(sample.severe_collision_count),
        }


def _scalar_range(value: Sequence[float]) -> ScalarRange:
    if len(value) != 2:
        raise ValueError("JSON scalar range requires two values")
    return ScalarRange(float(value[0]), float(value[1]))


def _pose_range(value: Mapping[str, Sequence[float]]) -> PoseRange:
    return PoseRange(*(_scalar_range(value[name]) for name in ("x", "y", "z", "yaw")))


def load_bimanual_task_specs(path: Path) -> dict[str, BimanualTaskSpec]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    randomization = DomainRandomization(
        *(
            _scalar_range(value["randomization"][name])
            for name in ("mass_scale", "friction_scale", "light_scale", "material_scale")
        )
    )
    specs: dict[str, BimanualTaskSpec] = {}
    for item in value["tasks"]:
        criteria = SuccessCriteria(**item["criteria"])
        reward = RewardWeights(**item.get("reward", {}))
        spec = BimanualTaskSpec(
            task_id=item["task_id"],
            scene_id=item["scene_id"],
            instruction=item["instruction"],
            objective=item["objective"],
            control_hz=float(value["control_hz"]),
            max_steps=int(item["max_steps"]),
            initial_base=_pose_range(item["initial_base"]),
            payload_reset=_pose_range(item["payload_reset"]),
            target_position=tuple(item["target_position"]),
            criteria=criteria,
            reward=reward,
            randomization=randomization,
        )
        if spec.task_id in specs:
            raise ValueError(f"duplicate bimanual task: {spec.task_id}")
        specs[spec.task_id] = spec
    return specs
