"""Engine-independent declarations for formal three-dimensional household tasks."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class NumericRange:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("range endpoints must be finite")
        if self.minimum > self.maximum:
            raise ValueError("range minimum exceeds maximum")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.minimum, self.maximum)

    @classmethod
    def from_value(cls, value: list[float]) -> "NumericRange":
        if len(value) != 2:
            raise ValueError("numeric range must have two endpoints")
        return cls(float(value[0]), float(value[1]))


@dataclass(frozen=True)
class PoseResetSpec:
    x: NumericRange
    y: NumericRange
    z: NumericRange
    yaw: NumericRange

    def sample(self, rng: random.Random) -> tuple[float, float, float, float]:
        return (
            self.x.sample(rng),
            self.y.sample(rng),
            self.z.sample(rng),
            self.yaw.sample(rng),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, list[float]]) -> "PoseResetSpec":
        ranges = (NumericRange.from_value(value[name]) for name in ("x", "y", "z", "yaw"))
        return cls(*ranges)


@dataclass(frozen=True)
class ManipulatedObjectSpec:
    object_id: str
    target_id: str
    reset: PoseResetSpec
    grasp_site_z_offset: float
    grip_fraction: float
    standoff_m: float

    def __post_init__(self) -> None:
        if not self.object_id or not self.target_id:
            raise ValueError("object and target IDs are required")
        if self.grasp_site_z_offset <= 0 or self.standoff_m <= 0:
            raise ValueError("grasp offset and standoff must be positive")
        if not 0 < self.grip_fraction <= 1:
            raise ValueError("grip fraction must be in (0, 1]")


@dataclass(frozen=True)
class ArticulationRequirement:
    articulation_id: str
    minimum_position: float

    def __post_init__(self) -> None:
        if not self.articulation_id or self.minimum_position <= 0:
            raise ValueError("articulation requirement is invalid")


@dataclass(frozen=True)
class RandomizationSpec:
    mass_scale: NumericRange
    friction_scale: NumericRange
    light_scale: NumericRange
    material_tint: NumericRange
    rgb_noise_std: NumericRange
    depth_dropout: NumericRange


@dataclass(frozen=True)
class Formal3DTaskSpec:
    task_id: str
    scene_id: str
    instruction: str
    max_steps: int
    control_hz: float
    hold_seconds: float
    initial_base: PoseResetSpec
    objects: tuple[ManipulatedObjectSpec, ...]
    randomization: RandomizationSpec
    articulation: ArticulationRequirement | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.scene_id or not self.instruction:
            raise ValueError("task identity and instruction are required")
        if self.max_steps <= 0 or self.control_hz <= 0 or self.hold_seconds < 2.0:
            raise ValueError("task timing is invalid")
        object_ids = {obj.object_id for obj in self.objects}
        if len(self.objects) < 2 or len(object_ids) != len(self.objects):
            raise ValueError("formal tasks need at least two unique manipulated objects")


def _randomization(value: Mapping[str, list[float]]) -> RandomizationSpec:
    ranges = {
        name: NumericRange.from_value(value[name])
        for name in RandomizationSpec.__annotations__
    }
    return RandomizationSpec(**ranges)


def _object(value: Mapping[str, Any]) -> ManipulatedObjectSpec:
    return ManipulatedObjectSpec(
        object_id=value["object_id"],
        target_id=value["target_id"],
        reset=PoseResetSpec.from_dict(value["reset"]),
        grasp_site_z_offset=float(value["grasp_site_z_offset"]),
        grip_fraction=float(value["grip_fraction"]),
        standoff_m=float(value["standoff_m"]),
    )


def load_formal_3d_tasks(path: Path) -> dict[str, Formal3DTaskSpec]:
    value = json.loads(path.read_text(encoding="utf-8"))
    randomization = _randomization(value["randomization"])
    tasks: dict[str, Formal3DTaskSpec] = {}
    for item in value["tasks"]:
        articulation_value = item.get("articulation")
        articulation = (
            ArticulationRequirement(
                articulation_value["articulation_id"],
                float(articulation_value["minimum_position"]),
            )
            if articulation_value
            else None
        )
        task = Formal3DTaskSpec(
            task_id=item["task_id"],
            scene_id=item["scene_id"],
            instruction=item["instruction"],
            max_steps=int(item["max_steps"]),
            control_hz=float(value["control_hz"]),
            hold_seconds=float(value["hold_seconds"]),
            initial_base=PoseResetSpec.from_dict(item["initial_base"]),
            objects=tuple(_object(obj) for obj in item["objects"]),
            randomization=randomization,
            articulation=articulation,
        )
        if task.task_id in tasks:
            raise ValueError(f"duplicate task ID: {task.task_id}")
        tasks[task.task_id] = task
    return tasks
