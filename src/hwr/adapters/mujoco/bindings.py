"""MuJoCo entity bindings for project-owned formal task declarations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MujocoObjectBinding:
    body: str
    joint: str
    collision_geom: str
    target_site: str


@dataclass(frozen=True)
class MujocoArticulationBinding:
    articulation_id: str
    joint: str
    handle_geom: str


@dataclass(frozen=True)
class MujocoTaskBinding:
    task_id: str
    model_path: Path
    objects: dict[str, MujocoObjectBinding]
    allowed_robot_contact_geoms: frozenset[str]
    articulation: MujocoArticulationBinding | None = None


def _articulation(value: dict[str, Any] | None) -> MujocoArticulationBinding | None:
    return MujocoArticulationBinding(**value) if value else None


def load_mujoco_task_bindings(path: Path, *, root: Path) -> dict[str, MujocoTaskBinding]:
    value = json.loads(path.read_text(encoding="utf-8"))
    bindings: dict[str, MujocoTaskBinding] = {}
    for item in value["bindings"]:
        binding = MujocoTaskBinding(
            task_id=item["task_id"],
            model_path=(root / item["model"]).resolve(),
            objects={
                object_id: MujocoObjectBinding(**object_value)
                for object_id, object_value in item["objects"].items()
            },
            allowed_robot_contact_geoms=frozenset(item["allowed_robot_contact_geoms"]),
            articulation=_articulation(item.get("articulation")),
        )
        if binding.task_id in bindings:
            raise ValueError(f"duplicate MuJoCo binding: {binding.task_id}")
        bindings[binding.task_id] = binding
    return bindings
