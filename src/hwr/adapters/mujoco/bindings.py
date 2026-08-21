"""MuJoCo entity bindings for project-owned formal task declarations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hwr.adapters.mujoco.contact_ledger import ALLOWED_CONTACT_ROLES


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
    allowed_robot_contact_roles: dict[str, frozenset[str]]
    articulation: MujocoArticulationBinding | None = None


def _articulation(value: dict[str, Any] | None) -> MujocoArticulationBinding | None:
    return MujocoArticulationBinding(**value) if value else None


def _allowed_contact_roles(
    value: object, allowed: frozenset[str]
) -> dict[str, frozenset[str]]:
    if not isinstance(value, dict):
        raise ValueError("allowed robot contact roles are required")
    if set(value) != set(ALLOWED_CONTACT_ROLES):
        raise ValueError("allowed robot contact role keys differ from the contract")
    roles: dict[str, frozenset[str]] = {}
    seen: set[str] = set()
    for role in ALLOWED_CONTACT_ROLES:
        names = value[role]
        if not isinstance(names, list) or not all(
            isinstance(name, str) and name for name in names
        ):
            raise ValueError(f"allowed robot contact role {role} must list geom names")
        if len(names) != len(set(names)):
            raise ValueError(f"allowed robot contact role {role} contains duplicates")
        overlap = seen & set(names)
        if overlap:
            raise ValueError(f"allowed robot contact roles overlap: {sorted(overlap)}")
        roles[role] = frozenset(names)
        seen.update(names)
    if seen != set(allowed):
        raise ValueError("allowed robot contact role union differs from legacy allow-list")
    return roles


def load_mujoco_task_bindings(path: Path, *, root: Path) -> dict[str, MujocoTaskBinding]:
    value = json.loads(path.read_text(encoding="utf-8"))
    bindings: dict[str, MujocoTaskBinding] = {}
    for item in value["bindings"]:
        allowed = frozenset(item["allowed_robot_contact_geoms"])
        binding = MujocoTaskBinding(
            task_id=item["task_id"],
            model_path=(root / item["model"]).resolve(),
            objects={
                object_id: MujocoObjectBinding(**object_value)
                for object_id, object_value in item["objects"].items()
            },
            allowed_robot_contact_geoms=allowed,
            allowed_robot_contact_roles=_allowed_contact_roles(
                item.get("allowed_robot_contact_roles"), allowed
            ),
            articulation=_articulation(item.get("articulation")),
        )
        if binding.task_id in bindings:
            raise ValueError(f"duplicate MuJoCo binding: {binding.task_id}")
        bindings[binding.task_id] = binding
    return bindings
