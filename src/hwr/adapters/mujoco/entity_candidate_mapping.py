"""Frozen geom-to-body role-table preflight for R0001-P50-E3."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import mujoco

from hwr.adapters.mujoco.model import MujocoModelBundle

MAPPING_SCHEMA = "hwr.p50-entity-role-table/v1"
ROLE_NAMES = frozenset(
    (
        "manipulated_object",
        "articulation",
        "target_container",
        "floor_support",
        "other_furniture",
        "robot",
    )
)


class EntityCandidateMappingError(ValueError):
    """Raised when the frozen body-role mapping is ambiguous."""

    def __init__(
        self, message: str, *, details: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.details = {} if details is None else dict(details)


@dataclass(frozen=True)
class EntityRole:
    label: str
    role: str
    instance: str | None

    def __post_init__(self) -> None:
        if self.role not in ROLE_NAMES:
            raise EntityCandidateMappingError("entity role is outside contract")
        if self.role == "manipulated_object" and not self.instance:
            raise EntityCandidateMappingError(
                "manipulated object requires exact instance"
            )


def build_entity_role_table(
    model: mujoco.MjModel,
    binding,
    *,
    robot_root_body: int,
) -> dict[str, object]:
    claims: dict[int, EntityRole] = {}

    def claim(body_id: int, role: EntityRole) -> None:
        previous = claims.get(body_id)
        if previous is not None and previous != role:
            raise EntityCandidateMappingError(
                "one body is occupied by multiple task roles",
                details={
                    "task_id": binding.task_id,
                    "body_id": body_id,
                    "body_name": _name(
                        model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    ),
                    "first": _role_record(previous),
                    "second": _role_record(role),
                },
            )
        claims[body_id] = role

    for instance, value in binding.objects.items():
        body_id = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, value.body
        )
        claim(
            body_id,
            EntityRole(
                f"object:{instance}", "manipulated_object", instance
            ),
        )
    if binding.articulation is not None:
        geom_id = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            binding.articulation.handle_geom,
        )
        claim(
            int(model.geom_bodyid[geom_id]),
            EntityRole(
                f"articulation:{binding.articulation.articulation_id}",
                "articulation",
                binding.articulation.articulation_id,
            ),
        )
    for role in ("target_container", "floor_support"):
        for geom_name in binding.allowed_robot_contact_roles[role]:
            geom_id = _required_id(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            claim(
                int(model.geom_bodyid[geom_id]),
                EntityRole(role, role, None),
            )
    robot_root = int(model.body_rootid[int(robot_root_body)])
    for body_id in range(model.nbody):
        if int(model.body_rootid[body_id]) == robot_root:
            claim(body_id, EntityRole("robot", "robot", None))
    bodies = []
    for body_id in range(model.nbody):
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        role = claims.get(body_id)
        if role is None and body_name:
            role = EntityRole(
                "other_furniture", "other_furniture", None
            )
        bodies.append(
            {
                "body_id": body_id,
                "body_name": body_name,
                "body_root_id": int(model.body_rootid[body_id]),
                **_role_record(role),
            }
        )
    geoms = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        role = claims.get(body_id)
        if role is None and body_name:
            role = EntityRole(
                "other_furniture", "other_furniture", None
            )
        if not body_name or not geom_name:
            role = None
        geoms.append(
            {
                "geom_id": geom_id,
                "geom_name": geom_name,
                "body_id": body_id,
                "body_name": body_name,
                "body_root_id": int(model.body_rootid[body_id]),
                **_role_record(role),
            }
        )
    payload = {
        "schema_version": MAPPING_SCHEMA,
        "task_id": binding.task_id,
        "body_count": len(bodies),
        "geom_count": len(geoms),
        "bodies": bodies,
        "geoms": geoms,
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def preflight_entity_role_tables(
    bindings: Mapping[str, object],
    task_ids: Sequence[str],
) -> dict[str, object]:
    reports = []
    for task_id in task_ids:
        binding = bindings[task_id]
        bundle = MujocoModelBundle.load(
            binding.model_path, object_joint_name=None
        )
        try:
            table = build_entity_role_table(
                bundle.model,
                binding,
                robot_root_body=bundle.ids.base_body,
            )
        except EntityCandidateMappingError as error:
            raise EntityCandidateMappingError(
                "frozen entity mapping preflight is infeasible",
                details={
                    "failed_task_id": task_id,
                    "completed_task_count": len(reports),
                    "completed_tasks": reports,
                    "conflict": error.details,
                    "episode_count": 0,
                    "physical_acquisition_count": 0,
                },
            ) from error
        reports.append(
            {
                "task_id": task_id,
                "decision": "mapping_preflight_passed",
                "mapping_sha256": table["sha256"],
                "body_count": table["body_count"],
                "geom_count": table["geom_count"],
            }
        )
    return {
        "decision": "mapping_preflight_passed",
        "task_count": len(reports),
        "tasks": reports,
        "episode_count": 0,
        "physical_acquisition_count": 0,
    }


def mujoco_runtime_version() -> str:
    return importlib.metadata.version("mujoco")


def _role_record(role: EntityRole | None) -> dict[str, object]:
    return {
        "label": "unknown" if role is None else role.label,
        "role": "unknown" if role is None else role.role,
        "instance": None if role is None else role.instance,
    }


def _required_id(
    model: mujoco.MjModel, kind: mujoco.mjtObj, name: str
) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise EntityCandidateMappingError(
            f"model is missing required entity: {name}"
        )
    return value


def _name(
    model: mujoco.MjModel, kind: mujoco.mjtObj, value: int
) -> str | None:
    return mujoco.mj_id2name(model, kind, value)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
