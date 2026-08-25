"""Legacy body-role and R0001-P50-E4 exact-geom mapping contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mujoco

from hwr.adapters.mujoco.model import MujocoModelBundle

MAPPING_SCHEMA = "hwr.p50-entity-role-table/v1"
EXACT_MAPPING_SCHEMA = "hwr.p50-e4-exact-geom-role-table/v1"
ALIAS_SCHEMA = "hwr.p50-e4-entity-candidate-aliases/v1"
ALIAS_PROPOSAL_ID = "R0001-P50-E4"
FROZEN_ALIAS_RECORDS = {
    "tidy_living_room_3d/v1": (
        ("toy_duck_visual", "toy_duck_collision", "manipulated_object", "duck"),
        ("toy_football_visual", "toy_football_collision", "manipulated_object",
         "football"),
        ("storage_basket_visual", "basket_bottom_collision",
         "target_container", None),
    ),
    "clear_dining_table_3d/v1": (
        ("dining_cup_visual", "dining_cup_collision",
         "manipulated_object", "cup"),
        ("dining_plate_visual", "dining_plate_collision",
         "manipulated_object", "plate"),
    ),
    "store_kitchen_items_3d/v1": (
        ("cleaner_yellow_visual", "cleaner_yellow_collision",
         "manipulated_object", "cleaner_yellow"),
        ("cleaner_pink_visual", "cleaner_pink_collision",
         "manipulated_object", "cleaner_pink"),
        ("drawer_handle_visual", "drawer_handle", "articulation", "drawer"),
    ),
}
FROZEN_VISIBLE_RECORDS = {
    "tidy_living_room_3d/v1": (
        ("toy_duck_visual", "manipulated_object", "duck"),
        ("toy_football_visual", "manipulated_object", "football"),
        ("storage_basket_visual", "target_container", None),
    ),
    "clear_dining_table_3d/v1": (
        ("dining_cup_visual", "manipulated_object", "cup"),
        ("dining_plate_visual", "manipulated_object", "plate"),
        ("cup_holder", "target_container", None),
        ("plate_holder", "target_container", None),
    ),
    "store_kitchen_items_3d/v1": (
        ("cleaner_yellow_visual", "manipulated_object", "cleaner_yellow"),
        ("cleaner_pink_visual", "manipulated_object", "cleaner_pink"),
        ("drawer_handle_visual", "articulation", "drawer"),
        ("drawer_bottom", "target_container", None),
        ("drawer_front", "target_container", None),
        ("drawer_back", "target_container", None),
        ("drawer_left", "target_container", None),
        ("drawer_right", "target_container", None),
        ("drawer_divider", "target_container", None),
    ),
}
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


@dataclass(frozen=True)
class EntityAlias:
    source_visual_geom: str
    canonical_exact_claimed_geom: str
    role: str
    instance: str | None

    def __post_init__(self) -> None:
        EntityRole(self.role, self.role, self.instance)
        if not self.source_visual_geom or not self.canonical_exact_claimed_geom:
            raise EntityCandidateMappingError("alias geom names must be nonempty")


@dataclass(frozen=True)
class TaskVisibleGeom:
    geom: str
    role: str
    instance: str | None

    def __post_init__(self) -> None:
        EntityRole(self.role, self.role, self.instance)
        if not self.geom:
            raise EntityCandidateMappingError("inventory geom name must be nonempty")


@dataclass(frozen=True)
class TaskAliasContract:
    task_id: str
    aliases: tuple[EntityAlias, ...]
    task_visible_inventory: tuple[TaskVisibleGeom, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise EntityCandidateMappingError("alias task id must be nonempty")


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


def load_entity_alias_contracts(path: Path) -> dict[str, TaskAliasContract]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "proposal_id",
        "tasks",
    }:
        raise EntityCandidateMappingError("alias document fields differ")
    if (
        value["schema_version"] != ALIAS_SCHEMA
        or value["proposal_id"] != ALIAS_PROPOSAL_ID
        or not isinstance(value["tasks"], list)
    ):
        raise EntityCandidateMappingError("alias document header differs")
    contracts: dict[str, TaskAliasContract] = {}
    for task in value["tasks"]:
        if not isinstance(task, dict) or set(task) != {
            "task_id",
            "aliases",
            "task_visible_inventory",
        }:
            raise EntityCandidateMappingError("alias task fields differ")
        aliases = tuple(_parse_alias(record) for record in task["aliases"])
        inventory = tuple(
            _parse_visible_geom(record)
            for record in task["task_visible_inventory"]
        )
        contract = TaskAliasContract(
            str(task["task_id"]), aliases, inventory
        )
        if contract.task_id in contracts:
            raise EntityCandidateMappingError("duplicate alias task")
        contracts[contract.task_id] = contract
    if tuple(contracts) != tuple(FROZEN_ALIAS_RECORDS):
        raise EntityCandidateMappingError(
            "frozen alias task order differs",
            details={"failure_kind": "frozen_alias_identity"},
        )
    for task_id, contract in contracts.items():
        alias_records = tuple(
            (
                value.source_visual_geom,
                value.canonical_exact_claimed_geom,
                value.role,
                value.instance,
            )
            for value in contract.aliases
        )
        visible_records = tuple(
            (value.geom, value.role, value.instance)
            for value in contract.task_visible_inventory
        )
        if (
            alias_records != FROZEN_ALIAS_RECORDS[task_id]
            or visible_records != FROZEN_VISIBLE_RECORDS[task_id]
        ):
            raise EntityCandidateMappingError(
                f"frozen alias inventory differs: {task_id}",
                details={"failure_kind": "frozen_alias_identity"},
            )
    return contracts


def build_exact_geom_role_table(
    model: mujoco.MjModel,
    binding,
    aliases: TaskAliasContract,
    *,
    robot_root_body: int,
) -> dict[str, object]:
    if aliases.task_id != binding.task_id:
        raise EntityCandidateMappingError(
            "alias task differs from binding",
            details={"failure_kind": "frozen_alias_identity"},
        )
    claims: dict[int, tuple[EntityRole, str]] = {}

    def claim(geom_name: str, role: EntityRole, source: str) -> None:
        geom_id = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        previous = claims.get(geom_id)
        if previous is not None and previous[0] != role:
            raise EntityCandidateMappingError(
                "one exact geom has conflicting task roles",
                details={
                    "failure_kind": "exact_geom_role_conflict",
                    "task_id": binding.task_id,
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "first": _role_record(previous[0]),
                    "first_source": previous[1],
                    "second": _role_record(role),
                    "second_source": source,
                },
            )
        claims[geom_id] = (role, source)

    object_geoms = {
        value.collision_geom for value in binding.objects.values()
    }
    if set(binding.allowed_robot_contact_roles["manipulated_object"]) != (
        object_geoms
    ):
        raise EntityCandidateMappingError(
            "binding manipulated-object exact geoms differ"
        )
    for instance, value in sorted(binding.objects.items()):
        claim(
            value.collision_geom,
            EntityRole(
                f"object:{instance}", "manipulated_object", instance
            ),
            f"binding.objects.{instance}.collision_geom",
        )
    articulation_geoms = set(
        binding.allowed_robot_contact_roles["articulation"]
    )
    expected_articulation = (
        set()
        if binding.articulation is None
        else {binding.articulation.handle_geom}
    )
    if articulation_geoms != expected_articulation:
        raise EntityCandidateMappingError(
            "binding articulation exact geoms differ"
        )
    if binding.articulation is not None:
        claim(
            binding.articulation.handle_geom,
            EntityRole(
                f"articulation:{binding.articulation.articulation_id}",
                "articulation",
                binding.articulation.articulation_id,
            ),
            "binding.articulation.handle_geom",
        )
    for role in ("target_container", "floor_support"):
        for geom_name in sorted(binding.allowed_robot_contact_roles[role]):
            claim(
                geom_name,
                EntityRole(role, role, None),
                f"binding.allowed_robot_contact_roles.{role}",
            )
    robot_root = int(model.body_rootid[int(robot_root_body)])
    for geom_id in range(model.ngeom):
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        if (
            geom_name
            and int(model.body_rootid[body_id]) == robot_root
        ):
            claim(geom_name, EntityRole("robot", "robot", None), "robot_root")

    alias_by_id = _validated_aliases(model, aliases, claims)
    inventory_by_id = _validated_inventory(
        model, aliases.task_visible_inventory
    )
    geoms = [
        _exact_geom_record(model, geom_id, claims, alias_by_id)
        for geom_id in range(model.ngeom)
    ]
    geom_by_id = {int(record["geom_id"]): record for record in geoms}
    unknown_inventory = []
    mismatched_inventory = []
    for geom_id, expected in inventory_by_id.items():
        actual = geom_by_id[geom_id]
        if actual["role"] == "unknown":
            unknown_inventory.append(expected.geom)
        if (
            actual["role"] != expected.role
            or actual["instance"] != expected.instance
        ):
            mismatched_inventory.append(
                {
                    "geom": expected.geom,
                    "expected": {
                        "role": expected.role,
                        "instance": expected.instance,
                    },
                    "actual": {
                        "role": actual["role"],
                        "instance": actual["instance"],
                    },
                }
            )
    if unknown_inventory or mismatched_inventory:
        raise EntityCandidateMappingError(
            "task-visible inventory does not resolve exactly",
            details={
                "failure_kind": "task_visible_inventory_mismatch",
                "unknown": unknown_inventory,
                "mismatches": mismatched_inventory,
            },
        )
    bodies = _exact_body_records(model, geoms)
    sites = [
        {
            "site_id": site_id,
            "site_name": _name(model, mujoco.mjtObj.mjOBJ_SITE, site_id),
            "body_id": int(model.site_bodyid[site_id]),
            "body_name": _name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.site_bodyid[site_id]),
            ),
            "label": "unknown_site",
            "role": "unknown_site",
            "instance": None,
        }
        for site_id in range(model.nsite)
    ]
    payload = {
        "schema_version": EXACT_MAPPING_SCHEMA,
        "task_id": binding.task_id,
        "body_count": len(bodies),
        "geom_count": len(geoms),
        "site_count": len(sites),
        "exact_claim_count": len(claims),
        "alias_count": len(alias_by_id),
        "exact_geom_role_conflict_count": 0,
        "task_visible_inventory_count": len(inventory_by_id),
        "task_visible_inventory_unknown_count": 0,
        "bodies": bodies,
        "geoms": geoms,
        "sites": sites,
        "background": {
            "object_id": -1,
            "object_type": -1,
            "label": "background",
            "role": "background",
            "instance": None,
        },
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def preflight_exact_geom_role_tables(
    bindings: Mapping[str, object],
    aliases: Mapping[str, TaskAliasContract],
    task_ids: Sequence[str],
) -> dict[str, dict[str, object]]:
    if set(aliases) != set(task_ids):
        raise EntityCandidateMappingError("alias task inventory differs")
    tables = {}
    for task_id in task_ids:
        binding = bindings[task_id]
        bundle = MujocoModelBundle.load(
            binding.model_path, object_joint_name=None
        )
        tables[task_id] = build_exact_geom_role_table(
            bundle.model,
            binding,
            aliases[task_id],
            robot_root_body=bundle.ids.base_body,
        )
    return tables


def classify_segmentation_entity(
    table: Mapping[str, object],
    object_id: int,
    object_type: int,
) -> dict[str, object]:
    if object_id == -1 and object_type == -1:
        return dict(table["background"])
    if object_type == int(mujoco.mjtObj.mjOBJ_GEOM):
        geoms = table["geoms"]
        if isinstance(geoms, list) and 0 <= object_id < len(geoms):
            return dict(geoms[object_id])
    if object_type == int(mujoco.mjtObj.mjOBJ_SITE):
        sites = table["sites"]
        if isinstance(sites, list) and 0 <= object_id < len(sites):
            return dict(sites[object_id])
    return {
        "object_id": object_id,
        "object_type": object_type,
        "label": "unknown",
        "role": "unknown",
        "instance": None,
    }


def mujoco_runtime_version() -> str:
    return importlib.metadata.version("mujoco")


def _role_record(role: EntityRole | None) -> dict[str, object]:
    return {
        "label": "unknown" if role is None else role.label,
        "role": "unknown" if role is None else role.role,
        "instance": None if role is None else role.instance,
    }


def _parse_alias(value: object) -> EntityAlias:
    if not isinstance(value, dict) or set(value) != {
        "source_visual_geom",
        "canonical_exact_claimed_geom",
        "role",
        "instance",
    }:
        raise EntityCandidateMappingError("alias fields differ")
    if not all(
        isinstance(value[name], str)
        for name in (
            "source_visual_geom",
            "canonical_exact_claimed_geom",
            "role",
        )
    ) or not (
        value["instance"] is None or isinstance(value["instance"], str)
    ):
        raise EntityCandidateMappingError("alias value types differ")
    return EntityAlias(**value)


def _parse_visible_geom(value: object) -> TaskVisibleGeom:
    if not isinstance(value, dict) or set(value) != {
        "geom",
        "role",
        "instance",
    }:
        raise EntityCandidateMappingError("inventory fields differ")
    if not all(
        isinstance(value[name], str) for name in ("geom", "role")
    ) or not (
        value["instance"] is None or isinstance(value["instance"], str)
    ):
        raise EntityCandidateMappingError("inventory value types differ")
    return TaskVisibleGeom(**value)


def _validated_aliases(
    model: mujoco.MjModel,
    aliases: TaskAliasContract,
    claims: Mapping[int, tuple[EntityRole, str]],
) -> dict[int, tuple[int, EntityRole]]:
    sources = [value.source_visual_geom for value in aliases.aliases]
    targets = [
        value.canonical_exact_claimed_geom for value in aliases.aliases
    ]
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        raise EntityCandidateMappingError(
            "alias source or target is duplicated",
            details={"failure_kind": "alias_contract_semantic"},
        )
    if set(sources) & set(targets):
        raise EntityCandidateMappingError(
            "alias chain or cycle is forbidden",
            details={"failure_kind": "alias_contract_semantic"},
        )
    result = {}
    for alias in aliases.aliases:
        source_id = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, alias.source_visual_geom
        )
        target_id = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            alias.canonical_exact_claimed_geom,
        )
        if source_id in claims:
            raise EntityCandidateMappingError(
                "alias source has an independent exact task claim",
                details={"failure_kind": "alias_contract_semantic"},
            )
        target_claim = claims.get(target_id)
        if target_claim is None:
            raise EntityCandidateMappingError(
                "alias target is not an exact claimed geom",
                details={"failure_kind": "alias_contract_semantic"},
            )
        if (
            target_claim[0].role != alias.role
            or target_claim[0].instance != alias.instance
        ):
            raise EntityCandidateMappingError(
                "alias role or instance differs from exact target",
                details={"failure_kind": "alias_contract_semantic"},
            )
        if int(model.geom_bodyid[source_id]) != int(
            model.geom_bodyid[target_id]
        ):
            raise EntityCandidateMappingError(
                "alias crosses body boundary",
                details={"failure_kind": "alias_contract_semantic"},
            )
        result[source_id] = (target_id, target_claim[0])
    return result


def _validated_inventory(
    model: mujoco.MjModel,
    inventory: Sequence[TaskVisibleGeom],
) -> dict[int, TaskVisibleGeom]:
    result = {}
    for item in inventory:
        geom_id = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, item.geom
        )
        if geom_id in result:
            raise EntityCandidateMappingError(
                "task-visible inventory contains duplicate geom"
            )
        result[geom_id] = item
    return result


def _exact_geom_record(
    model: mujoco.MjModel,
    geom_id: int,
    claims: Mapping[int, tuple[EntityRole, str]],
    aliases: Mapping[int, tuple[int, EntityRole]],
) -> dict[str, object]:
    body_id = int(model.geom_bodyid[geom_id])
    geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    canonical = None
    source = None
    claim_kind = "fallback"
    if geom_id in claims:
        role, source = claims[geom_id]
        canonical = geom_name
        claim_kind = "exact"
    elif geom_id in aliases:
        target_id, role = aliases[geom_id]
        canonical = _name(model, mujoco.mjtObj.mjOBJ_GEOM, target_id)
        source = "frozen_one_hop_same_body_alias"
        claim_kind = "alias"
    elif geom_name:
        role = EntityRole("other_furniture", "other_furniture", None)
    else:
        role = None
        claim_kind = "unknown"
    return {
        "geom_id": geom_id,
        "geom_name": geom_name,
        "body_id": body_id,
        "body_name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        "body_root_id": int(model.body_rootid[body_id]),
        "claim_kind": claim_kind,
        "claim_source": source,
        "canonical_exact_claimed_geom": canonical,
        **_role_record(role),
    }


def _exact_body_records(
    model: mujoco.MjModel,
    geoms: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = {
        body_id: [] for body_id in range(model.nbody)
    }
    for geom in geoms:
        grouped[int(geom["body_id"])].append(geom)
    return [
        {
            "body_id": body_id,
            "body_name": _name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ),
            "body_root_id": int(model.body_rootid[body_id]),
            "geom_ids": [int(value["geom_id"]) for value in grouped[body_id]],
            "geom_roles": sorted(
                {
                    str(value["role"])
                    for value in grouped[body_id]
                    if value["role"] != "unknown"
                }
            ),
        }
        for body_id in range(model.nbody)
    ]


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
