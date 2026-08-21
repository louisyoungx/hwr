"""Evaluator-private entity contact graph for formal MuJoCo tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import mujoco
import numpy as np

from hwr.adapters.mujoco.contact_ledger import CONTACT_CATEGORIES


MEASUREMENT_SCHEMA = "hwr.mujoco-entity-contact-graph/v1"
ROBOT_PARTS = ("base", "left_arm", "right_arm")
ROBOT_BODY_ROOT_NAMES = {
    "base": "robot_base",
    "left_arm": "left_shoulder_pan_link",
    "right_arm": "right_shoulder_pan_link",
}
ENTITY_ROLES = frozenset(
    ("floor_support", "manipulated_object", "target_container", "articulation", "forbidden")
)
P40_FIELDS = (
    "pair_peak_force", "category_peak_force", "cumulative_impulse",
    "contact_duration_seconds", "contact_point_count", "unique_pair_observation_count",
)
COUNT_FIELDS = (
    "physics_substep_count", "contact_point_count",
    "robot_environment_contact_point_count",
    "task_relevant_world_world_contact_point_count",
    "ignored_robot_self_contact_point_count",
    "ignored_world_world_contact_point_count", "missing_normal_force_count",
    "nonfinite_normal_force_count", "invalid_negative_normal_force_count",
    "unknown_mapping_count", "invalid_motion_state_count",
)
INTERACTION_FIELDS = (
    "same_entity_dual_arm_substep_count",
    "same_entity_dual_arm_entity_observation_count",
    "distinct_entity_dual_arm_substep_count",
    "distinct_entity_dual_arm_pair_observation_count", "single_arm_substep_count",
    "same_object_dual_arm_grasp_substep_count",
    "same_object_dual_arm_grasp_observation_count",
)
INVALID_FIELDS = COUNT_FIELDS[-5:]


class EntityContactGraphError(RuntimeError):
    """Raised when contact or mapping evidence violates the frozen contract."""


@dataclass(frozen=True)
class EntityContactPointObservation:
    geom1: int
    geom2: int
    normal_force: float | None

@dataclass(frozen=True)
class EntityMotionSource:
    kind: str
    index: int

    def __post_init__(self) -> None:
        if self.kind not in {"translation", "joint"} or self.index < 0:
            raise ValueError("entity motion source is invalid")

@dataclass
class _Accumulator:
    pair_peak: float = 0.0
    substep_peak: float = 0.0
    impulse: float = 0.0
    contact_duration_seconds: float = 0.0
    contact_point_count: int = 0
    pair_observation_count: int = 0

    def update(
        self,
        *,
        pair_peak: float,
        total_force: float,
        point_count: int,
        pair_count: int,
        timestep: float,
    ) -> None:
        self.pair_peak = max(self.pair_peak, pair_peak)
        self.substep_peak = max(self.substep_peak, total_force)
        self.impulse += total_force * timestep
        self.contact_duration_seconds += timestep * (total_force > 0.0)
        self.contact_point_count += point_count
        self.pair_observation_count += pair_count

    def category_dict(self, *, period: bool) -> dict[str, float | int]:
        return self._shared_dict() | {
            "pair_peak_force": self.pair_peak,
            "category_peak_force": self.substep_peak,
            "category_impulse" if period else "cumulative_impulse": self.impulse,
            "unique_pair_observation_count": self.pair_observation_count,
        }

    def edge_dict(self, *, period: bool) -> dict[str, float | int]:
        return self._shared_dict() | {
            "geom_pair_peak_force": self.pair_peak,
            "substep_peak_force": self.substep_peak,
            "period_impulse" if period else "cumulative_impulse": self.impulse,
            "unique_geom_pair_observation_count": self.pair_observation_count,
        }

    def _shared_dict(self) -> dict[str, float | int]:
        return {
            "contact_duration_seconds": self.contact_duration_seconds,
            "contact_point_count": self.contact_point_count,
        }

class EntityContactGraph:
    """Aggregate read-only contacts by robot part and task entity."""

    def __init__(
        self,
        *,
        all_geom_ids: Iterable[int],
        robot_part_by_geom: Mapping[int, str],
        entity_by_geom: Mapping[int, str],
        timestep: float,
        enabled: bool,
        motion_source_by_entity: Mapping[str, EntityMotionSource] | None = None,
        gripper_pad_groups: Mapping[
            str, tuple[Iterable[int], Iterable[int]]
        ] | None = None,
        geom_name_by_id: Mapping[int, str] | None = None,
        robot_body_roots: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.all_geom_ids = frozenset(int(value) for value in all_geom_ids)
        self.robot_part_by_geom = {
            int(geom): str(part) for geom, part in robot_part_by_geom.items()
        }
        self.entity_by_geom = {
            int(geom): str(entity) for geom, entity in entity_by_geom.items()
        }
        self.timestep = float(timestep)
        self.enabled = bool(enabled)
        self.motion_source_by_entity = dict(motion_source_by_entity or {})
        self.gripper_pad_groups = {
            part: (frozenset(first), frozenset(second))
            for part, (first, second) in (gripper_pad_groups or {}).items()
        }
        self.geom_name_by_id = {
            int(geom): str(name) for geom, name in (geom_name_by_id or {}).items()
        }
        self.robot_body_roots = {
            str(part): dict(identity)
            for part, identity in (robot_body_roots or {}).items()
        }
        self._validate_mapping()
        self.reset()

    def _validate_mapping(self) -> None:
        if not self.all_geom_ids:
            raise ValueError("entity contact graph requires geometry")
        if not math.isfinite(self.timestep) or self.timestep <= 0.0:
            raise ValueError("entity contact graph timestep must be positive")
        robot = set(self.robot_part_by_geom)
        environment = set(self.entity_by_geom)
        if robot & environment:
            raise ValueError("robot and environment geometry mappings overlap")
        if robot | environment != set(self.all_geom_ids):
            raise ValueError("entity contact graph geometry mapping is incomplete")
        unknown_parts = set(self.robot_part_by_geom.values()) - set(ROBOT_PARTS)
        if unknown_parts:
            raise ValueError(f"unknown robot parts: {sorted(unknown_parts)}")
        if set(self.robot_part_by_geom.values()) != set(ROBOT_PARTS):
            raise ValueError("all frozen robot parts must have geometry")
        if self.robot_body_roots:
            names = {
                part: value.get("body_name")
                for part, value in self.robot_body_roots.items()
            }
            if names != ROBOT_BODY_ROOT_NAMES:
                raise ValueError("robot body roots differ from the frozen contract")
            body_ids = tuple(
                value.get("body_id") for value in self.robot_body_roots.values()
            )
            if not all(isinstance(value, int) for value in body_ids):
                raise ValueError("robot body-root IDs are invalid")
            if len(body_ids) != len(set(body_ids)):
                raise ValueError("robot body roots overlap")
        for entity in self.entity_by_geom.values():
            _entity_role(entity)
        motion_entities = set(self.motion_source_by_entity)
        if motion_entities - set(self.entity_by_geom.values()):
            raise ValueError("motion source references an unknown entity")
        for entity, source in self.motion_source_by_entity.items():
            role = _entity_role(entity)
            expected = "joint" if role == "articulation" else "translation"
            if role not in {"manipulated_object", "articulation"}:
                raise ValueError("only objects and articulations may have motion")
            if source.kind != expected:
                raise ValueError("entity motion source kind differs from its role")
        if set(self.gripper_pad_groups) - {"left_arm", "right_arm"}:
            raise ValueError("gripper pads must belong to an arm")
        for part, groups in self.gripper_pad_groups.items():
            if not groups[0] or not groups[1] or groups[0] & groups[1]:
                raise ValueError("gripper pad groups must be nonempty and disjoint")
            if not set(groups[0] | groups[1]) <= robot:
                raise ValueError("gripper pad geometry is not robot geometry")
            if any(self.robot_part_by_geom[geom] != part for geom in groups[0] | groups[1]):
                raise ValueError("gripper pad geometry belongs to the wrong arm")

    def reset(self) -> None:
        self._episode_categories = _empty_categories()
        self._episode_robot_edges: dict[tuple[str, str], _Accumulator] = {}
        self._episode_world_edges: dict[tuple[str, str], _Accumulator] = {}
        self._period: dict[str, object] | None = None
        self._periods: list[dict[str, object]] = []
        self._substep_observations: list[dict[str, object]] = []
        self._counts = _empty_counts()
        self._interaction_counts = _empty_interaction_counts()

    def begin_control_period(self, motion_state: Mapping[str, object]) -> None:
        if not self.enabled:
            return
        if self._period is not None:
            raise EntityContactGraphError("entity contact graph period is active")
        self._period = {
            "categories": _empty_categories(),
            "robot_edges": {},
            "world_edges": {},
            "counts": _empty_counts(),
            "interactions": _empty_interaction_counts(),
            "contacted_entities": set(),
            "motion_start": self._normalize_motion_state(motion_state),
            "substeps": [],
        }

    def sample_mujoco_substep(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> None:
        if not self.enabled:
            return
        observations: list[EntityContactPointObservation] = []
        for index in range(data.ncon):
            contact = data.contact[index]
            normal: float | None
            force = np.empty(6, dtype=np.float64)
            try:
                mujoco.mj_contactForce(model, data, index, force)
                normal = float(force[0])
            except (ArithmeticError, RuntimeError, ValueError):
                normal = None
            observations.append(
                EntityContactPointObservation(
                    int(contact.geom1), int(contact.geom2), normal
                )
            )
        self.record_substep(observations)

    def record_substep(
        self, points: Iterable[EntityContactPointObservation]
    ) -> None:
        if not self.enabled:
            return
        period = self._require_period()
        points = tuple(points)
        pair_forces: dict[tuple[int, int], float] = {}
        pair_point_counts: dict[tuple[int, int], int] = {}
        pair_classes: dict[tuple[int, int], tuple[object, ...]] = {}
        invalid = False
        self._increment("physics_substep_count", period=period)
        self._increment("contact_point_count", len(points), period=period)
        for point in points:
            pair = tuple(sorted((int(point.geom1), int(point.geom2))))
            if pair[0] == pair[1] or not set(pair) <= self.all_geom_ids:
                self._increment("unknown_mapping_count", period=period)
                invalid = True
                continue
            classification = self._classify_pair(pair)
            pair_classes[pair] = classification
            self._count_classification(classification, period)
            normal = point.normal_force
            if normal is None:
                self._increment("missing_normal_force_count", period=period)
                invalid = True
                continue
            normal = float(normal)
            if not math.isfinite(normal):
                self._increment("nonfinite_normal_force_count", period=period)
                invalid = True
                continue
            if normal < 0.0:
                self._increment(
                    "invalid_negative_normal_force_count", period=period
                )
                invalid = True
                continue
            pair_forces[pair] = pair_forces.get(pair, 0.0) + normal
            pair_point_counts[pair] = pair_point_counts.get(pair, 0) + 1
        self._aggregate_valid_pairs(
            pair_forces, pair_point_counts, pair_classes, period
        )
        if invalid:
            raise EntityContactGraphError(
                "entity contact graph observed invalid force or mapping evidence"
            )

    def _classify_pair(self, pair: tuple[int, int]) -> tuple[object, ...]:
        first, second = pair
        first_robot = first in self.robot_part_by_geom
        second_robot = second in self.robot_part_by_geom
        if first_robot and second_robot:
            return ("robot_self",)
        if first_robot != second_robot:
            robot = first if first_robot else second
            environment = second if first_robot else first
            return (
                "robot_environment",
                robot,
                environment,
                self.robot_part_by_geom[robot],
                self.entity_by_geom[environment],
            )
        first_entity = self.entity_by_geom[first]
        second_entity = self.entity_by_geom[second]
        if _task_relevant_world_edge(first_entity, second_entity):
            edge = tuple(sorted((first_entity, second_entity)))
            return ("world_world", *edge)
        return ("ignored_world_world",)

    def _count_classification(
        self, classification: tuple[object, ...], period: Mapping[str, object]
    ) -> None:
        kind = classification[0]
        if kind == "robot_self":
            self._increment(
                "ignored_robot_self_contact_point_count", period=period
            )
        elif kind == "robot_environment":
            self._increment(
                "robot_environment_contact_point_count", period=period
            )
        elif kind == "world_world":
            self._increment(
                "task_relevant_world_world_contact_point_count", period=period
            )
        else:
            self._increment(
                "ignored_world_world_contact_point_count", period=period
            )

    def _aggregate_valid_pairs(
        self,
        pair_forces: Mapping[tuple[int, int], float],
        pair_point_counts: Mapping[tuple[int, int], int],
        pair_classes: Mapping[tuple[int, int], tuple[object, ...]],
        period: dict[str, object],
    ) -> None:
        category_pairs: dict[str, list[float]] = {
            category: [] for category in CONTACT_CATEGORIES
        }
        category_points = {category: 0 for category in CONTACT_CATEGORIES}
        robot_values: dict[tuple[str, str], list[tuple[float, int]]] = {}
        world_values: dict[tuple[str, str], list[tuple[float, int]]] = {}
        arm_entities = {"left_arm": set(), "right_arm": set()}
        grasp_pads: dict[tuple[str, str], set[int]] = {}
        for pair, force in pair_forces.items():
            classification = pair_classes[pair]
            points = pair_point_counts[pair]
            if classification[0] == "robot_environment":
                robot, _, part, entity = classification[1:]
                category = _entity_role(str(entity))
                category_pairs[category].append(force)
                category_points[category] += points
                edge = (str(part), str(entity))
                robot_values.setdefault(edge, []).append((force, points))
                if force > 0.0:
                    period["contacted_entities"].add(str(entity))
                if force > 0.0 and part in arm_entities:
                    arm_entities[str(part)].add(str(entity))
                    self._record_grasp_pad(
                        grasp_pads, str(part), str(entity), int(robot), force
                    )
            elif classification[0] == "world_world":
                edge = (str(classification[1]), str(classification[2]))
                world_values.setdefault(edge, []).append((force, points))
        for category in CONTACT_CATEGORIES:
            values = category_pairs[category]
            self._update_category(
                category,
                values,
                category_points[category],
                period,
            )
        self._update_edges(robot_values, period, robot=True)
        self._update_edges(world_values, period, robot=False)
        self._record_interactions(arm_entities, grasp_pads, period)

    def _update_category(
        self,
        category: str,
        pair_forces: list[float],
        point_count: int,
        period: Mapping[str, object],
    ) -> None:
        values = {
            "pair_peak": max(pair_forces, default=0.0),
            "total_force": sum(pair_forces),
            "point_count": point_count,
            "pair_count": len(pair_forces),
            "timestep": self.timestep,
        }
        period["categories"][category].update(**values)
        self._episode_categories[category].update(**values)

    def _update_edges(
        self,
        values: Mapping[tuple[str, str], list[tuple[float, int]]],
        period: Mapping[str, object],
        *,
        robot: bool,
    ) -> None:
        period_edges = period["robot_edges" if robot else "world_edges"]
        episode_edges = (
            self._episode_robot_edges if robot else self._episode_world_edges
        )
        for edge, observations in values.items():
            update = {
                "pair_peak": max(value[0] for value in observations),
                "total_force": sum(value[0] for value in observations),
                "point_count": sum(value[1] for value in observations),
                "pair_count": len(observations),
                "timestep": self.timestep,
            }
            period_edges.setdefault(edge, _Accumulator()).update(**update)
            episode_edges.setdefault(edge, _Accumulator()).update(**update)

    def _record_grasp_pad(
        self,
        grasp_pads: dict[tuple[str, str], set[int]],
        part: str,
        entity: str,
        robot_geom: int,
        force: float,
    ) -> None:
        if force <= 0.0 or _entity_role(entity) != "manipulated_object":
            return
        groups = self.gripper_pad_groups.get(part)
        if groups is None:
            return
        for index, group in enumerate(groups):
            if robot_geom in group:
                grasp_pads.setdefault((part, entity), set()).add(index)

    def _record_interactions(
        self,
        arm_entities: Mapping[str, set[str]],
        grasp_pads: Mapping[tuple[str, str], set[int]],
        period: Mapping[str, object],
    ) -> None:
        left = arm_entities["left_arm"]
        right = arm_entities["right_arm"]
        same = sorted(left & right)
        distinct = sorted(
            {tuple(sorted((one, two))) for one in left for two in right if one != two}
        )
        left_grasps = {
            entity
            for (part, entity), pads in grasp_pads.items()
            if part == "left_arm" and pads == {0, 1}
        }
        right_grasps = {
            entity
            for (part, entity), pads in grasp_pads.items()
            if part == "right_arm" and pads == {0, 1}
        }
        dual_grasps = sorted(left_grasps & right_grasps)
        observation = {
            "substep_index": len(self._substep_observations),
            "same_entity_dual_arm_contacts": same,
            "distinct_entity_dual_arm_contacts": [list(value) for value in distinct],
            "left_only_entities": sorted(left - right),
            "right_only_entities": sorted(right - left),
            "left_grasp_qualified_objects": sorted(left_grasps),
            "right_grasp_qualified_objects": sorted(right_grasps),
            "same_object_dual_arm_grasps": dual_grasps,
        }
        self._substep_observations.append(observation)
        period["substeps"].append(observation)
        increments = {
            "same_entity_dual_arm_substep_count": int(bool(same)),
            "same_entity_dual_arm_entity_observation_count": len(same),
            "distinct_entity_dual_arm_substep_count": int(bool(distinct)),
            "distinct_entity_dual_arm_pair_observation_count": len(distinct),
            "single_arm_substep_count": int(bool(left) != bool(right)),
            "same_object_dual_arm_grasp_substep_count": int(bool(dual_grasps)),
            "same_object_dual_arm_grasp_observation_count": len(dual_grasps),
        }
        for name, value in increments.items():
            self._interaction_counts[name] += value
            period["interactions"][name] += value

    def end_control_period(self, motion_state: Mapping[str, object]) -> dict[str, object]:
        if not self.enabled:
            return self.report()
        period = self._require_period()
        end = self._normalize_motion_state(motion_state)
        start = period["motion_start"]
        motions: dict[str, dict[str, object]] = {}
        for entity, source in self.motion_source_by_entity.items():
            displacement = _motion_delta(source.kind, start[entity], end[entity])
            associated = entity in period["contacted_entities"]
            motions[entity] = {
                "motion": displacement,
                "robot_contact_observed": associated,
                "contact_associated_motion": displacement if associated else 0.0,
            }
        report = {
            "period_index": len(self._periods),
            **dict(period["counts"]),
            "legacy_p40_categories": {
                category: period["categories"][category].category_dict(period=True)
                for category in CONTACT_CATEGORIES
            },
            "robot_environment_edges": _edge_reports(
                period["robot_edges"], robot=True, period=True
            ),
            "task_relevant_world_world_edges": _edge_reports(
                period["world_edges"], robot=False, period=True
            ),
            "interactions": dict(period["interactions"]),
            "substeps": list(period["substeps"]),
            "entity_motion": motions,
        }
        self._period = None
        self._periods.append(report)
        return report

    def capture_motion_state(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> dict[str, object]:
        state: dict[str, object] = {}
        for entity, source in self.motion_source_by_entity.items():
            if source.kind == "translation":
                state[entity] = tuple(
                    float(value) for value in data.geom_xpos[source.index]
                )
            else:
                address = int(model.jnt_qposadr[source.index])
                state[entity] = float(data.qpos[address])
        return state

    def _normalize_motion_state(
        self, motion_state: Mapping[str, object]
    ) -> dict[str, float | tuple[float, float, float]]:
        if set(motion_state) != set(self.motion_source_by_entity):
            self._counts["invalid_motion_state_count"] += 1
            raise EntityContactGraphError("entity motion state mapping is incomplete")
        normalized: dict[str, float | tuple[float, float, float]] = {}
        for entity, source in self.motion_source_by_entity.items():
            value = motion_state[entity]
            if source.kind == "translation":
                try:
                    vector = tuple(float(item) for item in value)
                except (TypeError, ValueError) as error:
                    self._counts["invalid_motion_state_count"] += 1
                    raise EntityContactGraphError(
                        "entity translation evidence is invalid"
                    ) from error
                if len(vector) != 3 or not all(map(math.isfinite, vector)):
                    self._counts["invalid_motion_state_count"] += 1
                    raise EntityContactGraphError(
                        "entity translation evidence is invalid"
                    )
                normalized[entity] = vector
            else:
                try:
                    scalar = float(value)
                except (TypeError, ValueError) as error:
                    self._counts["invalid_motion_state_count"] += 1
                    raise EntityContactGraphError(
                        "articulation position evidence is invalid"
                    ) from error
                if not math.isfinite(scalar):
                    self._counts["invalid_motion_state_count"] += 1
                    raise EntityContactGraphError(
                        "articulation position evidence is invalid"
                    )
                normalized[entity] = scalar
        return normalized

    def report(self) -> dict[str, object]:
        return {
            "schema_version": MEASUREMENT_SCHEMA,
            "enabled": self.enabled,
            "measurement_only": True,
            "timestep": self.timestep,
            "mapping": self.mapping_report(),
            "control_period_count": len(self._periods),
            **dict(self._counts),
            "contract_valid": not any(
                self._counts[name] for name in INVALID_FIELDS
            ),
            "legacy_p40_categories": {
                category: self._episode_categories[category].category_dict(period=False)
                for category in CONTACT_CATEGORIES
            },
            "robot_environment_edges": _edge_reports(
                self._episode_robot_edges, robot=True, period=False
            ),
            "task_relevant_world_world_edges": _edge_reports(
                self._episode_world_edges, robot=False, period=False
            ),
            "interactions": dict(self._interaction_counts),
            "substeps": list(self._substep_observations),
            "periods": list(self._periods),
        }

    def mapping_report(self) -> dict[str, object]:
        return {
            "robot_body_roots": self.robot_body_roots,
            "robot_geoms": [
                {
                    "geom_id": geom,
                    "geom_name": self.geom_name_by_id.get(geom, f"geom_{geom}"),
                    "robot_part": self.robot_part_by_geom[geom],
                }
                for geom in sorted(self.robot_part_by_geom)
            ],
            "environment_geoms": [
                {
                    "geom_id": geom,
                    "geom_name": self.geom_name_by_id.get(geom, f"geom_{geom}"),
                    "entity": self.entity_by_geom[geom],
                }
                for geom in sorted(self.entity_by_geom)
            ],
            "motion_sources": {
                entity: {"kind": source.kind, "index": source.index}
                for entity, source in sorted(self.motion_source_by_entity.items())
            },
            "gripper_pad_groups": {
                part: [sorted(groups[0]), sorted(groups[1])]
                for part, groups in sorted(self.gripper_pad_groups.items())
            },
        }

    def _increment(
        self, name: str, amount: int = 1, *, period: Mapping[str, object]
    ) -> None:
        self._counts[name] += amount
        period["counts"][name] += amount

    def _require_period(self) -> dict[str, object]:
        if self._period is None:
            raise EntityContactGraphError(
                "entity contact graph control period is not active"
            )
        return self._period


def resolve_robot_part_by_geom(
    model: mujoco.MjModel,
    robot_geoms: Iterable[int],
    root_names: Mapping[str, str] = ROBOT_BODY_ROOT_NAMES,
) -> tuple[dict[int, str], dict[str, dict[str, object]]]:
    if set(root_names) != set(ROBOT_PARTS):
        raise ValueError("robot body-root keys differ from the frozen contract")
    roots = {
        part: _model_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for part, name in root_names.items()
    }
    if len(set(roots.values())) != len(roots):
        raise ValueError("robot body roots overlap")
    base_root = int(model.body_rootid[roots["base"]])
    expected = {
        geom
        for geom in range(model.ngeom)
        if int(model.body_rootid[int(model.geom_bodyid[geom])]) == base_root
    }
    supplied = {int(geom) for geom in robot_geoms}
    if supplied != expected:
        raise ValueError("backend robot geometry set is incomplete")
    root_by_body = {body: part for part, body in roots.items()}
    result: dict[int, str] = {}
    for geom in sorted(supplied):
        body = int(model.geom_bodyid[geom])
        while body not in root_by_body and body != 0:
            body = int(model.body_parentid[body])
        if body not in root_by_body:
            raise ValueError("robot geometry has an unknown body root")
        result[geom] = root_by_body[body]
    if set(result.values()) != set(ROBOT_PARTS):
        raise ValueError("robot geometry does not cover every frozen robot part")
    identity = {
        part: {"body_name": root_names[part], "body_id": body}
        for part, body in roots.items()
    }
    return result, identity


def p40_conservation_differences(
    graph_report: Mapping[str, object], ledger_report: Mapping[str, object]
) -> dict[str, object]:
    graph_categories = graph_report["legacy_p40_categories"]
    ledger_categories = ledger_report["categories"]
    differences: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for category in CONTACT_CATEGORIES:
        fields: dict[str, float] = {}
        for field in P40_FIELDS:
            difference = abs(
                float(graph_categories[category][field])
                - float(ledger_categories[category][field])
            )
            fields[field] = difference
            maximum = max(maximum, difference)
        differences[category] = fields
    return {
        "scope": "robot_environment_only",
        "world_world_included": False,
        "categories": differences,
        "maximum_absolute_difference": maximum,
        "passed": maximum <= 1.0e-12,
    }


def _empty_categories() -> dict[str, _Accumulator]:
    return {category: _Accumulator() for category in CONTACT_CATEGORIES}


def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(COUNT_FIELDS, 0)


def _empty_interaction_counts() -> dict[str, int]:
    return dict.fromkeys(INTERACTION_FIELDS, 0)


def _edge_reports(
    edges: Mapping[tuple[str, str], _Accumulator],
    *,
    robot: bool,
    period: bool,
) -> list[dict[str, object]]:
    reports = []
    for edge, accumulator in sorted(edges.items()):
        identity = (
            {"robot_part": edge[0], "entity": edge[1]}
            if robot
            else {"entities": list(edge)}
        )
        reports.append({**identity, **accumulator.edge_dict(period=period)})
    return reports


def _entity_role(entity: str) -> str:
    role, separator, identifier = entity.partition(":")
    if separator != ":" or not identifier or role not in ENTITY_ROLES:
        raise ValueError(f"invalid task entity identity: {entity}")
    return role


def _task_relevant_world_edge(first: str, second: str) -> bool:
    first_role, second_role = _entity_role(first), _entity_role(second)
    roles = {first_role, second_role}
    if roles in (
        {"manipulated_object", "floor_support"},
        {"manipulated_object", "target_container"},
        {"manipulated_object", "articulation"},
    ):
        return True
    return (
        first_role == second_role == "manipulated_object" and first != second
    )


def _motion_delta(kind: str, start: object, end: object) -> float:
    if kind == "joint":
        return abs(float(end) - float(start))
    return math.sqrt(
        sum(
            (right - left) ** 2
            for left, right in zip(start, end, strict=True)
        )
    )


def _model_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise ValueError(f"formal MuJoCo model is missing {name}")
    return value
