"""Static reconstruction and audit for the frozen R0001-P61 contract."""

from __future__ import annotations

import ast
import copy
import math
from typing import Mapping, Sequence

from hwr.eval.stability import StabilityConfig

PROPOSAL_ID = "R0001-P61"
CONTRACT_SCHEMA = "hwr.interaction-contract/v1"
TRANSITIONS_SCHEMA = "hwr.p61-interaction-transitions/v1"
REPORT_SCHEMA = "hwr.p61-interaction-contract-report/v1"
TASK_IDS = ("tidy_living_room_3d/v1", "clear_dining_table_3d/v1", "store_kitchen_items_3d/v1")
TRANSITION_BLUEPRINTS = (
    ("R0001-P61-T01", TASK_IDS[0], "duck", None, ()),
    ("R0001-P61-T02", TASK_IDS[0], "football", None, ()),
    ("R0001-P61-T03", TASK_IDS[1], "cup", None, ()),
    ("R0001-P61-T04", TASK_IDS[1], "plate", None, ()),
    ("R0001-P61-T05", TASK_IDS[2], None, "drawer", ()),
    ("R0001-P61-T06", TASK_IDS[2], "cleaner_yellow", None, ("R0001-P61-T05",)),
    ("R0001-P61-T07", TASK_IDS[2], "cleaner_pink", None, ("R0001-P61-T05",)),
)
POLICY_VISIBLE_FIELDS = ("rgb_d", "dynamic_calibration", "proprioception",
                         "history", "phase", "safety_state", "language_encoding")
EVALUATOR_PRIVATE_FIELDS = ("task_transition_id", "object_or_articulation_identity",
                            "destination_target_identity", "target_volume_containment",
                            "drawer_requirement", "success_predicate",
                            "stability_predicate")
PRIMITIVE_INPUT_FIELDS = ("selected_candidate", "acquisition_base_pose",
                          "current_base_pose", "policy_visible_joint_positions",
                          "policy_visible_action_history")
PLANNER_CALL_STATE = {"fields": [], "transition_id_available": False,
                      "destination_target_available": False,
                      "validated_external_planner_present": False}
OBJECT_FORBIDDEN_ROLES = (
    "articulation", "target_container", "floor_support", "other_furniture",
    "robot", "site", "unknown", "mixed",
)
ARTICULATION_FORBIDDEN_ROLES = (
    "manipulated_object", "target_container", "floor_support", "other_furniture",
    "robot", "site", "unknown", "mixed",
)
CANDIDATE_FIELDS = (
    "center", "normal", "width", "prominence", "support_count", "view_count",
    "first_frame", "first_row", "first_column",
)
SERIALIZED_POLICY_FIELDS = (
    "observation_timestamp_ns", "sequence_id", "phase_index", "phase_step",
    "policy_rng_seed", "safety_state", "head_rgb_uint8", "head_depth_m",
    "head_depth_valid", "head_camera_intrinsics", "robot_from_head_camera",
    "proprioception", "executed_action_history", "history_available",
)
PRIMITIVE_ARGUMENTS = (
    "serialized_input", "candidate", "acquisition_base_pose", "post_selection_step",
)
SELECTOR_ARGUMENTS = ("candidates", "final_base_pose", "acquisition_base_pose")
PRIMITIVE_PHASES = (
    "B0_orient", "B1_approach", "B2_preposition", "B3_contact_approach",
    "B4_close", "B5_pull", "B6_retract", "B7_stop",
)
PRIVATE_ARGUMENT_TOKENS = frozenset(
    ("transition", "transition_id", "task_id", "task_target", "destination",
     "destination_target", "target_id", "object_id", "articulation_id",
     "entity_role", "reward", "success", "stage")
)


class InteractionContractError(ValueError): pass


def audit_interaction_contract(
    contract: Mapping[str, object],
    tasks_configuration: Mapping[str, object],
    bindings_configuration: Mapping[str, object],
    source_audit: Mapping[str, object],
) -> dict[str, object]:
    try:
        transitions = _reconstruct_transitions(
            contract, tasks_configuration, bindings_configuration
        )
        _require_source_audit(source_audit)
        initial = _audit_initial_microinteraction(
            contract, tasks_configuration, source_audit
        )
        full_task = _audit_full_task(transitions, source_audit)
        boundaries = _validate_boundaries(contract)
        source_checks = source_audit["checks"]
        checks = {
            "seven_transitions_reconstructed": len(transitions) == 7,
            "stable_transition_ids_unique": tuple(
                row["transition_id"] for row in transitions
            ) == tuple(item[0] for item in TRANSITION_BLUEPRINTS),
            "task_configuration_reconstruction_passed": True,
            "binding_configuration_reconstruction_passed": True,
            "runtime_predicate_reconstruction_passed": source_checks[
                "runtime_predicate_surface_verified"
            ],
            "four_information_boundaries_exact": True,
            "candidate_schema_audited": source_checks["candidate_schema_audited"],
            "policy_schema_audited": source_checks["policy_schema_audited"],
            "primitive_signature_audited": source_checks[
                "primitive_signature_audited"
            ],
            "selector_signature_audited": source_checks[
                "selector_signature_audited"
            ],
            "direct_call_graph_resolved": source_checks[
                "direct_call_graph_resolved"
            ],
            "direct_call_scope_declared": source_audit["analysis_scope"]["kind"]
            == "finite_static_same_function_direct_calls",
            "evaluator_annotation_isolated": source_checks[
                "evaluator_annotation_isolated"
            ],
            "initial_microinteraction_annotation_unique": initial["passed"],
            "full_task_assessed_transition_by_transition":
                len(full_task["transitions"]) == len(transitions),
        }
        valid = all(checks.values())
        gap = bool(full_task["contract_gap_present"]) or bool(
            initial["caller_role_gap_present"]
        )
        if not valid:
            decision = "invalid"
        elif gap:
            decision = "accepted as interaction-contract gap evidence"
        elif (full_task["all_transitions_uniquely_expressible"]
              and initial["validated_external_planner_present"]):
            decision = "rejected"
        else:
            decision = "invalid"
        transitions_document = {
            "schema_version": TRANSITIONS_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "transition_count": len(transitions),
            "transitions": transitions,
            "information_boundaries": boundaries,
            "initial_microinteraction": initial["annotations"],
            "source_boundary_evidence": source_audit,
        }
        return {
            "decision": decision,
            "validation_error": None,
            "checks": {**checks, "passed": valid},
            "full_task_contract": full_task,
            "initial_microinteraction_contract": {
                key: value for key, value in initial.items() if key != "annotations"
            },
            "transitions_document": transitions_document,
        }
    except InteractionContractError as error:
        return {
            "decision": "invalid",
            "validation_error": str(error),
            "checks": {"passed": False},
            "full_task_contract": None,
            "initial_microinteraction_contract": None,
            "transitions_document": {
                "schema_version": TRANSITIONS_SCHEMA,
                "proposal_id": PROPOSAL_ID,
                "transition_count": 0,
                "transitions": [],
                "information_boundaries": {},
                "initial_microinteraction": [],
                "source_boundary_evidence": {},
            },
        }


def _reconstruct_transitions(
    contract: Mapping[str, object],
    tasks_configuration: Mapping[str, object],
    bindings_configuration: Mapping[str, object],
) -> list[dict[str, object]]:
    _require_equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _require_equal(contract.get("proposal_id"), PROPOSAL_ID, "proposal ID")
    _require_equal(
        tasks_configuration.get("schema_version"),
        "hwr.formal-tasks/v1",
        "task schema",
    )
    _require_equal(
        bindings_configuration.get("schema_version"),
        "hwr.mujoco-formal-bindings/v1",
        "binding schema",
    )
    tasks = _indexed_objects(tasks_configuration.get("tasks"), "task_id", "tasks")
    bindings = _indexed_objects(
        bindings_configuration.get("bindings"), "task_id", "bindings"
    )
    _require_equal(tuple(tasks), TASK_IDS, "formal task order")
    _require_equal(set(bindings), set(TASK_IDS), "formal binding IDs")
    configured = contract.get("transitions")
    if not isinstance(configured, list):
        raise InteractionContractError("contract transitions must be a list")
    expected = [
        _transition_from_sources(
            {
                **tasks[task_id],
                "control_hz": tasks_configuration.get("control_hz"),
                "hold_seconds": tasks_configuration.get("hold_seconds"),
            },
            bindings[task_id],
            blueprint,
        )
        for blueprint in TRANSITION_BLUEPRINTS
        for _, task_id, _, _, _ in (blueprint,)
    ]
    _require_equal(configured, expected, "versioned transitions")
    return copy.deepcopy(expected)


def _transition_from_sources(
    task: Mapping[str, object],
    binding: Mapping[str, object],
    blueprint: tuple[str, str, str | None, str | None, tuple[str, ...]],
) -> dict[str, object]:
    transition_id, task_id, object_id, articulation_id, dependencies = blueprint
    _require_equal(task.get("task_id"), task_id, f"{transition_id} task")
    _require_equal(binding.get("task_id"), task_id, f"{transition_id} binding")
    common = {
        "transition_id": transition_id,
        "task_id": task_id,
        "dependencies": list(dependencies),
        "policy_visible_fields": list(POLICY_VISIBLE_FIELDS),
        "planner_call_state": copy.deepcopy(PLANNER_CALL_STATE),
        "primitive_input_fields": list(PRIMITIVE_INPUT_FIELDS),
        "evaluator_private_fields": list(EVALUATOR_PRIVATE_FIELDS),
    }
    if object_id is not None:
        return {**common, **_object_transition(
            task, binding, transition_id, object_id, dependencies)}
    return {**common, **_articulation_transition(
        task, binding, transition_id, str(articulation_id))}


def _object_transition(
    task: Mapping[str, object],
    binding: Mapping[str, object],
    transition_id: str,
    object_id: str,
    dependencies: tuple[str, ...],
) -> dict[str, object]:
    objects = _indexed_objects(task.get("objects"), "object_id", "task objects")
    task_object = objects.get(object_id)
    binding_objects = binding.get("objects")
    if not isinstance(task_object, Mapping) or not isinstance(binding_objects, Mapping):
        raise InteractionContractError(f"{transition_id} object source is missing")
    bound = binding_objects.get(object_id)
    if not isinstance(bound, Mapping):
        raise InteractionContractError(f"{transition_id} object binding is missing")
    target_id = _required_string(task_object, "target_id", transition_id)
    collision_geom = _required_string(bound, "collision_geom", transition_id)
    target_site = _required_string(bound, "target_site", transition_id)
    control_hz = _required_number(task, "control_hz", transition_id)
    hold_seconds = _required_number(task, "hold_seconds", transition_id)
    stability = StabilityConfig(control_hz, hold_seconds)
    condition: dict[str, object] = {
        "kind": "object_not_stably_contained",
        "object_id": object_id,
        "target_id": target_id,
    }
    if dependencies:
        requirement = _articulation_requirement(task, transition_id)
        condition["requires"] = {
            "kind": "articulation_position_at_least",
            **requirement,
        }
    return {
        "precondition": condition,
        "allowed_entity_instance_or_role": f"object:{object_id}",
        "forbidden_roles": list(OBJECT_FORBIDDEN_ROLES),
        "interaction_type": "pick-transport-place",
        "expected_state_change": {
            "kind": "object_stably_contained",
            "object_id": object_id,
            "target_id": target_id,
        },
        "evaluator_predicate": {
            "kind": "stable_target_volume",
            "object_id": object_id,
            "collision_geom": collision_geom,
            "target_id": target_id,
            "target_site": target_site,
            "containment_predicate": "TargetVolume.contains",
            "stability_predicate": "MultiObjectStabilityCriterion._stable",
            "control_hz": control_hz,
            "hold_seconds": hold_seconds,
            "required_hold_steps": math.ceil(control_hz * hold_seconds),
            "maximum_linear_speed_m_s": stability.max_linear_speed,
            "maximum_angular_speed_rad_s": stability.max_angular_speed,
        },
    }


def _articulation_transition(
    task: Mapping[str, object],
    binding: Mapping[str, object],
    transition_id: str,
    articulation_id: str,
) -> dict[str, object]:
    requirement = _articulation_requirement(task, transition_id)
    _require_equal(
        requirement["articulation_id"],
        articulation_id,
        f"{transition_id} articulation",
    )
    bound = binding.get("articulation")
    if not isinstance(bound, Mapping):
        raise InteractionContractError(f"{transition_id} articulation binding is missing")
    _require_equal(
        bound.get("articulation_id"), articulation_id, f"{transition_id} binding"
    )
    minimum = requirement["minimum_position_m"]
    return {
        "precondition": {
            "kind": "articulation_position_below",
            "articulation_id": articulation_id,
            "maximum_exclusive_m": minimum,
        },
        "allowed_entity_instance_or_role": f"articulation:{articulation_id}",
        "forbidden_roles": list(ARTICULATION_FORBIDDEN_ROLES),
        "interaction_type": "articulate-pull",
        "expected_state_change": {
            "kind": "articulation_position_at_least",
            "articulation_id": articulation_id,
            "minimum_position_m": minimum,
        },
        "evaluator_predicate": {
            "kind": "articulation_position_at_least",
            "articulation_id": articulation_id,
            "joint": _required_string(bound, "joint", transition_id),
            "handle_geom": _required_string(bound, "handle_geom", transition_id),
            "minimum_position_m": minimum,
            "runtime_predicate": (
                "MujocoFormalHouseholdDualArmBackend._articulation_satisfied"
            ),
        },
    }


def _articulation_requirement(
    task: Mapping[str, object], context: str
) -> dict[str, object]:
    value = task.get("articulation")
    if not isinstance(value, Mapping):
        raise InteractionContractError(f"{context} articulation requirement is missing")
    return {
        "articulation_id": _required_string(value, "articulation_id", context),
        "minimum_position_m": _required_number(value, "minimum_position", context),
    }


def _validate_boundaries(
    contract: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "evaluator_private": {"fields": list(EVALUATOR_PRIVATE_FIELDS)},
        "planner_call_state": copy.deepcopy(PLANNER_CALL_STATE),
        "policy_visible": {"fields": list(POLICY_VISIBLE_FIELDS)},
        "primitive_input": {
            "fields": list(PRIMITIVE_INPUT_FIELDS),
            "evaluator_private_fields_allowed": [],
        },
    }
    _require_equal(
        contract.get("information_boundaries"), expected, "information boundaries"
    )
    return expected


def _audit_initial_microinteraction(
    contract: Mapping[str, object],
    tasks_configuration: Mapping[str, object],
    source_audit: Mapping[str, object],
) -> dict[str, object]:
    expected = [
        _micro_annotation(
            TASK_IDS[0], ("object:duck", "object:football"), OBJECT_FORBIDDEN_ROLES
        ),
        _micro_annotation(
            TASK_IDS[1], ("object:cup", "object:plate"), OBJECT_FORBIDDEN_ROLES
        ),
        _micro_annotation(
            TASK_IDS[2], ("articulation:drawer",), ARTICULATION_FORBIDDEN_ROLES
        ),
    ]
    _require_equal(
        contract.get("initial_microinteraction"),
        expected,
        "initial microinteraction annotations",
    )
    tasks = _indexed_objects(tasks_configuration.get("tasks"), "task_id", "tasks")
    known = {
        task_id: {
            *(f"object:{item['object_id']}" for item in task["objects"]),
            *(
                ()
                if not isinstance(task.get("articulation"), Mapping)
                else (
                    f"articulation:{task['articulation']['articulation_id']}",
                )
            ),
        }
        for task_id, task in tasks.items()
    }
    roles_valid = all(
        set(item["allowed_entity_instance_or_roles"]) <= known[item["task_id"]]
        for item in expected
    )
    caller_records = source_audit["direct_call_graph"]["callers"]
    caller_role_available = any(
        record["selected_entity_role_available"]
        for record in caller_records
    )
    planner_present = any(
        record["planner_call_state_available"]
        for record in caller_records
    )
    return {
        "passed": roles_valid and len(expected) == 3,
        "evaluator_only_annotation_available_for_all_tasks": roles_valid,
        "caller_role_gap_present": not caller_role_available,
        "validated_external_planner_present": planner_present,
        "supporting_direct_callers": [
            record["caller_id"]
            for record in caller_records
            if record["selected_entity_role_available"]
        ],
        "annotations": expected,
    }


def _micro_annotation(
    task_id: str, allowed: Sequence[str], forbidden: Sequence[str]
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "scope": "reset-first-acquisition-plus-one-generic-primitive",
        "consumer": "future_evaluator_only",
        "allowed_entity_instance_or_roles": list(allowed),
        "forbidden_roles": list(forbidden),
    }


def _audit_full_task(
    transitions: Sequence[Mapping[str, object]],
    source_audit: Mapping[str, object],
) -> dict[str, object]:
    callers = source_audit["direct_call_graph"]["callers"]
    rows = []
    for transition in transitions:
        destination_required = (
            transition["interaction_type"] == "pick-transport-place"
        )
        compatible = [
            record
            for record in callers
            if _caller_supports_transition(record, transition)
        ]
        role_available = any(
            record["selected_entity_role_available"] for record in callers
        )
        interaction_available = any(
            transition["interaction_type"] in record["interaction_types"]
            for record in callers
        )
        destination_available = any(
            (
                record["destination_target_available"]
                if destination_required
                else record["articulation_threshold_available"]
            )
            for record in callers
        )
        reasons = []
        if not role_available:
            reasons.append("direct_caller_has_no_entity_or_role_identity")
        if not interaction_available:
            reasons.append("direct_caller_has_no_required_interaction_type")
        if not destination_available:
            reasons.append(
                "direct_caller_has_no_destination_target"
                if destination_required
                else "direct_caller_has_no_articulation_threshold"
            )
        if role_available and interaction_available and destination_available and not compatible:
            reasons.append("no_single_direct_caller_combines_required_fields")
        rows.append(
            {
                "transition_id": transition["transition_id"],
                "selected_entity_role_available": role_available,
                "interaction_type_available": interaction_available,
                "destination_available": destination_available,
                "destination_required": destination_required,
                "uniquely_expressible_and_implementable": bool(compatible),
                "supporting_direct_callers": [
                    record["caller_id"] for record in compatible
                ],
                "gap_reasons": reasons,
            }
        )
    all_expressible = all(
        row["uniquely_expressible_and_implementable"] for row in rows
    )
    return {
        "transition_count": len(rows),
        "transitions": rows,
        "all_transitions_uniquely_expressible": all_expressible,
        "contract_gap_present": not all_expressible,
        "analysis_scope": source_audit["analysis_scope"],
        "current_primitive_arguments": source_audit["primitive_function_arguments"],
        "current_candidate_fields": source_audit["candidate_fields"],
        "current_primitive_phases": source_audit["primitive_phases"],
    }


def _caller_supports_transition(
    caller: Mapping[str, object], transition: Mapping[str, object]
) -> bool:
    destination_required = transition["interaction_type"] == "pick-transport-place"
    return (
        caller["selected_entity_role_available"]
        and transition["interaction_type"] in caller["interaction_types"]
        and (
            caller["destination_target_available"]
            if destination_required
            else caller["articulation_threshold_available"]
        )
    )


def _require_source_audit(source_audit: Mapping[str, object]) -> None:
    required = {
        "analysis_scope", "candidate_fields", "serialized_policy_input_fields",
        "primitive_function_arguments", "selector_function_arguments",
        "primitive_phases", "direct_call_graph", "checks",
    }
    if not required <= set(source_audit):
        raise InteractionContractError("source audit is incomplete")


def source_requirement_fields(
    contract: Mapping[str, object],
) -> dict[str, frozenset[str]]:
    boundaries = contract.get("information_boundaries")
    transitions = contract.get("transitions")
    if not isinstance(boundaries, Mapping) or not isinstance(transitions, list):
        raise InteractionContractError("source requirements are missing")
    private = boundaries.get("evaluator_private")
    if not isinstance(private, Mapping) or not isinstance(private.get("fields"), list):
        raise InteractionContractError("private source requirements are missing")
    fields = frozenset(str(field) for field in private["fields"])
    return {
        "role_fields": fields
        & frozenset(("task_transition_id", "object_or_articulation_identity")),
        "interaction_fields": frozenset(("interaction_type",)),
        "destination_fields": fields & frozenset(("destination_target_identity",)),
        "threshold_fields": fields & frozenset(("drawer_requirement",)),
        "interaction_types": frozenset(
            str(transition["interaction_type"])
            for transition in transitions
            if isinstance(transition, Mapping)
        ),
    }


def runtime_predicates_verified(
    backend: ast.Module, stability: ast.Module
) -> bool:
    task_result = _method_node(
        backend, "MujocoFormalHouseholdDualArmBackend", "_task_result_after_step"
    )
    placement = _method_node(
        backend, "MujocoFormalHouseholdDualArmBackend", "_placement_sample"
    )
    articulation = _method_node(
        backend, "MujocoFormalHouseholdDualArmBackend", "_articulation_satisfied"
    )
    stable = _method_node(stability, "MultiObjectStabilityCriterion", "_stable")
    return (
        {"_articulation_satisfied", "update"} <= _called_attributes(task_result)
        and {"TargetVolume", "PlacementSample", "target_sites"}
        <= _node_symbols(placement)
        and {"_articulation_position", "minimum_position"}
        <= _node_symbols(articulation)
        and any(
            isinstance(operator, ast.GtE)
            for node in ast.walk(articulation)
            if isinstance(node, ast.Compare)
            for operator in node.ops
        )
        and {"contains", "max_linear_speed", "max_angular_speed"}
        <= _node_symbols(stable)
    )


def _called_attributes(node: ast.AST) -> set[str]:
    return {
        item.func.attr
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
    }


def _method_node(
    tree: ast.Module, class_name: str, name: str
) -> ast.FunctionDef:
    class_node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return next(
        item for item in class_node.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )


def _node_symbols(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)} | {
        item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
    }


def _indexed_objects(
    value: object, key: str, context: str
) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list):
        raise InteractionContractError(f"{context} must be a list")
    result: dict[str, Mapping[str, object]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise InteractionContractError(f"{context} entries must be objects")
        identifier = item.get(key)
        if not isinstance(identifier, str) or not identifier:
            raise InteractionContractError(f"{context} entry lacks {key}")
        if identifier in result:
            raise InteractionContractError(f"{context} contains duplicate {identifier}")
        result[identifier] = item
    return result


def _required_string(
    value: Mapping[str, object], key: str, context: str
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise InteractionContractError(f"{context} requires string {key}")
    return result


def _required_number(
    value: Mapping[str, object], key: str, context: str
) -> float:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise InteractionContractError(f"{context} requires numeric {key}")
    result = float(result)
    if not math.isfinite(result) or result <= 0.0:
        raise InteractionContractError(f"{context} requires positive finite {key}")
    return result


def _require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise InteractionContractError(f"{context} differs from frozen contract")
