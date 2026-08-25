"""Static reconstruction and audit for the frozen R0001-P61 contract."""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
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
    source_documents: Mapping[str, str],
) -> dict[str, object]:
    try:
        transitions = _reconstruct_transitions(
            contract, tasks_configuration, bindings_configuration
        )
        source_audit = _audit_sources(source_documents)
        initial = _audit_initial_microinteraction(contract, tasks_configuration)
        full_task = _audit_full_task(transitions, source_audit)
        boundaries = _validate_boundaries(contract)
        checks = {
            "seven_transitions_reconstructed": len(transitions) == 7,
            "stable_transition_ids_unique": tuple(
                row["transition_id"] for row in transitions
            ) == tuple(item[0] for item in TRANSITION_BLUEPRINTS),
            "task_configuration_reconstruction_passed": True,
            "binding_configuration_reconstruction_passed": True,
            "runtime_predicate_reconstruction_passed": source_audit[
                "runtime_predicate_surface_verified"
            ],
            "four_information_boundaries_exact": True,
            "primitive_signature_verified": source_audit[
                "primitive_signature_verified"
            ],
            "candidate_fields_verified": source_audit[
                "candidate_fields_verified"
            ],
            "serialized_policy_fields_verified": source_audit[
                "serialized_policy_fields_verified"
            ],
            "primitive_phases_verified": source_audit[
                "primitive_phases_verified"
            ],
            "candidate_has_no_private_identity": source_audit[
                "candidate_has_no_private_identity"
            ],
            "primitive_input_excludes_private_fields": source_audit[
                "primitive_input_excludes_private_fields"
            ],
            "primitive_callers_exclude_private_fields": source_audit[
                "primitive_callers_exclude_private_fields"
            ],
            "selector_boundary_excludes_private_fields": source_audit[
                "selector_boundary_excludes_private_fields"
            ],
            "evaluator_annotation_isolated": source_audit[
                "evaluator_annotation_isolated"
            ],
            "no_validated_external_planner": not source_audit[
                "validated_external_planner_present"
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
        elif gap and not source_audit["validated_external_planner_present"]:
            decision = "accepted as interaction-contract gap evidence"
        elif (full_task["all_transitions_uniquely_expressible"]
              and source_audit["validated_external_planner_present"]):
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
    return {
        "passed": roles_valid and len(expected) == 3,
        "evaluator_only_annotation_available_for_all_tasks": roles_valid,
        "caller_role_gap_present": True,
        "validated_external_planner_present": False,
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
    rows = []
    for transition in transitions:
        destination_required = (
            transition["interaction_type"] == "pick-transport-place"
        )
        rows.append(
            {
                "transition_id": transition["transition_id"],
                "selected_entity_role_available": False,
                "interaction_type_available": False,
                "destination_available": False,
                "destination_required": destination_required,
                "uniquely_expressible_and_implementable": False,
                "gap_reasons": [
                    "candidate_has_no_entity_or_role_identity",
                    "primitive_has_no_interaction_selector",
                    (
                        "primitive_has_no_destination_target"
                        if destination_required
                        else "primitive_has_no_articulation_threshold"
                    ),
                ],
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
        "current_primitive_arguments": list(
            source_audit["primitive_function_arguments"]
        ),
        "current_candidate_fields": list(source_audit["candidate_fields"]),
        "current_primitive_phases": list(source_audit["primitive_phases"]),
    }


def _audit_sources(
    source_documents: Mapping[str, str],
) -> dict[str, object]:
    target_path = "src/hwr/eval/target_selection.py"
    backend_path = "src/hwr/adapters/mujoco/formal_household_backend.py"
    stability_path = "src/hwr/eval/stability.py"
    for path in (target_path, backend_path, stability_path):
        if path not in source_documents:
            raise InteractionContractError(f"source audit is missing {path}")
    trees = {
        path: _parse_source(path, source)
        for path, source in source_documents.items()
    }
    target_tree = trees[target_path]
    candidate_fields = _class_fields(target_tree, "Candidate")
    policy_fields = _class_fields(target_tree, "PolicyVisibleInput")
    primitive_arguments = _function_arguments(target_tree, "primitive_action")
    selector_arguments = _function_arguments(target_tree, "select_candidate_index")
    phases = tuple(
        item[0] for item in _literal_assignment(target_tree, "PHASES")
    )
    calls = _function_calls(trees, "primitive_action")
    selector_calls = _function_calls(trees, "select_candidate_index")
    private_signature = PRIVATE_ARGUMENT_TOKENS & set(primitive_arguments)
    private_selector_signature = PRIVATE_ARGUMENT_TOKENS & set(selector_arguments)
    private_call_keywords = {
        keyword
        for call in calls
        for keyword in call["keywords"]
        if keyword in PRIVATE_ARGUMENT_TOKENS
    }
    private_selector_keywords = {
        keyword
        for call in selector_calls
        for keyword in call["keywords"]
        if keyword in PRIVATE_ARGUMENT_TOKENS
    }
    runtime_verified = _runtime_predicates_verified(
        trees[backend_path], trees[stability_path]
    )
    isolation_imports = _interaction_contract_imports(trees)
    allowed_imports = {"src/hwr/apps/audit_interaction_contract.py"}
    planner_present = bool(
        private_signature
        or private_selector_signature
        or private_call_keywords
        or private_selector_keywords
    )
    return {
        "candidate_fields": list(candidate_fields),
        "serialized_policy_input_fields": list(policy_fields),
        "primitive_function_arguments": list(primitive_arguments),
        "selector_function_arguments": list(selector_arguments),
        "primitive_phases": list(phases),
        "primitive_call_sites": calls,
        "selector_call_sites": selector_calls,
        "interaction_contract_importers": sorted(isolation_imports),
        "primitive_signature_verified": primitive_arguments == PRIMITIVE_ARGUMENTS,
        "candidate_fields_verified": candidate_fields == CANDIDATE_FIELDS,
        "serialized_policy_fields_verified": policy_fields == SERIALIZED_POLICY_FIELDS,
        "primitive_phases_verified": phases == PRIMITIVE_PHASES,
        "candidate_has_no_private_identity": not (
            PRIVATE_ARGUMENT_TOKENS & set(candidate_fields)
        ),
        "primitive_input_excludes_private_fields": not private_signature,
        "primitive_callers_exclude_private_fields": (
            bool(calls) and not private_call_keywords
            and _calls_match_boundary(calls, PRIMITIVE_ARGUMENTS)
        ),
        "selector_boundary_excludes_private_fields": (
            selector_arguments == SELECTOR_ARGUMENTS
            and bool(selector_calls)
            and not private_selector_signature
            and not private_selector_keywords
            and _calls_match_boundary(selector_calls, SELECTOR_ARGUMENTS)
        ),
        "runtime_predicate_surface_verified": runtime_verified,
        "evaluator_annotation_isolated": isolation_imports <= allowed_imports,
        "validated_external_planner_present": planner_present,
    }


def _runtime_predicates_verified(backend: ast.Module, stability: ast.Module) -> bool:
    backend_methods = _class_method_names(
        backend, "MujocoFormalHouseholdDualArmBackend"
    )
    target_methods = _class_method_names(stability, "TargetVolume")
    placement_methods = _class_method_names(
        stability, "MultiObjectStabilityCriterion"
    )
    task_result = _method_node(
        backend, "MujocoFormalHouseholdDualArmBackend", "_task_result_after_step"
    )
    placement_sample = _method_node(
        backend, "MujocoFormalHouseholdDualArmBackend", "_placement_sample"
    )
    articulation = _method_node(
        backend, "MujocoFormalHouseholdDualArmBackend", "_articulation_satisfied"
    )
    stable = _method_node(stability, "MultiObjectStabilityCriterion", "_stable")
    result_calls = {
        node.func.attr
        for node in ast.walk(task_result)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    articulation_comparators = {
        type(operator).__name__
        for node in ast.walk(articulation)
        if isinstance(node, ast.Compare)
        for operator in node.ops
    }
    stable_comparators = {
        type(operator).__name__
        for node in ast.walk(stable)
        if isinstance(node, ast.Compare)
        for operator in node.ops
    }
    return (
        {"_task_result_after_step", "_placement_sample", "_articulation_satisfied"}
        <= backend_methods
        and "contains" in target_methods
        and {"update", "_stable"} <= placement_methods
        and {"_articulation_satisfied", "update"} <= result_calls
        and {
            "object_geoms", "target_sites", "geom_xpos", "qvel", "site_xpos",
            "site_size", "TargetVolume", "PlacementSample",
        } <= _node_symbols(placement_sample)
        and {"_articulation_position", "minimum_position"} <= _node_symbols(articulation)
        and "GtE" in articulation_comparators
        and {"target", "contains", "max_linear_speed", "max_angular_speed"}
        <= _node_symbols(stable)
        and "LtE" in stable_comparators
    )


def _function_calls(
    trees: Mapping[str, ast.Module], function_name: str
) -> list[dict[str, object]]:
    calls = []
    for path, tree in sorted(trees.items()):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
            )
            if name != function_name:
                continue
            calls.append(
                {
                    "path": path,
                    "line": node.lineno,
                    "positional_argument_count": len(node.args),
                    "keywords": sorted(
                        keyword.arg
                        for keyword in node.keywords
                        if keyword.arg is not None
                    ),
                }
            )
    return calls


def _calls_match_boundary(
    calls: Sequence[Mapping[str, object]], arguments: Sequence[str]
) -> bool:
    allowed = set(arguments)
    return all(
        int(call["positional_argument_count"]) <= len(arguments)
        and set(call["keywords"]) <= allowed for call in calls
    )


def _interaction_contract_imports(
    trees: Mapping[str, ast.Module],
) -> set[str]:
    importers = set()
    for path, tree in trees.items():
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            if any(
                module == "hwr.eval.interaction_contract"
                or module.startswith("hwr.eval.interaction_contract.")
                for module in modules
            ):
                importers.add(path)
    return importers


def _parse_source(path: str, source: str) -> ast.Module:
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as error:
        raise InteractionContractError(f"source audit cannot parse {path}") from error


def _node_symbols(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)} | {
        item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
    }


def _class_fields(tree: ast.Module, name: str) -> tuple[str, ...]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return tuple(
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
            )
    raise InteractionContractError(f"source audit cannot find class {name}")


def _function_arguments(tree: ast.Module, name: str) -> tuple[str, ...]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            return tuple(argument.arg for argument in arguments)
    raise InteractionContractError(f"source audit cannot find function {name}")


def _literal_assignment(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise InteractionContractError(
                    f"source audit cannot evaluate {name}"
                ) from error
    raise InteractionContractError(f"source audit cannot find assignment {name}")


def _class_method_names(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            }
    raise InteractionContractError(f"source audit cannot find class {name}")


def _method_node(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise InteractionContractError(
        f"source audit cannot find method {class_name}.{name}"
    )


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
