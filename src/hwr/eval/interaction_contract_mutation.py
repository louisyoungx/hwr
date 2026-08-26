"""Counterfactual mutation harness for the frozen R0001-P61 audit."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

MUTATION_SCHEMA = "hwr.p72-interaction-contract-mutations/v1"
REPORT_SCHEMA = "hwr.p72-interaction-contract-mutation-report/v1"
PROPOSAL_ID = "R0001-P72-E1"
TARGET_SOURCE = "src/hwr/eval/target_selection.py"
CALLER_SOURCE = "src/hwr/adapters/mujoco/target_selection_diagnostic.py"
MUTATION_NAMES = (
    "candidate_schema_extra_field", "policy_schema_extra_field",
    "primitive_signature_extra_argument", "selector_signature_extra_argument",
    "primitive_phase_extra_value", "initial_annotation_remove_allowed_role",
    "initial_annotation_unknown_role", "initial_annotation_duplicate_task",
    "initial_annotation_consumer_changed",
    "role_field_without_independent_planner_state",
    "fully_expressive_direct_caller_positive_control",
    "remove_interaction_field_negative_control",
    "remove_destination_field_negative_control",
    "remove_articulation_threshold_negative_control",
)
BuildSourceAudit = Callable[[Mapping[str, str], Mapping[str, frozenset[str]]],
                            dict[str, object]]
AuditContract = Callable[[Mapping[str, object], Mapping[str, object],
                          Mapping[str, object], Mapping[str, object]],
                         dict[str, object]]
RequirementFields = Callable[[Mapping[str, object]],
                             dict[str, frozenset[str]]]

def audit_interaction_contract_mutations(
    contract: Mapping[str, object],
    tasks: Mapping[str, object],
    bindings: Mapping[str, object],
    sources: Mapping[str, str],
    p61_transitions: Mapping[str, object],
    *,
    build_source_audit: BuildSourceAudit,
    audit_contract: AuditContract,
    requirement_fields: RequirementFields,
) -> dict[str, object]:
    """Run all frozen mutations twice from independent clean copies."""
    baseline = _execute(
        contract,
        tasks,
        bindings,
        sources,
        build_source_audit,
        audit_contract,
        requirement_fields,
    )
    baseline_replay = _execute(
        contract, tasks, bindings, sources, build_source_audit, audit_contract,
        requirement_fields,
    )
    baseline_deterministic = canonical_bytes(baseline) == canonical_bytes(
        baseline_replay)
    p61_document_matches = (
        canonical_bytes(baseline["audit"]["transitions_document"])
        == canonical_bytes(p61_transitions)
    )
    annotation_matches = (
        canonical_bytes(contract["initial_microinteraction"])
        == canonical_bytes(p61_transitions.get("initial_microinteraction"))
    )

    states: dict[str, dict[str, object]] = {}
    records = []
    for name in MUTATION_NAMES:
        state = _mutation_state(name, contract, tasks, bindings, sources)
        first = _execute_state(
            state, build_source_audit, audit_contract, requirement_fields)
        replay_state = _mutation_state(name, contract, tasks, bindings, sources)
        second = _execute_state(
            replay_state, build_source_audit, audit_contract, requirement_fields)
        state["execution"] = first
        state["canonical_replay_bit_identical"] = canonical_bytes(
            first) == canonical_bytes(second)
        states[name] = state
        records.append(
            _mutation_record(name, state, baseline, states, contract)
        )

    harness_checks = {
        "fourteen_frozen_mutations_executed":
            tuple(record["mutation_id"] for record in records) == MUTATION_NAMES,
        "each_mutation_started_from_clean_copy":
            all(record["clean_copy_verified"] is True for record in records),
        "each_mutation_reached_parser_and_auditor":
            all(record["mutation_reached"] is True for record in records),
        "each_mutation_single_variable":
            all(record["single_variable_verified"] is True for record in records),
        "each_mutated_source_compiles":
            all(record["mutated_sources_compile"] is True for record in records),
        "each_mutation_replay_bit_identical": all(
            record["canonical_replay_bit_identical"] is True for record in records),
        "baseline_replay_bit_identical": baseline_deterministic,
        "baseline_matches_p61_transitions": p61_document_matches,
    }
    harness_valid = all(harness_checks.values())
    exact_residuals = [
        record["mutation_id"]
        for record in records[:5]
        if record["audit_fail_closed"] is not True
    ]
    planner_record = records[9]
    planner_residual = bool(
        planner_record["observations"]["planner_call_state_available"]
        and not planner_record["observations"][
            "independent_planner_state_or_call_evidence"])
    verdict_residuals = [
        record["mutation_id"]
        for record in records[10:]
        if record["expected_verdict_dependency_observed"] is not True
    ]
    p68_records = records[5:9]
    p68_gate = bool(harness_valid and annotation_matches and all(
        record["audit_fail_closed"] is True for record in p68_records))
    residuals = [
        *(
            {
                "kind": "exact_reference",
                "mutation_id": name,
            }
            for name in exact_residuals
        ),
        *(
            (
                {
                    "kind": "planner_evidence",
                    "mutation_id": planner_record["mutation_id"],
                },
            )
            if planner_residual
            else ()
        ),
        *(
            {
                "kind": "verdict_dependency",
                "mutation_id": name,
            }
            for name in verdict_residuals
        ),
    ]
    if not harness_valid:
        decision = "invalid"
    elif residuals:
        decision = "accepted as residual P61 contract gap evidence"
    else:
        decision = "accepted as P61 anti-self-certification audit"
    mutations = {
        "schema_version": MUTATION_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "baseline": {
            **baseline,
            "canonical_replay_bit_identical": baseline_deterministic,
            "p61_transitions_bit_identical": p61_document_matches,
            "initial_annotations_p61_bit_identical": annotation_matches,
        },
        "mutations": records,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "decision": decision,
        "harness_checks": {**harness_checks, "passed": harness_valid},
        "residuals": residuals,
        "residual_exact_reference_gaps": exact_residuals,
        "residual_planner_evidence_gap": planner_residual,
        "residual_verdict_dependency_gaps": verdict_residuals,
        "baseline_initial_annotations_p61_bit_identical": annotation_matches,
        "p68_dependency_mutations_fail_closed": all(
            record["audit_fail_closed"] is True for record in p68_records),
        "p68_dependency_gate_passed": p68_gate,
        "mutation_count": len(records),
        "residual_count_is_not_a_statistical_sample": True,
        "whole_program_planner_claim_allowed": False,
        "training_executed": False,
        "policy_inference_executed": False,
        "capability_claim_allowed": False,
        "task_success_claim_allowed": False,
        "generalization_claim_allowed": False,
        "hardware_safety_claim_allowed": False,
    }
    return {"mutations": mutations, "report": report}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
def _execute_state(
    state: Mapping[str, object],
    build_source_audit: BuildSourceAudit,
    audit_contract: AuditContract,
    requirement_fields: RequirementFields,
) -> dict[str, object]:
    return _execute(
        state["contract"], state["tasks"], state["bindings"], state["sources"],
        build_source_audit, audit_contract, requirement_fields,
    )
def _execute(
    contract: Mapping[str, object],
    tasks: Mapping[str, object],
    bindings: Mapping[str, object],
    sources: Mapping[str, str],
    build_source_audit: BuildSourceAudit,
    audit_contract: AuditContract,
    requirement_fields: RequirementFields,
) -> dict[str, object]:
    requirements = requirement_fields(contract)
    source_audit = build_source_audit(sources, requirements)
    audit = audit_contract(contract, tasks, bindings, source_audit)
    return {"source_audit": source_audit, "audit": audit}
def _mutation_state(
    name: str,
    contract: Mapping[str, object],
    tasks: Mapping[str, object],
    bindings: Mapping[str, object],
    sources: Mapping[str, str],
) -> dict[str, object]:
    state: dict[str, Any] = {
        "contract": copy.deepcopy(contract),
        "tasks": copy.deepcopy(tasks),
        "bindings": copy.deepcopy(bindings),
        "sources": copy.deepcopy(sources),
        "operations": [],
    }
    clean_identity = _input_identity(state)
    _apply_mutation(name, state)
    state["clean_identity"] = clean_identity
    state["mutated_identity"] = _input_identity(state)
    state["clean_copy_verified"] = clean_identity == _input_identity(
        {
            "contract": contract,
            "tasks": tasks,
            "bindings": bindings,
            "sources": sources,
        }
    )
    return state


def _apply_mutation(name: str, state: dict[str, Any]) -> None:
    sources = state["sources"]
    contract = state["contract"]
    operations = state["operations"]
    if name == "candidate_schema_extra_field":
        sources[TARGET_SOURCE] = _add_class_field(
            sources[TARGET_SOURCE], "Candidate", "p72_extra_candidate_field: int"
        )
        operations.append("Candidate.p72_extra_candidate_field")
    elif name == "policy_schema_extra_field":
        sources[TARGET_SOURCE] = _add_class_field(
            sources[TARGET_SOURCE],
            "PolicyVisibleInput",
            "p72_extra_policy_field: int",
        )
        operations.append("PolicyVisibleInput.p72_extra_policy_field")
    elif name == "primitive_signature_extra_argument":
        sources[TARGET_SOURCE] = _add_primitive_fields(
            sources[TARGET_SOURCE], ("p72_extra_primitive_argument",)
        )
        operations.append("primitive_action.p72_extra_primitive_argument")
    elif name == "selector_signature_extra_argument":
        sources[TARGET_SOURCE] = _add_selector_role(
            sources[TARGET_SOURCE], "p72_extra_selector_argument"
        )
        operations.append("select_candidate_index.p72_extra_selector_argument")
    elif name == "primitive_phase_extra_value":
        sources[TARGET_SOURCE] = _replace_once(
            sources[TARGET_SOURCE],
            '    ("B7_stop", 10),\n)',
            '    ("B7_stop", 10),\n    ("P72_extra_phase", 1),\n)',
        )
        operations.append("PHASES.P72_extra_phase")
    elif name == "initial_annotation_remove_allowed_role":
        del contract["initial_microinteraction"][0][
            "allowed_entity_instance_or_roles"
        ][0]
        operations.append("initial_microinteraction[0].allowed_roles.remove")
    elif name == "initial_annotation_unknown_role":
        contract["initial_microinteraction"][0][
            "allowed_entity_instance_or_roles"
        ][0] = "object:unknown_p72"
        operations.append("initial_microinteraction[0].allowed_roles.unknown")
    elif name == "initial_annotation_duplicate_task":
        contract["initial_microinteraction"].append(
            copy.deepcopy(contract["initial_microinteraction"][0])
        )
        operations.append("initial_microinteraction.duplicate_task")
    elif name == "initial_annotation_consumer_changed":
        contract["initial_microinteraction"][0]["consumer"] = "runtime_policy"
        operations.append("initial_microinteraction[0].consumer")
    elif name == "role_field_without_independent_planner_state":
        sources[TARGET_SOURCE] = _add_candidate_role(sources[TARGET_SOURCE])
        sources[TARGET_SOURCE] = _add_selector_role(
            sources[TARGET_SOURCE], "object_or_articulation_identity")
        sources[CALLER_SOURCE] = _add_role_to_existing_caller(
            sources[CALLER_SOURCE])
        operations.extend(
            (
                "Candidate.object_or_articulation_identity",
                "select_candidate_index.object_or_articulation_identity",
                "existing_direct_caller.literal_role",
            )
        )
    elif name in MUTATION_NAMES[10:]:
        capabilities = {
            "interaction_type",
            "destination_target_identity",
            "drawer_requirement",
        }
        if name == "remove_interaction_field_negative_control":
            capabilities.remove("interaction_type")
        elif name == "remove_destination_field_negative_control":
            capabilities.remove("destination_target_identity")
        elif name == "remove_articulation_threshold_negative_control":
            capabilities.remove("drawer_requirement")
        sources[TARGET_SOURCE] = _fully_expressive_source(
            sources[TARGET_SOURCE], capabilities)
        operations.extend(
            (
                "Candidate.object_or_articulation_identity",
                "select_candidate_index.object_or_articulation_identity",
                *(f"primitive_action.{field}" for field in sorted(capabilities)),
                "direct_planner_fixture",
            )
        )
    else:
        raise ValueError(f"unknown P72 mutation: {name}")


def _mutation_record(
    name: str,
    state: Mapping[str, object],
    baseline: Mapping[str, object],
    states: Mapping[str, Mapping[str, object]],
    clean_contract: Mapping[str, object],
) -> dict[str, object]:
    execution = state["execution"]
    reference = (
        states["fully_expressive_direct_caller_positive_control"]["execution"]
        if name in MUTATION_NAMES[11:]
        else baseline
    )
    observations = _observations(name, execution)
    source_delta = _projection_delta(
        _source_projection(reference["source_audit"]),
        _source_projection(execution["source_audit"]),
    )
    changed_inputs = _changed_inputs(
        state["clean_identity"], state["mutated_identity"])
    expected_inputs = _expected_changed_inputs(name)
    expected_semantics = _expected_semantic_delta(name)
    config_delta = _json_leaf_differences(
        clean_contract, state["contract"]
    )
    single_variable = (
        set(changed_inputs) == set(expected_inputs)
        and set(source_delta) == set(expected_semantics)
        and _config_delta_valid(name, config_delta)
    )
    return {
        "mutation_id": name,
        "comparison_reference": (
            "fully_expressive_direct_caller_positive_control"
            if name in MUTATION_NAMES[11:]
            else "clean_p61_baseline"
        ),
        "operations": list(state["operations"]),
        "changed_inputs": changed_inputs,
        "semantic_delta": source_delta,
        "config_delta": config_delta,
        "clean_copy_verified": state["clean_copy_verified"],
        "mutated_sources_compile": _sources_compile(
            state["sources"], changed_inputs),
        "mutation_reached": _mutation_reached(name, execution, observations),
        "single_variable_verified": single_variable,
        "canonical_replay_bit_identical": state[
            "canonical_replay_bit_identical"
        ],
        "canonical_sha256": hashlib.sha256(canonical_bytes(execution)).hexdigest(),
        "audit_decision": execution["audit"]["decision"],
        "audit_validation_error": execution["audit"]["validation_error"],
        "audit_fail_closed": execution["audit"]["decision"] == "invalid",
        "expected_verdict_dependency_observed": _verdict_dependency(name, execution),
        "observations": observations,
        "source_audit": execution["source_audit"],
        "audit": execution["audit"],
    }


def _source_projection(source_audit: Mapping[str, object]) -> dict[str, object]:
    callers = source_audit["direct_call_graph"]["callers"]
    return {
        "candidate_fields": source_audit["candidate_fields"],
        "policy_fields": source_audit["serialized_policy_input_fields"],
        "primitive_arguments": source_audit["primitive_function_arguments"],
        "selector_arguments": source_audit["selector_function_arguments"],
        "primitive_phases": source_audit["primitive_phases"],
        "candidate_role_fields": sorted(
            {field for row in callers for field in row["candidate_role_fields"]}
        ),
        "selected_entity_role_available": any(
            row["selected_entity_role_available"] for row in callers
        ),
        "planner_call_state_available": any(
            row["planner_call_state_available"] for row in callers
        ),
        "interaction_types": sorted(
            {value for row in callers for value in row["interaction_types"]}
        ),
        "destination_target_available": any(
            row["destination_target_available"] for row in callers
        ),
        "articulation_threshold_available": any(
            row["articulation_threshold_available"] for row in callers
        ),
    }


def _observations(
    name: str, execution: Mapping[str, object]
) -> dict[str, object]:
    source_audit = execution["source_audit"]
    audit = execution["audit"]
    projection = _source_projection(source_audit)
    rows = (
        []
        if audit["full_task_contract"] is None
        else audit["full_task_contract"]["transitions"]
    )
    return {
        "frozen_reference": source_audit["frozen_reference"],
        **projection,
        "planner_call_state_available": projection[
            "planner_call_state_available"
        ],
        "independent_planner_state_or_call_evidence": False,
        "all_transitions_uniquely_expressible": (
            None
            if audit["full_task_contract"] is None
            else audit["full_task_contract"][
                "all_transitions_uniquely_expressible"
            ]
        ),
        "gap_reasons_by_transition": {
            row["transition_id"]: row["gap_reasons"] for row in rows
        },
        "named_probe": name,
    }


def _mutation_reached(
    name: str,
    execution: Mapping[str, object],
    observations: Mapping[str, object],
) -> bool:
    frozen = observations["frozen_reference"]
    if name == MUTATION_NAMES[0]:
        return (
            "p72_extra_candidate_field" in observations["candidate_fields"]
            and frozen["candidate_fields_match"] is False
        )
    if name == MUTATION_NAMES[1]:
        return (
            "p72_extra_policy_field" in observations["policy_fields"]
            and frozen["policy_fields_match"] is False
        )
    if name == MUTATION_NAMES[2]:
        return (
            "p72_extra_primitive_argument" in observations["primitive_arguments"]
            and frozen["primitive_arguments_match"] is False
        )
    if name == MUTATION_NAMES[3]:
        return (
            "p72_extra_selector_argument" in observations["selector_arguments"]
            and frozen["selector_arguments_match"] is False
        )
    if name == MUTATION_NAMES[4]:
        return (
            "P72_extra_phase" in observations["primitive_phases"]
            and frozen["primitive_phases_match"] is False
        )
    if name in MUTATION_NAMES[5:9]:
        return (
            execution["audit"]["decision"] == "invalid"
            and "initial microinteraction annotations"
            in str(execution["audit"]["validation_error"])
        )
    if name == MUTATION_NAMES[9]:
        return observations["planner_call_state_available"] is True
    if name == MUTATION_NAMES[10]:
        return (
            execution["audit"]["decision"] == "rejected"
            and observations["all_transitions_uniquely_expressible"] is True
        )
    return _verdict_dependency(name, execution)


def _verdict_dependency(
    name: str, execution: Mapping[str, object]
) -> bool | None:
    audit = execution["audit"]
    if name == MUTATION_NAMES[10]:
        return (
            audit["decision"] == "rejected"
            and audit["full_task_contract"][
                "all_transitions_uniquely_expressible"
            ]
            is True
        )
    if name not in MUTATION_NAMES[11:]:
        return None
    rows = {
        row["transition_id"]: row
        for row in audit["full_task_contract"]["transitions"]
    }
    objects = [row for key, row in rows.items() if key != "R0001-P61-T05"]
    drawer = rows["R0001-P61-T05"]
    if name == MUTATION_NAMES[11]:
        return (
            audit["decision"] == "accepted as interaction-contract gap evidence"
            and all(
                "direct_caller_has_no_required_interaction_type"
                in row["gap_reasons"]
                for row in rows.values()
            )
        )
    if name == MUTATION_NAMES[12]:
        return (
            audit["decision"] == "accepted as interaction-contract gap evidence"
            and all(
                "direct_caller_has_no_destination_target" in row["gap_reasons"]
                for row in objects
            )
            and drawer["uniquely_expressible_and_implementable"] is True
        )
    return (
        audit["decision"] == "accepted as interaction-contract gap evidence"
        and all(row["uniquely_expressible_and_implementable"] for row in objects)
        and "direct_caller_has_no_articulation_threshold"
        in drawer["gap_reasons"]
    )


def _expected_changed_inputs(name: str) -> tuple[str, ...]:
    if name in MUTATION_NAMES[5:9]:
        return ("contract",)
    if name == MUTATION_NAMES[9]:
        return (CALLER_SOURCE, TARGET_SOURCE)
    return (TARGET_SOURCE,)


def _expected_semantic_delta(name: str) -> tuple[str, ...]:
    values = {
        MUTATION_NAMES[0]: ("candidate_fields",),
        MUTATION_NAMES[1]: ("policy_fields",),
        MUTATION_NAMES[2]: ("primitive_arguments",),
        MUTATION_NAMES[3]: ("selector_arguments",),
        MUTATION_NAMES[4]: ("primitive_phases",),
        MUTATION_NAMES[9]: (
            "candidate_fields",
            "candidate_role_fields",
            "planner_call_state_available",
            "selected_entity_role_available",
            "selector_arguments",
        ),
        MUTATION_NAMES[10]: (
            "articulation_threshold_available",
            "candidate_fields",
            "candidate_role_fields",
            "destination_target_available",
            "interaction_types",
            "planner_call_state_available",
            "primitive_arguments",
            "selected_entity_role_available",
            "selector_arguments",
        ),
        MUTATION_NAMES[11]: ("interaction_types", "primitive_arguments"),
        MUTATION_NAMES[12]: (
            "destination_target_available",
            "primitive_arguments",
        ),
        MUTATION_NAMES[13]: (
            "articulation_threshold_available",
            "primitive_arguments",
        ),
    }
    return values.get(name, ())


def _config_delta_valid(name: str, paths: Sequence[str]) -> bool:
    expected = {
        MUTATION_NAMES[5]: (
            "$.initial_microinteraction[0].allowed_entity_instance_or_roles",
        ),
        MUTATION_NAMES[6]: (
            "$.initial_microinteraction[0].allowed_entity_instance_or_roles[0]",
        ),
        MUTATION_NAMES[7]: ("$.initial_microinteraction",),
        MUTATION_NAMES[8]: ("$.initial_microinteraction[0].consumer",),
    }
    return tuple(paths) == expected.get(name, ())


def _input_identity(state: Mapping[str, object]) -> dict[str, str]:
    documents = {
        "contract": state["contract"],
        "tasks": state["tasks"],
        "bindings": state["bindings"],
    }
    result = {
        name: hashlib.sha256(canonical_bytes(value)).hexdigest()
        for name, value in documents.items()
    }
    result.update(
        {
            path: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for path, source in state["sources"].items()
        }
    )
    return result


def _changed_inputs(
    before: Mapping[str, str], after: Mapping[str, str]
) -> list[str]:
    return sorted(name for name in before if before[name] != after[name])


def _projection_delta(
    before: Mapping[str, object], after: Mapping[str, object]
) -> list[str]:
    return sorted(name for name in before if before[name] != after[name])


def _json_leaf_differences(
    before: object, after: object, path: str = "$"
) -> list[str]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        keys = sorted(set(before) | set(after))
        return [
            child
            for key in keys
            for child in _json_leaf_differences(
                before.get(key), after.get(key), f"{path}.{key}"
            )
        ]
    if (
        isinstance(before, Sequence)
        and not isinstance(before, (str, bytes))
        and isinstance(after, Sequence)
        and not isinstance(after, (str, bytes))
    ):
        if len(before) != len(after):
            return [path]
        return [
            child
            for index, (left, right) in enumerate(zip(before, after, strict=True))
            for child in _json_leaf_differences(
                left, right, f"{path}[{index}]"
            )
        ]
    if before != after:
        return [path]
    return []


def _sources_compile(
    sources: Mapping[str, str], changed_inputs: Sequence[str]
) -> bool:
    try:
        for path in changed_inputs:
            if path.endswith(".py"):
                compile(sources[path], path, "exec")
    except SyntaxError:
        return False
    return True


def _add_class_field(source: str, class_name: str, field: str) -> str:
    marker = (
        "    def canonical_key"
        if class_name == "Candidate"
        else "    @property\n    def base_pose"
    )
    return _replace_once(source, marker, f"    {field}\n\n{marker}")


def _add_candidate_role(source: str) -> str:
    return _add_class_field(
        source, "Candidate", "object_or_articulation_identity: str"
    )


def _add_selector_role(source: str, argument: str) -> str:
    return _replace_once(
        source,
        "    acquisition_base_pose: Sequence[float] = (0.0, 0.0, 0.0),\n"
        ") -> int:\n"
        "    if not candidates.candidates:",
        "    acquisition_base_pose: Sequence[float] = (0.0, 0.0, 0.0),\n"
        f"    {argument}: str | None = None,\n"
        ") -> int:\n"
        f"    _ = {argument}\n"
        "    if not candidates.candidates:",
    )


def _add_primitive_fields(source: str, fields: Sequence[str]) -> str:
    annotations = {
        "interaction_type": "str | None",
        "destination_target_identity": "str | None",
        "drawer_requirement": "float | None",
        "p72_extra_primitive_argument": "object | None",
    }
    arguments = "".join(
        f"    {field}: {annotations[field]} = None,\n" for field in fields
    )
    consumed = ", ".join(fields)
    body = (
        f"    _ = {consumed}\n"
        if fields
        else ""
    )
    if "interaction_type" in fields:
        body += (
            "    if interaction_type not in "
            "('pick-transport-place', 'articulate-pull'):\n"
            "        raise TargetSelectionContractError('interaction differs')\n"
        )
    return _replace_once(
        source,
        "    post_selection_step: int,\n"
        ") -> tuple[float, ...]:\n"
        "    value = deserialize_policy_input(serialized_input)",
        "    post_selection_step: int,\n"
        f"{arguments}"
        ") -> tuple[float, ...]:\n"
        f"{body}"
        "    value = deserialize_policy_input(serialized_input)",
    )


def _add_role_to_existing_caller(source: str) -> str:
    return _replace_once(
        source,
        "                        acquisition_base_pose=acquisition_pose,\n"
        "                    )",
        "                        acquisition_base_pose=acquisition_pose,\n"
        "                        object_or_articulation_identity='object:duck',\n"
        "                    )",
    )


def _fully_expressive_source(
    source: str, capabilities: set[str]
) -> str:
    source = _add_candidate_role(source)
    source = _add_selector_role(source, "object_or_articulation_identity")
    source = _add_primitive_fields(source, tuple(sorted(capabilities)))
    call_arguments = "".join(
        f"        {field}={field},\n" for field in sorted(capabilities)
    )
    signature_fields = "".join(
        f"    {field},\n" for field in sorted(capabilities)
    )
    return source + (
        "\n\ndef p72_direct_planner_fixture(\n"
        "    candidates, final_base_pose, payload, candidate,\n"
        "    acquisition_base_pose, post_selection_step,\n"
        "    object_or_articulation_identity,\n"
        f"{signature_fields}"
        "):\n"
        "    selected = select_candidate_index(\n"
        "        candidates, final_base_pose,\n"
        "        acquisition_base_pose=acquisition_base_pose,\n"
        "        object_or_articulation_identity=object_or_articulation_identity,\n"
        "    )\n"
        "    if selected < 0:\n"
        "        return None\n"
        "    return primitive_action(\n"
        "        payload, candidate, acquisition_base_pose, post_selection_step,\n"
        f"{call_arguments}"
        "    )\n"
    )


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"P72 source mutation anchor count differs: {old!r}")
    return source.replace(old, new, 1)
