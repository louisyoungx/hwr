from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from hwr.apps import audit_interaction_contract as app
from hwr.eval import interaction_contract as contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / app.FORMAL_CONTRACT
TASKS_PATH = ROOT / app.TASK_CONFIGURATION
BINDINGS_PATH = ROOT / app.BINDING_CONFIGURATION


@pytest.fixture(scope="module")
def inputs() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, str],
]:
    return (
        _read(CONTRACT_PATH),
        _read(TASKS_PATH),
        _read(BINDINGS_PATH),
        _sources(),
    )


@pytest.fixture(scope="module")
def audit(
    inputs: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, str],
    ],
) -> dict[str, object]:
    return contract.audit_interaction_contract(*inputs)


def test_frozen_sources_reconstruct_all_transitions(
    audit: dict[str, object],
) -> None:
    assert audit["decision"] == (
        "accepted as interaction-contract gap evidence"
    )
    assert audit["validation_error"] is None
    assert audit["checks"]["passed"] is True
    assert all(audit["checks"].values())
    transitions = audit["transitions_document"]
    assert transitions["schema_version"] == contract.TRANSITIONS_SCHEMA
    assert transitions["transition_count"] == 7
    assert [row["transition_id"] for row in transitions["transitions"]] == [
        item[0] for item in contract.TRANSITION_BLUEPRINTS
    ]
    assert [
        row["allowed_entity_instance_or_role"]
        for row in transitions["transitions"]
    ] == [
        "object:duck",
        "object:football",
        "object:cup",
        "object:plate",
        "articulation:drawer",
        "object:cleaner_yellow",
        "object:cleaner_pink",
    ]


def test_transition_order_only_encodes_kitchen_drawer_dependency(
    audit: dict[str, object],
) -> None:
    rows = {
        row["transition_id"]: row
        for row in audit["transitions_document"]["transitions"]
    }
    for transition_id in (
        "R0001-P61-T01",
        "R0001-P61-T02",
        "R0001-P61-T03",
        "R0001-P61-T04",
        "R0001-P61-T05",
    ):
        assert rows[transition_id]["dependencies"] == []
    assert rows["R0001-P61-T06"]["dependencies"] == ["R0001-P61-T05"]
    assert rows["R0001-P61-T07"]["dependencies"] == ["R0001-P61-T05"]
    for transition_id in ("R0001-P61-T06", "R0001-P61-T07"):
        requirement = rows[transition_id]["precondition"]["requires"]
        assert requirement == {
            "kind": "articulation_position_at_least",
            "articulation_id": "drawer",
            "minimum_position_m": 0.3,
        }


def test_runtime_predicates_and_information_boundaries_are_exact(
    audit: dict[str, object],
) -> None:
    transitions = audit["transitions_document"]
    rows = transitions["transitions"]
    object_rows = [row for row in rows if row["interaction_type"] != "articulate-pull"]
    drawer = next(row for row in rows if row["transition_id"].endswith("T05"))
    assert all(
        row["evaluator_predicate"]["required_hold_steps"] == 40
        and row["evaluator_predicate"]["maximum_linear_speed_m_s"] == 0.03
        and row["evaluator_predicate"]["maximum_angular_speed_rad_s"] == 0.15
        for row in object_rows
    )
    assert drawer["evaluator_predicate"]["minimum_position_m"] == 0.3
    assert transitions["information_boundaries"] == {
        "evaluator_private": {
            "fields": list(contract.EVALUATOR_PRIVATE_FIELDS)
        },
        "planner_call_state": contract.PLANNER_CALL_STATE,
        "policy_visible": {"fields": list(contract.POLICY_VISIBLE_FIELDS)},
        "primitive_input": {
            "fields": list(contract.PRIMITIVE_INPUT_FIELDS),
            "evaluator_private_fields_allowed": [],
        },
    }


def test_source_audit_proves_current_primitive_boundary(
    audit: dict[str, object],
) -> None:
    evidence = audit["transitions_document"]["source_boundary_evidence"]
    assert evidence["candidate_fields"] == list(contract.CANDIDATE_FIELDS)
    assert evidence["serialized_policy_input_fields"] == list(
        contract.SERIALIZED_POLICY_FIELDS
    )
    assert evidence["primitive_function_arguments"] == list(
        contract.PRIMITIVE_ARGUMENTS
    )
    assert evidence["primitive_phases"] == list(contract.PRIMITIVE_PHASES)
    assert evidence["validated_external_planner_present"] is False
    assert evidence["interaction_contract_importers"] == [
        "src/hwr/apps/audit_interaction_contract.py"
    ]
    assert evidence["primitive_call_sites"]
    assert all(
        set(call["keywords"]).isdisjoint(contract.PRIVATE_ARGUMENT_TOKENS)
        for call in evidence["primitive_call_sites"]
    )


def test_full_task_and_initial_contracts_are_separate(
    audit: dict[str, object],
) -> None:
    full_task = audit["full_task_contract"]
    initial = audit["initial_microinteraction_contract"]
    assert full_task["contract_gap_present"] is True
    assert full_task["all_transitions_uniquely_expressible"] is False
    assert all(
        not row["uniquely_expressible_and_implementable"]
        and "candidate_has_no_entity_or_role_identity" in row["gap_reasons"]
        for row in full_task["transitions"]
    )
    assert initial == {
        "passed": True,
        "evaluator_only_annotation_available_for_all_tasks": True,
        "caller_role_gap_present": True,
        "validated_external_planner_present": False,
    }
    annotations = audit["transitions_document"]["initial_microinteraction"]
    assert annotations[0]["allowed_entity_instance_or_roles"] == [
        "object:duck",
        "object:football",
    ]
    assert annotations[1]["allowed_entity_instance_or_roles"] == [
        "object:cup",
        "object:plate",
    ]
    assert annotations[2]["allowed_entity_instance_or_roles"] == [
        "articulation:drawer"
    ]


@pytest.mark.parametrize(
    ("document_index", "mutation"),
    (
        (
            0,
            lambda value: value["transitions"][0]["expected_state_change"].update(
                {"target_id": "wrong_target"}
            ),
        ),
        (
            1,
            lambda value: value["tasks"][0]["objects"][0].update(
                {"target_id": "wrong_target"}
            ),
        ),
        (
            2,
            lambda value: value["bindings"][0]["objects"]["duck"].update(
                {"target_site": "wrong_site"}
            ),
        ),
    ),
)
def test_config_task_and_binding_drift_fail_closed(
    inputs: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, str],
    ],
    document_index: int,
    mutation,
) -> None:
    changed = copy.deepcopy(list(inputs))
    mutation(changed[document_index])
    result = contract.audit_interaction_contract(*changed)
    assert result["decision"] == "invalid"
    assert result["checks"] == {"passed": False}
    assert result["validation_error"]
    assert result["transitions_document"]["transitions"] == []


@pytest.mark.parametrize(
    ("path", "change", "failed_check"),
    (
        (
            "src/hwr/eval/target_selection.py",
            lambda source: source.replace(
                "    post_selection_step: int,\n)",
                "    post_selection_step: int,\n    transition_id: str,\n)",
                1,
            ),
            "primitive_signature_verified",
        ),
        (
            "src/hwr/adapters/mujoco/formal_household_backend.py",
            lambda source: source.replace(
                ">= requirement.minimum_position",
                "< requirement.minimum_position",
                1,
            ),
            "runtime_predicate_reconstruction_passed",
        ),
        (
            "src/hwr/eval/target_selection.py",
            lambda source: (
                "from hwr.eval.interaction_contract import PROPOSAL_ID\n" + source
            ),
            "evaluator_annotation_isolated",
        ),
        (
            "src/hwr/adapters/mujoco/target_selection_diagnostic.py",
            lambda source: source
            + "\ndef _leaky_call():\n"
            + "    primitive_action(b'', None, (), 0, transition_id='private')\n",
            "primitive_callers_exclude_private_fields",
        ),
    ),
)
def test_source_boundary_drift_fails_closed(
    inputs: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, str],
    ],
    path: str,
    change,
    failed_check: str,
) -> None:
    changed = copy.deepcopy(inputs[3])
    changed[path] = change(changed[path])
    result = contract.audit_interaction_contract(
        inputs[0], inputs[1], inputs[2], changed
    )
    assert result["decision"] == "invalid"
    assert result["checks"][failed_check] is False
    assert result["checks"]["passed"] is False


def test_deterministic_reconstruction_is_bit_identical(
    inputs: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, str],
    ],
) -> None:
    first = contract.audit_interaction_contract(*inputs)
    second = contract.audit_interaction_contract(*inputs)
    assert app._canonical_bytes(first) == app._canonical_bytes(second)


def test_runner_writes_three_hash_bound_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "p61"
    provenance = {
        "checks": {"workspace_clean": True, "passed": True},
        "inputs": {
            app.FORMAL_CONTRACT.as_posix(): {
                "bytes": CONTRACT_PATH.stat().st_size,
                "sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            }
        },
    }
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        app, "_provenance", lambda root, source_commit: provenance
    )
    arguments = app.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--output",
            str(output),
        ]
    )
    result = app.run(arguments)
    assert result["decision"] == (
        "accepted as interaction-contract gap evidence"
    )
    assert set(path.name for path in output.iterdir()) == {
        "transitions.json",
        "report.json",
        "manifest.json",
    }
    transitions = _read(output / "transitions.json")
    report = _read(output / "report.json")
    manifest = _read(output / "manifest.json")
    assert transitions["transition_count"] == result["transition_count"] == 7
    assert report["decision"] == manifest["decision"] == result["decision"]
    assert report["source_commit"] == manifest["source_commit"] == "a" * 40
    assert report["checks"]["deterministic_reconstruction_bit_identical"] is True
    assert manifest["provenance"] == provenance
    assert manifest["environment"]["python"]
    assert manifest["environment"]["numpy"]
    assert "mujoco" in manifest["environment"]
    for flag in (*app.CLAIM_FLAGS, *app.UNCHANGED_FLAGS):
        assert report[flag] is False
        assert manifest[flag] is False
    for name, identity in manifest["artifacts"].items():
        payload = (output / name).read_bytes()
        assert identity == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    assert result["manifest_sha256"] == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()
    assert not output.with_name(output.name + ".tmp").exists()


def test_output_and_staging_overwrite_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        app._create_output(output, {"report.json": b"{}\n"})
    output.rmdir()
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir()
    with pytest.raises(FileExistsError):
        app._create_output(output, {"report.json": b"{}\n"})


def test_formal_paths_are_rejected_before_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        app,
        "_source_commit",
        lambda root: pytest.fail("provenance started before path rejection"),
    )
    with pytest.raises(ValueError, match="contract path"):
        app.run(
            app.build_parser().parse_args(
                [
                    "--contract",
                    str(tmp_path / "wrong.json"),
                    "--output",
                    str(tmp_path / "wrong"),
                ]
            )
        )
    with pytest.raises(ValueError, match="output path"):
        app.run(
            app.build_parser().parse_args(
                [
                    "--contract",
                    str(CONTRACT_PATH),
                    "--output",
                    str(tmp_path / "wrong"),
                ]
            )
        )


def test_provenance_fails_closed_and_frozen_trees_match() -> None:
    with pytest.raises(RuntimeError, match="historical_trees_match"):
        app._require_provenance(
            {
                "checks": {
                    "workspace_clean": True,
                    "historical_trees_match": False,
                    "passed": False,
                }
            }
        )
    frozen = app._frozen_file_status(
        ROOT, app.FROZEN_DOCUMENT_COMMIT, app.FROZEN_DOCUMENT_PATH
    )
    assert frozen["content_matches"] is True
    assert frozen["blob_matches"] is True
    for path, expected in app.HISTORICAL_TREES.items():
        assert app._git_output(ROOT, ("rev-parse", f"HEAD:{path}")) == expected


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sources() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src/hwr").rglob("*.py"))
    }
