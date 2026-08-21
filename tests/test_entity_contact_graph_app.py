from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from hwr.apps import evaluate_entity_contact_graph as app


def _evaluation(*, passed: bool = True) -> dict[str, object]:
    roots = {
        part: {"body_name": name, "body_id": index}
        for index, (part, name) in enumerate(app.ROBOT_BODY_ROOT_NAMES.items(), 1)
    }
    return {
        "fixture": {
            "schema_version": app.FIXTURE_SCHEMA,
            "passed": True,
            "classification_precision": 1.0,
            "classification_recall": 1.0,
            "contact_associated_motion_exclusion": app.SETTLING_EXCLUSION,
        },
        "tasks": [
            {
                "task_id": task_id,
                "legacy_trace_bit_identical": passed,
                "entity_contact_graph": {
                    "schema_version": app.MEASUREMENT_SCHEMA,
                    "mapping": {"robot_body_roots": roots},
                    "contact_associated_motion_exclusion": app.SETTLING_EXCLUSION,
                },
            }
            for task_id in app.TASK_IDS
        ],
        "checks": {
            "legacy_traces_bit_identical": passed,
            "all_checks": passed,
        },
        "passed": passed,
        "contact_associated_motion_exclusion": app.SETTLING_EXCLUSION,
        "physics": {
            task_id: {
                "mujoco_version": "test",
                "timestep": 0.002,
                "solver": 2,
                "iterations": 100,
                "tolerance": 1.0e-8,
                "substeps_per_control_period": 25,
            }
            for task_id in app.TASK_IDS
        },
        "robot_body_roots": {task_id: roots for task_id in app.TASK_IDS},
    }


def test_parser_requires_output() -> None:
    arguments = app.build_parser().parse_args(["--output", "runs/p40-e2"])

    assert arguments.output.as_posix() == "runs/p40-e2"


def test_evaluator_pairs_disabled_and_enabled_bit_identical_traces(
    tmp_path, monkeypatch
) -> None:
    calls = []
    zero_categories = {
        category: {
            "pair_peak_force": 0.0,
            "category_peak_force": 0.0,
            "cumulative_impulse": 0.0,
            "contact_duration_seconds": 0.0,
            "contact_point_count": 0,
            "unique_pair_observation_count": 0,
        }
        for category in app.CONTACT_CATEGORIES
    }
    roots = {
        part: {"body_name": name, "body_id": index}
        for index, (part, name) in enumerate(app.ROBOT_BODY_ROOT_NAMES.items(), 1)
    }

    def trace(task, binding, *, seed, enabled):
        calls.append((task.task_id, seed, enabled))
        return {
            "trace": [{"applied_action_vector": [0.0] * 16}],
            "fixed_hold_action": [0.0] * 16,
            "contact_ledger": {
                "contract_valid": True,
                "categories": zero_categories,
            },
            "entity_contact_graph": {
                "contract_valid": True,
                "contact_associated_motion_exclusion": app.SETTLING_EXCLUSION,
                "mapping": {
                    "robot_body_roots": roots,
                    "robot_geoms": [{}] * 48,
                },
                "legacy_p40_categories": zero_categories,
                "task_relevant_world_world_edges": [],
                "control_period_count": 1,
                "interactions": {
                    "same_entity_dual_arm_substep_count": 0,
                    "distinct_entity_dual_arm_substep_count": 0,
                    "single_arm_substep_count": 0,
                    "same_object_dual_arm_grasp_substep_count": 0,
                },
                **{name: 0 for name in app.INVALID_COUNT_FIELDS},
            },
            "physics": {"timestep": 0.002},
        }

    tasks = {
        task_id: SimpleNamespace(task_id=task_id) for task_id in app.TASK_IDS
    }
    bindings = dict.fromkeys(app.TASK_IDS, object())
    monkeypatch.setattr(
        app, "load_default_formal_household_catalogs", lambda root: (tasks, bindings)
    )
    monkeypatch.setattr(
        app,
        "_run_fixture",
        lambda: {
            "passed": True,
            "classification_precision": 1.0,
            "classification_recall": 1.0,
            "contact_associated_motion_exclusion": app.SETTLING_EXCLUSION,
        },
    )
    monkeypatch.setattr(app, "_run_trace", trace)

    evaluation = app._evaluate_contract(tmp_path)

    assert evaluation["passed"] is True
    assert all(
        value["legacy_trace_bit_identical"] for value in evaluation["tasks"]
    )
    assert calls == [
        item
        for task_index, task_id in enumerate(app.TASK_IDS)
        for item in (
            (task_id, app.BASE_SEED + task_index * app.SEED_STRIDE, False),
            (task_id, app.BASE_SEED + task_index * app.SEED_STRIDE, True),
        )
    ]


def test_runner_atomically_binds_contract_provenance_and_artifact_hashes(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "p40-e2"
    monkeypatch.setattr(app, "_require_clean_source", lambda root, binding: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(app, "_evaluate_contract", lambda root: _evaluation())

    result = app.run(app.build_parser().parse_args(["--output", str(output)]))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert result["decision"] == (
        "accepted as entity-contact measurement contract evidence"
    )
    assert report["schema_version"] == app.REPORT_SCHEMA
    assert manifest["schema_version"] == app.MANIFEST_SCHEMA
    assert report["measurement_schema"] == manifest["measurement_schema"]
    assert report["source_commit"] == manifest["source_commit"] == "a" * 40
    assert report["command"] == manifest["command"]
    assert manifest["frozen_document_commit"] == app.FROZEN_DOCUMENT_COMMIT
    assert manifest["frozen_document_commit_is_ancestor"] is True
    assert manifest["binding"] == {
        "path": "configs/adapters/mujoco/formal_3d_v1.json",
        "sha256": app.FROZEN_BINDING_SHA256,
        "bytes": app.FROZEN_BINDING_BYTES,
    }
    assert manifest["physics"] == report["physics"]
    assert manifest["robot_body_roots"] == report["robot_body_roots"]
    assert report["contact_associated_motion_exclusion"] == app.SETTLING_EXCLUSION
    assert manifest["contact_associated_motion_exclusion"] == app.SETTLING_EXCLUSION
    assert manifest["constants"]["excluded_initial_periods"] == 1
    for name, expected in app.CLAIM_FLAGS.items():
        assert report[name] is expected
        assert manifest[name] is expected
    assert report["legacy_runtime_behavior_unchanged"] is True
    assert manifest["legacy_runtime_behavior_unchanged"] is True
    for name, identity in manifest["artifacts"].items():
        content = (output / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    assert not output.with_name(output.name + ".tmp").exists()
    with pytest.raises(FileExistsError):
        app.run(app.build_parser().parse_args(["--output", str(output)]))


def test_failed_runner_publishes_failure_and_manifest_without_fake_report(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "failed"
    monkeypatch.setattr(app, "_require_clean_source", lambda root, binding: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "b" * 40)

    def fail(root):
        raise EntityFailure("injected nonfinite contact force")

    monkeypatch.setattr(app, "_evaluate_contract", fail)

    with pytest.raises(EntityFailure, match="nonfinite"):
        app.run(app.build_parser().parse_args(["--output", str(output)]))

    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert failure["schema_version"] == app.FAILURE_SCHEMA
    assert failure["error_type"] == "EntityFailure"
    assert manifest["status"] == "failed"
    assert manifest["frozen_document_commit_is_ancestor"] is False
    assert manifest["legacy_runtime_behavior_unchanged"] is False
    assert set(manifest["artifacts"]) == {"failure.json"}
    assert failure["action_causality_claim_allowed"] is False
    assert not (output / "report.json").exists()
    assert not output.with_name(output.name + ".tmp").exists()


class EntityFailure(RuntimeError):
    pass


def test_clean_source_gate_rejects_dirty_worktree(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M dirty.py\n"),
    )

    with pytest.raises(RuntimeError, match="clean committed source"):
        app._require_clean_source(tmp_path, _binding_identity())


def test_clean_source_gate_requires_ancestry_history_and_binding_identity(
    monkeypatch, tmp_path
) -> None:
    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(returncode=1),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="frozen document commit"):
        app._require_clean_source(tmp_path, _binding_identity())

    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=1),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="historical research-loop"):
        app._require_clean_source(tmp_path, _binding_identity())

    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="binding identity"):
        app._require_clean_source(
            tmp_path, {**_binding_identity(), "sha256": "f" * 64}
        )


def test_output_staging_is_removed_after_write_failure(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "write-failed"
    monkeypatch.setattr(app, "_require_clean_source", lambda root, binding: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "c" * 40)
    monkeypatch.setattr(app, "_evaluate_contract", lambda root: _evaluation())
    original = app._atomic_write
    writes = 0

    def fail_second(path, content):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected artifact write failure")
        original(path, content)

    monkeypatch.setattr(app, "_atomic_write", fail_second)

    with pytest.raises(OSError, match="injected"):
        app.run(app.build_parser().parse_args(["--output", str(output)]))

    assert (output / "failure.json").is_file()
    assert (output / "manifest.json").is_file()
    assert not (output / "report.json").exists()
    assert not output.with_name(output.name + ".tmp").exists()


def test_rejected_evaluation_never_claims_measurement_acceptance() -> None:
    report = app._build_report("d" * 40, ("python",), _evaluation(passed=False))

    assert report["decision"] == "rejected"
    assert report["legacy_runtime_behavior_unchanged"] is False
    assert report["measurement_only"] is True
    assert report["capability_claim_allowed"] is False
    assert report["hardware_safety_claim_allowed"] is False
    assert report["action_causality_claim_allowed"] is False


def _binding_identity() -> dict[str, object]:
    return {
        "path": "configs/adapters/mujoco/formal_3d_v1.json",
        "sha256": app.FROZEN_BINDING_SHA256,
        "bytes": app.FROZEN_BINDING_BYTES,
    }
