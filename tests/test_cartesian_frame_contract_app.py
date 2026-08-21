from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from hwr.apps import evaluate_cartesian_frame_contract as app


def test_frozen_matrix_replays_exact_candidate_and_rejects_legacy() -> None:
    first = app.evaluate_contract()
    second = app.evaluate_contract()

    assert first == second
    assert first["passed"] is True
    assert first["cell_count"] == 144
    assert first["expected_cell_count"] == 144
    assert first["legacy_counterexample_count"] == 48
    assert all(first["checks"].values())
    assert first["primitive_integration"]["passed"] is True
    assert all(first["primitive_integration"]["checks"].values())
    assert first["primitive_integration"]["case_count"] == 20
    assert first["primitive_integration"]["helper_call_count"] == 28
    assert max(first["candidate_error_maxima"].values()) <= app.TOLERANCE
    zero_yaw = [
        cell for cell in first["cells"] if cell["relative_yaw"] == 0.0
    ]
    assert len(zero_yaw) == 24
    assert all(
        cell["candidate_legacy_float64_bytes_identical"] for cell in zero_yaw
    )
    quarter_turns = [
        cell
        for cell in first["cells"]
        if abs(abs(cell["relative_yaw"]) - math.pi / 2.0) <= app.TOLERANCE
    ]
    assert all(
        cell["legacy"]["errors"]["angular"] >= math.pi / 2.0 - app.TOLERANCE
        for cell in quarter_turns
    )


def test_runner_writes_hash_bound_report_manifest_and_flags(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "p51"
    monkeypatch.setattr(app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)

    result = app.run(app.build_parser().parse_args(["--output", str(output)]))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert result["decision"] == (
        "accepted as Cartesian primitive correctness evidence"
    )
    assert report["source_commit"] == manifest["source_commit"] == "a" * 40
    assert report["command"] == manifest["command"]
    assert report["cell_count"] == 144
    assert report["checks"]["primitive_integration_passed"] is True
    assert manifest["checks"] == report["checks"]
    assert manifest["primitive_integration_checks"] == (
        report["primitive_integration"]["checks"]
    )
    assert manifest["frozen_document_commit"] == app.FROZEN_DOCUMENT_COMMIT
    assert manifest["model"] == {"executed": False, "identity": None}
    assert set(manifest["source_files"]) == {
        path.as_posix() for path in app.SOURCE_PATHS
    }
    for flag in (*app.CLAIM_FLAGS, *app.UNCHANGED_FLAGS):
        assert report[flag] is False
        assert manifest[flag] is False
    for name, identity in manifest["artifacts"].items():
        content = (output / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    assert result["report_sha256"] == manifest["artifacts"]["report.json"]["sha256"]
    assert result["manifest_sha256"] == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()
    assert not output.with_name(output.name + ".tmp").exists()

    with pytest.raises(FileExistsError):
        app.run(app.build_parser().parse_args(["--output", str(output)]))


def test_primitive_integration_rejects_helper_bypass(monkeypatch) -> None:
    def bypass_helper(payload, candidate, acquisition_pose, step):
        value = app.target_selection.deserialize_policy_input(payload)
        if candidate is None or value.safety_state != "ok":
            return (0.0, 0.0, *(0.0,) * 12, 0.20, 0.80)
        relative_yaw = value.base_pose[2] - acquisition_pose[2]
        return tuple(app._expected_primitive_action(step, relative_yaw))

    monkeypatch.setattr(app.target_selection, "primitive_action", bypass_helper)
    evaluation = app.evaluate_contract()

    assert evaluation["primitive_integration"]["checks"][
        "primitive_called_transform_for_both_arms"
    ] is False
    assert evaluation["passed"] is False
    assert app._build_report("a" * 40, [], evaluation)["decision"] == "rejected"


@pytest.mark.parametrize(
    ("index", "expected_check"),
    (
        (2, "target_contract_preserved"),
        (5, "arm_angular_commands_zero"),
        (14, "gripper_contract_preserved"),
    ),
)
def test_primitive_integration_rejects_action_field_tampering(
    monkeypatch, index: int, expected_check: str
) -> None:
    original = app.target_selection.primitive_action

    def tampered(payload, candidate, acquisition_pose, step):
        action = list(original(payload, candidate, acquisition_pose, step))
        value = app.target_selection.deserialize_policy_input(payload)
        if candidate is not None and value.safety_state == "ok":
            action[index] += 0.01
        return tuple(action)

    monkeypatch.setattr(app.target_selection, "primitive_action", tampered)
    evaluation = app.evaluate_contract()

    assert evaluation["primitive_integration"]["checks"][expected_check] is False
    assert evaluation["primitive_integration"]["checks"][
        "only_frozen_formula_changed"
    ] is False
    assert evaluation["passed"] is False
    assert app._build_report("a" * 40, [], evaluation)["decision"] == "rejected"


def test_primitive_integration_rejects_phase_hold_and_bounds_tampering(
    monkeypatch,
) -> None:
    original_phase = app.target_selection.phase_for_step
    monkeypatch.setattr(
        app.target_selection,
        "phase_for_step",
        lambda step: ("B7_stop", 0) if step == 400 else original_phase(step),
    )
    phase = app.evaluate_primitive_integration()
    assert phase["checks"]["phase_contract_preserved"] is False
    monkeypatch.setattr(app.target_selection, "phase_for_step", original_phase)

    original_action = app.target_selection.primitive_action

    def broken_hold(payload, candidate, acquisition_pose, step):
        action = list(original_action(payload, candidate, acquisition_pose, step))
        if candidate is None:
            action[2] = 0.01
        return tuple(action)

    monkeypatch.setattr(app.target_selection, "primitive_action", broken_hold)
    hold = app.evaluate_primitive_integration()
    assert hold["checks"]["hold_contract_preserved"] is False
    monkeypatch.setattr(app.target_selection, "primitive_action", original_action)

    maximum = app.target_selection.ACTION_MAXIMUM.copy()
    maximum[2] = 0.34
    monkeypatch.setattr(app.target_selection, "ACTION_MAXIMUM", maximum)
    bounds = app.evaluate_primitive_integration()
    assert bounds["checks"]["action_bounds_preserved"] is False


def test_failed_runner_publishes_failure_and_manifest(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "failed"
    monkeypatch.setattr(app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "b" * 40)
    monkeypatch.setattr(
        app,
        "evaluate_contract",
        lambda: (_ for _ in ()).throw(RuntimeError("injected matrix failure")),
    )

    with pytest.raises(RuntimeError, match="injected matrix failure"):
        app.run(app.build_parser().parse_args(["--output", str(output)]))

    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert failure["decision"] == "invalid"
    assert failure["error_type"] == "RuntimeError"
    assert manifest["status"] == "failed"
    assert set(manifest["artifacts"]) == {"failure.json"}
    assert not (output / "report.json").exists()
    assert all(failure[flag] is False for flag in app.CLAIM_FLAGS)
    assert all(failure[flag] is False for flag in app.UNCHANGED_FLAGS)


def test_clean_source_gate_requires_clean_tree_lineage_and_frozen_history(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M dirty.py\n"),
    )
    with pytest.raises(RuntimeError, match="clean committed source"):
        app._require_clean_source(tmp_path)

    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(returncode=1),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="frozen document commit"):
        app._require_clean_source(tmp_path)

    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=1),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="historical research-loop"):
        app._require_clean_source(tmp_path)
