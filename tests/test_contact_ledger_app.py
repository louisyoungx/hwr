from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from hwr.apps import evaluate_contact_ledger as app


def _evaluation(*, passed: bool = True) -> dict[str, object]:
    categories = {
        category: {
            "cumulative_impulse": 0.0,
            "pair_peak_force": 0.0,
            "category_peak_force": 0.0,
            "contact_duration_seconds": 0.0,
            "contact_point_count": 0,
            "unique_pair_observation_count": 0,
        }
        for category in app.CONTACT_CATEGORIES
    }
    fixture = {
        "schema_version": "hwr.mujoco-contact-ledger-timestep/v1",
        "fixture_xml_sha256": "f" * 64,
        "passed": True,
    }
    return {
        "timestep_fixture": fixture,
        "tasks": [
            {
                "task_id": task_id,
                "contact_ledger": {
                    "contract_valid": True,
                    "categories": categories,
                },
            }
            for task_id in app.TASK_IDS
        ],
        "checks": {
            "all_checks": passed,
            "legacy_traces_bit_identical": passed,
        },
        "passed": passed,
        "physics": {
            task_id: {"timestep": 0.002, "solver": 2}
            for task_id in app.TASK_IDS
        },
    }


def test_parser_requires_output() -> None:
    arguments = app.build_parser().parse_args(["--output", "runs/p40"])

    assert arguments.output.as_posix() == "runs/p40"


def test_runner_atomically_binds_source_input_physics_and_artifact_hashes(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "p40"
    monkeypatch.setattr(app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(app, "_evaluate_contract", lambda root: _evaluation())

    result = app.run(app.build_parser().parse_args(["--output", str(output)]))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert result["decision"] == "accepted as safety measurement contract evidence"
    assert report["source_commit"] == manifest["source_commit"] == "a" * 40
    assert report["command"] == manifest["command"]
    assert manifest["binding"]["path"] == "configs/adapters/mujoco/formal_3d_v1.json"
    assert manifest["physics"] == report["physics"]
    assert manifest["timestep_fixture"] == {
        "schema_version": "hwr.mujoco-contact-ledger-timestep/v1",
        "fixture_xml_sha256": "f" * 64,
    }
    assert report["capability_claim_allowed"] is False
    assert report["hardware_safety_claim_allowed"] is False
    assert report["measurement_only"] is True
    assert report["legacy_safety_decision_unchanged"] is True
    assert report["forbidden_force_threshold_is_hardware_safety_threshold"] is False
    assert manifest["capability_claim_allowed"] is False
    assert manifest["hardware_safety_claim_allowed"] is False
    for name, identity in manifest["artifacts"].items():
        content = (output / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    assert not output.with_name(output.name + ".tmp").exists()
    with pytest.raises(FileExistsError):
        app.run(app.build_parser().parse_args(["--output", str(output)]))


def test_failed_runner_publishes_failure_without_fake_report(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "failed"
    monkeypatch.setattr(app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "b" * 40)

    def fail(root):
        raise RuntimeError("injected nonfinite contact force")

    monkeypatch.setattr(app, "_evaluate_contract", fail)

    with pytest.raises(RuntimeError, match="nonfinite"):
        app.run(app.build_parser().parse_args(["--output", str(output)]))

    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "RuntimeError"
    assert manifest["status"] == "failed"
    assert manifest["legacy_safety_decision_unchanged"] is False
    assert set(manifest["artifacts"]) == {"failure.json"}
    assert not (output / "report.json").exists()
    assert not output.with_name(output.name + ".tmp").exists()


def test_clean_source_gate_rejects_dirty_worktree(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M dirty.py\n"),
    )

    with pytest.raises(RuntimeError, match="clean committed source"):
        app._require_clean_source(tmp_path)


def test_clean_source_gate_requires_frozen_ancestry_and_historical_docs(
    monkeypatch, tmp_path
) -> None:
    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(returncode=0),
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
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=1),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))

    with pytest.raises(RuntimeError, match="historical research-loop"):
        app._require_clean_source(tmp_path)


def test_output_staging_is_removed_after_write_failure(tmp_path, monkeypatch) -> None:
    output = tmp_path / "write-failed"
    monkeypatch.setattr(app, "_require_clean_source", lambda root: None)
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
