from __future__ import annotations

import hashlib
import json

import pytest

from hwr.apps import evaluate_factorial_benchmark_contract as app


SALT = "R0001-P36-E2-s20263602"


def _power_report(selected_n: int | None) -> dict[str, object]:
    return {
        "schema_version": "hwr.factorial-synthetic-power/v1",
        "selected_n": selected_n,
        "decision": "power_passed" if selected_n is not None else "inconclusive_power",
        "strata": [],
        "n_summaries": [],
    }


def _reset_smoke() -> dict[str, object]:
    return {
        "schema_version": "hwr.factorial-reset-only-smoke/v1",
        "reset_count": 27,
        "policy_inference_executed": False,
        "complete_episode_executed": False,
        "action_applied": False,
        "checks": {"exact_27_cell_coverage": True},
        "passed": True,
        "records": [],
    }


def test_parser_requires_output_and_salt() -> None:
    arguments = app.build_parser().parse_args(
        ["--output", "runs/p36", "--salt", SALT]
    )

    assert arguments.output.as_posix() == "runs/p36"
    assert arguments.salt == SALT


def test_app_atomically_binds_source_command_and_artifact_hashes(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "factorial"
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(app, "evaluate_synthetic_power", lambda design: _power_report(4))
    monkeypatch.setattr(app, "_reset_only_smoke", lambda root: _reset_smoke())
    arguments = app.build_parser().parse_args(
        ["--output", str(output), "--salt", SALT]
    )

    result = app.run(arguments)
    report = json.loads((output / "report.json").read_text())
    planned = json.loads((output / "planned-ledger.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    smoke = json.loads((output / "reset-only-smoke.json").read_text())

    assert result["decision"] == "accepted as balanced benchmark contract evidence"
    assert result["selected_n"] == 4
    assert report["source_commit"] == "a" * 40
    assert manifest["source_commit"] == report["source_commit"]
    assert manifest["command"] == report["invocation"]["command"]
    assert report["formal_seed_bank"] is False
    assert report["capability_claim_allowed"] is False
    assert report["closed_loop_success_available"] is False
    assert report["primary_ledger"] == "complete_challenge"
    assert report["full_profile_supported"] is False
    assert report["policy_inference_executed"] is False
    assert report["complete_episode_executed"] is False
    assert report["action_applied"] is False
    assert report["diagnostic_seed_lineage"] == manifest["diagnostic_seed_lineage"]
    assert report["diagnostic_seed_lineage"] == {
        "schema_version": "hwr.opaque-episode-seeds/v1",
        "commitment": (
            "f094032ccc029cc15979be8ffd636d956"
            "6500398f356256bc23efc5d8f88cdc9"
        ),
        "reveal": SALT,
        "commitment_verified": True,
        "environment_seed_mode": "derived",
        "role_enters_seed_derivation": False,
    }
    assert report["diagnostic_planned_ledger_available"] is True
    assert report["formal_capability_plan_usable"] is False
    assert planned["pair_count"] == 3 * 27 * 4
    assert planned["execution_count"] == 2 * 3 * 27 * 4
    assert smoke["reset_count"] == 27
    assert all(report["runner_integrity_fault_injection"].values())
    for name, identity in manifest["artifacts"].items():
        payload = (output / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    manifest_bytes = (output / "manifest.json").read_bytes()
    assert result["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert not output.with_name(output.name + ".tmp").exists()

    with pytest.raises(FileExistsError):
        app.run(arguments)


def test_inconclusive_power_publishes_no_usable_plan(tmp_path, monkeypatch) -> None:
    output = tmp_path / "inconclusive"
    monkeypatch.setattr(app, "_source_commit", lambda root: "b" * 40)
    monkeypatch.setattr(app, "evaluate_synthetic_power", lambda design: _power_report(None))
    monkeypatch.setattr(app, "_reset_only_smoke", lambda root: _reset_smoke())
    arguments = app.build_parser().parse_args(
        ["--output", str(output), "--salt", SALT]
    )

    result = app.run(arguments)
    report = json.loads((output / "report.json").read_text())
    planned = json.loads((output / "planned-ledger.json").read_text())

    assert result["decision"] == "inconclusive_power"
    assert result["selected_n"] is None
    assert report["diagnostic_planned_ledger_available"] is False
    assert report["formal_capability_plan_usable"] is False
    assert planned["decision"] == "inconclusive_power"
    assert planned["diagnostic_plan_available"] is False
    assert planned["formal_capability_plan_usable"] is False
    assert planned["replicate_count_per_slot_cell"] is None
    assert planned["pair_count"] == 0
    assert planned["execution_count"] == 0
    assert planned["pairs"] == []


def test_output_staging_is_removed_when_artifact_write_fails(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "failed"
    monkeypatch.setattr(app, "_source_commit", lambda root: "c" * 40)
    monkeypatch.setattr(app, "evaluate_synthetic_power", lambda design: _power_report(None))
    monkeypatch.setattr(app, "_reset_only_smoke", lambda root: _reset_smoke())
    original = app._atomic_write
    writes = 0

    def fail_second(path, content):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected write failure")
        original(path, content)

    monkeypatch.setattr(app, "_atomic_write", fail_second)
    arguments = app.build_parser().parse_args(
        ["--output", str(output), "--salt", SALT]
    )

    with pytest.raises(OSError, match="injected"):
        app.run(arguments)

    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()
