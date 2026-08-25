from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hwr.apps import evaluate_phase_entry_geometry as app
from hwr.eval import phase_entry_geometry as geometry
from hwr.eval.seed_contract import seed_commitment


ROOT = Path(__file__).resolve().parents[1]


def test_cli_freezes_output_and_salt_paths(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output"
    salt_file = tmp_path / "salt.txt"
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(app, "FORMAL_SALT_FILE", salt_file)
    parsed = app.build_parser().parse_args(
        ["--salt-file", str(salt_file), "--output", str(output)]
    )

    assert parsed.output == output
    assert parsed.salt_file == salt_file

    wrong = app.build_parser().parse_args(
        ["--salt-file", str(salt_file), "--output", str(tmp_path / "wrong")]
    )
    with pytest.raises(ValueError, match="output path"):
        app.run(wrong)
    wrong_salt = app.build_parser().parse_args(
        ["--salt-file", str(tmp_path / "wrong-salt"), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="salt path"):
        app.run(wrong_salt)


def test_execute_cohort_uses_natural_latency_and_stops_at_third_eligible(
    monkeypatch,
) -> None:
    salt = "ab" * 32
    monkeypatch.setattr(app, "SALT_COMMITMENT", seed_commitment(salt))
    monkeypatch.setattr(geometry, "SALT_COMMITMENT", seed_commitment(salt))
    monkeypatch.setattr(app, "_runtime_budget_failure", lambda *args: None)
    monkeypatch.setattr(app, "PhaseEntryGeometryMujoco", _CompleteBridge)
    monkeypatch.setattr(app, "load_default_formal_household_catalogs", _catalogs)
    _CompleteBridge.created = 0

    plan, audit, episodes, execution = app.execute_cohort(
        Path("/repo"),
        salt,
        "b" * 40,
        started=0.0,
    )

    assert execution["decision"] == "cohort_complete"
    assert len(plan["episodes"]) == 36
    assert len(audit["records"]) == 48
    assert episodes["physical_prefix_count"] == 36
    assert episodes["selected_episode_count"] == 36
    assert all(
        sum(
            row["latency_matched"]
            for row in audit["records"]
            if row["cell_id"] == cell.cell_id
        )
        == 3
        for cell in geometry.frozen_cells()
    )
    assert all(
        sum(
            row["physical_prefix_executed"]
            for row in audit["records"]
            if row["cell_id"] == cell.cell_id
        )
        == 3
        for cell in geometry.frozen_cells()
    )
    assert all(
        row["b2_action_generated"] is False
        and row["b2_action_executed"] is False
        and row["post_prefix_action_count"] == 0
        for row in episodes["records"]
    )


def test_infeasible_cell_enforces_sixteen_matched_limit_and_publishes_no_cohort(
    monkeypatch,
) -> None:
    salt = "bc" * 32
    monkeypatch.setattr(app, "_runtime_budget_failure", lambda *args: None)
    monkeypatch.setattr(app, "PhaseEntryGeometryMujoco", _IneligibleBridge)
    monkeypatch.setattr(app, "load_default_formal_household_catalogs", _catalogs)

    plan, audit, episodes, execution = app.execute_cohort(
        Path("/repo"),
        salt,
        "c" * 40,
        started=0.0,
    )

    assert execution["decision"] == "inconclusive_design_infeasible"
    assert plan["episodes"] == []
    assert plan["planned_episode_count"] == 0
    assert len(audit["records"]) == geometry.LATENCY_MATCH_LIMIT
    assert episodes["physical_prefix_count"] == geometry.LATENCY_MATCH_LIMIT
    assert execution["infeasible_cells"] == [
        {
            **geometry.frozen_cells()[0].to_dict(),
            "eligible_count": 0,
            "latency_matched_count": geometry.LATENCY_MATCH_LIMIT,
            "raw_seed_count": geometry.LATENCY_MATCH_LIMIT,
        }
    ]


def test_hard_safety_stops_globally_and_preserves_executed_prefix(
    monkeypatch,
) -> None:
    salt = "cd" * 32
    monkeypatch.setattr(app, "_runtime_budget_failure", lambda *args: None)
    monkeypatch.setattr(app, "PhaseEntryGeometryMujoco", _HardStopBridge)
    monkeypatch.setattr(app, "load_default_formal_household_catalogs", _catalogs)

    plan, audit, episodes, execution = app.execute_cohort(
        Path("/repo"),
        salt,
        "d" * 40,
        started=0.0,
    )

    assert execution["decision"] == "invalid"
    assert plan["hard_stop"]["reason"] == "severe_collision"
    assert len(audit["records"]) == 1
    assert episodes["physical_prefix_count"] == 1
    assert episodes["records"][0]["raw_prefix_trace"] == [
        {"step": 0, "terminal": False}
    ]
    assert plan["episodes"] == []


def test_run_writes_five_atomic_artifacts_with_bound_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "formal-output"
    salt_file = tmp_path / "p60-salt.txt"
    salt = "ef" * 32
    salt_file.write_text(salt + "\n", encoding="utf-8")
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(app, "FORMAL_SALT_FILE", salt_file)
    monkeypatch.setattr(app, "SALT_COMMITMENT", seed_commitment(salt))
    monkeypatch.setattr(app, "_source_commit", lambda root: "e" * 40)
    monkeypatch.setattr(app, "_source_identities", lambda root: {"source": "ok"})
    monkeypatch.setattr(app, "_require_clean_source", lambda *args: None)
    monkeypatch.setattr(app, "_require_disk_capacity", lambda *args: None)
    plan = {"schema_version": geometry.PLAN_SCHEMA}
    audit = {"schema_version": geometry.SEED_AUDIT_SCHEMA}
    episodes = {"schema_version": geometry.EPISODES_SCHEMA}
    monkeypatch.setattr(
        app,
        "execute_cohort",
        lambda *args, **kwargs: (
            plan,
            audit,
            episodes,
            {
                "decision": "cohort_complete",
                "infeasible_cells": [],
                "hard_stop": None,
            },
        ),
    )
    analysis = {
        "decision": "accepted as phase-entry necessary-geometry measurement evidence",
        "strict_diagnostic": "strict_phase_entry_deficit_supported",
        "nominal_diagnostic": "nominal_b2_support_deficit_rejected",
        "validation_error": None,
        "checks": {"passed": True},
        "summary": {"episode_count": 36},
    }
    monkeypatch.setattr(app, "analyze_evidence", lambda *args: analysis)
    monkeypatch.setattr(
        app,
        "_report",
        lambda source, execution, value, identities, deterministic_analysis: {
            "decision": value["decision"],
            "strict_diagnostic": value["strict_diagnostic"],
            "nominal_diagnostic": value["nominal_diagnostic"],
            "deterministic_analysis": deterministic_analysis,
        },
    )
    arguments = app.build_parser().parse_args(
        ["--salt-file", str(salt_file), "--output", str(output)]
    )

    result = app.run(arguments)

    assert result["decision"].startswith("accepted")
    assert {path.name for path in output.iterdir()} == {
        "plan.json",
        "seed-audit.json",
        "episodes.json",
        "report.json",
        "manifest.json",
    }
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["salt_input_identity"] == {
        "path": str(salt_file),
        "bytes": len(salt_file.read_bytes()),
        "sha256": hashlib.sha256(salt_file.read_bytes()).hexdigest(),
    }
    assert manifest["frozen_design"]["b2_action_allowed"] is False
    assert manifest["frozen_design"]["strict_and_nominal_decisions_separate"] is True
    assert manifest["policy_inference_executed"] is False
    for name, identity in manifest["artifacts"].items():
        content = (output / name).read_bytes()
        assert identity == {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    with pytest.raises(FileExistsError):
        app.run(arguments)


def test_frozen_document_provenance_and_historical_trees_are_live() -> None:
    identities = app._source_identities(ROOT)

    assert identities["frozen_document"]["commit_is_ancestor"] is True
    assert identities["frozen_document"]["content_matches"] is True
    assert identities["frozen_document"]["blob_matches"] is True
    assert identities["protected_frozen_source"]["passed"] is True
    assert identities["protected_frozen_source"]["changed_paths"] == []
    assert {
        path: app._git_output(ROOT, ("rev-parse", f"HEAD:{path}"))
        for path in app.HISTORICAL_TREES
    } == app.HISTORICAL_TREES


def test_adapter_source_has_no_b2_execution_or_p51_eligibility() -> None:
    source = (
        ROOT / "src/hwr/adapters/mujoco/phase_entry_geometry.py"
    ).read_text(encoding="utf-8")

    assert "relative_yaw_below_pi_over_6" not in source
    assert "first_treatment_guard" not in source
    assert "frame_legacy" not in source
    assert "frame_fixed" not in source
    assert "def _run_b2" not in source
    assert "\"b2_action_generated\": False" in source
    assert "\"b2_action_executed\": False" in source


def _catalogs(root):
    del root
    return (
        {task: SimpleNamespace(task_id=task) for task in geometry.TASK_IDS},
        {task: object() for task in geometry.TASK_IDS},
    )


class _CompleteBridge:
    created = 0

    def __init__(self, task, binding):
        del task, binding
        self.cell = geometry.frozen_cells()[type(self).created]
        type(self).created += 1
        self.calls = 0

    def sample_latencies(self, environment_seed):
        del environment_seed
        self.calls += 1
        return (
            (9, 9)
            if self.calls == 1
            else (
                self.cell.observation_latency_steps,
                self.cell.action_latency_steps,
            )
        )

    def inspect_prefix(self, environment_seed, policy_rng_seed):
        return _prefix(
            environment_seed,
            policy_rng_seed,
            eligible=True,
            observation_latency=self.cell.observation_latency_steps,
            action_latency=self.cell.action_latency_steps,
        )


class _IneligibleBridge:
    def __init__(self, task, binding):
        del task, binding

    def sample_latencies(self, environment_seed):
        del environment_seed
        return (1, 1)

    def inspect_prefix(self, environment_seed, policy_rng_seed):
        return _prefix(environment_seed, policy_rng_seed, eligible=False)


class _HardStopBridge:
    def __init__(self, task, binding):
        del task, binding

    def sample_latencies(self, environment_seed):
        del environment_seed
        return (1, 1)

    def inspect_prefix(self, environment_seed, policy_rng_seed):
        return {
            **_prefix(environment_seed, policy_rng_seed, eligible=False),
            "hard_safety_failure": True,
            "eligibility_reason": "severe_collision",
        }


def _prefix(
    environment_seed,
    policy_rng_seed,
    *,
    eligible,
    observation_latency=1,
    action_latency=1,
):
    trace = [{"step": 0, "terminal": False}]
    return {
        "environment_seed": environment_seed,
        "policy_rng_seed": policy_rng_seed,
        "eligible": eligible,
        "eligibility_reason": "eligible" if eligible else "candidate_set_empty",
        "hard_safety_failure": False,
        "runtime_observation_latency_steps": observation_latency,
        "runtime_action_latency_steps": action_latency,
        "latency_override_inactive": True,
        "raw_prefix_trace": trace,
        "raw_prefix_trace_sha256": geometry.canonical_sha256(trace),
        "b2_action_generated": False,
        "b2_action_executed": False,
        "post_prefix_action_count": 0,
    }
