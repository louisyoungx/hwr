from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from hwr.apps import evaluate_target_selection as app


def _power(selected: int | None = 54) -> dict[str, object]:
    return {
        "schema_version": app.POWER_SCHEMA,
        "selected_pair_count": selected,
        "decision": "power_passed" if selected == 54 else "inconclusive_power",
        "summaries": [],
    }


def _plan(salt: str) -> dict[str, object]:
    return {
        "schema_version": app.PLAN_SCHEMA,
        "proposal_id": app.PROPOSAL_ID,
        "mode": "smoke",
        "salt_commitment": hashlib.sha256(salt.encode()).hexdigest(),
        "salt_reveal": salt,
        "commitment_verified": True,
        "planned_pair_count": 6,
        "execution_count": 12,
        "pairs": [],
        "rejected_seed_audit": [],
    }


def _terminals() -> dict[str, object]:
    return {
        "schema_version": app.TERMINAL_SCHEMA,
        "mode": "smoke",
        "planned_pair_count": 6,
        "terminal_pair_count": 6,
        "records": [],
    }


def _analysis() -> dict[str, object]:
    return {
        "passed": True,
        "planned_pair_count": 6,
        "terminal_pair_count": 6,
        "unresolved_infrastructure": 0,
        "candidate_hash_equal": True,
        "same_index_bit_identity": True,
        "candidate_set_nonempty": True,
        "hard_guard": {"passed": True},
        "selector_comparison_executed": False,
    }


def _identities() -> dict[str, object]:
    return {
        "p40_e2": {
            "report": {"path": str(app.P40_REPORT), "sha256": app.P40_REPORT_SHA256, "bytes": 1},
            "manifest": {"path": str(app.P40_MANIFEST), "sha256": app.P40_MANIFEST_SHA256, "bytes": 1},
        },
        "binding": {"path": str(app.BINDING_PATH), "sha256": app.BINDING_SHA256, "bytes": app.BINDING_BYTES},
        "task_config": {
            "path": str(app.TASK_PATH),
            "sha256": app.TASK_SHA256,
            "bytes": app.TASK_BYTES,
        },
    }


def test_parser_enforces_frozen_mode_salts() -> None:
    smoke = app.build_parser().parse_args(
        ["--output", "runs/p41-smoke", "--salt", app.SMOKE_SALT, "--smoke"]
    )
    formal = app.build_parser().parse_args(
        ["--output", "runs/p41-formal", "--salt", app.FORMAL_SALT]
    )

    app._validate_mode(smoke)
    app._validate_mode(formal)
    with pytest.raises(ValueError, match="frozen salt"):
        app._validate_mode(
            app.build_parser().parse_args(
                ["--output", "x", "--salt", app.FORMAL_SALT, "--smoke"]
            )
        )


def test_exact_power_is_replayable_and_selects_frozen_54_pairs() -> None:
    first = app.evaluate_synthetic_power()
    second = app.evaluate_synthetic_power()

    assert first == second
    assert first["selected_pair_count"] == 54
    assert first["decision"] == "power_passed"
    assert first["candidate_pair_counts"] == [36, 54, 72, 90, 108]
    assert first["trials"] == 10_000
    assert first["seed"] == 20_264_102
    assert first["summaries"][0]["passes"] is False
    assert first["summaries"][1]["passes"] is True


def test_smoke_plan_uses_two_frozen_supported_cells_per_task(
    tmp_path, monkeypatch
) -> None:
    tasks = {
        task_id: SimpleNamespace(task_id=task_id)
        for task_id in app.TASK_IDS
    }
    monkeypatch.setattr(
        app,
        "load_default_formal_household_catalogs",
        lambda root: (tasks, dict.fromkeys(app.TASK_IDS, object())),
    )

    class Diagnostic:
        def __init__(self, task, binding):
            del binding
            self.task = task

        def sample_latencies(self, seed):
            del seed
            cell = int(self.task.task_id.split("-")[-1]) if "-" in self.task.task_id else 0
            del cell
            calls = getattr(self.task, "calls", 0)
            self.task.calls = calls + 1
            return ((1, 1), (2, 2))[calls % 2]

    monkeypatch.setattr(app, "TargetSelectionDiagnostic", Diagnostic)
    plan = app.build_plan(
        tmp_path,
        app.SMOKE_SALT,
        smoke=True,
        selected_pair_count=54,
    )

    assert plan["planned_pair_count"] == 6
    assert plan["execution_count"] == 12
    assert {
        (
            pair["task_id"],
            pair["observation_latency_steps"],
            pair["action_latency_steps"],
        )
        for pair in plan["pairs"]
    } == {
        (task_id, latency, latency)
        for task_id in app.TASK_IDS
        for latency in (1, 2)
    }


def test_unresolved_terminal_is_retained_without_crashing_smoke_analysis() -> None:
    pair = {
        "pair_id": "p",
        "task_id": app.TASK_IDS[0],
        "domain": "smoke",
        "observation_latency_steps": 1,
        "action_latency_steps": 1,
        "environment_seed": 1,
        "policy_rng_seed": 2,
    }
    terminal = app._unresolved_terminal(pair, "candidate", RuntimeError("boom"))
    analysis = app.analyze_terminals(
        {
            "planned_pair_count": 1,
            "terminal_pair_count": 1,
            "records": [terminal],
        },
        smoke=True,
    )

    assert terminal["resolved"] is False
    assert analysis["unresolved_infrastructure"] == 1
    assert analysis["planned_terminal_identity_complete"] is True
    assert analysis["passed"] is False


def test_missing_and_duplicate_terminals_make_smoke_inconclusive() -> None:
    pair = {
        "pair_id": "p",
        "task_id": app.TASK_IDS[0],
        "domain": "smoke",
        "observation_latency_steps": 1,
        "action_latency_steps": 1,
        "environment_seed": 1,
        "policy_rng_seed": 2,
    }
    terminal = app._unresolved_terminal(pair, "candidate", RuntimeError("boom"))

    missing = app.analyze_terminals(
        {"planned_pair_count": 2, "terminal_pair_count": 1, "records": [terminal]},
        smoke=True,
    )
    duplicate = app.analyze_terminals(
        {
            "planned_pair_count": 2,
            "terminal_pair_count": 2,
            "records": [terminal, terminal],
        },
        smoke=True,
    )

    assert missing["unresolved_infrastructure"] == 2
    assert missing["planned_terminal_identity_complete"] is False
    assert duplicate["unresolved_infrastructure"] == 3
    assert duplicate["planned_terminal_identity_complete"] is False


def test_smoke_runner_atomically_writes_five_hash_bound_artifacts(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "smoke"
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(app, "_source_identities", lambda root: _identities())
    monkeypatch.setattr(app, "_require_clean_source", lambda root, identities: None)
    monkeypatch.setattr(app, "evaluate_synthetic_power", lambda: _power())
    monkeypatch.setattr(
        app,
        "build_plan",
        lambda root, salt, **kwargs: _plan(salt),
    )
    monkeypatch.setattr(
        app,
        "execute_plan",
        lambda root, plan, **kwargs: _terminals(),
    )
    monkeypatch.setattr(
        app,
        "analyze_terminals",
        lambda terminals, **kwargs: _analysis(),
    )

    result = app.run(
        app.build_parser().parse_args(
            ["--output", str(output), "--salt", app.SMOKE_SALT, "--smoke"]
        )
    )
    report = json.loads((output / "report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())

    assert result["decision"] == "accepted as target-selection smoke contract evidence"
    assert report["mode"] == "smoke"
    assert report["selector_comparison_executed"] is False
    assert set(path.name for path in output.iterdir()) == {
        "report.json",
        "plan.json",
        "terminals.json",
        "power.json",
        "manifest.json",
    }
    for name, identity in manifest["artifacts"].items():
        payload = (output / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    assert manifest["task_config"] == _identities()["task_config"]
    assert not output.with_name(output.name + ".tmp").exists()
    assert all(report[name] is False for name in app.CLAIM_FLAGS)


def test_clean_source_gate_rejects_dirty_worktree(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M dirty.py\n"),
    )

    with pytest.raises(RuntimeError, match="clean committed source"):
        app._require_clean_source(tmp_path, _identities())
