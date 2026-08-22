from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hwr.apps import evaluate_cartesian_convergence as app
from hwr.eval import cartesian_convergence as convergence
from hwr.eval.seed_contract import seed_commitment


def test_cli_requires_mode_specific_inputs_and_frozen_commitment() -> None:
    parser = app.build_parser()
    build = parser.parse_args(
        [
            "--mode",
            "build-bank",
            "--output",
            "runs/bank",
            "--salt-file",
            "secret.txt",
        ]
    )
    app._validate_arguments(build)
    evaluate = parser.parse_args(
        [
            "--mode",
            "evaluate",
            "--output",
            "runs/eval",
            "--bank",
            "runs/bank/bank.json",
        ]
    )
    app._validate_arguments(evaluate)

    with pytest.raises(ValueError, match="only --bank"):
        app._validate_arguments(
            parser.parse_args(
                [
                    "--mode",
                    "evaluate",
                    "--output",
                    "x",
                    "--bank",
                    "bank.json",
                    "--salt-file",
                    "forbidden.txt",
                ]
            )
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--mode",
                "build-bank",
                "--output",
                "x",
                "--salt-commitment",
                "0" * 64,
            ]
        )


def test_build_bank_respects_raw_and_latency_matched_budgets(
    monkeypatch,
) -> None:
    salt = "ef" * 32
    monkeypatch.setattr(app, "SALT_COMMITMENT", seed_commitment(salt))
    monkeypatch.setattr(convergence, "SALT_COMMITMENT", seed_commitment(salt))
    monkeypatch.setattr(
        app,
        "load_default_formal_household_catalogs",
        lambda root: (
            {task: SimpleNamespace(task_id=task) for task in app.TASK_IDS},
            {task: object() for task in app.TASK_IDS},
        ),
    )
    monkeypatch.setattr(app, "CartesianConvergenceMujoco", _FakeBridge)
    monkeypatch.setattr(app, "_source_identities", lambda root: {"source": "ok"})
    _FakeBridge.created = 0

    bank, audit = app.build_bank(Path("/repo"), salt, "a" * 40)

    assert bank["eligible_pair_count"] == 36
    assert bank["infeasible_cells"] == []
    assert len(bank["pairs"]) == 36
    assert len(audit["records"]) == 48
    assert all(
        sum(
            record["latency_matched"]
            for record in audit["records"]
            if record["cell_id"] == cell.cell_id
        )
        == 3
        for cell in convergence.frozen_cells()
    )
    assert all(pair["replicate_ordinal"] in (0, 1, 2) for pair in bank["pairs"])


def test_build_bank_discards_partial_pairs_when_cell_is_infeasible(
    monkeypatch,
) -> None:
    salt = "12" * 32
    monkeypatch.setattr(app, "SALT_COMMITMENT", seed_commitment(salt))
    monkeypatch.setattr(convergence, "SALT_COMMITMENT", seed_commitment(salt))
    monkeypatch.setattr(app, "RAW_SEED_LIMIT", 4)
    monkeypatch.setattr(app, "LATENCY_MATCH_LIMIT", 2)
    monkeypatch.setattr(
        app,
        "load_default_formal_household_catalogs",
        lambda root: (
            {task: SimpleNamespace(task_id=task) for task in app.TASK_IDS},
            {task: object() for task in app.TASK_IDS},
        ),
    )
    monkeypatch.setattr(app, "CartesianConvergenceMujoco", _InfeasibleBridge)
    monkeypatch.setattr(app, "_source_identities", lambda root: {"source": "ok"})

    bank, audit = app.build_bank(Path("/repo"), salt, "b" * 40)

    assert bank["eligible_pair_count"] == 0
    assert bank["pairs"] == []
    assert len(bank["infeasible_cells"]) == 12
    counts = {
        cell.cell_id: sum(
            record["cell_id"] == cell.cell_id
            for record in audit["records"]
        )
        for cell in convergence.frozen_cells()
    }
    assert all(value <= 4 for value in counts.values())
    assert all(
        value["latency_matched_count"] <= 2
        for value in bank["infeasible_cells"]
    )


def test_evaluate_stops_after_hard_safety_and_preserves_terminal(
    monkeypatch,
) -> None:
    pair = _pair()
    bank = {
        "source_commit": "c" * 40,
        "pairs": [pair, {**pair, "pair_id": "b" * 64}],
    }
    calls = []

    class FakeEvaluator:
        def __init__(self, task, binding):
            pass

        def evaluate_pair(self, value):
            calls.append(value["pair_id"])
            return {
                **value,
                "resolved": True,
                "hard_safety_stop": True,
                "arms": {},
            }

    monkeypatch.setattr(
        app,
        "load_default_formal_household_catalogs",
        lambda root: (
            {task: SimpleNamespace(task_id=task) for task in app.TASK_IDS},
            {task: object() for task in app.TASK_IDS},
        ),
    )
    monkeypatch.setattr(app, "CartesianConvergenceMujoco", FakeEvaluator)

    terminals = app.evaluate_bank(Path("/repo"), bank)

    assert calls == [pair["pair_id"]]
    assert terminals["planned_pair_count"] == 2
    assert terminals["terminal_pair_count"] == 1


def test_evaluate_mode_never_reads_salt_and_writes_bound_artifacts(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "output"
    bank_path = tmp_path / "bank.json"
    bank_path.write_text("{}\n", encoding="utf-8")
    arguments = app.build_parser().parse_args(
        [
            "--mode",
            "evaluate",
            "--bank",
            str(bank_path),
            "--output",
            str(output),
        ]
    )
    bank = {"source_commit": "c" * 40, "pairs": []}
    terminals = {
        "schema_version": convergence.TERMINAL_SCHEMA,
        "planned_pair_count": 0,
        "terminal_pair_count": 0,
        "records": [],
    }
    analysis = {"decision": "inconclusive"}
    monkeypatch.setattr(app, "_source_commit", lambda root: "d" * 40)
    monkeypatch.setattr(app, "_source_identities", lambda root: {"source": "ok"})
    monkeypatch.setattr(app, "_require_clean_source", lambda root, values: None)
    monkeypatch.setattr(
        app,
        "_require_committed_bank",
        lambda root, path: {"path": "bank.json", "sha256": "e" * 64, "bytes": 3},
    )
    monkeypatch.setattr(app, "_read_json", lambda path: bank)
    monkeypatch.setattr(app, "validate_bank", lambda value: None)
    monkeypatch.setattr(app, "_require_bank_provenance", lambda *values: None)
    monkeypatch.setattr(app, "evaluate_bank", lambda root, value: terminals)
    monkeypatch.setattr(app, "analyze_terminals", lambda value: analysis)
    monkeypatch.setattr(
        app,
        "read_seed_salt",
        lambda path: pytest.fail("evaluate must not read salt"),
    )

    result = app.run(arguments)
    manifest = json.loads((output / "manifest.json").read_text())

    assert result["decision"] == "inconclusive"
    assert set(path.name for path in output.iterdir()) == {
        "terminals.json",
        "report.json",
        "manifest.json",
    }
    assert manifest["frozen_design"]["salt_commitment"] == app.SALT_COMMITMENT
    assert manifest["training_executed"] is False
    for name, identity in manifest["artifacts"].items():
        content = (output / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }


def test_failure_writes_failure_and_manifest(tmp_path, monkeypatch) -> None:
    output = tmp_path / "failure"
    arguments = app.build_parser().parse_args(
        [
            "--mode",
            "build-bank",
            "--output",
            str(output),
            "--salt-file",
            str(tmp_path / "missing-salt.txt"),
        ]
    )
    monkeypatch.setattr(app, "_source_commit", lambda root: "f" * 40)
    monkeypatch.setattr(app, "_source_identities", lambda root: {"source": "ok"})
    monkeypatch.setattr(app, "_require_clean_source", lambda root, values: None)

    with pytest.raises(FileNotFoundError):
        app.run(arguments)

    failure = json.loads((output / "failure.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert failure["decision"] == "invalid"
    assert manifest["status"] == "failed"
    assert set(manifest["artifacts"]) == {"failure.json"}


class _FakeBridge:
    created = 0

    def __init__(self, task, binding):
        self.task = task
        self.cell = convergence.frozen_cells()[self.created]
        type(self).created += 1
        self.calls = 0

    def sample_latencies(self, environment_seed):
        self.calls += 1
        return (
            (3, 3)
            if self.calls == 1
            else (
                self.cell.observation_latency_steps,
                self.cell.action_latency_steps,
            )
        )

    def inspect_prefix(self, environment_seed, policy_rng_seed):
        return _prefix()


class _InfeasibleBridge:
    def __init__(self, task, binding):
        pass

    def sample_latencies(self, environment_seed):
        return (1, 1)

    def inspect_prefix(self, environment_seed, policy_rng_seed):
        return {**_prefix(), "eligible": False, "eligibility_reason": "empty"}


def _prefix() -> dict[str, object]:
    candidate = {
        "center": [1.0, 0.0, 0.7],
        "normal": [-1.0, 0.0, 0.0],
        "width": 0.12,
        "prominence": 0.1,
        "support_count": 30,
        "view_count": 2,
        "first_frame": 0,
        "first_row": 20,
        "first_column": 30,
    }
    canonical = [1000, 0, 700, -10000, 0, 0, 120, 0, 20, 30, 100, 30, 2]
    payload = json.dumps(
        {"candidates": [canonical]}, separators=(",", ":"), sort_keys=True
    ).encode()
    actions = {
        "frame_legacy": [
            0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        "frame_fixed": [
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    }
    return {
        "eligible": True,
        "eligibility_reason": "eligible",
        "candidate_count": 1,
        "candidate_set_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_bytes_hex": payload.hex(),
        "selected_index": 0,
        "selected_record": candidate,
        "relative_yaw_at_b2": 1.0,
        "continuation_identity": {"identity": {"sha256": "a" * 64}},
        "prefix_trace_sha256": "b" * 64,
        "first_treatment_actions": actions,
        "first_treatment_guard": convergence.first_treatment_guard(
            actions["frame_legacy"], actions["frame_fixed"]
        ),
    }


def _pair() -> dict[str, object]:
    cell = convergence.frozen_cells()[0]
    return {
        **cell.to_dict(),
        "pair_id": "a" * 64,
        "planned_episode_id": "c" * 64,
        "replicate_ordinal": 0,
        "environment_seed": 1,
        "policy_rng_seed": 2,
        "role_order": list(convergence.ROLES),
        "first_treatment_guard": {},
    }
