from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hwr.apps import evaluate_candidate_acquisition as app
SALT = "0" * 64


class _Sampler:
    def __init__(self, task, binding) -> None:
        del binding
        self.task = task

    def sample_latencies(self, seed: int) -> tuple[int, int]:
        del seed
        return self.task.target


def _catalog() -> tuple[dict[str, object], dict[str, object]]:
    tasks = {
        task_id: SimpleNamespace(task_id=task_id, target=(1, 1))
        for task_id in app.TASK_IDS
    }
    return tasks, dict.fromkeys(app.TASK_IDS, object())


def _minimal_plan() -> dict[str, object]:
    return {
        "schema_version": app.PLAN_SCHEMA,
        "proposal_id": app.PROPOSAL_ID,
        "plan_id": app.PLAN_ID,
        "salt_commitment": app.SALT_COMMITMENT,
        "salt_reveal": SALT,
        "commitment_verified": True,
        "cells": [],
        "planned_episode_count": 0,
        "episodes": [],
        "rejected_seed_audit": [],
    }


def _identities() -> dict[str, object]:
    return {
        "binding": {"path": "binding", "sha256": "a" * 64, "bytes": 1},
        "task_config": {"path": "task", "sha256": "b" * 64, "bytes": 1},
        "recursive_xml": {},
        "sources": {},
        "historical_research_loop_trees": dict(app.HISTORICAL_TREES),
        "frozen_document": {
            "path": app.FROZEN_DOCUMENT_PATH.as_posix(),
            "sha256": "c" * 64,
            "bytes": 1,
        },
    }


def test_cli_keeps_exact_frozen_acquisition_paths() -> None:
    arguments = app.build_parser().parse_args(
        [
            "--output",
            app.FORMAL_OUTPUT.as_posix(),
            "--salt-file",
            app.FORMAL_SALT_FILE.as_posix(),
        ]
    )

    app._validate_arguments(arguments)
    wrong = app.build_parser().parse_args(
        ["--output", "runs/other", "--salt-file", app.FORMAL_SALT_FILE.as_posix()]
    )
    with pytest.raises(ValueError, match="frozen output"):
        app._validate_arguments(wrong)


def test_cli_keeps_exact_frozen_funnel_paths_and_forbids_salt() -> None:
    arguments = app.build_parser().parse_args(
        [
            "--mode",
            "funnel",
            "--capsules",
            app.FUNNEL_INPUT.as_posix(),
            "--output",
            app.FUNNEL_OUTPUT.as_posix(),
        ]
    )

    app._validate_arguments(arguments)
    with pytest.raises(ValueError, match="frozen capsules"):
        app._validate_arguments(
            app.build_parser().parse_args(
                [
                    "--mode",
                    "funnel",
                    "--capsules",
                    app.FUNNEL_INPUT.as_posix(),
                    "--output",
                    app.FUNNEL_OUTPUT.as_posix(),
                    "--salt-file",
                    app.FORMAL_SALT_FILE.as_posix(),
                ]
            )
        )


def test_plan_has_twelve_ordered_cells_and_two_natural_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, bindings = _catalog()
    monkeypatch.setattr(
        app,
        "load_default_formal_household_catalogs",
        lambda root: (tasks, bindings),
    )
    monkeypatch.setattr(app, "require_seed_reveal", lambda commitment, salt: None)

    class Sampler(_Sampler):
        def sample_latencies(self, seed: int) -> tuple[int, int]:
            del seed
            calls = getattr(self.task, "calls", 0)
            self.task.calls = calls + 1
            return (3, 3) if calls % 3 == 0 else self.task.target

    def factory(task, binding, observation_latency, action_latency):
        task.target = (observation_latency, action_latency)
        return Sampler(task, binding)

    monkeypatch.setattr(app, "_sampler_for_cell", factory)

    plan = app.build_plan(tmp_path, SALT)

    assert [
        (
            cell["task_id"],
            cell["observation_latency_steps"],
            cell["action_latency_steps"],
        )
        for cell in plan["cells"]
    ] == list(app.FROZEN_CELLS)
    assert plan["planned_episode_count"] == 24
    assert len(plan["episodes"]) == 24
    assert len(plan["rejected_seed_audit"]) == 12
    assert all(
        [episode["replicate_ordinal"] for episode in plan["episodes"]
         if episode["cell_id"] == cell["cell_id"]] == [0, 1]
        for cell in plan["cells"]
    )
    checked = [*plan["episodes"], *plan["rejected_seed_audit"]]
    assert len({row["environment_seed"] for row in checked}) == len(checked)
    assert len({row["policy_rng_seed"] for row in checked}) == len(checked)


def test_plan_stops_after_candidate_ordinal_95(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, bindings = _catalog()
    monkeypatch.setattr(
        app,
        "load_default_formal_household_catalogs",
        lambda root: (tasks, bindings),
    )
    monkeypatch.setattr(app, "require_seed_reveal", lambda commitment, salt: None)

    class Rejecting(_Sampler):
        def sample_latencies(self, seed: int) -> tuple[int, int]:
            del seed
            return (3, 3)

    monkeypatch.setattr(
        app,
        "_sampler_for_cell",
        lambda task, binding, observation, action: Rejecting(task, binding),
    )

    with pytest.raises(app.AcquisitionContractError, match="ordinal 95"):
        app.build_plan(tmp_path, SALT)


def test_terminal_ledger_rejects_missing_duplicate_unplanned_and_replacement() -> None:
    plan = {
        "episodes": [
            {
                "planned_episode_id": "a" * 64,
                "cell_id": "cell-0",
                "replicate_ordinal": 0,
            },
            {
                "planned_episode_id": "b" * 64,
                "cell_id": "cell-0",
                "replicate_ordinal": 1,
            },
        ]
    }
    valid = [
        {"planned_episode_id": "a" * 64, "replacement": False},
        {"planned_episode_id": "b" * 64, "replacement": False},
    ]

    assert app.validate_candidate_terminal_ledger(plan, valid)["passed"] is True
    for terminals in (
        valid[:1],
        [valid[0], valid[0]],
        [*valid, {"planned_episode_id": "c" * 64, "replacement": False}],
        [valid[0], {**valid[1], "replacement": True}],
    ):
        assert app.validate_candidate_terminal_ledger(plan, terminals)["passed"] is False


def _acceptance_fixture() -> tuple[dict[str, object], dict[str, object]]:
    episodes = [
        {
            "planned_episode_id": f"{index:064x}",
            "cell_id": f"cell-{index // 2:02d}",
        }
        for index in range(24)
    ]
    plan = {
        "planned_episode_count": 24,
        "cells": [
            {"cell_id": f"cell-{index:02d}"} for index in range(12)
        ],
        "episodes": episodes,
    }
    validation = {
        "trace_step_count": app.ACQUISITION_STEPS,
        "runtime_terminal": False,
        "action_bounds_valid": True,
        "safety_intervention_count": 0,
        "stale_action_applied_count": 0,
        "severe_collision_count": 0,
        "invalid_force_count": 0,
        "p40_conservation_maximum_difference": 0.0,
    }
    terminals = [
        {
            **episode,
            "replacement": False,
            "resolved": True,
            "trace_step_count": app.ACQUISITION_STEPS,
            "runtime_terminal": False,
            "planned_latency": {"observation_steps": 1, "action_steps": 1},
            "runtime_latency": {
                "observation_steps": 1,
                "action_steps": 1,
                "override_inactive": True,
            },
            "action_bounds_valid": True,
            "safety_intervention_count": 0,
            "stale_action_applied_count": 0,
            "severe_collision_count": 0,
            "invalid_force_count": 0,
            "p40_conservation_maximum_difference": 0.0,
            "validation_replay": dict(validation),
            "acquisition_failure": None,
            "candidate_count": 1,
        }
        for episode in episodes
    ]
    capsules = [
        {
            **episode,
            "offline_candidate_replay_bit_identical": True,
            "same_seed_validation_replay": True,
            "capture_enabled_disabled_identity": True,
            "anchor_blobs_complete": True,
        }
        for episode in episodes
    ]
    return plan, {
        "terminals": terminals,
        "capsules": {"episodes": capsules},
    }


@pytest.mark.parametrize(
    ("path", "value", "check"),
    (
        (("terminals", 0, "safety_intervention_count"), 1, "safety_intervention_zero"),
        (
            ("terminals", 0, "validation_replay", "safety_intervention_count"),
            1,
            "safety_intervention_zero",
        ),
        (("terminals", 0, "trace_step_count"), 994, "planned_step_terminal_semantics"),
        (
            ("terminals", 0, "runtime_latency", "observation_steps"),
            2,
            "runtime_latency_matches_plan",
        ),
    ),
)
def test_acceptance_rejects_safety_terminal_and_runtime_latency_drift(
    path: tuple[object, ...], value: object, check: str
) -> None:
    plan, execution = _acceptance_fixture()
    target = execution
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    report = app.analyze_acquisition(plan, execution)

    assert report["decision"] == "invalid"
    assert report["checks"][check] is False


def test_clean_gate_uses_all_untracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(tuple(command))
        if command[:2] == ("git", "status"):
            return SimpleNamespace(stdout="?? new.py\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="clean committed source"):
        app._require_clean_source(tmp_path, _identities())
    assert calls[0] == (
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
    )


def test_clean_gate_rejects_source_and_history_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(stdout="", returncode=1),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="source/config/XML"):
        app._require_clean_source(tmp_path, _identities())

    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(stdout="", returncode=0),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))
    identities = _identities()
    identities["historical_research_loop_trees"] = {}
    with pytest.raises(RuntimeError, match="historical research-loop"):
        app._require_clean_source(tmp_path, identities)


def test_runner_binds_clarified_frozen_document_commit() -> None:
    root = Path(app.__file__).resolve().parents[3]
    content = (root / app.FROZEN_DOCUMENT_PATH).read_bytes()
    committed = app.subprocess.run(
        (
            "git",
            "show",
            f"{app.FROZEN_DOCUMENT_COMMIT}:{app.FROZEN_DOCUMENT_PATH}",
        ),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout

    assert app.FROZEN_DOCUMENT_COMMIT == (
        "2d1752f2c0c8b9e39d7f3ebaa8e9ff0ec1d13f38"
    )
    assert content == committed


def test_plan_requires_committed_salt_before_catalog_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def load(root):
        nonlocal called
        called = True
        return _catalog()

    monkeypatch.setattr(app, "load_default_formal_household_catalogs", load)

    with pytest.raises(ValueError, match="does not match"):
        app.build_plan(tmp_path, SALT)
    assert called is False


def test_disk_gate_requires_five_gibibytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        app.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=5 * 1024**3 - 1),
    )

    with pytest.raises(RuntimeError, match="5GiB"):
        app._require_disk_capacity(tmp_path / "new" / "output")


def test_runner_writes_hash_bound_capsule_artifacts_and_failure_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "acquisition"
    monkeypatch.setattr(app, "_validate_arguments", lambda arguments: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "c" * 40)
    monkeypatch.setattr(app, "_source_identities", lambda root: _identities())
    monkeypatch.setattr(app, "_require_clean_source", lambda root, identities: None)
    monkeypatch.setattr(app, "_require_disk_capacity", lambda output: None)
    monkeypatch.setattr(app, "read_seed_salt", lambda path: SALT)
    monkeypatch.setattr(app, "require_seed_reveal", lambda commitment, salt: None)
    monkeypatch.setattr(app, "build_plan", lambda root, salt: _minimal_plan())
    monkeypatch.setattr(
        app,
        "execute_plan",
        lambda root, plan: {
            "terminals": [],
            "capsules": {
                "schema_version": app.CAPSULE_INDEX_SCHEMA,
                "episodes": [],
            },
            "binary_artifacts": {"blobs/example.bin": b"payload"},
        },
    )
    monkeypatch.setattr(
        app,
        "analyze_acquisition",
        lambda plan, execution: {
            "decision": "accepted as immutable acquisition evidence contract",
            "checks": {"passed": True},
        },
    )

    result = app.run(
        app.build_parser().parse_args(
            ["--output", str(output), "--salt-file", "ignored"]
        )
    )
    manifest = json.loads((output / "manifest.json").read_text())

    assert result["decision"].startswith("accepted")
    assert (output / "blobs/example.bin").read_bytes() == b"payload"
    assert set(path.name for path in output.iterdir()) == {
        "blobs",
        "plan.json",
        "capsules.json",
        "report.json",
        "manifest.json",
    }
    for name, identity in manifest["artifacts"].items():
        payload = (output / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    assert not output.with_name(output.name + ".tmp").exists()


def test_funnel_runner_analyzes_twice_and_writes_independent_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "funnel"
    source = tmp_path / "capsules"
    source.mkdir()
    analysis = {
        "episodes": [],
        "aggregate": {
            "episode_count": 24,
            "checks": {"passed": True},
        },
    }
    source_identity = {
        "path": str(source),
        "source_commit": "a" * 40,
        "report": {"path": "report.json", "sha256": "b" * 64, "bytes": 1},
        "manifest": {"path": "manifest.json", "sha256": "c" * 64, "bytes": 1},
    }
    calls = []
    monkeypatch.setattr(app, "_validate_arguments", lambda arguments: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "d" * 40)
    monkeypatch.setattr(app, "_source_identities", lambda root: _identities())
    monkeypatch.setattr(app, "_require_clean_source", lambda root, identities: None)

    def analyze(root, capsules, identities):
        calls.append((root, capsules, identities))
        return analysis, source_identity

    monkeypatch.setattr(app, "analyze_candidate_capsule_directory", analyze)
    result = app.run(
        app.build_parser().parse_args(
            [
                "--mode",
                "funnel",
                "--capsules",
                str(source),
                "--output",
                str(output),
            ]
        )
    )
    report = json.loads((output / "report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())

    assert len(calls) == 2
    assert result["decision"] == (
        "accepted as candidate-funnel measurement evidence"
    )
    assert report["report_replay_bit_identical"] is True
    assert report["descriptive_stage_is_not_causal_improvement_evidence"] is True
    assert manifest["report_only"] is True
    assert manifest["formal_candidate_output_modified"] is False
    assert set(path.name for path in output.iterdir()) == {
        "report.json",
        "manifest.json",
    }


def test_bound_blob_rejects_path_escape_and_tamper(tmp_path: Path) -> None:
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"data")
    identity = {
        "path": "blob.bin",
        "sha256": hashlib.sha256(b"data").hexdigest(),
        "bytes": 4,
    }

    assert app._read_bound_blob(tmp_path, identity) == b"data"
    with pytest.raises(app.CandidateFunnelContractError, match="escaped"):
        app._read_bound_blob(tmp_path, {**identity, "path": "../blob.bin"})
    with pytest.raises(app.CandidateFunnelContractError, match="differ"):
        app._read_bound_blob(tmp_path, {**identity, "sha256": "0" * 64})
