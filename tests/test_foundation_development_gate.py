from __future__ import annotations

import json
from pathlib import Path

import pytest

from hwr.apps.train_foundation_world_model import (
    _bind_development_readiness,
    build_parser,
)
from hwr.train.development_gate import (
    COMMITTED_SNAPSHOT_CHECKS,
    DEVELOPMENT_READY_SCHEMA,
    PROTECTED_PATHS,
    REQUIRED_DEVELOPMENT_CHECKS,
    current_commit,
    foundation_config_hashes,
    protected_tree_hashes,
    require_development_ready,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_training_cli_has_no_gate_bypass_option() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "development_ready" in destinations
    assert "skip_gate" not in destinations
    assert "expert" not in destinations
    assert "demonstration" not in destinations


def test_formal_training_cli_accepts_an_audited_independent_seed() -> None:
    arguments = build_parser().parse_args(
        ["--run-id", "replicate-a", "--seed", "20260815"]
    )

    assert arguments.seed == 20260815


def test_development_gate_requires_the_full_named_evidence_set() -> None:
    assert REQUIRED_DEVELOPMENT_CHECKS == {
        "protected_tree",
        "algorithm_audit",
        "configuration",
        "model_selection",
        "foundation_dependencies",
        "weights",
        "architecture",
        "python_size",
        "tests",
        "foundation_inference",
        "training_semantics",
    }


def test_development_gate_protects_accelerator_memory_implementation() -> None:
    assert "src/hwr/train/accelerator_memory.py" in PROTECTED_PATHS
    assert "src/hwr/train/foundation_visual_update.py" in PROTECTED_PATHS
    assert "src/hwr/adapters/mujoco/formal_household_backend.py" in PROTECTED_PATHS
    assert "src/hwr/train/development_semantics.py" in PROTECTED_PATHS


def test_development_gate_rejects_missing_or_stale_report(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        require_development_ready(ROOT, tmp_path / "missing.json")
    path = tmp_path / "ready.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": DEVELOPMENT_READY_SCHEMA,
                "source_commit": "stale",
                "foundation_config_sha256": foundation_config_hashes(ROOT),
                "protected_tree_sha256": protected_tree_hashes(ROOT),
                "checks": {"tests": {"passed": True}},
            }
        )
    )
    with pytest.raises(RuntimeError, match="source commit"):
        require_development_ready(ROOT, path)


def test_development_gate_rejects_matching_but_incomplete_report(tmp_path) -> None:
    path = tmp_path / "ready.json"
    report = {
        "schema_version": DEVELOPMENT_READY_SCHEMA,
        "source_commit": current_commit(ROOT),
        "foundation_config_sha256": foundation_config_hashes(ROOT),
        "protected_tree_sha256": protected_tree_hashes(ROOT),
        "training_unlocked": True,
        "checks": {
            "tests": {"passed": True},
            "architecture": {"passed": True},
        },
    }
    path.write_text(json.dumps(report))

    with pytest.raises(RuntimeError, match="checks are incomplete"):
        require_development_ready(ROOT, path)


def test_development_gate_accepts_exact_complete_snapshot_bound_report(
    tmp_path,
) -> None:
    path = tmp_path / "ready.json"
    commit = current_commit(ROOT)
    checks = {name: {"passed": True} for name in REQUIRED_DEVELOPMENT_CHECKS}
    for name in COMMITTED_SNAPSHOT_CHECKS:
        checks[name]["source_commit"] = commit
    report = {
        "schema_version": DEVELOPMENT_READY_SCHEMA,
        "source_commit": commit,
        "foundation_config_sha256": foundation_config_hashes(ROOT),
        "protected_tree_sha256": protected_tree_hashes(ROOT),
        "training_unlocked": True,
        "checks": checks,
    }
    path.write_text(json.dumps(report))

    assert require_development_ready(ROOT, path) == report


def test_development_gate_rejects_snapshot_evidence_from_another_commit(
    tmp_path,
) -> None:
    path = tmp_path / "ready.json"
    commit = current_commit(ROOT)
    checks = {name: {"passed": True} for name in REQUIRED_DEVELOPMENT_CHECKS}
    for name in COMMITTED_SNAPSHOT_CHECKS:
        checks[name]["source_commit"] = commit
    checks["tests"]["source_commit"] = "stale"
    report = {
        "schema_version": DEVELOPMENT_READY_SCHEMA,
        "source_commit": commit,
        "foundation_config_sha256": foundation_config_hashes(ROOT),
        "protected_tree_sha256": protected_tree_hashes(ROOT),
        "training_unlocked": True,
        "checks": checks,
    }
    path.write_text(json.dumps(report))

    with pytest.raises(RuntimeError, match="snapshot evidence differs"):
        require_development_ready(ROOT, path)


def test_training_run_copies_and_rechecks_development_readiness(tmp_path) -> None:
    source = tmp_path / "ready.json"
    source.write_text('{"training_unlocked": true}')
    run = tmp_path / "run"

    digest = _bind_development_readiness(run, source, resume=False)

    assert (run / "development-ready.json").read_bytes() == source.read_bytes()
    assert _bind_development_readiness(run, source, resume=True) == digest
    source.write_text('{"training_unlocked": false}')
    with pytest.raises(ValueError, match="readiness differs"):
        _bind_development_readiness(run, source, resume=True)
