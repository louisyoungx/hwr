from __future__ import annotations

import json
from pathlib import Path

import pytest

from hwr.apps.train_foundation_world_model import build_parser
from hwr.train.development_gate import (
    DEVELOPMENT_READY_SCHEMA,
    current_commit,
    foundation_config_hashes,
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
                "checks": {"tests": {"passed": True}},
            }
        )
    )
    with pytest.raises(RuntimeError, match="source commit"):
        require_development_ready(ROOT, path)


def test_development_gate_accepts_matching_complete_report(tmp_path) -> None:
    path = tmp_path / "ready.json"
    report = {
        "schema_version": DEVELOPMENT_READY_SCHEMA,
        "source_commit": current_commit(ROOT),
        "foundation_config_sha256": foundation_config_hashes(ROOT),
        "checks": {
            "tests": {"passed": True},
            "architecture": {"passed": True},
        },
    }
    path.write_text(json.dumps(report))

    assert require_development_ready(ROOT, path) == report
