import hashlib
import json

import pytest

from hwr.apps.evaluate_foundation_world_model import (
    ABLATIONS,
    _require_action_causality,
    build_parser,
)


def test_foundation_evaluation_defaults_match_fixed_acceptance_protocol() -> None:
    arguments = build_parser().parse_args(["runs/example"])

    assert arguments.seed_count == 20
    assert arguments.video_seed_count == 1
    assert ABLATIONS == ("none", "lock_left", "lock_right")


def test_foundation_evaluation_has_no_exploration_or_training_switch() -> None:
    destinations = {action.dest for action in build_parser()._actions}

    assert "exploration" not in destinations
    assert "train" not in destinations
    assert "expert" not in destinations


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _causality_run(tmp_path):
    run = tmp_path / "run"
    report = run / "diagnostics/action-causality/update-000000001/report.json"
    _write_json(report, {"assessment": {"passed": True}})
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    diagnostics = {
        "action_causality_report_sha256": digest,
        "action_causality_passed": True,
    }
    _write_json(
        run / "checkpoints/update-000000001/manifest.json",
        {"training_diagnostics": diagnostics},
    )
    _write_json(
        run / "deployments/update-000000001/manifest.json",
        {"training_diagnostics": diagnostics},
    )
    _write_json(
        run / "latest.json",
        {
            "training_checkpoint": "checkpoints/update-000000001",
            "deployment": "deployments/update-000000001",
            "action_causality_report": (
                "diagnostics/action-causality/update-000000001/report.json"
            ),
            "action_causality_sha256": digest,
        },
    )
    return run, report


def test_evaluation_requires_one_hash_bound_causality_report(tmp_path) -> None:
    run, report = _causality_run(tmp_path)

    assert _require_action_causality(run) == report

    report.write_text('{"assessment":{"passed":false}}', encoding="utf-8")
    with pytest.raises(ValueError, match="report hash differs"):
        _require_action_causality(run)


def test_evaluation_rejects_checkpoint_causality_provenance_drift(tmp_path) -> None:
    run, _ = _causality_run(tmp_path)
    manifest = run / "checkpoints/update-000000001/manifest.json"
    value = json.loads(manifest.read_text())
    value["training_diagnostics"]["action_causality_report_sha256"] = "0" * 64
    _write_json(manifest, value)

    with pytest.raises(ValueError, match="checkpoint and action causality"):
        _require_action_causality(run)
