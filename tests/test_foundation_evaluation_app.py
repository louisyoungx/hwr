import hashlib
import json

import pytest

from hwr.apps.evaluate_foundation_world_model import (
    ABLATIONS,
    _artifact_manifest,
    _require_action_causality,
    _unseen_seeds,
    _video_acceptance,
    build_parser,
)
from hwr.world_model import (
    ACTION_CAUSALITY_COMPONENTS,
    CounterfactualCausalityReport,
    CounterfactualComponentReport,
    assess_action_causality,
)
from hwr.train.foundation_registry import foundation_lineage
from hwr.train.development_gate import (
    COMMITTED_SNAPSHOT_CHECKS,
    DEVELOPMENT_READY_SCHEMA,
    REQUIRED_DEVELOPMENT_CHECKS,
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


def test_unseen_seeds_exclude_training_and_causality_holdout(tmp_path) -> None:
    run = tmp_path / "run"
    _write_json(
        run / "run-manifest.json",
        {"training_config": {"seed": 100}},
    )
    (run / "episodes.jsonl").write_text(
        json.dumps({"seed": 500}) + "\n", encoding="utf-8"
    )
    _write_json(
        run / "causality-holdout/autonomous/manifest.json",
        {"shards": [{"seed": 105229}]},
    )

    seeds = _unseen_seeds(run, 3, 500)

    assert seeds == (209958, 314687, 419416)


def test_video_evidence_requires_successful_uncut_four_view_episode_per_task() -> None:
    views = {
        name: f"{name}.mp4"
        for name in (
            "third_person",
            "head_rgb",
            "left_wrist_rgb",
            "right_wrist_rgb",
        )
    }
    videos = [
        {"task_id": "task-a/v1", "success": True, "uncut": True, "views": views},
        {"task_id": "task-b/v1", "success": True, "uncut": False, "views": views},
    ]

    report = _video_acceptance(videos, ("task-a/v1", "task-b/v1"), 1)

    assert report["successful_uncut_videos_per_task"] == {
        "task-a/v1": 1,
        "task-b/v1": 0,
    }
    assert report["passed"] is False


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _causality_run(tmp_path):
    run = tmp_path / "run"
    (run / "episodes.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run / "episodes.jsonl").write_text("")
    checks = {name: {"passed": True} for name in REQUIRED_DEVELOPMENT_CHECKS}
    for name in COMMITTED_SNAPSHOT_CHECKS:
        checks[name]["source_commit"] = "abc123"
    readiness = run / "development-ready.json"
    _write_json(
        readiness,
        {
            "schema_version": DEVELOPMENT_READY_SCHEMA,
            "source_commit": "abc123",
            "training_unlocked": True,
            "checks": checks,
        },
    )
    training_manifest = run / "replay/autonomous/manifest.json"
    audit_manifest = run / "causality-holdout/autonomous/manifest.json"
    _write_json(training_manifest, {"dataset_id": "training"})
    _write_json(audit_manifest, {"dataset_id": "causality-holdout"})
    _write_json(
        run / "run-manifest.json",
        {
            "schema_version": "hwr.foundation-online-run/v3",
            "source_commit": "abc123",
            "development_ready": {
                "schema_version": DEVELOPMENT_READY_SCHEMA,
                "sha256": _digest(readiness),
                "path": "development-ready.json",
            },
            "lineage": foundation_lineage("abc123"),
            "training_config": {
                "causality_audit_windows_per_task": 1,
                "minimum_action_causality_ratio": 1.05,
                "minimum_action_causality_horizon_fraction": 0.60,
            },
            "tasks": [{"task_id": "task-a/v1"}],
        },
    )
    report = run / "diagnostics/action-causality/update-000000001/report.json"
    component_reports = {
        name: CounterfactualComponentReport(
            1.0, 1.2, 1.2, (1.0, 1.0), (1.2, 1.2)
        )
        for name in ACTION_CAUSALITY_COMPONENTS
    }
    raw_report = CounterfactualCausalityReport(
        5.0,
        6.0,
        1.2,
        (5.0, 5.0),
        (6.0, 6.0),
        (0.1, 0.1),
        component_reports,
        ACTION_CAUSALITY_COMPONENTS,
    )
    raw_assessment = assess_action_causality(raw_report)
    _write_json(
        report,
        {
            "schema_version": "hwr.foundation-action-causality/v4",
            "action_source": "actual_executed_action",
            "safety_action_source": "actor_proposal",
            "counterfactual_pairing": "proposal-executed-pair/v1",
            "counterfactual_transform": "deterministic-global-derangement/v1",
            "partition_key": "task_id",
            "partitions": {
                "task-a/v1": {
                    "report": raw_report.to_dict(),
                    "assessment": raw_assessment,
                }
            },
            "assessment": {
                **raw_assessment,
                "passed": True,
                "aggregate_passed": True,
                "all_partitions_passed": True,
                "partition_count": 1,
            },
            "report": raw_report.to_dict(),
            "window_selection": [
                {
                    "task_id": "task-a/v1",
                    "episode_id": "episode-1",
                    "transition_start": 0,
                    "transition_stop": 16,
                }
            ],
            "holdout_collector": "foundation-causality-holdout/v1",
            "source_commit": "abc123",
            "update_count": 1,
            "training_data_manifest_sha256": _digest(training_manifest),
            "audit_data_manifest_sha256": _digest(audit_manifest),
        },
    )
    digest = _digest(report)
    diagnostics = {
        "action_causality_report_sha256": digest,
        "action_causality_passed": True,
    }
    checkpoint_artifact = run / "checkpoints/update-000000001/training-state.pt"
    checkpoint_artifact.parent.mkdir(parents=True)
    checkpoint_artifact.write_bytes(b"checkpoint")
    _write_json(
        run / "checkpoints/update-000000001/manifest.json",
        {
            "schema_version": "hwr.foundation-training-checkpoint/v1",
            "artifact_file": "training-state.pt",
            "artifact_sha256": _digest(checkpoint_artifact),
            "data_manifest_sha256": _digest(training_manifest),
            "training_diagnostics": diagnostics,
            "update_count": 1,
            "lineage": foundation_lineage("abc123"),
        },
    )
    deployment_artifact = run / "deployments/update-000000001/deployable-state.pt"
    deployment_artifact.parent.mkdir(parents=True)
    deployment_artifact.write_bytes(b"deployment")
    _write_json(
        run / "deployments/update-000000001/manifest.json",
        {
            "schema_version": "hwr.foundation-deployment/v1",
            "artifact_file": "deployable-state.pt",
            "artifact_sha256": _digest(deployment_artifact),
            "training_checkpoint_sha256": _digest(checkpoint_artifact),
            "source_commit": "abc123",
            "training_diagnostics": diagnostics,
        },
    )
    _write_json(
        run / "latest.json",
        {
            "schema_version": "hwr.foundation-online-latest/v1",
            "training_checkpoint": "checkpoints/update-000000001",
            "deployment": "deployments/update-000000001",
            "action_causality_report": (
                "diagnostics/action-causality/update-000000001/report.json"
            ),
            "action_causality_sha256": digest,
            "update_count": 1,
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


def test_evaluation_rejects_causality_holdout_manifest_drift(tmp_path) -> None:
    run, _ = _causality_run(tmp_path)
    _write_json(
        run / "causality-holdout/autonomous/manifest.json",
        {"dataset_id": "changed-after-audit"},
    )

    with pytest.raises(ValueError, match="data provenance differs"):
        _require_action_causality(run)


def test_evaluation_rejects_tampered_no_expert_run_lineage(tmp_path) -> None:
    run, _ = _causality_run(tmp_path)
    manifest = run / "run-manifest.json"
    value = json.loads(manifest.read_text())
    value["lineage"]["demonstration_datasets"] = ["forbidden"]
    _write_json(manifest, value)

    with pytest.raises(ValueError, match="no-expert lineage differs"):
        _require_action_causality(run)


def test_evaluation_rejects_development_readiness_hash_drift(tmp_path) -> None:
    run, _ = _causality_run(tmp_path)
    readiness = run / "development-ready.json"
    value = json.loads(readiness.read_text())
    value["training_unlocked"] = False
    _write_json(readiness, value)

    with pytest.raises(ValueError, match="readiness artifact differs"):
        _require_action_causality(run)


def test_evaluation_manifest_hashes_training_data_model_and_gate_artifacts(
    tmp_path,
) -> None:
    run, _ = _causality_run(tmp_path)
    output = tmp_path / "evaluation"
    output.mkdir()
    _write_json(output / "report.json", {"episodes": []})
    _write_json(output / "acceptance.json", {"passed": True})

    manifest = _artifact_manifest(output, run, (31,), ())

    assert manifest["schema_version"] == "hwr.foundation-evaluation-run/v2"
    assert {
        "training/development-ready.json",
        "training/episodes.jsonl",
        "training/replay-manifest.json",
        "training/causality-holdout-manifest.json",
        "training/action-causality.json",
        "training/checkpoint-manifest.json",
        "training/checkpoint-artifact",
        "training/deployment-manifest.json",
        "training/deployment-artifact",
    } <= set(manifest["artifacts"])
    assert all(
        len(identity["sha256"]) == 64 and identity["bytes"] >= 0
        for identity in manifest["artifacts"].values()
    )


def test_evaluation_rejects_missing_task_partition(tmp_path) -> None:
    run, report = _causality_run(tmp_path)
    value = json.loads(report.read_text())
    value["partitions"] = {}
    _write_json(report, value)
    latest = run / "latest.json"
    latest_value = json.loads(latest.read_text())
    latest_value["action_causality_sha256"] = _digest(report)
    _write_json(latest, latest_value)

    with pytest.raises(ValueError, match="partition evidence is incomplete"):
        _require_action_causality(run)


def test_evaluation_recomputes_component_causality_instead_of_trusting_passed(
    tmp_path,
) -> None:
    run, report = _causality_run(tmp_path)
    value = json.loads(report.read_text())
    raw = value["report"]
    safety = raw["component_reports"]["safety"]
    safety["shuffled_error"] = 1.0
    safety["shuffled_to_true_ratio"] = 1.0
    safety["shuffled_horizon_errors"] = [1.0, 1.0]
    raw["shuffled_action_error"] = 5.8
    raw["shuffled_to_true_ratio"] = 1.16
    raw["shuffled_horizon_errors"] = [5.8, 5.8]
    _write_json(report, value)
    latest = run / "latest.json"
    latest_value = json.loads(latest.read_text())
    latest_value["action_causality_sha256"] = _digest(report)
    _write_json(latest, latest_value)

    with pytest.raises(ValueError, match="assessment differs from evidence"):
        _require_action_causality(run)
