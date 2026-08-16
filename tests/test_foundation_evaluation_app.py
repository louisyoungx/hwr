import hashlib
import json
from types import SimpleNamespace

import pytest

import hwr.apps.evaluate_foundation_world_model as evaluation_app
from hwr.apps.evaluate_foundation_world_model import (
    ABLATIONS,
    _artifact_manifest,
    _require_action_causality,
    _unseen_seeds,
    _video_acceptance,
    build_parser,
)
from hwr.data.foundation_cache import FoundationCacheKey
from hwr.perception.foundation import language_source_sha256
from hwr.world_model import (
    ACTION_CAUSALITY_COMPONENTS,
    CounterfactualCausalityReport,
    CounterfactualComponentReport,
    assess_action_causality,
)
from hwr.train.foundation_registry import foundation_lineage
from hwr.train.foundation_holdout import HOLDOUT_COLLECTOR
from hwr.train.development_gate import (
    COMMITTED_SNAPSHOT_CHECKS,
    DEVELOPMENT_READY_SCHEMA,
    REQUIRED_DEVELOPMENT_CHECKS,
)


def test_foundation_evaluation_defaults_match_fixed_acceptance_protocol() -> None:
    arguments = build_parser().parse_args(["runs/example"])

    assert arguments.seed_count == 40
    assert arguments.video_seed_count == 1
    assert ABLATIONS == ("none", "lock_left", "lock_right")


def test_foundation_evaluation_has_no_exploration_or_training_switch() -> None:
    destinations = {action.dest for action in build_parser()._actions}

    assert "exploration" not in destinations
    assert "train" not in destinations
    assert "expert" not in destinations


def _minimal_run_arguments(tmp_path, evaluation_id: str):
    run = tmp_path / "run"
    _write_json(
        run / "run-manifest.json",
        {
            "training_config": {
                "seed": 7,
                "camera_width": 32,
                "camera_height": 32,
            }
        },
    )
    arguments = build_parser().parse_args(
        [
            str(run),
            "--output-root",
            str(tmp_path / "evaluations"),
            "--evaluation-id",
            evaluation_id,
            "--seed-count",
            "1",
        ]
    )
    return run, arguments


def _patch_evaluation_preamble(monkeypatch, run, task):
    monkeypatch.setattr(
        evaluation_app, "_require_action_causality", lambda path: path
    )
    monkeypatch.setattr(
        evaluation_app,
        "_unseen_seeds",
        lambda path, count, start: (31,),
    )
    monkeypatch.setattr(
        evaluation_app,
        "load_default_formal_household_catalogs",
        lambda root: ({"task-a/v1": task}, {"task-a/v1": object()}),
    )
    monkeypatch.setattr(
        evaluation_app,
        "load_foundation_model_locks",
        lambda lock_path, model_root: {"qwen3-embedding-0.6b": object()},
    )


def test_language_materialization_finishes_before_first_environment(
    tmp_path, monkeypatch
) -> None:
    run, arguments = _minimal_run_arguments(tmp_path, "ordered")
    task = SimpleNamespace(max_steps=1, control_hz=20.0)
    _patch_evaluation_preamble(monkeypatch, run, task)
    events = []
    policy = SimpleNamespace(close=lambda: events.append("policy-closed"))
    monkeypatch.setattr(
        evaluation_app,
        "Qwen3LanguageProvider",
        lambda lock, device: object(),
    )

    def materialize(run_path, output_path, tasks, provider):
        events.append("language-materialized")
        return SimpleNamespace(resolver=object())

    def build_policy(run_path, resolver, *, device):
        assert events == ["language-materialized"]
        events.append("policy-built")
        return policy

    def backend(*args, **kwargs):
        assert events == ["language-materialized", "policy-built"]
        events.append("environment-built")
        return object()

    def evaluate(task_id, max_steps, environment_factory, policy, seeds, **kwargs):
        environment_factory()
        raise RuntimeError("stop after first environment")

    monkeypatch.setattr(
        evaluation_app, "materialize_evaluation_language", materialize
    )
    monkeypatch.setattr(evaluation_app, "_policy", build_policy)
    monkeypatch.setattr(
        evaluation_app, "MujocoFormalHouseholdDualArmBackend", backend
    )
    monkeypatch.setattr(evaluation_app, "evaluate_bimanual_policy", evaluate)

    with pytest.raises(RuntimeError, match="stop after first environment"):
        evaluation_app.run(arguments)

    assert events[:3] == [
        "language-materialized",
        "policy-built",
        "environment-built",
    ]


def test_missing_language_weights_fail_before_policy_or_episode(
    tmp_path, monkeypatch
) -> None:
    run, arguments = _minimal_run_arguments(tmp_path, "missing-weights")
    task = SimpleNamespace(max_steps=1, control_hz=20.0)
    _patch_evaluation_preamble(monkeypatch, run, task)

    def missing_provider(lock, device):
        raise FileNotFoundError("missing Qwen3 weights")

    def unexpected(*args, **kwargs):
        pytest.fail("evaluation advanced past missing language weights")

    monkeypatch.setattr(evaluation_app, "Qwen3LanguageProvider", missing_provider)
    monkeypatch.setattr(
        evaluation_app, "materialize_evaluation_language", unexpected
    )
    monkeypatch.setattr(evaluation_app, "_policy", unexpected)
    monkeypatch.setattr(evaluation_app, "evaluate_bimanual_policy", unexpected)
    monkeypatch.setattr(
        evaluation_app, "MujocoFormalHouseholdDualArmBackend", unexpected
    )

    with pytest.raises(FileNotFoundError, match="missing Qwen3 weights"):
        evaluation_app.run(arguments)


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


def _evaluation_language(output, run) -> None:
    cache = output / "evaluation-language/cache"
    entries = []
    for task_index in range(3):
        for instruction_index in range(3):
            text = f"评测指令 {task_index}-{instruction_index}"
            source = language_source_sha256(text, "zh-CN")
            key = FoundationCacheKey(
                "language", source, "e" * 64, "f" * 64
            )
            cache_key = key.digest
            path = cache / "language" / cache_key[:2] / f"{cache_key}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"embedding-{source}".encode())
            entries.append(
                {
                    "task_id": f"task-{task_index}/v1",
                    "locale": "zh-CN",
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "source_sha256": source,
                    "cache_key": cache_key,
                    "encoder_lock_sha256": "e" * 64,
                    "preprocess_sha256": "f" * 64,
                    "output_dimension": 6,
                    "path": path.relative_to(output / "evaluation-language").as_posix(),
                    "file_sha256": _digest(path),
                    "bytes": path.stat().st_size,
                }
            )
    language_index = run / "features/language.json"
    language_index.parent.mkdir(parents=True, exist_ok=True)
    language_index.write_bytes(b"fixture-language-index")
    training_embedding = run / "feature-cache/language/fixture.npz"
    training_embedding.parent.mkdir(parents=True, exist_ok=True)
    training_embedding.write_bytes(b"fixture-training-embedding")
    replay = run / "replay/autonomous/manifest.json"
    training_inputs = {
        "run_path": str(run.resolve()),
        "language_index": {
            "path": language_index.relative_to(run).as_posix(),
            "sha256": _digest(language_index),
            "bytes": language_index.stat().st_size,
        },
        "replay_manifest": {
            "path": replay.relative_to(run).as_posix(),
            "sha256": _digest(replay),
            "bytes": replay.stat().st_size,
        },
        "embedding_files": [
            {
                "path": training_embedding.relative_to(run).as_posix(),
                "sha256": _digest(training_embedding),
                "bytes": training_embedding.stat().st_size,
                "cache_key": "a" * 64,
            }
        ],
    }
    _write_json(
        output / "evaluation-language/manifest.json",
        {
            "schema_version": "hwr.foundation-evaluation-language/v1",
            "instruction_count": 9,
            "task_count": 3,
            "encoder": {
                "lock_sha256": "e" * 64,
                "output_dimension": 6,
            },
            "preprocess_sha256": "f" * 64,
            "training_inputs": training_inputs,
            "instructions": entries,
            "isolation": {
                "evaluation_only": True,
                "training_artifacts_unchanged": True,
                "training_run": str(run.resolve()),
            },
        },
    )


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
            "schema_version": "hwr.foundation-online-run/v4",
            "source_commit": "abc123",
            "development_ready": {
                "schema_version": DEVELOPMENT_READY_SCHEMA,
                "sha256": _digest(readiness),
                "path": "development-ready.json",
            },
            "lineage": foundation_lineage("abc123"),
            "training_config": {
                "seed": 7,
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
    physical_components = {
        name: component_reports[name]
        for name in ("visual_latent", "proprioception")
    }
    physical_report = CounterfactualCausalityReport(
        2.0,
        2.4,
        1.2,
        (2.0, 2.0),
        (2.4, 2.4),
        (0.1, 0.1),
        physical_components,
        ("visual_latent", "proprioception"),
    )
    physical_assessment = assess_action_causality(physical_report)
    statistics = {
        "count": 1,
        "reports": [raw_report.to_dict()],
        "shuffled_to_true_ratios": [1.2],
        "ratio_p05": 1.2,
        "ratio_median": 1.2,
        "ratio_p95": 1.2,
        "lower_bound_passed": True,
        "passed_fraction": 1.0,
        "all_reports_passed": True,
        "robust_passed": True,
    }
    physical_statistics = {
        **statistics,
        "reports": [physical_report.to_dict()],
    }
    _write_json(
        report,
        {
            "schema_version": "hwr.foundation-action-causality/v6",
            "action_source": "actual_executed_action",
            "safety_action_source": "actor_proposal",
            "counterfactual_pairing": "proposal-executed-pair/v1",
            "counterfactual_transform": "deterministic-global-derangement/v1",
            "partition_key": "task_id",
            "partitions": {
                "task-a/v1": {
                    "report": raw_report.to_dict(),
                    "assessment": raw_assessment,
                    "shuffle_statistics": statistics,
                    "one_step_action_utilization": {
                        "report": physical_report.to_dict(),
                        "assessment": physical_assessment,
                        "shuffle_statistics": physical_statistics,
                    },
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
            "shuffle_repeats": 1,
            "shuffle_statistics": statistics,
            "one_step_action_utilization": {
                "conditioning": "teacher-forced-posterior-state/v1",
                "physical_components": ["visual_latent", "proprioception"],
                "report": physical_report.to_dict(),
                "assessment": physical_assessment,
                "shuffle_statistics": physical_statistics,
            },
            "window_selection": [
                {
                    "task_id": "task-a/v1",
                    "episode_id": "episode-1",
                    "transition_start": 0,
                    "transition_stop": 16,
                }
            ],
            "holdout_collector": HOLDOUT_COLLECTOR,
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
        "actor_readiness_unlocked": True,
        "task_actor_update_count": 1,
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
    _write_json(output / "acceptance.json", {
        "schema_version": "hwr.foundation-per-seed-acceptance/v1",
        "passed": False,
        "per_seed_passed": True,
        "formal_passed": False,
    })
    _evaluation_language(output, run)

    manifest = _artifact_manifest(output, run, (31,), ())

    assert manifest["schema_version"] == "hwr.foundation-evaluation-run/v3"
    assert manifest["per_seed_passed"] is True
    assert manifest["formal_passed"] is False
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
        "evaluation-language/manifest.json",
    } <= set(manifest["artifacts"])
    assert len(
        [
            name
            for name in manifest["artifacts"]
            if name.startswith("evaluation-language/cache/")
        ]
    ) == 9
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
