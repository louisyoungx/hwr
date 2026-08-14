from __future__ import annotations

import hashlib
import json

from hwr.apps.aggregate_foundation_evaluations import (
    aggregate_foundation_evaluations,
)


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _identity(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def _evaluation(tmp_path, index: int, training_seed: int):
    run = tmp_path / f"run-{index}"
    run_manifest = {
        "schema_version": "hwr.foundation-online-run/v4",
        "source_commit": "shared-commit",
        "development_ready": {"sha256": "d" * 64},
        "training_config": {"seed": training_seed, "episodes": 120},
        "tasks": [{"task_id": "task-a/v1"}],
        "preprocessing": {"fingerprint": "same"},
        "execution": {"device": "mps"},
        "lineage": {"expert_policies": []},
    }
    _write_json(run / "run-manifest.json", run_manifest)
    (run / "episodes.jsonl").write_text(
        json.dumps({"seed": training_seed + 100}) + "\n", encoding="utf-8"
    )
    _write_json(
        run / "causality-holdout/autonomous/manifest.json",
        {"shards": [{"seed": training_seed + 200}]},
    )
    deployment_manifest = run / "deployments/update-000000001/manifest.json"
    _write_json(deployment_manifest, {"artifact_sha256": f"{index:064x}"})
    _write_json(
        run / "latest.json",
        {"deployment": "deployments/update-000000001"},
    )
    evaluation = tmp_path / f"evaluation-{index}"
    _write_json(evaluation / "report.json", {"episodes": []})
    _write_json(
        evaluation / "acceptance.json",
        {
            "schema_version": "hwr.foundation-per-seed-acceptance/v1",
            "passed": False,
            "per_seed_passed": True,
            "formal_passed": False,
        },
    )
    _write_json(
        evaluation / "manifest.json",
        {
            "schema_version": "hwr.foundation-evaluation-run/v3",
            "training_run": str(run),
            "training_seed": training_seed,
            "per_seed_passed": True,
            "formal_passed": False,
            "source_commit": "shared-commit",
            "training_run_manifest_sha256": _identity(
                run / "run-manifest.json"
            )["sha256"],
            "deployment_manifest_sha256": _identity(deployment_manifest)["sha256"],
            "unseen_seeds": [9000001, 9000002],
            "artifacts": {
                "evaluation/report.json": _identity(evaluation / "report.json"),
                "evaluation/acceptance.json": _identity(
                    evaluation / "acceptance.json"
                ),
            },
        },
    )
    return evaluation


def test_three_distinct_seed_evaluations_are_required_for_formal_passage(
    tmp_path,
) -> None:
    evaluations = [
        _evaluation(tmp_path, index, 1000 + index * 1000)
        for index in range(1, 4)
    ]

    result = aggregate_foundation_evaluations(
        evaluations, tmp_path / "aggregate"
    )

    assert result["passed"] is True
    assert result["formal_passed"] is True
    assert result["checks"]["minimum_three_training_seeds"] is True
    assert result["checks"]["training_seed_sets_disjoint"] is True


def test_duplicate_training_seed_cannot_pass_formal_aggregate(tmp_path) -> None:
    evaluations = [
        _evaluation(tmp_path, index, 1000 if index < 3 else 3000)
        for index in range(1, 4)
    ]

    result = aggregate_foundation_evaluations(
        evaluations, tmp_path / "aggregate"
    )

    assert result["passed"] is False
    assert result["checks"]["distinct_training_seeds"] is False
    assert result["checks"]["training_seed_sets_disjoint"] is False
