from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from hwr.policy.latent_actions import LatentActionScaling
from hwr.train.foundation_dashboard import load_dashboard_snapshot
from hwr.train.foundation_metrics import (
    FoundationMetricsProgress,
    FoundationMetricsStore,
    mean_metrics,
    summarize_action_coverage,
)


def test_metric_mean_supports_lower_frequency_visual_measurements() -> None:
    result = mean_metrics(
        (
            {"world/total": 2.0, "visual/total": 4.0},
            {"world/total": 1.0},
            {"world/total": 3.0, "visual/total": 2.0},
        )
    )

    assert result == {"visual/total": 3.0, "world/total": 2.0}


def _episode(task_id: str, offset: float = 0.0):
    proposed = np.zeros((3, 16), np.float32)
    proposed[:, :14] = np.asarray((-0.5, 0.0, 0.5))[:, None] + offset
    proposed[:, 14:] = np.asarray(((0, 1), (1, 0), (1, 0)), np.float32)
    executed = proposed.copy()
    executed[1, 0] = 0.0
    return SimpleNamespace(
        task_id=task_id,
        arrays={
            "actor_proposal": proposed,
            "executed_action": executed,
            "reward": np.asarray((0.0, 1.0, 2.0)),
            "safety_intervention": np.asarray((0.0, 1.0, 0.0)),
        },
        metadata={"success": True, "result_metrics": {"contact": 1.0}},
    )


def test_metrics_store_is_atomic_and_cycle_records_are_immutable(tmp_path) -> None:
    store = FoundationMetricsStore(
        tmp_path, source_commit="abc", target_episodes=12
    )
    store.publish_progress(FoundationMetricsProgress("updating", 1, 4, 3, 10, 4))
    store.publish_cycle(1, {"update_count": 10, "training": {"world/total": 2.0}})

    latest = json.loads((tmp_path / "metrics/latest.json").read_text())
    assert latest["record_type"] == "cycle"
    assert not list((tmp_path / "metrics").glob("*.tmp"))
    store.publish_cycle(1, {"update_count": 10, "training": {"world/total": 2.0}})
    store.publish_cycle(2, {"update_count": 20})
    assert len(store.rollback_after(1)) == 1


def test_action_coverage_is_task_agnostic_and_dashboard_reads_small_artifacts(
    tmp_path,
) -> None:
    summary = summarize_action_coverage(
        [_episode("a"), _episode("b", 0.1)], LatentActionScaling()
    )
    assert summary["transition_count"] == 6
    assert 0.0 < summary["active_dimension_fraction"] <= 1.0
    assert summary["proposal_execution_changed_fraction"] > 0.0
    (tmp_path / "run-manifest.json").write_text("{}")
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "latest.json").write_text('{"stage":"updating"}')
    (tmp_path / "episodes.jsonl").write_text(
        '{"episode_index":0,"success":true}\n', encoding="utf-8"
    )

    snapshot = load_dashboard_snapshot(tmp_path)

    assert snapshot["episodes"]["success_count"] == 1
    assert snapshot["metrics"]["stage"] == "updating"
