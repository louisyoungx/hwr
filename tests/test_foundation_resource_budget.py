from __future__ import annotations

import pytest

from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_resource_budget import foundation_storage_estimate


def test_formal_compact_storage_estimate_excludes_holdout_teacher_features() -> None:
    config = FoundationOnlineTrainingConfig(
        causality_holdout_episodes_per_task=16,
        causality_audit_windows_per_task=64,
        causality_holdout_transitions_per_episode=64,
        replay_windows_per_episode=2,
        minimum_collision_positive_episodes_per_task=8,
        minimum_collision_negative_episodes_per_task=8,
    )

    report = foundation_storage_estimate(config, task_count=3)

    assert report["holdout"]["transitions"] == 3 * 16 * (64 + 16)
    assert report["holdout"]["teacher_visual_features"] is False
    assert report["estimated_gib"] < config.maximum_estimated_run_storage_gib


def test_config_rejects_replay_capacity_that_cannot_retain_gate_evidence() -> None:
    with pytest.raises(ValueError, match="evidence unreachable"):
        FoundationOnlineTrainingConfig(
            replay_transition_capacity=3 * 16,
            replay_windows_per_episode=2,
            minimum_collision_positive_episodes_per_task=1,
            minimum_collision_negative_episodes_per_task=1,
        )
