"""Static storage envelope and runtime free-space preflight for foundation runs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Mapping

from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig


RESOURCE_PREFLIGHT_SCHEMA = "hwr.foundation-resource-preflight/v2"
_GIB = 1024**3


def foundation_storage_estimate(
    config: FoundationOnlineTrainingConfig, *, task_count: int
) -> dict[str, object]:
    if task_count <= 0:
        raise ValueError("foundation resource estimate requires tasks")
    replay_shards = config.episodes * config.replay_windows_per_episode
    replay_transitions = min(
        config.replay_transition_capacity,
        replay_shards * config.sequence_transitions,
    )
    replay_observations = replay_transitions + replay_shards
    visual_shards = min(
        replay_shards,
        config.episodes * config.visual_supervision_windows_per_episode,
    )
    visual_transitions = min(
        replay_transitions,
        visual_shards * config.sequence_transitions,
    )
    visual_observations = visual_transitions + visual_shards
    holdout_shards = task_count * config.causality_holdout_episodes_per_task
    holdout_transitions = (
        holdout_shards * config.causality_holdout_transitions_per_episode
    )
    holdout_observations = holdout_transitions + holdout_shards
    collision_shards = (
        task_count * config.collision_validation_holdout_episodes_per_task
    )
    collision_transitions = (
        collision_shards
        * config.collision_validation_holdout_transitions_per_episode
    )
    holdout_shards += collision_shards
    holdout_transitions += collision_transitions
    holdout_observations += collision_transitions + collision_shards
    raw_bytes_per_observation = (
        config.camera_width * config.camera_height * (3 * 3 + 4 + 1)
    )
    raw_bytes = (replay_observations + holdout_observations) * raw_bytes_per_observation
    teacher_cache_bytes = (
        visual_observations * config.estimated_teacher_cache_bytes_per_observation
    )
    checkpoint_bytes = (
        config.published_checkpoint_retention
        * config.estimated_checkpoint_bytes
    )
    estimated = raw_bytes + teacher_cache_bytes + checkpoint_bytes
    return {
        "schema_version": RESOURCE_PREFLIGHT_SCHEMA,
        "replay": {
            "shards": replay_shards,
            "transitions": replay_transitions,
            "observations": replay_observations,
            "visual_supervision_shards": visual_shards,
            "visual_supervision_observations": visual_observations,
        },
        "holdout": {
            "shards": holdout_shards,
            "transitions": holdout_transitions,
            "observations": holdout_observations,
            "teacher_visual_features": False,
        },
        "estimated_bytes": estimated,
        "estimated_gib": estimated / _GIB,
        "budget_gib": config.maximum_estimated_run_storage_gib,
        "within_configured_budget": (
            estimated <= config.maximum_estimated_run_storage_gib * _GIB
        ),
    }


def require_foundation_resource_budget(
    run_path: Path,
    config: FoundationOnlineTrainingConfig,
    *,
    task_count: int,
) -> dict[str, object]:
    estimate = foundation_storage_estimate(config, task_count=task_count)
    usage = shutil.disk_usage(run_path)
    required_free = max(
        config.minimum_free_storage_gib * _GIB,
        int(estimate["estimated_bytes"]) * 5 // 4,
    )
    report = {
        **estimate,
        "filesystem": {
            "free_bytes": usage.free,
            "free_gib": usage.free / _GIB,
            "required_free_bytes": required_free,
            "required_free_gib": required_free / _GIB,
        },
        "passed": bool(estimate["within_configured_budget"])
        and usage.free >= required_free,
    }
    _write_report(run_path / "resource-preflight.json", report)
    if report["passed"] is not True:
        raise RuntimeError(
            "foundation resource preflight failed: estimated storage or free space "
            "exceeds the configured envelope"
        )
    return report


def _write_report(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
