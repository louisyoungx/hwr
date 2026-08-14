"""Task-independent physical interaction coverage for Actor admission."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


INTERACTION_COVERAGE_SCHEMA = "hwr.foundation-interaction-coverage/v1"


def summarize_interaction_coverage(
    replay_path: Path,
    replay_manifest: Mapping[str, object],
    *,
    minimum_displacement: float,
) -> dict[str, object]:
    """Report contact, controlled motion, and collision classes by task."""
    del replay_path
    if minimum_displacement <= 0.0:
        raise ValueError("interaction displacement threshold must be positive")
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for shard in replay_manifest.get("shards", ()):
        metadata = shard.get("metadata", {})
        audit = metadata.get("interaction_audit", {})
        if not isinstance(audit, Mapping):
            raise ValueError("replay interaction audit is invalid")
        grouped.setdefault(str(shard["task_id"]), []).append(audit)
    if not grouped:
        raise ValueError("interaction coverage replay is empty")
    partitions = {
        task_id: _summarize_task(values, minimum_displacement)
        for task_id, values in sorted(grouped.items())
    }
    return {
        "schema_version": INTERACTION_COVERAGE_SCHEMA,
        "partition_key": "task_id",
        "minimum_displacement": minimum_displacement,
        "episode_count": sum(
            int(value["episode_count"]) for value in partitions.values()
        ),
        "partitions": partitions,
        "task_semantic_fields": [],
    }


def _summarize_task(
    values: list[Mapping[str, object]], minimum_displacement: float
) -> dict[str, object]:
    numeric = [_numeric_audit(value) for value in values]
    unilateral = [
        item["left_contact_steps"] > 0.0 or item["right_contact_steps"] > 0.0
        for item in numeric
    ]
    bilateral = [item["simultaneous_contact_steps"] > 0.0 for item in numeric]
    rigid = [
        item["maximum_controlled_rigid_displacement"] >= minimum_displacement
        for item in numeric
    ]
    articulated = [
        item["maximum_controlled_articulation_displacement"]
        >= minimum_displacement
        for item in numeric
    ]
    collisions = [item["severe_collision_count"] > 0.0 for item in numeric]
    return {
        "episode_count": len(numeric),
        "unilateral_contact_episode_count": sum(unilateral),
        "bilateral_contact_episode_count": sum(bilateral),
        "controlled_rigid_motion_episode_count": sum(rigid),
        "controlled_articulation_episode_count": sum(articulated),
        "controlled_motion_episode_count": sum(
            left or right for left, right in zip(rigid, articulated, strict=True)
        ),
        "severe_collision_positive_episode_count": sum(collisions),
        "severe_collision_negative_episode_count": sum(not item for item in collisions),
    }


def _numeric_audit(value: Mapping[str, object]) -> dict[str, float]:
    names = (
        "left_contact_steps",
        "right_contact_steps",
        "simultaneous_contact_steps",
        "maximum_controlled_rigid_displacement",
        "maximum_controlled_articulation_displacement",
        "severe_collision_count",
    )
    result = {name: float(value.get(name, 0.0)) for name in names}
    if not all(np.isfinite(item) and item >= 0.0 for item in result.values()):
        raise ValueError("replay interaction audit contains invalid values")
    return result
