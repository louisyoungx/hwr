"""Episode-level physical metrics for bimanual training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PhysicalProgressStatistics:
    minimum_target_distance: float
    maximum_articulation_position: float
    maximum_controlled_target_progress: float
    maximum_controlled_articulation_progress: float


def transition_safety_cost(
    runtime_info: Mapping[str, object], metrics: Mapping[str, float]
) -> bool:
    """Label autonomous transitions from runtime intervention or severe contact."""
    return bool(runtime_info["safety_intervened"]) or float(
        metrics["severe_collisions"]
    ) > 0.0


def physical_progress_statistics(
    states: list[tuple[float, ...]],
) -> PhysicalProgressStatistics:
    """Summarize privileged progress for audit without exposing it to the Actor."""
    if not states:
        raise ValueError("physical progress statistics require at least one state")
    return PhysicalProgressStatistics(
        minimum_target_distance=min(
            math.dist(state[0:3], state[12:15]) for state in states
        ),
        maximum_articulation_position=max(state[6] for state in states),
        maximum_controlled_target_progress=max(state[60] for state in states),
        maximum_controlled_articulation_progress=max(state[61] for state in states),
    )


def physical_progress_record_fields(
    states: list[tuple[float, ...]],
) -> dict[str, float]:
    summary = physical_progress_statistics(states)
    return {
        "minimum_target_distance": summary.minimum_target_distance,
        "maximum_articulation_position": summary.maximum_articulation_position,
        "maximum_controlled_target_progress": (
            summary.maximum_controlled_target_progress
        ),
        "maximum_controlled_articulation_progress": (
            summary.maximum_controlled_articulation_progress
        ),
    }


def bilateral_near_statistics(
    states: list[tuple[float, ...]], threshold: float = 0.10
) -> tuple[int, int]:
    total = 0
    current = 0
    longest = 0
    for state in states:
        near = max(state[24], state[25]) <= threshold
        total += int(near)
        current = current + 1 if near else 0
        longest = max(longest, current)
    return total, longest
