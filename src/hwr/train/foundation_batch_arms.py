"""Frozen batch-pair schedules for the R0001-P05 replay experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Protocol

import numpy as np

from hwr.train.foundation_sequence_reservoir import source_episode_id


BATCH_ARMS = ("duplicate", "same_source", "cross_source")
BATCH_SCHEDULE_SCHEMA = "hwr.foundation-batch-arm-schedule/v1"


class BatchArmLoader(Protocol):
    def __len__(self) -> int: ...

    def window_metadata(self, index: int) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class BatchArmStep:
    anchor_index: int
    duplicate_index: int
    same_source_index: int
    cross_source_index: int

    def indices(self, arm: str) -> tuple[int, int]:
        if arm not in BATCH_ARMS:
            raise ValueError(f"unknown batch arm: {arm}")
        return self.anchor_index, int(getattr(self, f"{arm}_index"))


@dataclass(frozen=True)
class BatchArmSchedule:
    seed: int
    visual_update_interval: int
    steps: tuple[BatchArmStep, ...]
    eligible_indices: tuple[int, ...]
    excluded_indices: tuple[int, ...]
    schema_version: str = BATCH_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.seed < 0
            or self.visual_update_interval <= 0
            or not self.steps
            or not self.eligible_indices
            or set(self.eligible_indices) & set(self.excluded_indices)
        ):
            raise ValueError("batch arm schedule is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "visual_update_interval": self.visual_update_interval,
            "eligible_indices": list(self.eligible_indices),
            "excluded_indices": list(self.excluded_indices),
            "steps": [asdict(value) for value in self.steps],
        }


def build_batch_arm_schedule(
    loader: BatchArmLoader,
    *,
    seed: int,
    updates: int,
    visual_update_interval: int = 4,
) -> BatchArmSchedule:
    if seed < 0 or min(updates, visual_update_interval) <= 0 or len(loader) <= 0:
        raise ValueError("batch arm schedule dimensions are invalid")
    identities = tuple(_window_identity(loader.window_metadata(index)) for index in range(len(loader)))
    by_source: dict[str, list[int]] = {}
    by_stratum: dict[tuple[object, ...], list[int]] = {}
    for index, identity in enumerate(identities):
        by_source.setdefault(identity["source_episode_id"], []).append(index)
        by_stratum.setdefault(identity["stratum"], []).append(index)
    eligible = tuple(
        index
        for index, identity in enumerate(identities)
        if _has_same_source_candidate(index, identity, identities, by_source)
        and _has_cross_source_candidate(index, identity, identities, by_stratum)
    )
    if not eligible:
        raise ValueError("batch arm schedule has no eligible anchors")
    excluded = tuple(index for index in range(len(loader)) if index not in set(eligible))
    rng = np.random.default_rng(seed)
    visual_eligible = tuple(
        index for index in eligible if identities[index]["visual_supervision"] is True
    )
    if not visual_eligible:
        raise ValueError("batch arm schedule has no visual supervision anchors")
    anchors = tuple(
        int(
            rng.choice(
                np.asarray(
                    visual_eligible
                    if update % visual_update_interval == 0
                    else eligible,
                    np.int64,
                )
            )
        )
        for update in range(updates)
    )
    source_offsets: dict[tuple[str, bool], int] = {}
    stratum_offsets: dict[tuple[tuple[object, ...], str], int] = {}
    steps = []
    for anchor_index in anchors:
        identity = identities[anchor_index]
        same = _cycle_candidate(
            _same_source_candidates(anchor_index, identity, identities, by_source),
            source_offsets,
            (identity["source_episode_id"], identity["visual_supervision"]),
        )
        cross = _cycle_candidate(
            _cross_source_candidates(anchor_index, identity, identities, by_stratum),
            stratum_offsets,
            (identity["stratum"], identity["source_episode_id"]),
        )
        steps.append(BatchArmStep(anchor_index, anchor_index, same, cross))
    return BatchArmSchedule(
        seed, visual_update_interval, tuple(steps), eligible, excluded
    )


def audit_batch_arm_schedule(
    loader: BatchArmLoader, schedule: BatchArmSchedule
) -> dict[str, object]:
    identities = tuple(_window_identity(loader.window_metadata(index)) for index in range(len(loader)))
    arms = {}
    for arm in BATCH_ARMS:
        source_counts = []
        unique_counts = []
        strata_match = []
        for step in schedule.steps:
            left, right = step.indices(arm)
            pair = (identities[left], identities[right])
            source_counts.append(len({value["source_episode_id"] for value in pair}))
            unique_counts.append(len({left, right}))
            strata_match.append(pair[0]["stratum"] == pair[1]["stratum"])
        expected_sources = 2 if arm == "cross_source" else 1
        expected_windows = 1 if arm == "duplicate" else 2
        arms[arm] = {
            "batch_count": len(schedule.steps),
            "source_episodes_per_batch_min": min(source_counts),
            "source_episodes_per_batch_max": max(source_counts),
            "unique_windows_per_batch_min": min(unique_counts),
            "unique_windows_per_batch_max": max(unique_counts),
            "all_strata_match": all(strata_match),
            "passed": (
                set(source_counts) == {expected_sources}
                and set(unique_counts) == {expected_windows}
                and all(strata_match)
            ),
        }
    return {
        "schema_version": "hwr.foundation-batch-arm-schedule-audit/v1",
        "seed": schedule.seed,
        "updates": len(schedule.steps),
        "visual_update_interval": schedule.visual_update_interval,
        "visual_update_anchors_valid": all(
            identities[step.anchor_index]["visual_supervision"] is True
            for update, step in enumerate(schedule.steps)
            if update % schedule.visual_update_interval == 0
        ),
        "eligible_window_count": len(schedule.eligible_indices),
        "excluded_window_count": len(schedule.excluded_indices),
        "arms": arms,
        "passed": (
            all(value["passed"] for value in arms.values())
            and all(
                identities[step.anchor_index]["visual_supervision"] is True
                for update, step in enumerate(schedule.steps)
                if update % schedule.visual_update_interval == 0
            )
        ),
    }


def _window_identity(metadata: Mapping[str, object]) -> dict[str, object]:
    episode = metadata.get("metadata", {})
    if not isinstance(episode, Mapping):
        raise TypeError("batch arm window metadata is incomplete")
    source = source_episode_id(metadata)
    task_id = str(metadata.get("task_id", ""))
    reason = str(episode.get("result_reason", ""))
    visual = episode.get("visual_supervision")
    if not source or not task_id or not reason or not isinstance(visual, bool):
        raise ValueError("batch arm window identity is incomplete")
    terminal_collision = (
        reason == "severe_collision"
        and int(metadata.get("transition_stop", -1))
        == int(metadata.get("transition_count", -2))
    )
    return {
        "source_episode_id": source,
        "task_id": task_id,
        "result_reason": reason,
        "visual_supervision": visual,
        "terminal_collision": terminal_collision,
        "stratum": (task_id, reason, visual, terminal_collision),
    }


def _has_same_source_candidate(index, identity, identities, by_source) -> bool:
    return bool(_same_source_candidates(index, identity, identities, by_source))


def _has_cross_source_candidate(index, identity, identities, by_stratum) -> bool:
    return bool(_cross_source_candidates(index, identity, identities, by_stratum))


def _same_source_candidates(index, identity, identities, by_source) -> tuple[int, ...]:
    return tuple(
        candidate
        for candidate in by_source[identity["source_episode_id"]]
        if candidate != index
        and identities[candidate]["visual_supervision"] == identity["visual_supervision"]
    )


def _cross_source_candidates(index, identity, identities, by_stratum) -> tuple[int, ...]:
    del index
    return tuple(
        candidate
        for candidate in by_stratum[identity["stratum"]]
        if identities[candidate]["source_episode_id"] != identity["source_episode_id"]
    )


def _cycle_candidate(candidates, offsets, key) -> int:
    if not candidates:
        raise RuntimeError("batch arm schedule candidate pool is empty")
    ordered = tuple(sorted(candidates))
    offset = offsets.get(key, 0)
    offsets[key] = offset + 1
    return ordered[offset % len(ordered)]
