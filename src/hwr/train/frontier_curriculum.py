"""Automatic reset curriculum built only from self-discovered physical states."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from hwr.core.state_snapshot import PhysicalStateSnapshot


@dataclass(frozen=True)
class FrontierCurriculumConfig:
    capacity_per_task: int = 16
    reset_probability: float = 0.50
    discovery_reach_meters: float = 0.06
    bilateral_reach_meters: float = 0.10
    score_distance_scale_meters: float = 0.08

    def __post_init__(self) -> None:
        if self.capacity_per_task <= 0:
            raise ValueError("frontier capacity must be positive")
        if not 0.0 <= self.reset_probability <= 1.0:
            raise ValueError("frontier reset probability must be in [0, 1]")
        if min(
            self.discovery_reach_meters,
            self.bilateral_reach_meters,
            self.score_distance_scale_meters,
        ) <= 0.0:
            raise ValueError("frontier distance scales must be positive")
        if self.bilateral_reach_meters < self.discovery_reach_meters:
            raise ValueError("bilateral frontier reach cannot be stricter than discovery")


@dataclass(frozen=True)
class FrontierOutcome:
    left_reach_distance: float
    right_reach_distance: float
    left_contact: bool
    right_contact: bool
    severe_collision: bool = False

    def __post_init__(self) -> None:
        distances = (self.left_reach_distance, self.right_reach_distance)
        if min(distances) < 0.0 or not all(math.isfinite(item) for item in distances):
            raise ValueError("frontier reach distances must be finite and non-negative")


@dataclass(frozen=True)
class FrontierEntry:
    snapshot: PhysicalStateSnapshot
    outcome: FrontierOutcome
    score: float
    signature: int
    source_episode: int
    source_step: int


class OutcomeFrontierCurriculum:
    """Reuse autonomous discoveries as reset states without prescribing actions."""

    def __init__(
        self,
        task_ids: Sequence[str],
        config: FrontierCurriculumConfig | None = None,
    ) -> None:
        identities = tuple(sorted(set(task_ids)))
        if not identities:
            raise ValueError("frontier curriculum requires task identities")
        self.task_ids = identities
        self.config = config or FrontierCurriculumConfig()
        self.entries: dict[str, list[FrontierEntry]] = {
            task_id: [] for task_id in identities
        }
        self.reset_count = 0

    def outcome_from_metrics(self, metrics: Mapping[str, float]) -> FrontierOutcome:
        return FrontierOutcome(
            left_reach_distance=float(metrics["left_reach_distance"]),
            right_reach_distance=float(metrics["right_reach_distance"]),
            left_contact=float(metrics["left_contact"]) > 0.5,
            right_contact=float(metrics["right_contact"]) > 0.5,
            severe_collision=float(metrics["severe_collisions"]) > 0.0,
        )

    def qualifies(self, outcome: FrontierOutcome) -> bool:
        return not outcome.severe_collision and (
            outcome.left_contact
            or outcome.right_contact
            or min(
                outcome.left_reach_distance, outcome.right_reach_distance
            ) <= self.config.discovery_reach_meters
            or max(
                outcome.left_reach_distance, outcome.right_reach_distance
            ) <= self.config.bilateral_reach_meters
        )

    def consider(
        self,
        task_id: str,
        snapshot: PhysicalStateSnapshot,
        outcome: FrontierOutcome,
        *,
        source_episode: int,
        source_step: int,
    ) -> bool:
        if task_id not in self.entries or snapshot.task_id != task_id:
            raise ValueError("frontier candidate task identity differs")
        if not self.qualifies(outcome):
            return False
        score = self._score(outcome)
        signature = self._signature(outcome)
        candidate = FrontierEntry(
            snapshot, outcome, score, signature, source_episode, source_step
        )
        values = self.entries[task_id]
        same = [item for item in values if item.signature == signature]
        signature_capacity = max(1, self.config.capacity_per_task // 4)
        if len(same) >= signature_capacity:
            worst = min(same, key=lambda item: item.score)
            if score <= worst.score + 1.0e-6:
                return False
            values.remove(worst)
        elif len(values) >= self.config.capacity_per_task:
            worst = min(values, key=lambda item: item.score)
            if score <= worst.score + 1.0e-6:
                return False
            values.remove(worst)
        values.append(candidate)
        values.sort(key=lambda item: (-item.score, item.source_episode, item.source_step))
        return True

    def select(
        self, task_id: str, rng: np.random.Generator
    ) -> FrontierEntry | None:
        try:
            values = self.entries[task_id]
        except KeyError as exc:
            raise ValueError(f"frontier curriculum does not know {task_id}") from exc
        if not values or rng.random() >= self.config.reset_probability:
            return None
        self.reset_count += 1
        candidate_count = max(1, (len(values) + 1) // 2)
        return values[int(rng.integers(0, candidate_count))]

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "hwr.outcome-frontier-curriculum/v1",
            "config": asdict(self.config),
            "entry_counts": {
                task_id: len(values) for task_id, values in self.entries.items()
            },
            "reset_count": self.reset_count,
            "action_outputs": False,
            "actor_input_fields": [],
            "task_stages": False,
            "source": "autonomous_physical_state_discovery",
            "selection": "uniform_among_top_score_half",
            "score": "exp(-max(left_reach,right_reach)/scale)",
            "contact_affects_score": False,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "task_ids": self.task_ids,
            "config": asdict(self.config),
            "reset_count": self.reset_count,
            "entries": {
                task_id: [
                    {
                        "snapshot": asdict(item.snapshot),
                        "outcome": asdict(item.outcome),
                        "score": item.score,
                        "signature": item.signature,
                        "source_episode": item.source_episode,
                        "source_step": item.source_step,
                    }
                    for item in values
                ]
                for task_id, values in self.entries.items()
            },
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if tuple(value["task_ids"]) != self.task_ids:
            raise ValueError("frontier checkpoint task identities differ")
        if dict(value["config"]) != asdict(self.config):
            raise ValueError("frontier checkpoint configuration differs")
        self.reset_count = int(value["reset_count"])
        states = value["entries"]
        for task_id in self.task_ids:
            entries = [
                FrontierEntry(
                    snapshot=PhysicalStateSnapshot(**item["snapshot"]),
                    outcome=FrontierOutcome(**item["outcome"]),
                    score=self._score(FrontierOutcome(**item["outcome"])),
                    signature=self._signature(FrontierOutcome(**item["outcome"])),
                    source_episode=int(item["source_episode"]),
                    source_step=int(item["source_step"]),
                )
                for item in states[task_id]
            ]
            entries.sort(
                key=lambda item: (
                    -item.score, item.source_episode, item.source_step
                )
            )
            self.entries[task_id] = entries

    def _score(self, outcome: FrontierOutcome) -> float:
        worst = max(outcome.left_reach_distance, outcome.right_reach_distance)
        return float(math.exp(-worst / self.config.score_distance_scale_meters))

    def _signature(self, outcome: FrontierOutcome) -> int:
        contact = int(outcome.left_contact) | (int(outcome.right_contact) << 1)
        if contact:
            return contact
        near = int(
            outcome.left_reach_distance <= self.config.discovery_reach_meters
        ) | (
            int(outcome.right_reach_distance <= self.config.discovery_reach_meters)
            << 1
        )
        return 4 + near
