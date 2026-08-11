"""Automatic reset curriculum built only from self-discovered physical states."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from hwr.core.state_snapshot import PhysicalStateSnapshot


FRONTIER_QUALIFICATION_COUNTERS = (
    "observed",
    "qualified",
    "severe_collision",
    "unsupported",
    "payload_linear_speed",
    "payload_angular_speed",
    "target_beyond_workspace",
    "target_regression",
    "not_near",
)


@dataclass(frozen=True)
class FrontierCurriculumConfig:
    capacity_per_task: int = 16
    reset_probability: float = 0.50
    discovery_reach_meters: float = 0.06
    bilateral_reach_meters: float = 0.10
    score_distance_scale_meters: float = 0.08
    score_target_scale_meters: float = 0.50
    score_articulation_scale_meters: float = 0.15
    maximum_candidate_target_distance_meters: float = 1.10
    maximum_target_regression_meters: float = 0.15
    maximum_payload_linear_speed: float = 0.05
    maximum_payload_angular_speed: float = 0.15
    signature_uniform_fraction: float = 0.20
    maximum_entries_per_source_signature: int = 2
    minimum_contact_stability_steps: int = 40

    def __post_init__(self) -> None:
        if self.capacity_per_task <= 0:
            raise ValueError("frontier capacity must be positive")
        if not 0.0 <= self.reset_probability <= 1.0:
            raise ValueError("frontier reset probability must be in [0, 1]")
        if not 0.0 <= self.signature_uniform_fraction <= 1.0:
            raise ValueError("frontier signature uniform fraction must be in [0, 1]")
        if self.maximum_entries_per_source_signature <= 0:
            raise ValueError("frontier per-source signature capacity must be positive")
        if self.minimum_contact_stability_steps <= 0:
            raise ValueError("frontier contact stability steps must be positive")
        if min(
            self.discovery_reach_meters,
            self.bilateral_reach_meters,
            self.score_distance_scale_meters,
            self.score_target_scale_meters,
            self.score_articulation_scale_meters,
            self.maximum_candidate_target_distance_meters,
            self.maximum_target_regression_meters,
            self.maximum_payload_linear_speed,
            self.maximum_payload_angular_speed,
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
    support_contact: bool = True
    payload_linear_speed: float = 0.0
    payload_angular_speed: float = 0.0
    target_distance: float = 10.0
    articulation_position: float = 0.0
    initial_target_distance: float = 10.0
    task_progress_observed: bool = False

    def __post_init__(self) -> None:
        physical = (
            self.left_reach_distance,
            self.right_reach_distance,
            self.payload_linear_speed,
            self.payload_angular_speed,
            self.target_distance,
            self.initial_target_distance,
        )
        if min(physical) < 0.0 or not all(
            math.isfinite(item) for item in (*physical, self.articulation_position)
        ):
            raise ValueError("frontier physical values must be finite and non-negative")


@dataclass(frozen=True)
class FrontierEntry:
    snapshot: PhysicalStateSnapshot
    outcome: FrontierOutcome
    score: float
    signature: int
    source_episode: int
    source_step: int
    contact_stability_steps: int


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
        self.reset_validation_count = 0
        self.reset_validation_success_count = 0
        self.reset_validation_failure_count = 0
        self.qualification_counts = {
            task_id: {
                name: 0 for name in FRONTIER_QUALIFICATION_COUNTERS
            }
            for task_id in identities
        }

    def outcome_from_metrics(self, metrics: Mapping[str, float]) -> FrontierOutcome:
        return FrontierOutcome(
            left_reach_distance=float(metrics["left_reach_distance"]),
            right_reach_distance=float(metrics["right_reach_distance"]),
            left_contact=float(metrics["left_contact"]) > 0.5,
            right_contact=float(metrics["right_contact"]) > 0.5,
            severe_collision=float(metrics["severe_collisions"]) > 0.0,
            support_contact=float(
                metrics.get("physical_support_contact", metrics["support_contact"])
            ) > 0.5,
            payload_linear_speed=float(metrics["payload_linear_speed"]),
            payload_angular_speed=float(metrics["payload_angular_speed"]),
            target_distance=float(metrics.get("target_distance", 10.0)),
            articulation_position=float(metrics.get("articulation_position", 0.0)),
            initial_target_distance=float(
                metrics.get("initial_target_distance", 10.0)
            ),
            task_progress_observed=(
                "target_distance" in metrics
                and "articulation_position" in metrics
            ),
        )

    def advance_contact_stability(
        self,
        outcome: FrontierOutcome,
        previous_signature: int,
        previous_steps: int,
    ) -> tuple[int, int]:
        signature = int(outcome.left_contact) | (int(outcome.right_contact) << 1)
        if signature == 0:
            return 0, 0
        if signature == previous_signature:
            return signature, previous_steps + 1
        return signature, 1

    def qualifies(self, outcome: FrontierOutcome) -> bool:
        return not self._qualification_failures(outcome)

    def observe(self, task_id: str, outcome: FrontierOutcome) -> bool:
        """Audit one autonomous state without changing qualification semantics."""
        if task_id not in self.qualification_counts:
            raise ValueError(f"frontier curriculum does not know {task_id}")
        failures = self._qualification_failures(outcome)
        counts = self.qualification_counts[task_id]
        counts["observed"] += 1
        if not failures:
            counts["qualified"] += 1
        for failure in failures:
            counts[failure] += 1
        return not failures

    def _qualification_failures(
        self, outcome: FrontierOutcome
    ) -> tuple[str, ...]:
        physically_supported = (
            outcome.support_contact
            or outcome.left_contact
            or outcome.right_contact
        )
        progress_not_degraded = (
            not outcome.task_progress_observed
            or outcome.target_distance
            <= self.config.maximum_candidate_target_distance_meters
        ) and (
            not outcome.task_progress_observed
            or outcome.target_distance
            <= outcome.initial_target_distance
            + self.config.maximum_target_regression_meters
        )
        near = (
            outcome.left_contact
            or outcome.right_contact
            or min(
                outcome.left_reach_distance, outcome.right_reach_distance
            ) <= self.config.discovery_reach_meters
            or max(
                outcome.left_reach_distance, outcome.right_reach_distance
            ) <= self.config.bilateral_reach_meters
        )
        failures = []
        failures.extend(["severe_collision"] * outcome.severe_collision)
        failures.extend(["unsupported"] * (not physically_supported))
        failures.extend(
            ["payload_linear_speed"]
            * (
                outcome.payload_linear_speed
                > self.config.maximum_payload_linear_speed
            )
        )
        failures.extend(
            ["payload_angular_speed"]
            * (
                outcome.payload_angular_speed
                > self.config.maximum_payload_angular_speed
            )
        )
        failures.extend(
            ["target_beyond_workspace"]
            * (
                outcome.task_progress_observed
                and outcome.target_distance
                > self.config.maximum_candidate_target_distance_meters
            )
        )
        failures.extend(
            ["target_regression"]
            * (outcome.task_progress_observed and not progress_not_degraded)
        )
        failures.extend(["not_near"] * (not near))
        return tuple(failures)

    def consider(
        self,
        task_id: str,
        snapshot: PhysicalStateSnapshot,
        outcome: FrontierOutcome,
        *,
        source_episode: int,
        source_step: int,
        contact_stability_steps: int = 0,
    ) -> bool:
        if task_id not in self.entries or snapshot.task_id != task_id:
            raise ValueError("frontier candidate task identity differs")
        if not self.qualifies(outcome):
            return False
        if (
            (outcome.left_contact or outcome.right_contact)
            and contact_stability_steps < self.config.minimum_contact_stability_steps
        ):
            return False
        score = self._score(outcome)
        signature = self._signature(outcome)
        candidate = FrontierEntry(
            snapshot,
            outcome,
            score,
            signature,
            source_episode,
            source_step,
            contact_stability_steps,
        )
        values = self.entries[task_id]
        same = [item for item in values if item.signature == signature]
        same_source = [
            item
            for item in same
            if item.source_episode == source_episode
        ]
        signature_capacity = max(1, self.config.capacity_per_task // 4)
        if len(same_source) >= self.config.maximum_entries_per_source_signature:
            earliest = min(same_source, key=lambda item: item.source_step)
            replaceable = [item for item in same_source if item is not earliest]
            worst = min(replaceable or same_source, key=self._retention_rank)
            if not self._outperforms(candidate, worst):
                return False
            values.remove(worst)
        elif len(same) >= signature_capacity:
            worst = min(same, key=self._retention_rank)
            if not self._outperforms(candidate, worst):
                return False
            values.remove(worst)
        elif len(values) >= self.config.capacity_per_task:
            worst = min(values, key=self._retention_rank)
            if not self._outperforms(candidate, worst):
                return False
            values.remove(worst)
        values.append(candidate)
        values.sort(key=self._display_rank)
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
        eligible = self._eligible_entries(values)
        signatures = sorted({item.signature for item in eligible})
        best_scores = np.asarray(
            [
                max(item.score for item in eligible if item.signature == signature)
                for signature in signatures
            ],
            dtype=np.float64,
        )
        quality = best_scores / best_scores.sum()
        uniform = np.full(len(signatures), 1.0 / len(signatures))
        mix = self.config.signature_uniform_fraction
        probabilities = mix * uniform + (1.0 - mix) * quality
        signature = signatures[int(rng.choice(len(signatures), p=probabilities))]
        matching = [item for item in eligible if item.signature == signature]
        sources = sorted({item.source_episode for item in matching})
        source_scores = np.asarray(
            [
                max(item.score for item in matching if item.source_episode == source)
                for source in sources
            ],
            dtype=np.float64,
        )
        source_quality = source_scores / source_scores.sum()
        source_uniform = np.full(len(sources), 1.0 / len(sources))
        source_probabilities = mix * source_uniform + (1.0 - mix) * source_quality
        source = sources[int(rng.choice(len(sources), p=source_probabilities))]
        matching = [item for item in matching if item.source_episode == source]
        candidate_count = max(1, (len(matching) + 1) // 2)
        return matching[int(rng.integers(0, candidate_count))]

    def find(
        self,
        task_id: str,
        source_episode: int,
        source_step: int,
    ) -> FrontierEntry | None:
        return next(
            (
                item
                for item in self.entries.get(task_id, ())
                if item.source_episode == source_episode
                and item.source_step == source_step
            ),
            None,
        )

    def discard_tasks(
        self, task_ids: Sequence[str]
    ) -> dict[str, dict[str, object]]:
        identities = tuple(dict.fromkeys(task_ids))
        unknown = sorted(set(identities) - set(self.task_ids))
        if unknown:
            raise ValueError(
                "frontier cannot discard unknown tasks: " + ", ".join(unknown)
            )
        discarded = {}
        for task_id in identities:
            discarded[task_id] = {
                "entry_count": len(self.entries[task_id]),
                "qualification_counts": dict(
                    self.qualification_counts[task_id]
                ),
            }
            self.entries[task_id] = []
            self.qualification_counts[task_id] = {
                name: 0 for name in FRONTIER_QUALIFICATION_COUNTERS
            }
        return discarded

    def report_reset_outcome(
        self, entry: FrontierEntry, contact_steps: int
    ) -> bool | None:
        if entry.signature not in (1, 2, 3) or not entry.snapshot.runtime_state:
            return None
        if contact_steps < 0:
            raise ValueError("frontier reset contact steps cannot be negative")
        self.reset_validation_count += 1
        reproduced = contact_steps >= self.config.minimum_contact_stability_steps
        if reproduced:
            self.reset_validation_success_count += 1
            return True
        self.reset_validation_failure_count += 1
        values = self.entries[entry.snapshot.task_id]
        values[:] = [
            item
            for item in values
            if not (
                item.signature == entry.signature
                and item.source_episode == entry.source_episode
                and item.source_step == entry.source_step
            )
        ]
        return False

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "hwr.outcome-frontier-curriculum/v1",
            "config": asdict(self.config),
            "entry_counts": {
                task_id: len(values) for task_id, values in self.entries.items()
            },
            "qualification_counts": {
                task_id: dict(counts)
                for task_id, counts in self.qualification_counts.items()
            },
            "reset_count": self.reset_count,
            "reset_validation_count": self.reset_validation_count,
            "reset_validation_success_count": self.reset_validation_success_count,
            "reset_validation_failure_count": self.reset_validation_failure_count,
            "action_outputs": False,
            "actor_input_fields": [],
            "task_stages": False,
            "source": "autonomous_physical_state_discovery",
            "snapshot_state": (
                "positions_velocities_and_controller_loads_never_actor_inputs"
            ),
            "snapshot_migration": (
                "complete_state_replaces_and_suppresses_legacy_within_signature"
            ),
            "reset_validation": (
                "outcome_only_contact_reproduction_under_task_free_dwell"
            ),
            "selection": (
                "quality_weighted_signature_and_source_with_diversity_floor"
            ),
            "signature_uniform_fraction": self.config.signature_uniform_fraction,
            "maximum_entries_per_source_signature": (
                self.config.maximum_entries_per_source_signature
            ),
            "minimum_contact_stability_steps": (
                self.config.minimum_contact_stability_steps
            ),
            "score": (
                "bilateral_reach_quality_times_one_plus_target_and_articulation_quality"
            ),
            "contact_affects_score": False,
            "physical_stability_filter": {
                "requires_support_or_arm_contact": True,
                "maximum_candidate_target_distance_meters": (
                    self.config.maximum_candidate_target_distance_meters
                ),
                "maximum_target_regression_meters": (
                    self.config.maximum_target_regression_meters
                ),
                "maximum_payload_linear_speed": (
                    self.config.maximum_payload_linear_speed
                ),
                "maximum_payload_angular_speed": (
                    self.config.maximum_payload_angular_speed
                ),
            },
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "task_ids": self.task_ids,
            "config": asdict(self.config),
            "reset_count": self.reset_count,
            "reset_validation_count": self.reset_validation_count,
            "reset_validation_success_count": self.reset_validation_success_count,
            "reset_validation_failure_count": self.reset_validation_failure_count,
            "qualification_counts": self.qualification_counts,
            "entries": {
                task_id: [
                    {
                        "snapshot": asdict(item.snapshot),
                        "outcome": asdict(item.outcome),
                        "score": item.score,
                        "signature": item.signature,
                        "source_episode": item.source_episode,
                        "source_step": item.source_step,
                        "contact_stability_steps": item.contact_stability_steps,
                    }
                    for item in values
                ]
                for task_id, values in self.entries.items()
            },
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if tuple(value["task_ids"]) != self.task_ids:
            raise ValueError("frontier checkpoint task identities differ")
        saved_config = dict(value["config"])
        current_config = asdict(self.config)
        mutable_fields = (
            "reset_probability",
            "signature_uniform_fraction",
            "maximum_entries_per_source_signature",
            "minimum_contact_stability_steps",
            "score_target_scale_meters",
            "score_articulation_scale_meters",
            "maximum_candidate_target_distance_meters",
            "maximum_target_regression_meters",
        )
        for name in mutable_fields:
            saved_config.pop(name, None)
            current_config.pop(name)
        if saved_config != current_config:
            raise ValueError("frontier checkpoint configuration differs")
        self.reset_count = int(value["reset_count"])
        self.reset_validation_count = int(value.get("reset_validation_count", 0))
        self.reset_validation_success_count = int(
            value.get("reset_validation_success_count", 0)
        )
        self.reset_validation_failure_count = int(
            value.get("reset_validation_failure_count", 0)
        )
        saved_qualification_counts = value.get("qualification_counts", {})
        self.qualification_counts = {
            task_id: {
                name: int(
                    saved_qualification_counts.get(task_id, {}).get(name, 0)
                )
                for name in FRONTIER_QUALIFICATION_COUNTERS
            }
            for task_id in self.task_ids
        }
        states = value["entries"]
        for task_id in self.task_ids:
            entries = []
            for item in states[task_id]:
                outcome = _restore_frontier_outcome(item["outcome"])
                entries.append(
                    FrontierEntry(
                        snapshot=PhysicalStateSnapshot(**item["snapshot"]),
                        outcome=outcome,
                        score=self._score(outcome),
                        signature=self._signature(outcome),
                        source_episode=int(item["source_episode"]),
                        source_step=int(item["source_step"]),
                        contact_stability_steps=int(
                            item.get("contact_stability_steps", 0)
                        ),
                    )
                )
            entries = [
                item
                for item in entries
                if self.qualifies(item.outcome)
                and not (
                    item.signature == 3
                    and item.contact_stability_steps
                    < self.config.minimum_contact_stability_steps
                )
            ]
            entries = self._prune_source_duplicates(entries)
            entries.sort(key=self._display_rank)
            self.entries[task_id] = entries

    def _prune_source_duplicates(
        self, entries: Sequence[FrontierEntry]
    ) -> list[FrontierEntry]:
        retained: list[FrontierEntry] = []
        limit = self.config.maximum_entries_per_source_signature
        groups = sorted({(item.signature, item.source_episode) for item in entries})
        for signature, source_episode in groups:
            matching = [
                item
                for item in entries
                if item.signature == signature
                and item.source_episode == source_episode
            ]
            matching.sort(key=self._display_rank)
            earliest = min(matching, key=lambda item: item.source_step)
            selected = [earliest]
            selected.extend(
                item for item in matching if item is not earliest
            )
            retained.extend(selected[:limit])
        return retained

    def _eligible_entries(
        self, entries: Sequence[FrontierEntry]
    ) -> list[FrontierEntry]:
        eligible: list[FrontierEntry] = []
        for signature in sorted({item.signature for item in entries}):
            matching = [item for item in entries if item.signature == signature]
            complete = [item for item in matching if item.snapshot.runtime_state]
            eligible.extend(complete or matching)
        return eligible

    def _retention_rank(self, item: FrontierEntry) -> tuple[bool, float]:
        return bool(item.snapshot.runtime_state), item.score

    def _outperforms(
        self, candidate: FrontierEntry, incumbent: FrontierEntry
    ) -> bool:
        candidate_complete = bool(candidate.snapshot.runtime_state)
        incumbent_complete = bool(incumbent.snapshot.runtime_state)
        if candidate_complete != incumbent_complete:
            return candidate_complete
        return candidate.score > incumbent.score + 1.0e-6

    def _display_rank(self, item: FrontierEntry) -> tuple[int, float, int, int]:
        return (
            -int(bool(item.snapshot.runtime_state)),
            -item.score,
            item.source_episode,
            item.source_step,
        )

    def _score(self, outcome: FrontierOutcome) -> float:
        worst = max(outcome.left_reach_distance, outcome.right_reach_distance)
        reach = math.exp(-worst / self.config.score_distance_scale_meters)
        target = 0.0
        articulation = 0.0
        if outcome.task_progress_observed:
            target = math.exp(
                -outcome.target_distance / self.config.score_target_scale_meters
            )
            articulation = 1.0 - math.exp(
                -max(0.0, outcome.articulation_position)
                / self.config.score_articulation_scale_meters
            )
        return float(reach * (1.0 + target + articulation))

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


def _restore_frontier_outcome(value: Mapping[str, object]) -> FrontierOutcome:
    fields = dict(value)
    fields.setdefault(
        "task_progress_observed",
        "target_distance" in fields
        and "articulation_position" in fields,
    )
    return FrontierOutcome(**fields)
