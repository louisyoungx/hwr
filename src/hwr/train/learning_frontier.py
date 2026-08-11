"""Task-agnostic reset curriculum driven by generic RL learning signals."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.core.embodied import DualArmObservation


LEARNING_FRONTIER_SCHEMA = "hwr.task-agnostic-learning-frontier/v1"
LEARNING_SIGNAL_NAMES = (
    "state_novelty",
    "td_error",
    "reward_improvement",
    "failure_boundary",
)


@dataclass(frozen=True)
class LearningFrontierConfig:
    capacity_per_task: int = 16
    reset_probability: float = 0.50
    candidates_per_episode: int = 4
    signature_uniform_fraction: float = 0.20
    maximum_entries_per_source_signature: int = 2
    history_size: int = 256

    def __post_init__(self) -> None:
        if min(
            self.capacity_per_task,
            self.candidates_per_episode,
            self.maximum_entries_per_source_signature,
            self.history_size,
        ) <= 0:
            raise ValueError("learning frontier capacities must be positive")
        if not 0.0 <= self.reset_probability <= 1.0:
            raise ValueError("learning frontier reset probability must be in [0, 1]")
        if not 0.0 <= self.signature_uniform_fraction <= 1.0:
            raise ValueError("learning frontier mixture must be in [0, 1]")


@dataclass(frozen=True)
class LearningSignal:
    state_novelty: float
    td_error: float
    reward_improvement: float
    failure_boundary: float
    safe: bool = True

    def __post_init__(self) -> None:
        values = (
            self.state_novelty,
            self.td_error,
            self.reward_improvement,
            self.failure_boundary,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("learning frontier signals must be finite")
        if min(self.state_novelty, self.td_error, self.failure_boundary) < 0.0:
            raise ValueError("learning frontier magnitudes cannot be negative")


@dataclass(frozen=True)
class LearningFrontierCandidate:
    snapshot: PhysicalStateSnapshot
    state_embedding: tuple[float, ...]
    signal: LearningSignal
    source_episode: int
    source_step: int

    def __post_init__(self) -> None:
        if not self.state_embedding or not all(
            math.isfinite(value) for value in self.state_embedding
        ):
            raise ValueError("learning frontier state embedding must be finite")
        if min(self.source_episode, self.source_step) < 0:
            raise ValueError("learning frontier source coordinates cannot be negative")


@dataclass(frozen=True)
class LearningFrontierEntry:
    snapshot: PhysicalStateSnapshot
    state_embedding: tuple[float, ...]
    signal: LearningSignal
    score: float
    signature: int
    source_episode: int
    source_step: int


class LearningFrontierBackend(Protocol):
    def reset(
        self,
        *,
        seed: int,
        task_id: str,
        initial_state: PhysicalStateSnapshot | None = None,
    ) -> DualArmObservation: ...

    def capture_state_snapshot(self) -> PhysicalStateSnapshot: ...


@dataclass(frozen=True)
class PreparedLearningFrontierReset:
    observation: DualArmObservation
    reset_seed: int
    applied: bool
    validated: bool
    reproduced: bool


class TaskAgnosticLearningFrontier:
    """Keep difficult autonomous states without reading task geometry or names."""

    def __init__(
        self,
        task_ids: Sequence[str],
        config: LearningFrontierConfig | None = None,
    ) -> None:
        identities = tuple(sorted(set(task_ids)))
        if not identities:
            raise ValueError("learning frontier requires task identities")
        self.task_ids = identities
        self.config = config or LearningFrontierConfig()
        self.entries: dict[str, list[LearningFrontierEntry]] = {
            task_id: [] for task_id in identities
        }
        self.signal_history = {
            task_id: {
                name: deque(maxlen=self.config.history_size)
                for name in LEARNING_SIGNAL_NAMES
            }
            for task_id in identities
        }
        self.reset_count = 0
        self.legacy_discarded_entry_count = 0

    def state_novelty(self, task_id: str, state: Sequence[float]) -> float:
        values = np.asarray(state, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("frontier novelty state must be a finite vector")
        entries = self.entries[task_id]
        if not entries:
            return 1.0
        reference = np.asarray(
            [entry.state_embedding for entry in entries], dtype=np.float64
        )
        value_norm = np.linalg.norm(values)
        reference_norm = np.linalg.norm(reference, axis=1)
        denominator = np.maximum(value_norm * reference_norm, 1.0e-12)
        similarity = reference @ values / denominator
        return float(np.clip(1.0 - similarity.max(), 0.0, 2.0))

    def consider_episode(
        self,
        task_id: str,
        candidates: Sequence[LearningFrontierCandidate],
    ) -> int:
        if task_id not in self.entries:
            raise ValueError(f"learning frontier does not know {task_id}")
        safe = [
            item
            for item in candidates
            if item.signal.safe and item.snapshot.task_id == task_id
        ]
        ranked = sorted(
            (
                self._entry(task_id, candidate)
                for candidate in safe
            ),
            key=lambda item: (item.score, item.source_step),
            reverse=True,
        )[: self.config.candidates_per_episode]
        for candidate in candidates:
            self._record_signal(task_id, candidate.signal)
        for entry in ranked:
            self._insert(task_id, entry)
        return len(ranked)

    def select(
        self, task_id: str, rng: np.random.Generator
    ) -> LearningFrontierEntry | None:
        candidates = self.entries[task_id]
        if not candidates or rng.random() >= self.config.reset_probability:
            return None
        signatures = sorted({entry.signature for entry in candidates})
        best = np.asarray(
            [max(item.score for item in candidates if item.signature == signature)
             for signature in signatures],
            dtype=np.float64,
        )
        weighted = best / best.sum() if best.sum() > 0.0 else np.ones_like(best) / len(best)
        uniform = np.ones_like(weighted) / len(weighted)
        mixture = self.config.signature_uniform_fraction
        probabilities = mixture * uniform + (1.0 - mixture) * weighted
        signature = signatures[int(rng.choice(len(signatures), p=probabilities))]
        pool = [item for item in candidates if item.signature == signature]
        scores = np.asarray([item.score for item in pool], dtype=np.float64)
        probabilities = scores / scores.sum() if scores.sum() > 0.0 else None
        self.reset_count += 1
        return pool[int(rng.choice(len(pool), p=probabilities))]

    def find(
        self, task_id: str, source_episode: int, source_step: int
    ) -> LearningFrontierEntry | None:
        return next(
            (
                item
                for item in self.entries.get(task_id, ())
                if item.source_episode == source_episode
                and item.source_step == source_step
            ),
            None,
        )

    def discard_tasks(self, task_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        discarded = {}
        for task_id in task_ids:
            if task_id not in self.entries:
                raise ValueError(f"learning frontier does not know {task_id}")
            discarded[task_id] = {"entry_count": len(self.entries[task_id])}
            self.entries[task_id].clear()
            for values in self.signal_history[task_id].values():
                values.clear()
        return discarded

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": LEARNING_FRONTIER_SCHEMA,
            "config": asdict(self.config),
            "metrics": list(LEARNING_SIGNAL_NAMES),
            "task_semantic_fields": [],
            "distance_thresholds": False,
            "action_outputs": False,
            "task_stages": False,
            "candidate_counts": {
                task_id: len(values) for task_id, values in self.entries.items()
            },
            "reset_count": self.reset_count,
            "legacy_discarded_entry_count": self.legacy_discarded_entry_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": LEARNING_FRONTIER_SCHEMA,
            "task_ids": self.task_ids,
            "config": asdict(self.config),
            "entries": {
                task_id: [asdict(item) for item in values]
                for task_id, values in self.entries.items()
            },
            "signal_history": {
                task_id: {name: list(values) for name, values in history.items()}
                for task_id, history in self.signal_history.items()
            },
            "reset_count": self.reset_count,
            "legacy_discarded_entry_count": self.legacy_discarded_entry_count,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if tuple(value["task_ids"]) != self.task_ids:
            raise ValueError("learning frontier checkpoint tasks differ")
        if value.get("schema_version") != LEARNING_FRONTIER_SCHEMA:
            self.legacy_discarded_entry_count += sum(
                len(items) for items in value.get("entries", {}).values()
            )
            return
        saved_config = dict(value["config"])
        current_config = asdict(self.config)
        for name in ("reset_probability", "signature_uniform_fraction"):
            saved_config[name] = current_config[name]
        if saved_config != current_config:
            raise ValueError("learning frontier checkpoint configuration differs")
        for task_id in self.task_ids:
            self.entries[task_id] = [
                _restore_entry(item) for item in value["entries"][task_id]
            ]
            for name in LEARNING_SIGNAL_NAMES:
                self.signal_history[task_id][name].extend(
                    float(item)
                    for item in value["signal_history"][task_id][name]
                )
        self.reset_count = int(value["reset_count"])
        self.legacy_discarded_entry_count = int(
            value.get("legacy_discarded_entry_count", 0)
        )

    def _entry(
        self, task_id: str, candidate: LearningFrontierCandidate
    ) -> LearningFrontierEntry:
        values = (
            candidate.signal.state_novelty,
            candidate.signal.td_error,
            candidate.signal.reward_improvement,
            candidate.signal.failure_boundary,
        )
        ranks = tuple(
            self._percentile(task_id, name, value)
            for name, value in zip(LEARNING_SIGNAL_NAMES, values, strict=True)
        )
        signature = int(np.argmax(np.asarray(ranks)))
        return LearningFrontierEntry(
            candidate.snapshot,
            candidate.state_embedding,
            candidate.signal,
            float(sum(ranks) / len(ranks)),
            signature,
            candidate.source_episode,
            candidate.source_step,
        )

    def _percentile(self, task_id: str, name: str, value: float) -> float:
        history = self.signal_history[task_id][name]
        if not history:
            return 0.5
        values = np.asarray(history, dtype=np.float64)
        return float((np.count_nonzero(values <= value) + 0.5) / (len(values) + 1.0))

    def _record_signal(self, task_id: str, signal: LearningSignal) -> None:
        for name in LEARNING_SIGNAL_NAMES:
            self.signal_history[task_id][name].append(float(getattr(signal, name)))

    def _insert(self, task_id: str, candidate: LearningFrontierEntry) -> None:
        entries = self.entries[task_id]
        same_source = [
            item
            for item in entries
            if item.source_episode == candidate.source_episode
            and item.signature == candidate.signature
        ]
        if len(same_source) >= self.config.maximum_entries_per_source_signature:
            weakest = min(same_source, key=lambda item: item.score)
            if weakest.score >= candidate.score:
                return
            entries.remove(weakest)
        entries.append(candidate)
        entries.sort(key=lambda item: item.score, reverse=True)
        del entries[self.config.capacity_per_task :]


def _restore_entry(value: Mapping[str, object]) -> LearningFrontierEntry:
    signal = LearningSignal(**value["signal"])
    snapshot = PhysicalStateSnapshot(**value["snapshot"])
    return LearningFrontierEntry(
        snapshot,
        tuple(float(item) for item in value["state_embedding"]),
        signal,
        float(value["score"]),
        int(value["signature"]),
        int(value["source_episode"]),
        int(value["source_step"]),
    )


def prepare_learning_frontier_reset(
    environment: LearningFrontierBackend,
    entry: LearningFrontierEntry | None,
    *,
    task_id: str,
    episode_seed: int,
    source_seed: int,
) -> PreparedLearningFrontierReset:
    if entry is None:
        observation = environment.reset(seed=episode_seed, task_id=task_id)
        return PreparedLearningFrontierReset(
            observation, episode_seed, False, False, False
        )
    observation = environment.reset(
        seed=source_seed,
        task_id=task_id,
        initial_state=entry.snapshot,
    )
    restored = environment.capture_state_snapshot()
    reproduced = _snapshot_close(entry.snapshot, restored)
    if not reproduced:
        observation = environment.reset(seed=episode_seed, task_id=task_id)
    return PreparedLearningFrontierReset(
        observation,
        source_seed if reproduced else episode_seed,
        reproduced,
        True,
        reproduced,
    )


def _snapshot_close(
    expected: PhysicalStateSnapshot, actual: PhysicalStateSnapshot
) -> bool:
    if (
        expected.task_id != actual.task_id
        or expected.backend_fingerprint != actual.backend_fingerprint
    ):
        return False
    pairs = (
        (expected.generalized_positions, actual.generalized_positions),
        (expected.generalized_velocities, actual.generalized_velocities),
    )
    return all(
        len(left) == len(right)
        and np.allclose(left, right, rtol=1.0e-7, atol=1.0e-8)
        for left, right in pairs
    )
