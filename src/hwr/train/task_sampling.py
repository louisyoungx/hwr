"""Task allocation from scale-free, task-agnostic RL learning signals."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


TASK_SAMPLING_SCHEMA = "hwr.task-agnostic-learning-sampling/v5"


@dataclass(frozen=True)
class OutcomeAdaptiveTaskSamplingConfig:
    window: int = 12
    initial_cycles: int = 2
    minimum_probability: float = 0.12
    temperature: float = 0.75
    maximum_probability: float = 0.55

    def __post_init__(self) -> None:
        if min(self.window, self.initial_cycles) <= 0:
            raise ValueError("task sampling history dimensions must be positive")
        if not 0.0 <= self.minimum_probability < 1.0:
            raise ValueError("task sampling probability floor is invalid")
        if self.temperature <= 0.0:
            raise ValueError("task sampling temperature must be positive")
        if not self.minimum_probability < self.maximum_probability <= 1.0:
            raise ValueError("task sampling probability ceiling is invalid")


@dataclass(frozen=True)
class TaskOutcome:
    episode_return: float
    state_novelty: float
    td_error: float
    reward_improvement: float
    failure_boundary: float
    success: bool
    safety_intervention_rate: float

    def __post_init__(self) -> None:
        values = (
            self.episode_return,
            self.state_novelty,
            self.td_error,
            self.reward_improvement,
            self.failure_boundary,
            self.safety_intervention_rate,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("task sampling outcome values must be finite")
        if min(
            self.state_novelty,
            self.td_error,
            self.failure_boundary,
            self.safety_intervention_rate,
        ) < 0.0:
            raise ValueError("task sampling magnitudes cannot be negative")


class OutcomeAdaptiveTaskSampler:
    """Allocate data collection without object, task, or geometry features."""

    def __init__(
        self,
        task_ids: Sequence[str],
        config: OutcomeAdaptiveTaskSamplingConfig | None = None,
    ) -> None:
        identities = tuple(sorted(set(task_ids)))
        if not identities:
            raise ValueError("task sampler requires task identities")
        self.task_ids = identities
        self.config = config or OutcomeAdaptiveTaskSamplingConfig()
        if self.config.minimum_probability * len(identities) >= 1.0:
            raise ValueError("task sampling probability floors consume all mass")
        if len(identities) > 1 and self.config.maximum_probability * len(identities) < 1.0:
            raise ValueError("task sampling probability ceilings cannot sum to one")
        self.history = {
            task_id: deque(maxlen=self.config.window) for task_id in identities
        }
        self.credits = {task_id: 0.0 for task_id in identities}
        self.sample_count = 0
        self.legacy_discarded_outcome_count = 0

    def sample(self, rng: np.random.Generator) -> tuple[str, float]:
        del rng
        initial = self.config.initial_cycles * len(self.task_ids)
        if self.sample_count < initial:
            task_id = self.task_ids[self.sample_count % len(self.task_ids)]
            self.sample_count += 1
            return task_id, 1.0
        probabilities = self.probabilities()
        for task_id, probability in probabilities.items():
            self.credits[task_id] += probability
        task_id = max(self.task_ids, key=lambda name: self.credits[name])
        self.credits[task_id] -= 1.0
        self.sample_count += 1
        return task_id, probabilities[task_id]

    def reward_improvement(self, task_id: str, episode_return: float) -> float:
        history = self.history[task_id]
        if not history:
            return 0.0
        baseline = sum(item.episode_return for item in history) / len(history)
        return float(episode_return - baseline)

    def record(self, task_id: str, outcome: TaskOutcome) -> None:
        try:
            self.history[task_id].append(outcome)
        except KeyError as exc:
            raise ValueError(f"task sampler does not know {task_id}") from exc

    def discard_tasks(
        self, task_ids: Sequence[str]
    ) -> dict[str, dict[str, float | int]]:
        identities = tuple(dict.fromkeys(task_ids))
        unknown = sorted(set(identities) - set(self.task_ids))
        if unknown:
            raise ValueError(
                "task sampler cannot discard unknown tasks: "
                + ", ".join(unknown)
            )
        discarded = {}
        for task_id in identities:
            discarded[task_id] = {
                "history_count": len(self.history[task_id]),
                "credit": self.credits[task_id],
            }
            self.history[task_id].clear()
            self.credits[task_id] = 0.0
        return discarded

    def probabilities(self) -> dict[str, float]:
        features = np.asarray(
            [self._features(task_id) for task_id in self.task_ids],
            dtype=np.float64,
        )
        priorities = np.zeros(len(self.task_ids), dtype=np.float64)
        for column in range(features.shape[1]):
            priorities += _midranks(features[:, column])
        logits = priorities / self.config.temperature
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        floor = self.config.minimum_probability
        probabilities = floor + (1.0 - floor * len(self.task_ids)) * weights
        probabilities = _cap_probabilities(
            probabilities,
            max(self.config.maximum_probability, 1.0 / len(self.task_ids)),
        )
        return {
            task_id: float(probability)
            for task_id, probability in zip(
                self.task_ids, probabilities, strict=True
            )
        }

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": TASK_SAMPLING_SCHEMA,
            "config": asdict(self.config),
            "probabilities": self.probabilities(),
            "metrics": [
                "state_novelty",
                "td_error",
                "reward_improvement_speed",
                "failure_boundary",
            ],
            "task_semantic_fields": [],
            "distance_thresholds": False,
            "actor_input_fields": [],
            "action_outputs": False,
            "task_stages": False,
            "legacy_discarded_outcome_count": self.legacy_discarded_outcome_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": TASK_SAMPLING_SCHEMA,
            "task_ids": self.task_ids,
            "config": asdict(self.config),
            "sample_count": self.sample_count,
            "credits": dict(self.credits),
            "history": {
                task_id: [asdict(outcome) for outcome in outcomes]
                for task_id, outcomes in self.history.items()
            },
            "legacy_discarded_outcome_count": self.legacy_discarded_outcome_count,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if tuple(value["task_ids"]) != self.task_ids:
            raise ValueError("task sampler checkpoint tasks differ")
        self.sample_count = int(value["sample_count"])
        self.credits = {
            task_id: float(value["credits"][task_id])
            for task_id in self.task_ids
        }
        if value.get("schema_version") != TASK_SAMPLING_SCHEMA:
            self.legacy_discarded_outcome_count = int(
                value.get("legacy_discarded_outcome_count", 0)
            ) + sum(
                len(items) for items in value.get("history", {}).values()
            )
            self.sample_count = 0
            self.credits = {task_id: 0.0 for task_id in self.task_ids}
            return
        saved_config = dict(value["config"])
        current_config = asdict(self.config)
        for name in ("temperature", "maximum_probability"):
            saved_config[name] = current_config[name]
        if saved_config != current_config:
            raise ValueError("task sampler checkpoint configuration differs")
        for task_id in self.task_ids:
            self.history[task_id].clear()
            self.history[task_id].extend(
                TaskOutcome(**item) for item in value["history"][task_id]
            )
        self.legacy_discarded_outcome_count = int(
            value.get("legacy_discarded_outcome_count", 0)
        )

    def _features(self, task_id: str) -> tuple[float, ...]:
        history = self.history[task_id]
        if not history:
            return (1.0, 1.0, 1.0, 1.0)
        mean = lambda name: sum(getattr(item, name) for item in history) / len(history)
        return (
            mean("state_novelty"),
            mean("td_error"),
            -mean("reward_improvement"),
            mean("failure_boundary"),
        )


def _cap_probabilities(values: np.ndarray, maximum: float) -> np.ndarray:
    result = np.zeros_like(values)
    active = np.ones(len(values), dtype=bool)
    remaining = 1.0
    while active.any():
        weights = values[active]
        allocated = remaining * weights / weights.sum()
        high = allocated > maximum
        indices = np.flatnonzero(active)
        if not high.any():
            result[indices] = allocated
            break
        capped = indices[high]
        result[capped] = maximum
        active[capped] = False
        remaining -= maximum * len(capped)
    return result


def _midranks(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for target in np.unique(values):
        mask = values == target
        below = np.count_nonzero(values < target)
        result[mask] = (below + 0.5 * np.count_nonzero(mask)) / len(values)
    return result
