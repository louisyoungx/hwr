"""Outcome-adaptive task sampling without task stages or action prescriptions."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OutcomeAdaptiveTaskSamplingConfig:
    window: int = 12
    initial_cycles: int = 2
    minimum_probability: float = 0.12
    temperature: float = 0.25
    reach_scale_meters: float = 0.15

    def __post_init__(self) -> None:
        if min(self.window, self.initial_cycles) <= 0:
            raise ValueError("task sampling history dimensions must be positive")
        if not 0.0 <= self.minimum_probability < 1.0:
            raise ValueError("task sampling probability floor is invalid")
        if min(self.temperature, self.reach_scale_meters) <= 0:
            raise ValueError("task sampling scales must be positive")


@dataclass(frozen=True)
class TaskOutcome:
    left_contact_steps: int
    right_contact_steps: int
    simultaneous_contact_steps: int
    minimum_left_reach_distance: float
    minimum_right_reach_distance: float

    def __post_init__(self) -> None:
        steps = (
            self.left_contact_steps,
            self.right_contact_steps,
            self.simultaneous_contact_steps,
        )
        distances = (
            self.minimum_left_reach_distance,
            self.minimum_right_reach_distance,
        )
        if min(steps) < 0 or min(distances) < 0 or not all(
            math.isfinite(value) for value in distances
        ):
            raise ValueError("task sampling outcome values are invalid")


class OutcomeAdaptiveTaskSampler:
    """Spend collection on low-competence outcomes while retaining coverage."""

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
        self.history = {
            task_id: deque(maxlen=self.config.window) for task_id in identities
        }
        self.sample_count = 0

    def sample(self, rng: np.random.Generator) -> tuple[str, float]:
        initial = self.config.initial_cycles * len(self.task_ids)
        if self.sample_count < initial:
            task_id = self.task_ids[self.sample_count % len(self.task_ids)]
            self.sample_count += 1
            return task_id, 1.0
        probabilities = self.probabilities()
        values = np.asarray([probabilities[name] for name in self.task_ids])
        index = int(rng.choice(len(self.task_ids), p=values))
        self.sample_count += 1
        return self.task_ids[index], float(values[index])

    def record(self, task_id: str, outcome: TaskOutcome) -> None:
        try:
            self.history[task_id].append(outcome)
        except KeyError as exc:
            raise ValueError(f"task sampler does not know {task_id}") from exc

    def probabilities(self) -> dict[str, float]:
        competence = np.asarray(
            [self._competence(task_id) for task_id in self.task_ids],
            dtype=np.float64,
        )
        logits = (1.0 - competence) / self.config.temperature
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        floor = self.config.minimum_probability
        probabilities = floor + (1.0 - floor * len(self.task_ids)) * weights
        return {
            task_id: float(probability)
            for task_id, probability in zip(
                self.task_ids, probabilities, strict=True
            )
        }

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "hwr.outcome-adaptive-task-sampling/v1",
            "config": asdict(self.config),
            "probabilities": self.probabilities(),
            "actor_input_fields": [],
            "action_outputs": False,
            "task_stages": False,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "task_ids": self.task_ids,
            "config": asdict(self.config),
            "sample_count": self.sample_count,
            "history": {
                task_id: [asdict(outcome) for outcome in outcomes]
                for task_id, outcomes in self.history.items()
            },
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if tuple(value["task_ids"]) != self.task_ids:
            raise ValueError("task sampler checkpoint tasks differ")
        if dict(value["config"]) != asdict(self.config):
            raise ValueError("task sampler checkpoint configuration differs")
        self.sample_count = int(value["sample_count"])
        for task_id in self.task_ids:
            self.history[task_id].clear()
            self.history[task_id].extend(
                TaskOutcome(**item) for item in value["history"][task_id]
            )

    def _competence(self, task_id: str) -> float:
        outcomes = self.history[task_id]
        if not outcomes:
            return 0.0
        values = []
        for outcome in outcomes:
            worst_reach = max(
                outcome.minimum_left_reach_distance,
                outcome.minimum_right_reach_distance,
            )
            reach = math.exp(-worst_reach / self.config.reach_scale_meters)
            sides = 0.5 * (
                (outcome.left_contact_steps > 0)
                + (outcome.right_contact_steps > 0)
            )
            concurrent = min(1.0, outcome.simultaneous_contact_steps / 10.0)
            values.append(0.35 * reach + 0.30 * sides + 0.35 * concurrent)
        return sum(values) / len(values)
