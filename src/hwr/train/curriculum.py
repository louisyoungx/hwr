"""Automatic domain-randomization curriculum driven only by episode outcomes."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CurriculumConfig:
    window: int = 20
    promote_success_rate: float = 0.70
    demote_success_rate: float = 0.25
    step: float = 0.10
    initial_level: float = 0.10

    def __post_init__(self) -> None:
        if self.window <= 0 or not 0.0 < self.step <= 1.0:
            raise ValueError("curriculum window and step must be positive")
        values = (
            self.promote_success_rate,
            self.demote_success_rate,
            self.initial_level,
        )
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("curriculum rates and level must be in [0, 1]")
        if self.demote_success_rate >= self.promote_success_rate:
            raise ValueError("curriculum promotion must exceed demotion threshold")


@dataclass(frozen=True)
class CurriculumUpdate:
    task_id: str
    previous_level: float
    level: float
    success_rate: float
    severe_collision_rate: float
    changed: bool


class AutomaticCurriculum:
    """Expand randomization after competence and contract it after regressions."""

    def __init__(self, task_ids: Sequence[str], config: CurriculumConfig) -> None:
        unique = tuple(dict.fromkeys(task_ids))
        if not unique:
            raise ValueError("curriculum requires task identities")
        self.config = config
        self._levels = {task_id: config.initial_level for task_id in unique}
        self._success = {
            task_id: deque(maxlen=config.window) for task_id in unique
        }
        self._severe = {
            task_id: deque(maxlen=config.window) for task_id in unique
        }

    def level(self, task_id: str) -> float:
        try:
            return self._levels[task_id]
        except KeyError as error:
            raise KeyError(f"unknown curriculum task: {task_id}") from error

    def record(
        self, task_id: str, *, success: bool, severe_collision: bool
    ) -> CurriculumUpdate:
        previous = self.level(task_id)
        self._success[task_id].append(bool(success))
        self._severe[task_id].append(bool(severe_collision))
        success_rate = sum(self._success[task_id]) / len(self._success[task_id])
        severe_rate = sum(self._severe[task_id]) / len(self._severe[task_id])
        level = previous
        full_window = len(self._success[task_id]) == self.config.window
        if full_window and success_rate >= self.config.promote_success_rate and severe_rate == 0:
            level = min(1.0, previous + self.config.step)
            self._clear_window(task_id)
        elif full_window and (
            success_rate <= self.config.demote_success_rate or severe_rate > 0
        ):
            level = max(0.0, previous - self.config.step)
            self._clear_window(task_id)
        self._levels[task_id] = level
        return CurriculumUpdate(
            task_id,
            previous,
            level,
            success_rate,
            severe_rate,
            level != previous,
        )

    def _clear_window(self, task_id: str) -> None:
        self._success[task_id].clear()
        self._severe[task_id].clear()

    def state_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "levels": dict(self._levels),
            "success": {name: list(values) for name, values in self._success.items()},
            "severe": {name: list(values) for name, values in self._severe.items()},
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if value["config"] != asdict(self.config):
            raise ValueError("curriculum checkpoint configuration differs")
        if set(value["levels"]) != set(self._levels):
            raise ValueError("curriculum checkpoint task set differs")
        self._levels = {name: float(level) for name, level in value["levels"].items()}
        for name in self._levels:
            self._success[name].extend(bool(item) for item in value["success"][name])
            self._severe[name].extend(bool(item) for item in value["severe"][name])
