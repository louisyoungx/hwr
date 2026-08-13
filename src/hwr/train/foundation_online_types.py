"""Public contracts and evidence records for foundation online training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from hwr.core.runtime import RuntimeBackend
from hwr.perception.foundation import (
    FrozenLanguageFeatureProvider,
    FrozenVisionFeatureProvider,
)


@dataclass(frozen=True)
class FoundationTaskInterface:
    task_id: str
    maximum_steps: int

    def __post_init__(self) -> None:
        if not self.task_id or self.maximum_steps <= 0:
            raise ValueError("foundation task interface is invalid")


class FoundationEnvironmentFactory(Protocol):
    def __call__(
        self, task_id: str, camera_width: int, camera_height: int
    ) -> RuntimeBackend: ...


@dataclass(frozen=True)
class FoundationProviderFactories:
    vision_language: Callable[[], FrozenVisionFeatureProvider]
    dense_vision: Callable[[], FrozenVisionFeatureProvider]
    language: Callable[[], FrozenLanguageFeatureProvider]


@dataclass(frozen=True)
class FoundationEpisodeRecord:
    episode_index: int
    task_id: str
    seed: int
    action_source: str
    episode_return: float
    success: bool
    safety_intervention_rate: float
    environment_steps: int
    update_count: int
    state_novelty: float = 0.0
    td_error: float = 0.0
    reward_improvement: float = 0.0
    failure_boundary: float = 0.0


@dataclass(frozen=True)
class FoundationOnlineTrainingResult:
    records: tuple[FoundationEpisodeRecord, ...]
    update_count: int
    replay_path: Path
    latest_checkpoint: Path
    latest_deployment: Path
    latest_action_causality_report: Path
