"""Runner-facing orchestration for independently timed holdout phases."""

from __future__ import annotations

from typing import Mapping

from hwr.core.runtime import RuntimeBackend
from hwr.data.autonomous_trajectory import AppendableAutonomousTrajectoryStore
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.policy.latent_actions import LatentActionScaling
from hwr.train.foundation_exploration import RandomRLExplorationConfig
from hwr.train.foundation_holdout import (
    COLLISION_VALIDATION_PHASE,
    SYSTEM_IDENTIFICATION_PHASE,
    collect_causality_holdout,
)
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_online_types import FoundationTaskInterface


def prepare_foundation_system_holdout(
    store: AppendableAutonomousTrajectoryStore,
    environments: Mapping[str, RuntimeBackend],
    tasks: Mapping[str, FoundationTaskInterface],
    preprocessor: HighResolutionVisionPreprocessor,
    action_scaling: LatentActionScaling,
    exploration: RandomRLExplorationConfig,
    config: FoundationOnlineTrainingConfig,
    *,
    source_commit: str,
) -> None:
    _prepare(
        store,
        environments,
        tasks,
        preprocessor,
        action_scaling,
        exploration,
        config,
        source_commit=source_commit,
        phase=SYSTEM_IDENTIFICATION_PHASE,
        episodes_per_task=config.causality_holdout_episodes_per_task,
        windows_per_episode=(
            config.causality_audit_windows_per_task
            // config.causality_holdout_episodes_per_task
        ),
        retained_transitions=config.causality_holdout_transitions_per_episode,
        collision_balanced=False,
        collision_positive_episodes=None,
    )


def prepare_foundation_collision_holdout(
    store: AppendableAutonomousTrajectoryStore,
    environments: Mapping[str, RuntimeBackend],
    tasks: Mapping[str, FoundationTaskInterface],
    preprocessor: HighResolutionVisionPreprocessor,
    action_scaling: LatentActionScaling,
    exploration: RandomRLExplorationConfig,
    config: FoundationOnlineTrainingConfig,
    *,
    source_commit: str,
) -> None:
    _prepare(
        store,
        environments,
        tasks,
        preprocessor,
        action_scaling,
        exploration,
        config,
        source_commit=source_commit,
        phase=COLLISION_VALIDATION_PHASE,
        episodes_per_task=config.collision_validation_holdout_episodes_per_task,
        windows_per_episode=1,
        retained_transitions=(
            config.collision_validation_holdout_transitions_per_episode
        ),
        collision_balanced=True,
        collision_positive_episodes=(
            config.minimum_collision_validation_positive_episodes_per_task
        ),
    )


def _prepare(
    store: AppendableAutonomousTrajectoryStore,
    environments: Mapping[str, RuntimeBackend],
    tasks: Mapping[str, FoundationTaskInterface],
    preprocessor: HighResolutionVisionPreprocessor,
    action_scaling: LatentActionScaling,
    exploration: RandomRLExplorationConfig,
    config: FoundationOnlineTrainingConfig,
    *,
    source_commit: str,
    phase: str,
    episodes_per_task: int,
    windows_per_episode: int,
    retained_transitions: int,
    collision_balanced: bool,
    collision_positive_episodes: int | None,
) -> None:
    collect_causality_holdout(
        store,
        environments,
        {task_id: task.maximum_steps for task_id, task in tasks.items()},
        preprocessor,
        action_scaling,
        exploration_config=exploration,
        episodes_per_task=episodes_per_task,
        windows_per_episode=windows_per_episode,
        sequence_transitions=config.sequence_transitions,
        retained_transitions_per_episode=retained_transitions,
        maximum_attempts_per_episode=(
            config.causality_holdout_maximum_attempts_per_episode
        ),
        base_seed=config.seed,
        source_commit=source_commit,
        holdout_phase=phase,
        collision_balanced=collision_balanced,
        collision_positive_episodes=collision_positive_episodes,
    )
