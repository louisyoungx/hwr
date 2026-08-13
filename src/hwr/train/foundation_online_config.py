"""Validated configuration for the unified foundation online RL runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hwr.train.foundation_exploration import RandomRLExplorationConfig
from hwr.world_model.evaluation import ActionCausalityCriteria


@dataclass(frozen=True)
class FoundationOnlineTrainingConfig:
    episodes: int = 120
    initial_random_episodes: int = 6
    collection_episodes_per_cycle: int = 3
    updates_per_cycle: int = 200
    batch_size: int = 2
    sequence_transitions: int = 16
    camera_width: int = 256
    camera_height: int = 192
    augmentation_probability: float = 0.5
    checkpoint_interval_cycles: int = 1
    replay_transition_capacity: int = 18000
    published_checkpoint_retention: int = 3
    minimum_action_causality_ratio: float = 1.05
    minimum_action_causality_horizon_fraction: float = 0.60
    causality_holdout_episodes_per_task: int = 2
    causality_audit_windows_per_task: int = 8
    causality_audit_batch_size: int = 2
    random_exploration_motion_correlation: float = 0.96
    random_exploration_gripper_flip_probability: float = 0.05
    learning_signal_windows_per_episode: int = 4
    metrics_publish_interval_updates: int = 10
    seed: int = 20260812

    def __post_init__(self) -> None:
        positive = (
            self.episodes,
            self.initial_random_episodes,
            self.collection_episodes_per_cycle,
            self.updates_per_cycle,
            self.batch_size,
            self.sequence_transitions,
            self.camera_width,
            self.camera_height,
            self.checkpoint_interval_cycles,
            self.replay_transition_capacity,
            self.published_checkpoint_retention,
            self.causality_holdout_episodes_per_task,
            self.causality_audit_windows_per_task,
            self.causality_audit_batch_size,
            self.learning_signal_windows_per_episode,
            self.metrics_publish_interval_updates,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("foundation online training dimensions are invalid")
        if self.initial_random_episodes > self.episodes:
            raise ValueError("initial random Episodes exceed total Episodes")
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError("foundation augmentation probability is invalid")
        if min(self.camera_width, self.camera_height) < 160:
            raise ValueError("foundation online training requires high-resolution cameras")
        if self.replay_transition_capacity < self.sequence_transitions * 3:
            raise ValueError("foundation replay capacity cannot retain one window per task")
        if self.causality_audit_windows_per_task % self.causality_audit_batch_size:
            raise ValueError("causality audit batch size must divide task window count")
        ActionCausalityCriteria(
            self.minimum_action_causality_ratio,
            self.minimum_action_causality_horizon_fraction,
        )
        RandomRLExplorationConfig(
            self.random_exploration_motion_correlation,
            self.random_exploration_gripper_flip_probability,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
