"""Validated configuration for the unified foundation online RL runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hwr.train.foundation_exploration import RandomRLExplorationConfig
from hwr.train.learning_frontier import LearningFrontierConfig
from hwr.world_model.evaluation import ActionCausalityCriteria


@dataclass(frozen=True)
class FoundationOnlineTrainingConfig:
    episodes: int = 120
    minimum_actor_readiness_episodes: int = 12
    actor_readiness_consecutive_passes: int = 2
    minimum_active_action_dimension_fraction: float = 0.75
    minimum_action_effective_rank: float = 6.0
    minimum_data_action_probe_ratio: float = 1.05
    minimum_data_action_probe_ratio_p05: float = 1.01
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
    causality_shuffle_repeats: int = 5
    random_exploration_motion_correlation: float = 0.96
    random_exploration_gripper_flip_probability: float = 0.05
    learning_signal_windows_per_episode: int = 4
    learning_frontier_capacity_per_task: int = 16
    learning_frontier_reset_probability: float = 0.20
    learning_frontier_candidates_per_episode: int = 4
    learning_frontier_signature_uniform_fraction: float = 0.20
    learning_frontier_maximum_entries_per_source_signature: int = 2
    metrics_publish_interval_updates: int = 10
    seed: int = 20260812

    def __post_init__(self) -> None:
        positive = (
            self.episodes,
            self.minimum_actor_readiness_episodes,
            self.actor_readiness_consecutive_passes,
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
            self.causality_shuffle_repeats,
            self.learning_signal_windows_per_episode,
            self.learning_frontier_capacity_per_task,
            self.learning_frontier_candidates_per_episode,
            self.learning_frontier_maximum_entries_per_source_signature,
            self.metrics_publish_interval_updates,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("foundation online training dimensions are invalid")
        if self.minimum_actor_readiness_episodes > self.episodes:
            raise ValueError("Actor readiness Episodes exceed total Episodes")
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
        LearningFrontierConfig(
            capacity_per_task=self.learning_frontier_capacity_per_task,
            reset_probability=self.learning_frontier_reset_probability,
            candidates_per_episode=self.learning_frontier_candidates_per_episode,
            signature_uniform_fraction=(
                self.learning_frontier_signature_uniform_fraction
            ),
            maximum_entries_per_source_signature=(
                self.learning_frontier_maximum_entries_per_source_signature
            ),
        )
        if not 0.0 < self.minimum_active_action_dimension_fraction <= 1.0:
            raise ValueError("minimum active action dimension fraction is invalid")
        if self.minimum_action_effective_rank <= 0.0:
            raise ValueError("minimum action effective rank is invalid")
        if min(
            self.minimum_data_action_probe_ratio,
            self.minimum_data_action_probe_ratio_p05,
        ) <= 1.0:
            raise ValueError("data action probe ratios must exceed one")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
