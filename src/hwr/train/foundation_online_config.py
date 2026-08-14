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
    minimum_interaction_displacement: float = 0.01
    minimum_contact_episodes_per_task: int = 1
    minimum_controlled_motion_episodes_per_task: int = 1
    minimum_collision_positive_episodes_per_task: int = 1
    minimum_collision_negative_episodes_per_task: int = 1
    minimum_collision_validation_positive_episodes_per_task: int = 8
    minimum_collision_validation_negative_episodes_per_task: int = 8
    minimum_collision_validation_recall: float = 0.80
    minimum_collision_validation_pr_auc: float = 0.50
    maximum_collision_validation_brier_score: float = 0.10
    calibration_early_stop_episodes: int = 24
    collection_episodes_per_cycle: int = 3
    updates_per_cycle: int = 200
    actor_warmup_minimum_updates: int = 200
    actor_warmup_maximum_updates: int = 1000
    actor_warmup_window_updates: int = 50
    actor_warmup_stable_windows: int = 3
    actor_warmup_maximum_gradient_norm: float = 100.0
    actor_warmup_maximum_return_relative_range: float = 0.25
    actor_warmup_minimum_motion_entropy: float = 0.0
    actor_warmup_minimum_gripper_entropy: float = -0.5
    severe_collision_batch_fraction: float = 0.25
    batch_size: int = 2
    sequence_transitions: int = 16
    camera_width: int = 256
    camera_height: int = 192
    augmentation_probability: float = 0.5
    checkpoint_interval_cycles: int = 1
    replay_transition_capacity: int = 18000
    replay_windows_per_episode: int = 1
    published_checkpoint_retention: int = 3
    minimum_action_causality_ratio: float = 1.05
    minimum_action_causality_horizon_fraction: float = 0.60
    causality_holdout_episodes_per_task: int = 2
    causality_audit_windows_per_task: int = 8
    causality_audit_batch_size: int = 2
    causality_shuffle_repeats: int = 5
    causality_holdout_maximum_attempts_per_episode: int = 8
    causality_holdout_transitions_per_episode: int = 64
    maximum_estimated_run_storage_gib: float = 30.0
    minimum_free_storage_gib: float = 35.0
    estimated_teacher_cache_bytes_per_observation: int = 2_800_000
    estimated_checkpoint_bytes: int = 2_000_000_000
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
            self.minimum_contact_episodes_per_task,
            self.minimum_controlled_motion_episodes_per_task,
            self.calibration_early_stop_episodes,
            self.collection_episodes_per_cycle,
            self.updates_per_cycle,
            self.actor_warmup_minimum_updates,
            self.actor_warmup_maximum_updates,
            self.actor_warmup_window_updates,
            self.actor_warmup_stable_windows,
            self.batch_size,
            self.sequence_transitions,
            self.camera_width,
            self.camera_height,
            self.checkpoint_interval_cycles,
            self.replay_transition_capacity,
            self.replay_windows_per_episode,
            self.published_checkpoint_retention,
            self.causality_holdout_episodes_per_task,
            self.causality_audit_windows_per_task,
            self.causality_audit_batch_size,
            self.causality_shuffle_repeats,
            self.causality_holdout_maximum_attempts_per_episode,
            self.causality_holdout_transitions_per_episode,
            self.estimated_teacher_cache_bytes_per_observation,
            self.estimated_checkpoint_bytes,
            self.learning_signal_windows_per_episode,
            self.learning_frontier_capacity_per_task,
            self.learning_frontier_candidates_per_episode,
            self.learning_frontier_maximum_entries_per_source_signature,
            self.metrics_publish_interval_updates,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("foundation online training dimensions are invalid")
        if min(
            self.minimum_collision_positive_episodes_per_task,
            self.minimum_collision_negative_episodes_per_task,
        ) < 0:
            raise ValueError("collision coverage counts cannot be negative")
        if min(
            self.minimum_collision_validation_positive_episodes_per_task,
            self.minimum_collision_validation_negative_episodes_per_task,
        ) < 0 or (
            self.minimum_collision_validation_positive_episodes_per_task
            + self.minimum_collision_validation_negative_episodes_per_task
            <= 0
        ):
            raise ValueError("collision validation counts are invalid")
        if self.minimum_actor_readiness_episodes > self.episodes:
            raise ValueError("Actor readiness Episodes exceed total Episodes")
        if self.calibration_early_stop_episodes > self.episodes:
            raise ValueError("calibration early stop exceeds total Episodes")
        if not (
            self.actor_warmup_window_updates
            <= self.actor_warmup_minimum_updates
            <= self.actor_warmup_maximum_updates
        ):
            raise ValueError("Actor warmup update bounds are invalid")
        if (
            self.actor_warmup_minimum_updates % self.actor_warmup_window_updates
            or self.actor_warmup_maximum_updates % self.actor_warmup_window_updates
            or self.actor_warmup_stable_windows
            > self.actor_warmup_minimum_updates // self.actor_warmup_window_updates
        ):
            raise ValueError("Actor warmup windows are invalid")
        if self.actor_warmup_maximum_gradient_norm <= 0.0:
            raise ValueError("Actor warmup gradient limit must be positive")
        if not 0.0 <= self.actor_warmup_maximum_return_relative_range <= 1.0:
            raise ValueError("Actor warmup return stability limit is invalid")
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError("foundation augmentation probability is invalid")
        if not 0.0 <= self.severe_collision_batch_fraction <= 1.0:
            raise ValueError("severe collision batch fraction is invalid")
        if min(self.camera_width, self.camera_height) < 160:
            raise ValueError("foundation online training requires high-resolution cameras")
        if self.replay_transition_capacity < self.sequence_transitions * 3:
            raise ValueError("foundation replay capacity cannot retain one window per task")
        retained_per_source = self.sequence_transitions * self.replay_windows_per_episode
        required_sources = max(
            self.minimum_contact_episodes_per_task,
            self.minimum_controlled_motion_episodes_per_task,
            self.minimum_collision_positive_episodes_per_task
            + self.minimum_collision_negative_episodes_per_task,
        )
        if self.replay_transition_capacity // 3 < retained_per_source * required_sources:
            raise ValueError("foundation replay capacity makes Actor evidence unreachable")
        if self.causality_audit_windows_per_task % self.causality_audit_batch_size:
            raise ValueError("causality audit batch size must divide task window count")
        if (
            self.causality_audit_windows_per_task
            % self.causality_holdout_episodes_per_task
        ):
            raise ValueError("causality windows must balance across holdout Episodes")
        windows_per_holdout = (
            self.causality_audit_windows_per_task
            // self.causality_holdout_episodes_per_task
        )
        if self.causality_holdout_transitions_per_episode < (
            windows_per_holdout * self.sequence_transitions
        ):
            raise ValueError("compact causality holdout cannot supply audit windows")
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
        if self.minimum_interaction_displacement <= 0.0:
            raise ValueError("minimum interaction displacement must be positive")
        collision_limits = (
            self.minimum_collision_validation_recall,
            self.minimum_collision_validation_pr_auc,
            self.maximum_collision_validation_brier_score,
        )
        if any(not 0.0 <= value <= 1.0 for value in collision_limits):
            raise ValueError("collision validation limits are invalid")
        if min(
            self.maximum_estimated_run_storage_gib,
            self.minimum_free_storage_gib,
        ) <= 0.0:
            raise ValueError("foundation storage budgets must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
