"""Configuration contract for local no-demonstration bimanual RL."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BimanualRLTrainingConfig:
    episodes: int = 120
    episode_step_limit: int | None = None
    replay_capacity: int = 80_000
    batch_size: int = 64
    learning_starts: int = 512
    updates_per_environment_step: float = 0.25
    initial_random_episodes: int = 9
    random_action_hold_steps: int = 8
    exploration_noise: float = 0.18
    exploration_correlation: float = 0.85
    action_smoothing: float = 0.65
    gripper_exploration_probability: float = 0.35
    gripper_exploration_hold_steps: int = 16
    policy_gripper_hold_steps: int = 12
    reflection_coupled_exploration_probability: float = 0.60
    paired_gripper_exploration_probability: float = 0.60
    global_random_burst_probability: float = 0.01
    global_random_burst_steps: int = 8
    actuator_dwell_probability: float = 0.0
    actuator_dwell_steps: int = 240
    actuator_initial_dwell_probability: float = 0.0
    actuator_dwell_closed_probability: float = 0.50
    frontier_reset_probability: float = 0.50
    frontier_capacity_per_task: int = 16
    frontier_signature_uniform_fraction: float = 0.20
    frontier_max_entries_per_source_signature: int = 2
    frontier_minimum_contact_stability_steps: int = 40
    frontier_reset_validation_steps: int = 40
    failure_replay_fraction: float = 0.25
    discovery_replay_fraction: float = 0.25
    progress_replay_fraction: float = 0.35
    safety_replay_fraction: float = 0.15
    visual_temporal_contrastive_weight: float = 0.05
    n_step_horizon: int = 8
    actor_learning_rate: float = 3.0e-5
    final_actor_learning_rate: float = 1.0e-5
    actor_learning_rate_decay_updates: int = 6500
    seed: int = 20260810
    device: str = "cpu"
    raw_image_width: int = 64
    raw_image_height: int = 48
    image_width: int = 32
    image_height: int = 24
    point_count: int = 32
    language_dim: int = 64
    hidden_dim: int = 64
    attention_heads: int = 4
    transformer_layers: int = 1

    def __post_init__(self) -> None:
        positive = (
            self.episodes,
            self.replay_capacity,
            self.batch_size,
            self.learning_starts,
            self.random_action_hold_steps,
            self.gripper_exploration_hold_steps,
            self.policy_gripper_hold_steps,
            self.global_random_burst_steps,
            self.actuator_dwell_steps,
            self.frontier_capacity_per_task,
            self.frontier_max_entries_per_source_signature,
            self.frontier_minimum_contact_stability_steps,
            self.frontier_reset_validation_steps,
            self.raw_image_width,
            self.raw_image_height,
            self.image_width,
            self.image_height,
            self.point_count,
            self.language_dim,
            self.hidden_dim,
            self.attention_heads,
            self.transformer_layers,
            self.n_step_horizon,
            self.actor_learning_rate,
            self.final_actor_learning_rate,
            self.actor_learning_rate_decay_updates,
        )
        if min(positive) <= 0 or self.initial_random_episodes < 0:
            raise ValueError("bimanual training dimensions must be positive")
        if (
            self.frontier_reset_validation_steps
            < self.frontier_minimum_contact_stability_steps
        ):
            raise ValueError("frontier reset validation cannot be shorter than stability")
        if self.episode_step_limit is not None and self.episode_step_limit <= 0:
            raise ValueError("bimanual episode step limit must be positive when set")
        fractions = (
            self.updates_per_environment_step,
            self.exploration_noise,
            self.exploration_correlation,
            self.action_smoothing,
            self.gripper_exploration_probability,
            self.reflection_coupled_exploration_probability,
            self.paired_gripper_exploration_probability,
            self.global_random_burst_probability,
            self.actuator_dwell_probability,
            self.actuator_initial_dwell_probability,
            self.actuator_dwell_closed_probability,
            self.frontier_reset_probability,
            self.frontier_signature_uniform_fraction,
            self.failure_replay_fraction,
            self.discovery_replay_fraction,
            self.progress_replay_fraction,
            self.safety_replay_fraction,
            self.visual_temporal_contrastive_weight,
        )
        if min(fractions) < 0 or any(value > 1 for value in fractions[1:]):
            raise ValueError("bimanual training fractions are invalid")
        replay_fraction = (
            self.failure_replay_fraction
            + self.discovery_replay_fraction
            + self.progress_replay_fraction
            + self.safety_replay_fraction
        )
        if replay_fraction > 1.0 + 1e-9:
            raise ValueError("bimanual replay fractions exceed one batch")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
