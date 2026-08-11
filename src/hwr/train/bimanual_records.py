"""Serializable per-Episode records for bimanual reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingEpisodeRecord:
    episode: int
    task_id: str
    seed: int
    steps: int
    reward: float
    success: bool
    severe_collisions: int
    maximum_concurrent_steps: int
    left_contact_steps: int
    right_contact_steps: int
    simultaneous_contact_steps: int
    stable_steps: int
    minimum_left_reach_distance: float
    minimum_right_reach_distance: float
    minimum_worst_side_reach_distance: float
    curriculum_level: float
    replay_size: int
    updates: int
    bilateral_near_steps: int = 0
    maximum_bilateral_near_steps: int = 0
    safety_interventions: int = 0
    sampling_probability: float = 1.0 / 3.0
    actor_updates: int = 0
    mean_critic_loss: float = 0.0
    mean_safety_loss: float = 0.0
    mean_actor_loss: float = 0.0
    mean_actor_reward_value: float = 0.0
    mean_actor_safety_risk: float = 0.0
    mean_reward_critic_disagreement: float = 0.0
    mean_safety_critic_disagreement: float = 0.0
    mean_actor_motion_ratio: float = 0.0
    maximum_actor_motion_ratio: float = 0.0
    mean_actor_entropy: float = 0.0
    mean_actor_motion_log_standard_deviation: float = 0.0
    mean_actor_gripper_log_standard_deviation: float = 0.0
    actor_learning_rate: float = 0.0
    frontier_reset: bool = False
    frontier_source_episode: int = -1
    frontier_source_step: int = -1
    environment_reset_seed: int = -1
    frontier_source_signature: int = -1
    frontier_reset_contact_steps: int = 0
    frontier_reset_validated: bool = False
    frontier_reset_reproduced: bool = False
    frontier_reset_applied: bool = False
    minimum_target_distance: float = 0.0
    maximum_articulation_position: float = 0.0
    maximum_controlled_target_progress: float = 0.0
    maximum_controlled_articulation_progress: float = 0.0
