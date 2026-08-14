"""Differentiable latent rollouts driven only by the current RL Actor."""

from __future__ import annotations

from typing import NamedTuple

import torch

from hwr.policy.latent_actor import LatentActor
from hwr.policy.latent_actions import LatentActionScaling, scale_latent_action
from hwr.world_model.distributions import reward_expectation
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMState


class ImaginedTrajectory(NamedTuple):
    features: torch.Tensor
    next_features: torch.Tensor
    actions: torch.Tensor
    executed_actions: torch.Tensor
    action_rewrite_magnitudes: torch.Tensor
    log_probabilities: torch.Tensor
    motion_entropies: torch.Tensor
    gripper_entropies: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    safety_probabilities: torch.Tensor
    severe_collision_probabilities: torch.Tensor
    uncertainties: torch.Tensor


def imagine_trajectory(
    world_model: ActionConditionedWorldModel,
    actor: LatentActor,
    initial_state: RSSMState,
    *,
    horizon: int,
    action_scaling: LatentActionScaling | None = None,
) -> ImaginedTrajectory:
    if horizon <= 0:
        raise ValueError("imagination horizon must be positive")
    state = initial_state
    features: list[torch.Tensor] = []
    next_features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    executed_actions: list[torch.Tensor] = []
    action_rewrites: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    motion_entropies: list[torch.Tensor] = []
    gripper_entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    continues: list[torch.Tensor] = []
    safety: list[torch.Tensor] = []
    severe_collisions: list[torch.Tensor] = []
    uncertainties: list[torch.Tensor] = []
    for _ in range(horizon):
        feature = world_model.rssm.features(state)
        sample = actor.sample(feature)
        action = (
            scale_latent_action(sample.action, action_scaling)
            if action_scaling is not None and sample.action.shape[-1] == 16
            else sample.action
        )
        safety_logits = world_model.predict_safety_intervention(feature, action)
        executed_action = world_model.predict_executed_action(feature, action)
        collision_logits = world_model.predict_severe_collision(
            feature, executed_action
        )
        state, _, ensemble = world_model.rssm.step_prior(
            state, executed_action, sample=True
        )
        next_feature = world_model.rssm.features(state)
        _, _, reward_logits, continue_logits = world_model.decode_features(
            next_feature
        )
        ensemble_probability = ensemble.softmax(dim=-1)
        features.append(feature)
        next_features.append(next_feature)
        actions.append(action)
        executed_actions.append(executed_action)
        action_rewrites.append(
            (executed_action - action).square().mean(dim=-1).sqrt()
        )
        log_probabilities.append(sample.log_probability)
        motion_entropies.append(sample.motion_entropy)
        gripper_entropies.append(sample.gripper_entropy)
        rewards.append(
            reward_expectation(
                reward_logits, limit=world_model.config.reward_symlog_limit
            )
        )
        continues.append(continue_logits.sigmoid())
        safety.append(safety_logits.sigmoid())
        severe_collisions.append(collision_logits.sigmoid())
        uncertainties.append(
            ensemble_probability.var(dim=1, unbiased=False).mean(dim=(-1, -2))
        )
    return ImaginedTrajectory(
        *(torch.stack(values, dim=1) for values in (
            features,
            next_features,
            actions,
            executed_actions,
            action_rewrites,
            log_probabilities,
            motion_entropies,
            gripper_entropies,
            rewards,
            continues,
            safety,
            severe_collisions,
            uncertainties,
        ))
    )
