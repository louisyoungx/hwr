"""Differentiable latent rollouts driven only by the current RL Actor."""

from __future__ import annotations

from typing import NamedTuple

import torch

from hwr.policy.latent_actor import LatentActor
from hwr.world_model.distributions import reward_expectation
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMState


class ImaginedTrajectory(NamedTuple):
    features: torch.Tensor
    next_features: torch.Tensor
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    motion_entropies: torch.Tensor
    gripper_entropies: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    safety_probabilities: torch.Tensor
    uncertainties: torch.Tensor


def imagine_trajectory(
    world_model: ActionConditionedWorldModel,
    actor: LatentActor,
    initial_state: RSSMState,
    *,
    horizon: int,
) -> ImaginedTrajectory:
    if horizon <= 0:
        raise ValueError("imagination horizon must be positive")
    state = initial_state
    features: list[torch.Tensor] = []
    next_features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    motion_entropies: list[torch.Tensor] = []
    gripper_entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    continues: list[torch.Tensor] = []
    safety: list[torch.Tensor] = []
    uncertainties: list[torch.Tensor] = []
    for _ in range(horizon):
        feature = world_model.rssm.features(state)
        sample = actor.sample(feature)
        state, _, ensemble = world_model.rssm.step_prior(
            state, sample.action, sample=True
        )
        next_feature = world_model.rssm.features(state)
        _, _, reward_logits, continue_logits, safety_logits = world_model.decode_features(
            next_feature
        )
        ensemble_probability = ensemble.softmax(dim=-1)
        features.append(feature)
        next_features.append(next_feature)
        actions.append(sample.action)
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
        uncertainties.append(
            ensemble_probability.var(dim=1, unbiased=False).mean(dim=(-1, -2))
        )
    return ImaginedTrajectory(
        *(torch.stack(values, dim=1) for values in (
            features,
            next_features,
            actions,
            log_probabilities,
            motion_entropies,
            gripper_entropies,
            rewards,
            continues,
            safety,
            uncertainties,
        ))
    )
