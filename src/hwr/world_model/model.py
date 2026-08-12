"""Action-conditioned world model and open-loop prior rollout."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from hwr.world_model.config import WorldModelConfig
from hwr.world_model.rssm import CategoricalRSSM, RSSMSequence, RSSMState


class WorldModelOutput(NamedTuple):
    sequence: RSSMSequence
    features: torch.Tensor
    visual_prediction: torch.Tensor
    proprioception_prediction: torch.Tensor
    reward_logits: torch.Tensor
    continue_logits: torch.Tensor
    safety_logits: torch.Tensor


class WorldModelPriorRollout(NamedTuple):
    states: RSSMSequence
    features: torch.Tensor
    visual_prediction: torch.Tensor
    proprioception_prediction: torch.Tensor
    reward_logits: torch.Tensor
    continue_logits: torch.Tensor
    safety_logits: torch.Tensor
    uncertainty: torch.Tensor


class ActionConditionedWorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_dimension, config.hidden_dimension),
            nn.LayerNorm(config.hidden_dimension),
            nn.SiLU(),
        )
        self.language_encoder = nn.Sequential(
            nn.Linear(config.language_dimension, config.hidden_dimension // 2),
            nn.LayerNorm(config.hidden_dimension // 2),
            nn.SiLU(),
        )
        self.proprioception_encoder = nn.Sequential(
            nn.Linear(config.proprioception_dimension, config.hidden_dimension // 2),
            nn.LayerNorm(config.hidden_dimension // 2),
            nn.SiLU(),
        )
        combined = config.hidden_dimension * 2
        self.observation_encoder = nn.Sequential(
            nn.Linear(combined, config.observation_embedding_dimension),
            nn.LayerNorm(config.observation_embedding_dimension),
            nn.SiLU(),
        )
        self.rssm = CategoricalRSSM(config)
        feature = config.feature_dimension
        self.visual_head = _head(feature, config.hidden_dimension, config.visual_dimension)
        self.proprioception_head = _head(
            feature, config.hidden_dimension, config.proprioception_dimension
        )
        self.reward_head = _head(feature, config.hidden_dimension, config.reward_bins)
        self.continue_head = _head(feature, config.hidden_dimension, 1)
        self.safety_head = _head(feature, config.hidden_dimension, 1)

    def observe(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        proprioception: torch.Tensor,
        executed_actions: torch.Tensor,
    ) -> WorldModelOutput:
        self._check_observation_shapes(
            visual, language, proprioception, executed_actions
        )
        embedding = self.encode_observations(visual, language, proprioception)
        sequence = self.rssm.observe(embedding, executed_actions)
        features = torch.cat(
            (sequence.deterministic, sequence.stochastic), dim=-1
        )
        decoded = self._decode(features)
        return WorldModelOutput(sequence, features, *decoded)

    def encode_observations(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        history = visual.shape[1]
        language_features = self.language_encoder(language)[:, None].expand(-1, history, -1)
        return self.observation_encoder(
            torch.cat(
                (
                    self.visual_encoder(visual),
                    language_features,
                    self.proprioception_encoder(proprioception),
                ),
                dim=-1,
            )
        )

    def initial_posterior(
        self, visual: torch.Tensor, language: torch.Tensor, proprioception: torch.Tensor
    ) -> RSSMState:
        if visual.ndim != 2 or proprioception.ndim != 2:
            raise ValueError("world model initial posterior requires one observation")
        empty_actions = visual.new_zeros(
            visual.shape[0], 0, self.config.action_dimension
        )
        output = self.observe(
            visual[:, None], language, proprioception[:, None], empty_actions
        )
        return self.rssm.posterior_state(output.sequence)

    def posterior_step(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        proprioception: torch.Tensor,
        *,
        previous: RSSMState | None,
        executed_action: torch.Tensor | None,
        sample: bool = False,
    ) -> RSSMState:
        if visual.ndim != 2 or proprioception.ndim != 2:
            raise ValueError("world model posterior step requires one observation")
        embedding = self.encode_observations(
            visual[:, None], language, proprioception[:, None]
        )[:, 0]
        return self.rssm.update_posterior(
            embedding,
            previous=previous,
            executed_action=executed_action,
            sample=sample,
        )

    def rollout_prior(
        self, initial: RSSMState, actions: torch.Tensor, *, sample: bool = False
    ) -> WorldModelPriorRollout:
        if actions.ndim != 3 or actions.shape[-1] != self.config.action_dimension:
            raise ValueError("world model rollout actions have an invalid shape")
        state = initial
        deterministic: list[torch.Tensor] = []
        stochastic: list[torch.Tensor] = []
        priors: list[torch.Tensor] = []
        ensembles: list[torch.Tensor] = []
        for index in range(actions.shape[1]):
            state, prior, ensemble = self.rssm.step_prior(
                state, actions[:, index], sample=sample
            )
            deterministic.append(state.deterministic)
            stochastic.append(state.stochastic)
            priors.append(prior)
            ensembles.append(ensemble)
        if not deterministic:
            raise ValueError("world model prior rollout requires at least one action")
        deterministic_tensor = torch.stack(deterministic, dim=1)
        stochastic_tensor = torch.stack(stochastic, dim=1)
        prior_tensor = torch.stack(priors, dim=1)
        ensemble_tensor = torch.stack(ensembles, dim=1)
        placeholder = torch.zeros_like(prior_tensor)
        sequence = RSSMSequence(
            deterministic_tensor,
            stochastic_tensor,
            prior_tensor,
            placeholder,
            ensemble_tensor,
        )
        features = torch.cat((deterministic_tensor, stochastic_tensor), dim=-1)
        decoded = self._decode(features)
        probabilities = ensemble_tensor.softmax(dim=-1)
        uncertainty = probabilities.var(dim=2, unbiased=False).mean(dim=(-1, -2))
        return WorldModelPriorRollout(sequence, features, *decoded, uncertainty)

    def _decode(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.visual_head(features),
            self.proprioception_head(features),
            self.reward_head(features),
            self.continue_head(features).squeeze(-1),
            self.safety_head(features).squeeze(-1),
        )

    def decode_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent features without exposing any action answer."""
        if features.shape[-1] != self.config.feature_dimension:
            raise ValueError("world model decoded feature dimension is invalid")
        return self._decode(features)

    def _check_observation_shapes(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        proprioception: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        if visual.ndim != 3:
            raise ValueError("world model visual input must be batch-time-feature")
        batch, observations = visual.shape[:2]
        expected = {
            "visual": (batch, observations, self.config.visual_dimension),
            "language": (batch, self.config.language_dimension),
            "proprioception": (
                batch,
                observations,
                self.config.proprioception_dimension,
            ),
            "executed_actions": (
                batch,
                observations - 1,
                self.config.action_dimension,
            ),
        }
        actual = {
            "visual": tuple(visual.shape),
            "language": tuple(language.shape),
            "proprioception": tuple(proprioception.shape),
            "executed_actions": tuple(actions.shape),
        }
        mismatches = {
            name: (actual[name], shape)
            for name, shape in expected.items()
            if actual[name] != shape
        }
        if mismatches:
            raise ValueError(f"world model tensor shapes are invalid: {mismatches}")


def _head(input_dimension: int, hidden_dimension: int, output_dimension: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.LayerNorm(hidden_dimension),
        nn.SiLU(),
        nn.Linear(hidden_dimension, output_dimension),
    )
