"""Categorical recurrent state-space dynamics conditioned on executed actions."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from hwr.world_model.config import WorldModelConfig


class RSSMState(NamedTuple):
    deterministic: torch.Tensor
    stochastic: torch.Tensor


class RSSMSequence(NamedTuple):
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    prior_logits: torch.Tensor
    posterior_logits: torch.Tensor
    ensemble_prior_logits: torch.Tensor


class CategoricalRSSM(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        stochastic = config.stochastic_dimension
        self.transition_input = nn.Sequential(
            nn.Linear(stochastic + config.action_dimension, config.hidden_dimension),
            nn.LayerNorm(config.hidden_dimension),
            nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(
            config.hidden_dimension, config.deterministic_dimension
        )
        self.prior = nn.Sequential(
            nn.Linear(config.deterministic_dimension, config.hidden_dimension),
            nn.SiLU(),
            nn.Linear(
                config.hidden_dimension,
                config.stochastic_variables * config.stochastic_classes,
            ),
        )
        self.posterior = nn.Sequential(
            nn.Linear(
                config.deterministic_dimension + config.observation_embedding_dimension,
                config.hidden_dimension,
            ),
            nn.SiLU(),
            nn.Linear(
                config.hidden_dimension,
                config.stochastic_variables * config.stochastic_classes,
            ),
        )
        self.ensemble = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.deterministic_dimension, config.hidden_dimension),
                    nn.SiLU(),
                    nn.Linear(
                        config.hidden_dimension,
                        config.stochastic_variables * config.stochastic_classes,
                    ),
                )
                for _ in range(config.prior_ensemble)
            ]
        )

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            torch.zeros(batch_size, self.config.deterministic_dimension, device=device),
            torch.zeros(batch_size, self.config.stochastic_dimension, device=device),
        )

    def observe(
        self, observation_embeddings: torch.Tensor, actions: torch.Tensor
    ) -> RSSMSequence:
        batch, observations, dimension = observation_embeddings.shape
        if dimension != self.config.observation_embedding_dimension:
            raise ValueError("RSSM observation embedding dimension is invalid")
        if actions.shape != (
            batch,
            observations - 1,
            self.config.action_dimension,
        ):
            raise ValueError("RSSM actions must connect consecutive observations")
        state = self.initial(batch, observation_embeddings.device)
        deterministic: list[torch.Tensor] = []
        stochastic: list[torch.Tensor] = []
        priors: list[torch.Tensor] = []
        posteriors: list[torch.Tensor] = []
        ensembles: list[torch.Tensor] = []
        for index in range(observations):
            if index:
                state, prior, ensemble = self.step_prior(
                    state, actions[:, index - 1], sample=self.training
                )
            else:
                prior = self._logits(self.prior(state.deterministic))
                ensemble = self._ensemble_logits(state.deterministic)
            posterior = self._logits(
                self.posterior(
                    torch.cat((state.deterministic, observation_embeddings[:, index]), dim=-1)
                )
            )
            state = RSSMState(
                state.deterministic,
                self._sample(posterior, sample=self.training),
            )
            deterministic.append(state.deterministic)
            stochastic.append(state.stochastic)
            priors.append(prior)
            posteriors.append(posterior)
            ensembles.append(ensemble)
        return RSSMSequence(
            torch.stack(deterministic, dim=1),
            torch.stack(stochastic, dim=1),
            torch.stack(priors, dim=1),
            torch.stack(posteriors, dim=1),
            torch.stack(ensembles, dim=1),
        )

    def step_prior(
        self, state: RSSMState, action: torch.Tensor, *, sample: bool
    ) -> tuple[RSSMState, torch.Tensor, torch.Tensor]:
        if action.shape != (state.deterministic.shape[0], self.config.action_dimension):
            raise ValueError("RSSM prior action shape is invalid")
        transition = self.transition_input(torch.cat((state.stochastic, action), dim=-1))
        deterministic = self.recurrent(transition, state.deterministic)
        logits = self._logits(self.prior(deterministic))
        stochastic = self._sample(logits, sample=sample)
        return (
            RSSMState(deterministic, stochastic),
            logits,
            self._ensemble_logits(deterministic),
        )

    def posterior_state(self, sequence: RSSMSequence, index: int = -1) -> RSSMState:
        return RSSMState(
            sequence.deterministic[:, index], sequence.stochastic[:, index]
        )

    def update_posterior(
        self,
        observation_embedding: torch.Tensor,
        *,
        previous: RSSMState | None,
        executed_action: torch.Tensor | None,
        sample: bool,
    ) -> RSSMState:
        batch = observation_embedding.shape[0]
        if observation_embedding.shape != (
            batch,
            self.config.observation_embedding_dimension,
        ):
            raise ValueError("RSSM posterior observation embedding shape is invalid")
        if previous is None:
            if executed_action is not None:
                raise ValueError("initial RSSM posterior cannot consume an earlier action")
            prior = self.initial(batch, observation_embedding.device)
        else:
            if executed_action is None:
                raise ValueError("subsequent RSSM posterior requires the executed action")
            prior, _, _ = self.step_prior(previous, executed_action, sample=sample)
        logits = self._logits(
            self.posterior(
                torch.cat((prior.deterministic, observation_embedding), dim=-1)
            )
        )
        return RSSMState(prior.deterministic, self._sample(logits, sample=sample))

    def features(self, state: RSSMState) -> torch.Tensor:
        return torch.cat((state.deterministic, state.stochastic), dim=-1)

    def _logits(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(
            value.shape[0],
            self.config.stochastic_variables,
            self.config.stochastic_classes,
        )

    def _ensemble_logits(self, deterministic: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self._logits(head(deterministic)) for head in self.ensemble], dim=1
        )

    def _sample(self, logits: torch.Tensor, *, sample: bool) -> torch.Tensor:
        probabilities = logits.softmax(dim=-1)
        if self.config.categorical_unimix:
            uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
            probabilities = (
                (1.0 - self.config.categorical_unimix) * probabilities
                + self.config.categorical_unimix * uniform
            )
        if sample:
            discrete = torch.distributions.OneHotCategorical(probs=probabilities).sample()
            discrete = discrete + probabilities - probabilities.detach()
        else:
            indices = probabilities.argmax(dim=-1)
            discrete = nn.functional.one_hot(
                indices, self.config.stochastic_classes
            ).to(probabilities.dtype)
        return discrete.flatten(-2)
