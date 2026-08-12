"""Deployment-only posterior state filter stripped of prediction heads."""

from __future__ import annotations

import copy

import torch
from torch import nn

from hwr.world_model.config import WorldModelConfig
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import CategoricalRSSM, RSSMState


class DeployableWorldModelStateFilter(nn.Module):
    """Observation encoders and RSSM posterior needed by the learned Actor."""

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
        self.observation_encoder = nn.Sequential(
            nn.Linear(config.hidden_dimension * 2, config.observation_embedding_dimension),
            nn.LayerNorm(config.observation_embedding_dimension),
            nn.SiLU(),
        )
        self.rssm = CategoricalRSSM(config)

    @classmethod
    def from_world_model(
        cls, world_model: ActionConditionedWorldModel
    ) -> "DeployableWorldModelStateFilter":
        deployment = cls(copy.deepcopy(world_model.config))
        for name in (
            "visual_encoder",
            "language_encoder",
            "proprioception_encoder",
            "observation_encoder",
            "rssm",
        ):
            getattr(deployment, name).load_state_dict(
                getattr(world_model, name).state_dict()
            )
        return deployment

    def encode_observation(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        batch = visual.shape[0]
        if visual.shape != (batch, self.config.visual_dimension):
            raise ValueError("deployment visual state shape is invalid")
        if language.shape != (batch, self.config.language_dimension):
            raise ValueError("deployment language state shape is invalid")
        if proprioception.shape != (batch, self.config.proprioception_dimension):
            raise ValueError("deployment proprioception state shape is invalid")
        return self.observation_encoder(
            torch.cat(
                (
                    self.visual_encoder(visual),
                    self.language_encoder(language),
                    self.proprioception_encoder(proprioception),
                ),
                dim=-1,
            )
        )

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
        embedding = self.encode_observation(visual, language, proprioception)
        return self.rssm.update_posterior(
            embedding,
            previous=previous,
            executed_action=executed_action,
            sample=sample,
        )

    def features(self, state: RSSMState) -> torch.Tensor:
        return self.rssm.features(state)
