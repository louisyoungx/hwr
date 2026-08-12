from __future__ import annotations

import torch

from hwr.world_model import (
    ActionConditionedWorldModel,
    DeployableWorldModelStateFilter,
    WorldModelConfig,
)


def _config() -> WorldModelConfig:
    return WorldModelConfig(
        visual_dimension=8,
        language_dimension=6,
        proprioception_dimension=5,
        action_dimension=16,
        observation_embedding_dimension=12,
        deterministic_dimension=10,
        stochastic_variables=3,
        stochastic_classes=4,
        hidden_dimension=16,
        prior_ensemble=2,
        reward_bins=11,
        formal=False,
    )


def test_deployment_filter_matches_training_world_model_posterior() -> None:
    torch.manual_seed(4)
    training = ActionConditionedWorldModel(_config()).eval()
    deployment = DeployableWorldModelStateFilter.from_world_model(training).eval()
    visual = torch.randn(2, 8)
    language = torch.randn(2, 6)
    proprioception = torch.randn(2, 5)

    expected = training.posterior_step(
        visual, language, proprioception, previous=None, executed_action=None
    )
    actual = deployment.posterior_step(
        visual, language, proprioception, previous=None, executed_action=None
    )

    torch.testing.assert_close(actual.deterministic, expected.deterministic)
    torch.testing.assert_close(actual.stochastic, expected.stochastic)


def test_deployment_filter_contains_no_training_prediction_heads() -> None:
    deployment = DeployableWorldModelStateFilter(_config())

    names = {name for name, _ in deployment.named_modules()}
    assert not {"reward_head", "continue_head", "safety_head", "visual_head"} & names
    assert all("critic" not in name and "teacher" not in name for name in names)
