from __future__ import annotations

import torch
import pytest

from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.policy.latent_value import LatentValueModel
from hwr.train.imagination_rl import (
    ImaginationActorCritic,
    ImaginationRLConfig,
    lambda_returns,
    optimize_imagination_step,
)
from hwr.world_model import ActionConditionedWorldModel, WorldModelConfig


def _models():
    world_config = WorldModelConfig(
        visual_dimension=8,
        language_dimension=6,
        proprioception_dimension=5,
        action_dimension=3,
        observation_embedding_dimension=16,
        deterministic_dimension=16,
        stochastic_variables=4,
        stochastic_classes=4,
        hidden_dimension=32,
        prior_ensemble=3,
        reward_bins=21,
        formal=False,
    )
    world = ActionConditionedWorldModel(world_config)
    actor = LatentActor(
        LatentActorConfig(
            world_config.feature_dimension,
            action_dimension=3,
            hidden_dimension=32,
            hidden_layers=2,
            formal=False,
        )
    )
    value = LatentValueModel(
        world_config.feature_dimension, bins=21, hidden_dimension=32, hidden_layers=2
    )
    config = ImaginationRLConfig(
        horizon=5,
        value_bins=21,
        value_symlog_limit=5.0,
    )
    return world, actor, value, config


def test_latent_actor_outputs_canonical_ranges_without_scene_inputs() -> None:
    config = LatentActorConfig(12, hidden_dimension=16, hidden_layers=1)
    actor = LatentActor(config)
    sample = actor.sample(torch.randn(4, 12))

    assert sample.action.shape == (4, 16)
    assert torch.all(sample.action[:, :14].abs() <= 1.0)
    assert torch.all((sample.action[:, 14:] >= 0.0) & (sample.action[:, 14:] <= 1.0))
    assert sample.log_probability.shape == (4,)
    assert actor.deterministic(torch.randn(4, 12)).shape == (4, 16)


def test_imagination_uses_actor_actions_and_world_model_outcomes() -> None:
    world, actor, value, config = _models()
    algorithm = ImaginationActorCritic(world, actor, value, config)
    initial = world.rssm.initial(3, torch.device("cpu"))

    losses, trajectory = algorithm.losses(initial)

    assert trajectory.actions.shape == (3, 5, 3)
    assert trajectory.next_features.shape == (3, 5, world.config.feature_dimension)
    assert set(losses) == {
        "actor", "value", "imagined_reward", "imagined_return",
        "imagined_safety", "imagined_uncertainty",
        "td_error",
    }
    assert all(torch.isfinite(value) for value in losses.values())


def test_slow_value_inherits_exact_value_device_and_dtype() -> None:
    world, actor, value, config = _models()
    value.to(dtype=torch.float64)

    algorithm = ImaginationActorCritic(world, actor, value, config)

    current = tuple(value.parameters())
    slow = tuple(algorithm.slow_value.parameters())
    assert len(current) == len(slow)
    assert all(left.device == right.device for left, right in zip(current, slow))
    assert all(left.dtype == right.dtype for left, right in zip(current, slow))


def test_imagination_optimization_updates_actor_and_value_not_world_model() -> None:
    world, actor, value, config = _models()
    algorithm = ImaginationActorCritic(world, actor, value, config)
    initial = world.rssm.initial(3, torch.device("cpu"))
    world_before = [parameter.detach().clone() for parameter in world.parameters()]
    actor_before = actor.mean_head.weight.detach().clone()
    value_before = value.network[-1].weight.detach().clone()

    metrics = optimize_imagination_step(
        algorithm,
        initial,
        torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        torch.optim.Adam(value.parameters(), lr=1.0e-3),
    )

    assert torch.any(actor.mean_head.weight != actor_before)
    assert torch.any(value.network[-1].weight != value_before)
    assert all(torch.equal(before, after) for before, after in zip(world_before, world.parameters()))
    assert all(parameter.requires_grad for parameter in world.parameters())
    assert metrics["imagined_return"] == metrics["imagined_return"]


def test_imagination_failure_restores_world_model_gradient_state(monkeypatch) -> None:
    world, actor, value, config = _models()
    algorithm = ImaginationActorCritic(world, actor, value, config)
    initial = world.rssm.initial(3, torch.device("cpu"))

    def fail(_initial):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(algorithm, "losses", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        optimize_imagination_step(
            algorithm,
            initial,
            torch.optim.Adam(actor.parameters(), lr=1.0e-3),
            torch.optim.Adam(value.parameters(), lr=1.0e-3),
        )

    assert all(parameter.requires_grad for parameter in world.parameters())


def test_lambda_returns_match_one_step_extremes() -> None:
    reward = torch.tensor([[1.0, 2.0, 3.0]])
    discount = torch.full_like(reward, 0.5)
    values = torch.tensor([[10.0, 20.0, 30.0]])

    one_step = lambda_returns(reward, discount, values, lambda_=0.0)
    monte_carlo = lambda_returns(reward, discount, values, lambda_=1.0)

    assert torch.allclose(one_step, torch.tensor([[11.0, 17.0, 18.0]]))
    assert torch.allclose(monte_carlo, torch.tensor([[6.5, 11.0, 18.0]]))


def test_formal_imagination_uses_same_physical_action_units_as_replay() -> None:
    world_config = WorldModelConfig(
        visual_dimension=8,
        language_dimension=6,
        proprioception_dimension=5,
        action_dimension=16,
        observation_embedding_dimension=16,
        deterministic_dimension=16,
        stochastic_variables=4,
        stochastic_classes=4,
        hidden_dimension=32,
        prior_ensemble=3,
        reward_bins=21,
        formal=False,
    )
    world = ActionConditionedWorldModel(world_config)
    actor = LatentActor(
        LatentActorConfig(
            world_config.feature_dimension,
            hidden_dimension=32,
            hidden_layers=2,
            formal=False,
        )
    )
    value = LatentValueModel(
        world_config.feature_dimension, bins=21, hidden_dimension=32, hidden_layers=2
    )
    algorithm = ImaginationActorCritic(
        world, actor, value, ImaginationRLConfig(horizon=2, value_bins=21)
    )

    _, trajectory = algorithm.losses(world.rssm.initial(2, torch.device("cpu")))

    assert torch.all(trajectory.actions[..., 0].abs() <= 0.18)
    assert torch.all(trajectory.actions[..., 1].abs() <= 0.50)
    assert torch.all(trajectory.actions[..., 2:14].abs() <= 0.35)
    assert torch.all((trajectory.actions[..., 14:] >= 0.0))
    assert torch.all((trajectory.actions[..., 14:] <= 1.0))
