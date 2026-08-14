from __future__ import annotations

import torch
import pytest

from hwr.world_model import (
    ActionCausalityCriteria,
    ActionConditionedWorldModel,
    CounterfactualCausalityReport,
    CounterfactualComponentReport,
    WorldModelConfig,
    WorldModelLoss,
    WorldModelLossConfig,
    WorldModelTargets,
    assess_action_causality,
    aggregate_action_causality_reports,
    deterministic_action_derangement,
    evaluate_action_causality,
    evaluate_one_step_action_utilization,
)
from hwr.world_model.distributions import reward_expectation, two_hot_symlog


def _config() -> WorldModelConfig:
    return WorldModelConfig(
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


def _inputs(config: WorldModelConfig, batch: int = 2, transitions: int = 4):
    return (
        torch.randn(batch, transitions + 1, config.visual_dimension),
        torch.randn(batch, config.language_dimension),
        torch.randn(batch, transitions + 1, config.proprioception_dimension),
        torch.randn(batch, transitions, config.action_dimension),
        torch.randn(batch, transitions, config.action_dimension),
    )


def test_action_conditioned_world_model_observe_and_prior_shapes() -> None:
    config = _config()
    model = ActionConditionedWorldModel(config)
    visual, language, proprioception, proposals, actions = _inputs(config)

    output = model.observe(visual, language, proprioception, proposals, actions)
    initial = model.rssm.posterior_state(output.sequence, 0)
    rollout = model.rollout_prior(initial, actions, proposals, sample=False)

    assert output.features.shape == (2, 5, config.feature_dimension)
    assert output.reward_logits.shape == (2, 5, 21)
    assert output.sequence.ensemble_prior_logits.shape == (2, 5, 3, 4, 4)
    assert output.safety_logits.shape == (2, 4)
    assert rollout.visual_prediction.shape == (2, 4, 8)
    assert rollout.uncertainty.shape == (2, 4)
    assert torch.isfinite(rollout.features).all()


def test_world_model_losses_train_dynamics_and_outcome_heads() -> None:
    config = _config()
    model = ActionConditionedWorldModel(config)
    visual, language, proprioception, proposals, actions = _inputs(config)
    output = model.observe(visual, language, proprioception, proposals, actions)
    targets = WorldModelTargets(
        visual=visual,
        proprioception=proprioception,
        reward=torch.randn(2, 4),
        continues=torch.ones(2, 4),
        safety_interventions=torch.zeros(2, 4),
        severe_collisions=torch.zeros(2, 4),
    )
    objective = WorldModelLoss(config, WorldModelLossConfig())

    losses = objective(output, targets)
    losses["total"].backward()

    assert set(losses) == {
        "visual", "proprioception", "reward", "continue", "safety",
        "severe_collision",
        "dynamics", "representation", "ensemble", "total",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    assert model.rssm.recurrent.weight_hh.grad is not None
    assert model.reward_head[-1].weight.grad is not None
    assert model.safety_head[-1].weight.grad is not None
    assert model.severe_collision_head[-1].weight.grad is not None


def test_actor_proposals_affect_safety_head_but_not_executed_dynamics() -> None:
    config = _config()
    model = ActionConditionedWorldModel(config).eval()
    model.safety_head = torch.nn.Linear(
        config.feature_dimension + config.action_dimension, 1, bias=False
    )
    with torch.no_grad():
        model.safety_head.weight.zero_()
        model.safety_head.weight[:, -config.action_dimension :] = 1.0
    visual, language, proprioception, _, actions = _inputs(config)
    first_proposals = torch.zeros_like(actions)
    second_proposals = torch.ones_like(actions)

    first = model.observe(
        visual, language, proprioception, first_proposals, actions
    )
    second = model.observe(
        visual, language, proprioception, second_proposals, actions
    )

    torch.testing.assert_close(first.features, second.features)
    assert torch.all(second.safety_logits > first.safety_logits)


def test_action_shuffle_counterfactual_reports_open_loop_errors() -> None:
    config = _config()
    model = ActionConditionedWorldModel(config)
    visual, language, proprioception, proposals, actions = _inputs(config)

    report = evaluate_action_causality(
        model,
        visual,
        language,
        proprioception,
        proposals,
        actions,
        torch.randn(2, 4),
        torch.ones(2, 4),
        torch.zeros(2, 4),
    )

    assert report.true_action_error >= 0.0
    assert report.shuffled_action_error >= 0.0
    assert len(report.true_horizon_errors) == 4
    assert len(report.uncertainty_by_horizon) == 4
    assert report.error_components == (
        "visual_latent",
        "proprioception",
        "reward",
        "continue",
        "safety",
    )
    assert tuple(report.component_reports) == report.error_components
    assert all(
        len(component.true_horizon_errors) == 4
        for component in report.component_reports.values()
    )
    assert model.training
    assert report.sample_count == 2


def test_one_step_action_utilization_holds_posterior_states_fixed() -> None:
    config = _config()
    model = ActionConditionedWorldModel(config)
    visual, language, proprioception, proposals, actions = _inputs(config)

    report = evaluate_one_step_action_utilization(
        model, visual, language, proprioception, proposals, actions, shuffle_seed=3
    )

    assert report.error_components == ("visual_latent", "proprioception")
    assert report.sample_count == actions.shape[0] * actions.shape[1]
    assert len(report.true_horizon_errors) == 1
    assert model.training


def test_action_derangement_is_deterministic_value_preserving_and_has_no_fixed_points() -> None:
    actions = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()

    first = deterministic_action_derangement(actions, seed=17)
    second = deterministic_action_derangement(actions, seed=17)

    torch.testing.assert_close(first, second)
    assert not torch.any(torch.all(first == actions, dim=-1))
    assert {
        tuple(value.tolist()) for value in first.flatten(0, 1)
    } == {tuple(value.tolist()) for value in actions.flatten(0, 1)}


def test_action_causality_aggregation_weights_sequence_counts() -> None:
    first_components = {
        name: CounterfactualComponentReport(
            0.5, 1.0, 2.0, (0.5, 0.5), (1.0, 1.0)
        )
        for name in ("visual_latent", "proprioception")
    }
    second_components = {
        name: CounterfactualComponentReport(
            1.5, 2.0, 4.0 / 3.0, (1.5, 1.5), (2.0, 2.0)
        )
        for name in ("visual_latent", "proprioception")
    }
    first = CounterfactualCausalityReport(
        1.0, 2.0, 2.0, (1.0, 1.0), (2.0, 2.0), (0.1, 0.2),
        first_components,
        sample_count=1,
    )
    second = CounterfactualCausalityReport(
        3.0, 4.0, 4.0 / 3.0, (3.0, 3.0), (4.0, 4.0), (0.3, 0.4),
        second_components,
        sample_count=3,
    )

    aggregate = aggregate_action_causality_reports((first, second))

    assert aggregate.true_horizon_errors == pytest.approx((2.5, 2.5))
    assert aggregate.shuffled_horizon_errors == pytest.approx((3.5, 3.5))
    assert aggregate.shuffled_to_true_ratio == pytest.approx(1.4)
    assert aggregate.sample_count == 4


def test_action_causality_gate_requires_ratio_and_most_horizons_to_degrade() -> None:
    components = {
        name: CounterfactualComponentReport(
            1.0,
            1.125,
            1.125,
            (1.0, 1.0, 1.0, 1.0),
            (1.1, 1.2, 0.9, 1.3),
        )
        for name in ("visual_latent", "proprioception")
    }
    report = CounterfactualCausalityReport(
        true_action_error=2.0,
        shuffled_action_error=2.25,
        shuffled_to_true_ratio=1.125,
        true_horizon_errors=(2.0, 2.0, 2.0, 2.0),
        shuffled_horizon_errors=(2.2, 2.4, 1.8, 2.6),
        uncertainty_by_horizon=(0.1, 0.2, 0.3, 0.4),
        component_reports=components,
    )

    assessment = assess_action_causality(
        report,
        ActionCausalityCriteria(1.05, 0.60),
    )

    assert assessment["passed"] is True
    assert assessment["worse_horizon_fraction"] == 0.75
    failed = assess_action_causality(
        report,
        ActionCausalityCriteria(1.25, 0.60),
    )
    assert failed["passed"] is False


def test_action_causality_gate_rejects_one_action_independent_prediction_head() -> None:
    passing = CounterfactualComponentReport(
        1.0, 1.2, 1.2, (1.0, 1.0), (1.2, 1.2)
    )
    ignored = CounterfactualComponentReport(
        1.0, 1.0, 1.0, (1.0, 1.0), (1.0, 1.0)
    )
    report = CounterfactualCausalityReport(
        2.0,
        2.2,
        1.1,
        (2.0, 2.0),
        (2.2, 2.2),
        (0.1, 0.1),
        {"visual_latent": passing, "safety": ignored},
        ("visual_latent", "safety"),
    )

    assessment = assess_action_causality(report)

    assert assessment["aggregate_passed"] is True
    assert assessment["components"]["visual_latent"]["passed"] is True
    assert assessment["components"]["safety"]["passed"] is False
    assert assessment["passed"] is False


def test_distributional_reward_round_trip_is_finite() -> None:
    values = torch.tensor([-10.0, -1.0, 0.0, 2.0, 30.0])
    targets = two_hot_symlog(values, bins=21, limit=5.0)
    logits = torch.log(targets + 1.0e-6)
    reconstructed = reward_expectation(logits, limit=5.0)

    assert torch.allclose(targets.sum(dim=-1), torch.ones(5))
    assert torch.isfinite(reconstructed).all()
    assert reconstructed[0] < reconstructed[2] < reconstructed[-1]


def test_formal_world_model_rejects_noncanonical_action_dimension() -> None:
    with pytest.raises(ValueError, match="canonical 16-D"):
        WorldModelConfig(action_dimension=8)
