from __future__ import annotations

import torch

from hwr.eval.action_input_contribution import (
    aggregate_action_input_contribution,
    canonical_normalize_actions,
    evaluate_action_input_contribution,
)
from hwr.world_model import ActionConditionedWorldModel, WorldModelConfig


def _fixture():
    config = WorldModelConfig(
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
        action_minimum=(-0.1, -0.5, 0.0),
        action_maximum=(0.1, 0.5, 1.0),
        formal=False,
    )
    model = ActionConditionedWorldModel(config)
    model.eval()
    batch, transitions = 2, 8
    visual = torch.randn(batch, transitions + 1, 8)
    language = torch.randn(batch, 6)
    proprioception = torch.randn(batch, transitions + 1, 5)
    proposals = torch.zeros(batch, transitions, 3)
    actions = torch.empty(batch, transitions, 3).uniform_(-0.1, 0.1)
    actions[..., 1] *= 5.0
    actions[..., 2] = torch.rand(batch, transitions)
    observed = model.observe(
        visual, language, proprioception, proposals, actions
    )
    return model, observed.sequence, actions


def test_canonical_action_normalization_uses_runtime_bounds() -> None:
    model, _, _ = _fixture()
    actions = torch.tensor([[[-0.1, 0.0, 1.0], [0.1, -0.5, 0.0]]])

    normalized = canonical_normalize_actions(actions, model.config)

    torch.testing.assert_close(
        normalized,
        torch.tensor([[[-1.0, 0.0, 1.0], [1.0, -1.0, -1.0]]]),
    )


def test_action_contribution_report_is_finite_and_bounded() -> None:
    model, sequence, actions = _fixture()

    report = evaluate_action_input_contribution(model, sequence, actions)

    assert report["canonical_actions_finite"]
    assert report["canonical_actions_in_bounds"]
    assert report["stochastic_contribution_rms"] > 0.0
    assert report["raw_action_contribution_rms"] > 0.0
    assert report["canonical_action_contribution_rms"] > 0.0
    assert len(report["weights"]["action_column_norms"]) == 3


def test_action_contribution_aggregate_applies_frozen_checks() -> None:
    model, sequence, actions = _fixture()
    report = evaluate_action_input_contribution(model, sequence, actions)
    report["raw_action_to_stochastic_ratio"] = 0.1
    report["canonical_to_raw_contribution_gain"] = 2.0

    aggregate = aggregate_action_input_contribution(
        [report for _ in range(24)]
    )

    assert aggregate["episode_count"] == 24
    assert aggregate["episodes_passing_contribution_conditions"] == 24
    assert set(aggregate["assessment"]["checks"]) == {
        "raw_action_to_stochastic_ratio_below_0_20",
        "canonical_to_raw_gain_at_least_1_50",
        "canonical_actions_finite",
        "canonical_actions_in_bounds",
        "at_least_20_of_24_episodes_pass",
    }
