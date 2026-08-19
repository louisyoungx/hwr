from __future__ import annotations

import torch

from hwr.eval.free_nats_deadzone import (
    aggregate_free_nats_deadzone,
    evaluate_free_nats_deadzone,
)
from hwr.world_model import ActionConditionedWorldModel, WorldModelConfig
from hwr.world_model.rssm import RSSMSequence


def _fixture_report():
    torch.manual_seed(13)
    model = ActionConditionedWorldModel(
        WorldModelConfig(
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
    )
    model.eval()
    batch, transitions = 2, 8
    visual = torch.randn(batch, transitions + 1, 8)
    language = torch.randn(batch, 6)
    proprioception = torch.randn(batch, transitions + 1, 5)
    proposals = torch.randn(batch, transitions, 3)
    actions = torch.randn(batch, transitions, 3, requires_grad=True)
    observed = model.observe(
        visual, language, proprioception, proposals, actions
    )
    prior = observed.sequence.prior_logits
    target = prior.detach().clone()
    pattern = torch.tensor((0.45, -0.45, 0.15, -0.15))
    target[:, 1:] = target[:, 1:] + pattern
    sequence = RSSMSequence(
        observed.sequence.deterministic,
        observed.sequence.stochastic,
        observed.sequence.prior_logits,
        target,
        observed.sequence.ensemble_prior_logits,
    )
    return evaluate_free_nats_deadzone(model, sequence, actions)


def test_free_nats_deadzone_recovers_raw_gradient_below_current_floor() -> None:
    report = _fixture_report()

    assert report["raw_kl"]["below_1_fraction"] == 1.0
    assert report["conditions"]["current"]["parameter_gradient_norm"] == 0.0
    assert report["conditions"]["raw"]["parameter_gradient_norm"] > 1.0e-6
    assert report["conditions"]["candidate"]["parameter_gradient_norm"] > 1.0e-6
    assert report["conditions"]["candidate"]["action_gradient_norm"] > 1.0e-6
    assert report["candidate_raw_parameter_gradient_cosine"] > 0.99


def test_free_nats_aggregate_applies_frozen_diagnostic_checks() -> None:
    report = _fixture_report()

    aggregate = aggregate_free_nats_deadzone([report for _ in range(24)])

    assert aggregate["episode_count"] == 24
    assert aggregate["assessment"]["passed"]
    assert aggregate["assessment"]["decision"] == "diagnostic_passed"
