from __future__ import annotations

import torch

from hwr.apps.evaluate_posterior_overshooting import (
    select_source_episode_windows,
)
from hwr.eval.posterior_overshooting import (
    aggregate_posterior_overshooting,
    build_overshooting_pairs,
    evaluate_posterior_overshooting,
)
from hwr.world_model import ActionConditionedWorldModel, WorldModelConfig


def _model() -> ActionConditionedWorldModel:
    return ActionConditionedWorldModel(
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


def _report():
    torch.manual_seed(7)
    model = _model()
    model.eval()
    batch, transitions = 2, 8
    visual = torch.randn(batch, transitions + 1, 8)
    language = torch.randn(batch, 6)
    proprioception = torch.randn(batch, transitions + 1, 5)
    proposals = torch.randn(batch, transitions, 3)
    actions = torch.randn(batch, transitions, 3)
    observed = model.observe(
        visual, language, proprioception, proposals, actions
    )
    return evaluate_posterior_overshooting(
        model, observed.sequence, actions, horizons=(1, 2, 4, 8)
    )


def test_overshooting_pairs_are_causal_and_in_bounds() -> None:
    pairs = build_overshooting_pairs(16, 8)

    assert pairs.starts[0] == 0
    assert pairs.targets[0] == 8
    assert pairs.starts[-1] == 8
    assert pairs.targets[-1] == 16
    assert all(target - start == 8 for start, target in zip(pairs.starts, pairs.targets))


def test_overshooting_report_has_finite_action_gradient_and_stopped_targets() -> None:
    report = _report()

    assert report["action_gradient_finite"]
    assert report["action_gradient_norm"] > 0.0
    assert report["target_requires_gradient"] is False
    assert set(report["conditions"]) == {
        "true_action",
        "zero_action",
        "shifted_action",
    }
    assert set(report["conditions"]["true_action"]["horizon_total_losses"]) == {
        "1",
        "2",
        "4",
        "8",
    }


def test_overshooting_aggregate_applies_frozen_thresholds() -> None:
    report = _report()
    reports = [report for _ in range(24)]

    aggregate = aggregate_posterior_overshooting(reports)

    assert aggregate["episode_count"] == 24
    assert set(aggregate["assessment"]["checks"]) == {
        "all_losses_finite",
        "true_better_on_three_of_four_horizons",
        "true_at_least_five_percent_better_than_zero",
        "true_at_least_five_percent_better_than_shifted",
        "finite_action_gradient",
        "action_gradient_norm_above_1e_6",
        "posterior_targets_stopped",
    }


class _WindowLoader:
    def __len__(self):
        return 48

    def window_metadata(self, index: int):
        source = index // 2
        return {
            "episode_id": f"window-{index}",
            "task_id": f"task-{source % 3}",
            "seed": source,
            "transition_start": (index % 2) * 16,
            "transition_stop": (index % 2 + 1) * 16,
            "metadata": {
                "sequence_reservoir": {
                    "source_episode_id": f"source-{source}",
                }
            },
        }


def test_source_window_selection_is_deterministic_and_one_per_source() -> None:
    loader = _WindowLoader()

    first = select_source_episode_windows(loader, seed=20_261_306)
    second = select_source_episode_windows(loader, seed=20_261_306)

    assert first == second
    assert len(first) == 24
    assert set(first) == {f"source-{index}" for index in range(24)}
    assert all(
        index in (
            int(source.removeprefix("source-")) * 2,
            int(source.removeprefix("source-")) * 2 + 1,
        )
        for source, index in first.items()
    )
