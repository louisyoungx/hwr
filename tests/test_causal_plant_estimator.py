from __future__ import annotations

import numpy as np
import pytest

from hwr.core.embodied import (
    DUAL_ARM_ACTION_MAXIMUM,
    DUAL_ARM_ACTION_MINIMUM,
)
from hwr.eval.causal_plant_estimator import (
    PLANT_STABLE_START,
    action_out_of_bounds_rate,
    deterministic_proposal_derangement,
    estimate_causal_plant_actions,
    normalized_action_rmse,
)
from hwr.eval.causal_plant_evaluation import (
    PlantEpisode,
    evaluate_causal_plant_estimator,
)


def _plant_episode(
    *,
    task_id: str,
    seed: int,
    correlation: float,
    latency: int,
    transitions: int = 64,
) -> PlantEpisode:
    rng = np.random.default_rng(seed)
    lower = np.asarray(DUAL_ARM_ACTION_MINIMUM)
    upper = np.asarray(DUAL_ARM_ACTION_MAXIMUM)
    proposals = rng.uniform(lower, upper, size=(transitions, 16))
    proposals[:, 14:] = (proposals[:, 14:] > 0.5).astype(np.float64)
    applied = np.zeros_like(proposals)
    applied[:, 14:] = 0.25
    gain = 0.93 + 0.01 * (seed % 5)
    for step in range(transitions):
        if step >= latency:
            applied[step] = proposals[step - latency]
            applied[step, :14] *= gain
    applied = np.clip(applied, lower, upper)
    return PlantEpisode(
        task_id,
        seed,
        correlation,
        latency,
        proposals,
        applied,
        np.zeros(transitions, np.bool_),
    )


@pytest.mark.parametrize("latency", (0, 1, 2, 3))
def test_estimator_recovers_fixed_fifo_and_gain(latency: int) -> None:
    episode = _plant_episode(
        task_id="fixture/v1",
        seed=17 + latency,
        correlation=0.50,
        latency=latency,
    )

    prediction = estimate_causal_plant_actions(
        episode.proposals,
        episode.applied_actions,
        episode.safety_interventions,
    )
    stable = prediction.stable

    assert np.flatnonzero(stable)[0] == PLANT_STABLE_START
    assert np.all(prediction.selected_lag[stable] == latency)
    assert normalized_action_rmse(
        prediction.predicted_action,
        episode.applied_actions,
        stable,
    ) < 1.0e-10
    assert action_out_of_bounds_rate(prediction.predicted_action) == 0.0


def test_estimator_uses_only_past_feedback() -> None:
    episode = _plant_episode(
        task_id="fixture/v1",
        seed=31,
        correlation=0.96,
        latency=2,
    )
    first = estimate_causal_plant_actions(
        episode.proposals,
        episode.applied_actions,
        episode.safety_interventions,
    )
    changed_actions = episode.applied_actions.copy()
    changed_actions[45:] *= -1.0
    second = estimate_causal_plant_actions(
        episode.proposals,
        changed_actions,
        episode.safety_interventions,
    )

    np.testing.assert_allclose(
        first.predicted_action[:45], second.predicted_action[:45]
    )
    np.testing.assert_array_equal(
        first.selected_lag[:45], second.selected_lag[:45]
    )


def test_derangement_is_deterministic_and_has_no_fixed_points() -> None:
    proposals = np.arange(64 * 16, dtype=np.float64).reshape(64, 16)

    first = deterministic_proposal_derangement(proposals, seed=19)
    second = deterministic_proposal_derangement(proposals, seed=19)

    np.testing.assert_array_equal(first, second)
    assert not np.any(np.all(first == proposals, axis=1))
    assert sorted(first[:, 0]) == sorted(proposals[:, 0])


def test_report_accepts_synthetic_confirmation_contract() -> None:
    development = [
        _plant_episode(
            task_id=task,
            seed=1000 + task_index * 100 + index,
            correlation=correlation,
            latency=latency,
        )
        for task_index, task in enumerate(("a", "b", "c"))
        for correlation in (0.50, 0.96)
        for latency in (0, 1)
        for index in range(8)
    ]
    confirmation = [
        _plant_episode(
            task_id=task,
            seed=2000 + task_index * 1000 + int(correlation * 100) * 10
            + latency * 100
            + index,
            correlation=correlation,
            latency=latency,
        )
        for task_index, task in enumerate(("a", "b", "c"))
        for correlation in (0.50, 0.96)
        for latency in (1, 2, 3)
        for index in range(8)
    ]

    report = evaluate_causal_plant_estimator(development, confirmation)

    assert report["decision"] == "accepted"
    assert report["development"]["passed"]
    assert report["confirmation"]["passed"]
    assert len(report["confirmation"]["partitions"]) == 18


def test_missing_confirmation_is_inconclusive() -> None:
    episode = _plant_episode(
        task_id="a",
        seed=7,
        correlation=0.50,
        latency=1,
    )

    report = evaluate_causal_plant_estimator((episode,) * 8, ())

    assert report["decision"] == "inconclusive"
    assert "confirmation_episode_count" in report["invalid_experiment_reasons"]
