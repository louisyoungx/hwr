from __future__ import annotations

import numpy as np

from hwr.train import (
    OutcomeAdaptiveTaskSampler,
    OutcomeAdaptiveTaskSamplingConfig,
    TaskOutcome,
)


def _outcome(
    left: int, right: int, simultaneous: int, distance: float
) -> TaskOutcome:
    return TaskOutcome(left, right, simultaneous, distance, distance)


def test_outcome_sampler_prioritizes_weak_task_without_starving_others() -> None:
    config = OutcomeAdaptiveTaskSamplingConfig(
        initial_cycles=1, minimum_probability=0.10
    )
    sampler = OutcomeAdaptiveTaskSampler(("basket", "drawer", "tray"), config)
    for _ in range(6):
        sampler.record("basket", _outcome(0, 0, 0, 0.22))
        sampler.record("drawer", _outcome(20, 0, 0, 0.10))
        sampler.record("tray", _outcome(20, 20, 12, 0.04))

    probabilities = sampler.probabilities()

    assert probabilities["basket"] > probabilities["drawer"]
    assert probabilities["drawer"] > probabilities["tray"]
    assert all(value >= 0.10 for value in probabilities.values())
    assert np.isclose(sum(probabilities.values()), 1.0)


def test_outcome_sampler_initial_coverage_and_state_are_reproducible() -> None:
    config = OutcomeAdaptiveTaskSamplingConfig(initial_cycles=1)
    sampler = OutcomeAdaptiveTaskSampler(("basket", "drawer", "tray"), config)
    rng = np.random.default_rng(7)

    initial = [sampler.sample(rng)[0] for _ in range(3)]
    sampler.record("basket", _outcome(0, 0, 0, 0.2))
    restored = OutcomeAdaptiveTaskSampler(("basket", "drawer", "tray"), config)
    restored.load_state_dict(sampler.state_dict())

    assert initial == ["basket", "drawer", "tray"]
    assert restored.state_dict() == sampler.state_dict()
    assert sampler.audit()["actor_input_fields"] == []


def test_outcome_sampler_uses_weighted_fair_credits_not_random_luck() -> None:
    config = OutcomeAdaptiveTaskSamplingConfig(initial_cycles=1)
    sampler = OutcomeAdaptiveTaskSampler(("basket", "drawer", "tray"), config)
    for _ in range(6):
        sampler.record("basket", _outcome(0, 0, 0, 0.22))
        sampler.record("drawer", _outcome(20, 0, 0, 0.10))
        sampler.record("tray", _outcome(20, 20, 12, 0.04))
    rng = np.random.default_rng(999)
    selected = [sampler.sample(rng)[0] for _ in range(33)]
    adaptive = selected[3:]

    assert adaptive.count("basket") > adaptive.count("drawer")
    assert adaptive.count("drawer") > adaptive.count("tray")
    assert max(adaptive.count(name) for name in sampler.task_ids) < 20
