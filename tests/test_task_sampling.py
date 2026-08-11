from __future__ import annotations

import numpy as np

from hwr.train import (
    OutcomeAdaptiveTaskSampler,
    OutcomeAdaptiveTaskSamplingConfig,
    TaskOutcome,
)


def _outcome(
    *,
    episode_return: float = -1.0,
    novelty: float = 0.5,
    td_error: float = 0.5,
    improvement: float = 0.0,
    failure: float = 1.0,
) -> TaskOutcome:
    return TaskOutcome(
        episode_return,
        novelty,
        td_error,
        improvement,
        failure,
        success=not bool(failure),
        safety_cost_rate=0.0,
    )


def test_sampler_prioritizes_generic_learning_pressure_without_starvation() -> None:
    config = OutcomeAdaptiveTaskSamplingConfig(
        initial_cycles=1, minimum_probability=0.10
    )
    sampler = OutcomeAdaptiveTaskSampler(("a", "b", "c"), config)
    for _ in range(6):
        sampler.record("a", _outcome(novelty=0.9, td_error=0.9, improvement=-1.0))
        sampler.record("b", _outcome(novelty=0.5, td_error=0.5, improvement=0.0))
        sampler.record("c", _outcome(novelty=0.1, td_error=0.1, improvement=1.0, failure=0.0))

    probabilities = sampler.probabilities()

    assert probabilities["a"] > probabilities["b"] > probabilities["c"]
    assert all(value >= 0.10 for value in probabilities.values())
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert sampler.audit()["distance_thresholds"] is False
    assert sampler.audit()["task_semantic_fields"] == []


def test_sampler_initial_coverage_state_and_reward_improvement() -> None:
    config = OutcomeAdaptiveTaskSamplingConfig(initial_cycles=1)
    sampler = OutcomeAdaptiveTaskSampler(("a", "b", "c"), config)
    rng = np.random.default_rng(7)

    initial = [sampler.sample(rng)[0] for _ in range(3)]
    sampler.record("a", _outcome(episode_return=2.0))
    assert sampler.reward_improvement("a", 3.5) == 1.5
    restored = OutcomeAdaptiveTaskSampler(("a", "b", "c"), config)
    restored.load_state_dict(sampler.state_dict())

    assert initial == ["a", "b", "c"]
    assert restored.state_dict() == sampler.state_dict()
    assert sampler.audit()["actor_input_fields"] == []


def test_sampler_uses_weighted_fair_credits_not_random_luck() -> None:
    config = OutcomeAdaptiveTaskSamplingConfig(initial_cycles=1)
    sampler = OutcomeAdaptiveTaskSampler(("a", "b", "c"), config)
    for _ in range(6):
        sampler.record("a", _outcome(novelty=0.9, td_error=0.9, improvement=-1.0))
        sampler.record("b", _outcome(novelty=0.5, td_error=0.5))
        sampler.record("c", _outcome(novelty=0.1, td_error=0.1, improvement=1.0, failure=0.0))
    selected = [sampler.sample(np.random.default_rng(999))[0] for _ in range(33)]
    adaptive = selected[3:]

    assert adaptive.count("a") > adaptive.count("b") > adaptive.count("c")
    assert max(adaptive.count(name) for name in sampler.task_ids) < 22


def test_sampler_discards_only_changed_task_history() -> None:
    sampler = OutcomeAdaptiveTaskSampler(("a", "b", "c"))
    sampler.record("a", _outcome())
    sampler.record("c", _outcome())
    sampler.credits["c"] = 0.4

    discarded = sampler.discard_tasks(("c",))

    assert discarded["c"] == {"history_count": 1, "credit": 0.4}
    assert len(sampler.history["a"]) == 1
    assert len(sampler.history["c"]) == 0


def test_sampler_discards_legacy_geometry_histories() -> None:
    sampler = OutcomeAdaptiveTaskSampler(("a",))
    legacy = sampler.state_dict()
    legacy.pop("schema_version")
    legacy["history"]["a"] = [{"minimum_left_reach_distance": 0.1}]
    legacy["sample_count"] = 19
    legacy["credits"]["a"] = 0.7

    sampler.load_state_dict(legacy)

    assert not sampler.history["a"]
    assert sampler.legacy_discarded_outcome_count == 1
    assert sampler.sample_count == 0
    assert sampler.credits["a"] == 0.0
