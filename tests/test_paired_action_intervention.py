from __future__ import annotations

import numpy as np

from hwr.eval.paired_action_collection import (
    BranchTrace,
    branch_order,
    paired_direction,
    paired_effect_from_traces,
)
from hwr.eval.paired_action_intervention import (
    PAIRED_HORIZONS,
    PairedEpisodeEffect,
    analyze_paired_effects,
    blind_injection_power,
    clopper_pearson_lower,
    clopper_pearson_upper,
    holm_correction,
    paired_cross_moment,
)


def _trace(sign: float, direction: np.ndarray, *, identical: bool = False) -> BranchTrace:
    steps = 17
    normalized = np.zeros((steps, 16), np.float64)
    normalized[:, :14] = sign * 0.5 / np.sqrt(14.0) * direction
    scales = np.asarray((0.18, 0.50, *([0.35] * 12), 1.0, 1.0))
    action = normalized * scales
    proprioception = np.zeros((steps, 37), np.float64)
    for index in range(steps):
        proprioception[index, 6:12] = (
            sign * (index + 1) * direction[2:8] * 0.01
        )
        proprioception[index, 18:24] = (
            sign * (index + 1) * direction[8:14] * 0.01
        )
        proprioception[index, 29:31] = sign * (index + 1) * direction[:2] * 0.01
    if identical:
        action = np.abs(action)
        proprioception = np.abs(proprioception)
    return BranchTrace(
        action,
        action,
        proprioception,
        proprioception.copy(),
        np.zeros(steps),
        np.zeros(steps, np.bool_),
        np.zeros(steps, np.int64),
        np.zeros(steps, np.bool_),
        tuple("[]" for _ in range(steps)),
        np.arange(20, dtype=np.float64),
    )


def _effect(task: str, index: int) -> PairedEpisodeEffect:
    rng = np.random.default_rng(index)
    first = {h: rng.normal(size=14) for h in PAIRED_HORIZONS}
    matrix = np.zeros((14, 16), np.float64)
    matrix[:, :14] = np.eye(14)
    matrix[:2, 14:] = np.eye(2)
    outcome = {
        h: first[h] @ matrix
        for h in PAIRED_HORIZONS
    }
    return PairedEpisodeEffect(
        task,
        index,
        index,
        first,
        outcome,
        (1.0,) * 17,
        0.2,
        True,
        0,
        0,
        False,
    )


def test_direction_and_branch_order_are_deterministic() -> None:
    np.testing.assert_array_equal(paired_direction(1, 7), paired_direction(1, 7))
    assert branch_order(1, 7) == branch_order(1, 7)
    assert set(branch_order(1, 7)) == {"plus", "minus", "sham_a", "sham_b"}


def test_trace_pair_builds_all_horizons_and_exact_sham() -> None:
    direction = np.ones(14, np.float64)
    plus = _trace(1.0, direction)
    minus = _trace(-1.0, direction)
    sham = _trace(1.0, direction, identical=True)

    effect, audit = paired_effect_from_traces(
        "fixture/v1",
        9,
        0,
        direction,
        {"plus": plus, "minus": minus, "sham_a": sham, "sham_b": sham},
    )

    assert effect.sham_equal
    assert set(effect.first_stage) == set(PAIRED_HORIZONS)
    assert set(effect.outcome) == set(PAIRED_HORIZONS)
    assert audit["actual_action_start"] == 0
    assert audit["actual_action_difference_rms"] > 0.1
    assert audit["minimum_direction_cosine"] > 0.999
    assert audit["first_stage_relative_asymmetry"] == 0.0


def test_cross_moment_and_holm_detect_strong_effect() -> None:
    effects = [
        _effect(task, index)
        for task in ("a", "b", "c")
        for index in range(16)
    ]

    report = analyze_paired_effects(effects, permutation_count=999, base_seed=17)

    assert report["passed"]
    assert all(value["observed_statistic"] > 0 for value in report["partitions"].values())
    assert all(value["holm_passed"] for value in report["partitions"].values())
    assert paired_cross_moment(np.eye(4), np.eye(4)) > 0.0
    correction = holm_correction(
        [(f"p{index}", 0.0001) for index in range(12)], alpha=0.05
    )
    assert all(value["passed"] for value in correction.values())


def test_clopper_pearson_bounds_are_conservative() -> None:
    assert clopper_pearson_upper(0, 64, 0.95) < 0.05
    assert clopper_pearson_lower(900, 1000, 0.95) > 0.88


def test_blind_injection_uses_familywise_calibration() -> None:
    effects = [
        _effect(task, index)
        for task in ("a", "b", "c")
        for index in range(8)
    ]

    report = blind_injection_power(
        effects, trials=20, base_seed=23
    )

    assert report["trials"] == 20
    assert len(report["trial_results"]) == 20
    assert 0.0 <= report["null_false_positive_rate"] <= 1.0
    assert 0.0 <= report["planted_power"] <= 1.0
