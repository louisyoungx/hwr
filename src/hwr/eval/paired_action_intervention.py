"""Paired physical action intervention statistics for R0001-P17."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PAIRED_INTERVENTION_SCHEMA = "hwr.paired-action-intervention/v1"
PAIRED_HORIZONS = (1, 4, 8, 16)
PAIRED_PERMUTATIONS = 999
PAIRED_INJECTION_TRIALS = 1_000
PAIRED_ALPHA = 0.05
PAIRED_BASE_SEED = 20_261_017


@dataclass(frozen=True)
class PairedEpisodeEffect:
    task_id: str
    seed: int
    episode_index: int
    first_stage: Mapping[int, np.ndarray]
    outcome: Mapping[int, np.ndarray]
    direction_cosines: tuple[float, ...]
    action_difference_rms: float
    sham_equal: bool
    safety_interventions: int
    severe_collisions: int
    terminated_early: bool

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or self.seed < 0
            or self.episode_index < 0
            or set(self.first_stage) != set(PAIRED_HORIZONS)
            or set(self.outcome) != set(PAIRED_HORIZONS)
            or self.action_difference_rms < 0.0
            or self.safety_interventions < 0
            or self.severe_collisions < 0
        ):
            raise ValueError("paired action Episode identity is invalid")
        first_stage = {
            horizon: np.asarray(value, np.float64)
            for horizon, value in self.first_stage.items()
        }
        outcome = {
            horizon: np.asarray(value, np.float64)
            for horizon, value in self.outcome.items()
        }
        if any(value.shape != (14,) for value in first_stage.values()) or any(
            value.shape != (16,) for value in outcome.values()
        ):
            raise ValueError("paired action Episode dimensions are invalid")
        if not all(
            np.isfinite(value).all()
            for value in (*first_stage.values(), *outcome.values())
        ) or not all(math.isfinite(value) for value in self.direction_cosines):
            raise ValueError("paired action Episode contains non-finite values")
        object.__setattr__(self, "first_stage", first_stage)
        object.__setattr__(self, "outcome", outcome)


def analyze_paired_effects(
    episodes: Sequence[PairedEpisodeEffect],
    *,
    permutation_count: int = PAIRED_PERMUTATIONS,
    base_seed: int = PAIRED_BASE_SEED,
) -> dict[str, object]:
    """Run the frozen 12-test confirmation family."""
    if permutation_count <= 0 or base_seed < 0:
        raise ValueError("paired action permutation configuration is invalid")
    values = tuple(episodes)
    tasks = _validate_episodes(values)
    partitions = {}
    p_values = []
    for task_index, task_id in enumerate(tasks):
        selected = tuple(value for value in values if value.task_id == task_id)
        for horizon in PAIRED_HORIZONS:
            first_stage = np.stack(
                [value.first_stage[horizon] for value in selected]
            )
            outcome = np.stack([value.outcome[horizon] for value in selected])
            result = permutation_test(
                first_stage,
                outcome,
                permutations=permutation_count,
                seed=base_seed + task_index * 104_729 + horizon * 1_009,
            )
            name = f"{task_id}|{horizon}"
            partitions[name] = result
            p_values.append((name, float(result["p_value"])))
    holm = holm_correction(p_values, alpha=PAIRED_ALPHA)
    for name, result in partitions.items():
        result["holm_adjusted_p"] = holm[name]["adjusted_p"]
        result["holm_passed"] = holm[name]["passed"]
    passed = all(
        result["observed_statistic"] > 0.0 and result["holm_passed"]
        for result in partitions.values()
    )
    return {
        "schema_version": PAIRED_INTERVENTION_SCHEMA,
        "permutations": permutation_count,
        "alpha": PAIRED_ALPHA,
        "episode_count": len(values),
        "tasks": list(tasks),
        "horizons": list(PAIRED_HORIZONS),
        "partitions": partitions,
        "holm": holm,
        "passed": passed,
    }


def permutation_test(
    first_stage: np.ndarray,
    outcome: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """Run the exact Episode-level sign-flip test."""
    x = np.asarray(first_stage, np.float64)
    y = np.asarray(outcome, np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or len(x) < 8
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("paired action regression inputs are invalid")
    observed = paired_cross_moment(x, y)
    rng = np.random.default_rng(seed)
    null_statistics = np.empty(permutations, np.float64)
    for index in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(len(y), 1))
        null_statistics[index] = paired_cross_moment(x, signs * y)
    p_value = (1 + int(np.sum(null_statistics >= observed))) / (
        permutations + 1
    )
    return {
        "observed_statistic": observed,
        "p_value": p_value,
        "null_p95": float(np.quantile(null_statistics, 0.95)),
        "null_maximum": float(null_statistics.max()),
        "first_stage_rms": float(np.sqrt(np.mean(np.square(x)))),
        "outcome_rms": float(np.sqrt(np.mean(np.square(y)))),
    }


def paired_cross_moment(first_stage: np.ndarray, outcome: np.ndarray) -> float:
    x = np.asarray(first_stage, np.float64)
    y = np.asarray(outcome, np.float64)
    x_energy = float(np.square(x).sum(axis=1).mean())
    y_energy = float(np.square(y).sum(axis=1).mean())
    if min(x_energy, y_energy) <= 1.0e-12:
        return 0.0
    moment = x.T @ y / len(x)
    return float(np.square(moment).sum() / (x_energy * y_energy))


def holm_correction(
    p_values: Sequence[tuple[str, float]], *, alpha: float
) -> dict[str, dict[str, object]]:
    if (
        not p_values
        or not 0.0 < alpha < 1.0
        or len({name for name, _ in p_values}) != len(p_values)
        or any(not 0.0 <= value <= 1.0 for _, value in p_values)
    ):
        raise ValueError("Holm correction inputs are invalid")
    ordered = sorted(p_values, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted = {}
    running = 0.0
    rejected = True
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, (count - index) * p_value)
        threshold = alpha / (count - index)
        rejected = rejected and p_value <= threshold
        adjusted[name] = {
            "raw_p": p_value,
            "adjusted_p": min(1.0, running),
            "threshold": threshold,
            "passed": rejected,
        }
    return adjusted


def blind_injection_power(
    episodes: Sequence[PairedEpisodeEffect],
    *,
    trials: int = PAIRED_INJECTION_TRIALS,
    base_seed: int = PAIRED_BASE_SEED,
) -> dict[str, object]:
    """Measure family-wise null FPR and planted-effect power."""
    if trials <= 0 or base_seed < 0:
        raise ValueError("paired action injection configuration is invalid")
    values = tuple(episodes)
    tasks = _validate_episodes(values)
    response = _response_matrices(tasks, base_seed)
    keys = [
        (task_id, horizon)
        for task_id in tasks
        for horizon in PAIRED_HORIZONS
    ]
    first_stage = {
        key: np.stack(
            [
                value.first_stage[key[1]]
                for value in values
                if value.task_id == key[0]
            ]
        )
        for key in keys
    }
    null_statistics = {key: np.empty(trials, np.float64) for key in keys}
    planted_statistics = {key: np.empty(trials, np.float64) for key in keys}
    for trial in range(trials):
        rng = np.random.default_rng(base_seed + trial * 1_000_003 + 53)
        for key in keys:
            x = first_stage[key]
            normalized = x / np.maximum(
                np.sqrt(np.square(x).mean(axis=1, keepdims=True)), 1.0e-12
            )
            noise = rng.standard_normal((len(x), 16))
            noise *= 0.5 / max(
                float(np.sqrt(np.mean(np.square(noise)))), 1.0e-12
            )
            signal = 0.5 * (normalized @ response[key])
            null_statistics[key][trial] = paired_cross_moment(x, noise)
            planted_statistics[key][trial] = paired_cross_moment(
                x, signal + noise
            )
    null_passes = 0
    planted_passes = 0
    trial_results = []
    for trial in range(trials):
        null_p = []
        planted_p = []
        for key in keys:
            null_reference = np.delete(null_statistics[key], trial)
            null_p.append(
                (
                    f"{key[0]}|{key[1]}",
                    (1 + int(np.sum(null_reference >= null_statistics[key][trial])))
                    / trials,
                )
            )
            planted_p.append(
                (
                    f"{key[0]}|{key[1]}",
                    (
                        1
                        + int(
                            np.sum(
                                null_statistics[key]
                                >= planted_statistics[key][trial]
                            )
                        )
                    )
                    / (trials + 1),
                )
            )
        null_passed = all(
            value["passed"]
            for value in holm_correction(null_p, alpha=PAIRED_ALPHA).values()
        )
        planted_passed = all(
            value["passed"]
            for value in holm_correction(planted_p, alpha=PAIRED_ALPHA).values()
        )
        null_passes += int(null_passed)
        planted_passes += int(planted_passed)
        trial_results.append(
            {
                "trial": trial,
                "null_passed": null_passed,
                "planted_passed": planted_passed,
            }
        )
    null_upper = clopper_pearson_upper(null_passes, trials, 0.95)
    power_lower = clopper_pearson_lower(planted_passes, trials, 0.95)
    return {
        "trials": trials,
        "null_calibration_trials": trials,
        "null_passes": null_passes,
        "planted_passes": planted_passes,
        "null_false_positive_rate": null_passes / trials,
        "null_false_positive_rate_upper_95": null_upper,
        "planted_power": planted_passes / trials,
        "planted_power_lower_95": power_lower,
        "passed": null_upper <= 0.05 and power_lower >= 0.80,
        "trial_results": trial_results,
    }


def clopper_pearson_upper(successes: int, trials: int, confidence: float) -> float:
    if successes < 0 or successes > trials or trials <= 0:
        raise ValueError("binomial count is invalid")
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    return _bisect_probability(
        lambda probability: _binomial_cdf(successes, trials, probability),
        alpha,
        lower=successes / trials,
        upper=1.0,
        decreasing=True,
    )


def clopper_pearson_lower(successes: int, trials: int, confidence: float) -> float:
    if successes < 0 or successes > trials or trials <= 0:
        raise ValueError("binomial count is invalid")
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    return _bisect_probability(
        lambda probability: 1.0
        - _binomial_cdf(successes - 1, trials, probability),
        alpha,
        lower=0.0,
        upper=successes / trials,
        decreasing=False,
    )


def _validate_episodes(
    episodes: Sequence[PairedEpisodeEffect],
) -> tuple[str, ...]:
    tasks = tuple(sorted({value.task_id for value in episodes}))
    if not tasks or any(
        len([value for value in episodes if value.task_id == task]) < 4
        for task in tasks
    ):
        raise ValueError("paired action task coverage is insufficient")
    return tasks


def _response_matrices(
    tasks: Sequence[str], base_seed: int
) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for task_index, task_id in enumerate(tasks):
        for horizon in PAIRED_HORIZONS:
            rng = np.random.default_rng(
                base_seed + task_index * 104_729 + horizon * 1_009
            )
            matrix = rng.standard_normal((14, 16))
            norms = np.linalg.norm(matrix, axis=0, keepdims=True)
            result[(task_id, horizon)] = matrix / norms
    return result


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return float(successes >= trials)
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * math.log(probability)
        + (trials - index) * math.log1p(-probability)
        for index in range(successes + 1)
    ]
    maximum = max(logs)
    return math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)


def _bisect_probability(
    function,
    target: float,
    *,
    lower: float,
    upper: float,
    decreasing: bool,
) -> float:
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        value = function(midpoint)
        move_lower = value > target if decreasing else value < target
        if move_lower:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0
