"""Monte Carlo decision power for frozen action-identifiability designs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from hwr.train.foundation_action_probe import (
    ACTION_PROBE_BOOTSTRAP_CONTRACT,
    ACTION_PROBE_HORIZONS,
    _synchronized_horizon_bootstrap,
)


POWER_REPORT_SCHEMA = "hwr.action-probe-power/v1"
POWER_PROPOSAL_ID = "R0001-P16"
POWER_ARMS = ("fragmented_7x16", "continuous_same_7_starts", "continuous_all_starts")
POWER_CONDITIONS = ("null", "planted", "permutation")
POWER_TRIALS = 500
POWER_BOOTSTRAP_SAMPLES = 200
POWER_BASE_SEED = 20_260_916
POWER_RIDGE = 1.0e-3
POWER_EFFECT_SCALE = 0.5
POWER_NOISE_RMS = 0.5
POWER_NOISE_CORRELATION = 0.8
POWER_TRANSITIONS = 112
POWER_OUTPUT_DIMENSION = 16


@dataclass(frozen=True)
class PowerEpisode:
    episode_id: str
    task_id: str
    split: str
    correlation: float
    state: np.ndarray
    action: np.ndarray
    content_sha256: str

    def __post_init__(self) -> None:
        state = np.asarray(self.state, np.float64)
        action = np.asarray(self.action, np.float64)
        content = hashlib.sha256(
            state.tobytes() + action.tobytes()
        ).hexdigest()
        if (
            not self.episode_id
            or not self.task_id
            or self.split not in {"training", "holdout"}
            or not 0.0 <= self.correlation < 1.0
            or state.ndim != 2
            or state.shape[0] != POWER_TRANSITIONS + 1
            or action.shape != (POWER_TRANSITIONS, 16)
            or len(self.content_sha256) != 64
            or content != self.content_sha256
            or not np.isfinite(state).all()
            or not np.isfinite(action).all()
        ):
            raise ValueError("action probe power Episode is invalid")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "action", action)


@dataclass(frozen=True)
class _Rows:
    episode_id: str
    state: np.ndarray
    action: np.ndarray
    canonical_indices: np.ndarray
@dataclass(frozen=True)
class _RidgeDesign:
    training_operator: np.ndarray
    evaluation: np.ndarray
    rows: int
    columns: int
    rank: int
    condition_number: float

    @classmethod
    def build(
        cls, training: np.ndarray, evaluation: np.ndarray, *, ridge: float
    ) -> "_RidgeDesign":
        mean = training.mean(axis=0)
        scale = training.std(axis=0)
        scale[scale < 1.0e-8] = 1.0
        fitted = (training - mean) / scale
        evaluated = (evaluation - mean) / scale
        fitted = np.concatenate((fitted, np.ones((len(fitted), 1))), axis=1)
        evaluated = np.concatenate(
            (evaluated, np.ones((len(evaluated), 1))), axis=1
        )
        penalty = np.eye(fitted.shape[1], dtype=np.float64) * ridge
        penalty[-1, -1] = 0.0
        gram = fitted.T @ fitted + penalty
        operator = np.linalg.solve(gram, fitted.T)
        return cls(
            operator,
            evaluated,
            len(fitted),
            fitted.shape[1],
            int(np.linalg.matrix_rank(fitted)),
            float(np.linalg.cond(gram)),
        )

    def predict(self, target: np.ndarray) -> np.ndarray:
        prediction = self.evaluation @ (self.training_operator @ target)
        if not np.isfinite(prediction).all():
            raise ValueError("action probe power prediction is non-finite")
        return prediction
@dataclass(frozen=True)
class _ProbeDesign:
    arm: str
    correlation: float
    task_id: str
    horizon: int
    training_refs: tuple[tuple[str, str, np.ndarray], ...]
    holdout_refs: tuple[tuple[str, str, np.ndarray], ...]
    state_model: _RidgeDesign
    action_model: _RidgeDesign
    holdout_lengths: tuple[int, ...]
def run_action_probe_power(
    episodes: Sequence[PowerEpisode],
    *,
    trials: int = POWER_TRIALS,
    bootstrap_samples: int = POWER_BOOTSTRAP_SAMPLES,
    base_seed: int = POWER_BASE_SEED,
) -> dict[str, object]:
    """Run all frozen null, planted-effect, and permutation trials."""
    if trials <= 0 or bootstrap_samples <= 0 or base_seed < 0:
        raise ValueError("action probe power run dimensions are invalid")
    episode_values = tuple(episodes)
    _require_episode_plan(episode_values)
    designs = _build_designs(episode_values)
    content_check = _content_check(episode_values)
    counters = _new_counters(designs)
    trial_reports = []
    for trial_index in range(trials):
        targets = _trial_targets(episode_values, trial_index, base_seed)
        if not all(
            np.array_equal(
                targets["null"][key], targets["zero_action"][key]
            )
            for key in targets["null"]
        ):
            raise RuntimeError("zero action coefficient differs from null")
        trial_reports.append(
            _evaluate_trial(
                designs,
                targets,
                trial_index=trial_index,
                bootstrap_samples=bootstrap_samples,
                base_seed=base_seed,
                counters=counters,
            )
        )
    summaries = _counter_summaries(counters, trials)
    formal = trials == POWER_TRIALS
    qualified = [
        arm
        for arm, summary in summaries.items()
        if summary["null_false_positive_rate"] <= 0.05
        and summary["planted_detection_power"] >= 0.80
        and summary["permutation_false_positive_rate"] <= 0.05
    ]
    return {
        "schema_version": POWER_REPORT_SCHEMA,
        "proposal_id": POWER_PROPOSAL_ID,
        "mode": "formal" if formal else "smoke",
        "decision": (
            "accepted"
            if formal and qualified
            else "rejected"
            if formal
            else "smoke_only"
        ),
        "p14_route": (
            "eligible"
            if formal and "continuous_all_starts" in qualified
            else "blocked"
            if formal
            else "smoke_only"
        ),
        "trial_count_per_condition": trials,
        "bootstrap_samples_per_trial": bootstrap_samples,
        "base_seed": base_seed,
        "ridge": POWER_RIDGE,
        "effect_scale": POWER_EFFECT_SCALE,
        "noise_rms": POWER_NOISE_RMS,
        "noise_correlation": POWER_NOISE_CORRELATION,
        "output_dimension": POWER_OUTPUT_DIMENSION,
        "bootstrap_contract": ACTION_PROBE_BOOTSTRAP_CONTRACT,
        "arms": summaries,
        "qualified_arms": qualified,
        "content_check": content_check,
        "designs": _design_report(designs),
        "trials": trial_reports,
    }

def arm_indices(arm: str, horizon: int) -> np.ndarray:
    """Return the frozen canonical start indices for one design arm."""
    if arm not in POWER_ARMS or horizon not in ACTION_PROBE_HORIZONS:
        raise ValueError("unknown action probe power design")
    if arm == "continuous_all_starts":
        return np.arange(POWER_TRANSITIONS - horizon + 1, dtype=np.int64)
    if arm == "continuous_same_7_starts":
        return np.arange(0, POWER_TRANSITIONS, 16, dtype=np.int64)
    return np.concatenate(
        [
            np.arange(
                block,
                block + 16 - horizon + 1,
                dtype=np.int64,
            )
            for block in range(0, POWER_TRANSITIONS, 16)
        ]
    )

def _build_designs(
    episodes: Sequence[PowerEpisode],
) -> dict[tuple[str, float, str, int], _ProbeDesign]:
    result = {}
    correlations = sorted({episode.correlation for episode in episodes})
    tasks = sorted({episode.task_id for episode in episodes})
    for arm in POWER_ARMS:
        for correlation in correlations:
            for task_id in tasks:
                selected = [
                    episode
                    for episode in episodes
                    if episode.correlation == correlation
                    and episode.task_id == task_id
                ]
                for horizon in ACTION_PROBE_HORIZONS:
                    training = _design_rows(selected, "training", arm, horizon)
                    holdout = _design_rows(selected, "holdout", arm, horizon)
                    train_state = np.concatenate([row.state for row in training])
                    train_action = np.concatenate([row.action for row in training])
                    test_state = np.concatenate([row.state for row in holdout])
                    test_action = np.concatenate([row.action for row in holdout])
                    result[(arm, correlation, task_id, horizon)] = _ProbeDesign(
                        arm,
                        correlation,
                        task_id,
                        horizon,
                        tuple(
                            ("training", row.episode_id, row.canonical_indices)
                            for row in training
                        ),
                        tuple(
                            ("holdout", row.episode_id, row.canonical_indices)
                            for row in holdout
                        ),
                        _RidgeDesign.build(
                            train_state, test_state, ridge=POWER_RIDGE
                        ),
                        _RidgeDesign.build(
                            np.concatenate((train_state, train_action), axis=1),
                            np.concatenate((test_state, test_action), axis=1),
                            ridge=POWER_RIDGE,
                        ),
                        tuple(len(row.state) for row in holdout),
                    )
    return result

def _design_rows(
    episodes: Sequence[PowerEpisode],
    split: str,
    arm: str,
    horizon: int,
) -> list[_Rows]:
    indices = arm_indices(arm, horizon)
    rows = []
    for episode in episodes:
        if episode.split != split:
            continue
        controllable = _controllable_state(episode.state)
        actions = np.stack(
            [
                episode.action[index : index + horizon].mean(axis=0)
                for index in indices
            ]
        )
        rows.append(
            _Rows(
                episode.episode_id,
                episode.state[indices],
                actions,
                indices,
            )
        )
        if len(controllable[indices + horizon]) != len(indices):
            raise RuntimeError("action probe power target indexing differs")
    if len(rows) != 8:
        raise ValueError("action probe power split requires eight Episodes")
    return rows

def _trial_targets(
    episodes: Sequence[PowerEpisode],
    trial_index: int,
    base_seed: int,
) -> dict[str, dict[tuple[float, str, int, str, str], np.ndarray]]:
    state_rng = np.random.default_rng(
        base_seed + trial_index * 1_000_003 + 11
    )
    action_rng = np.random.default_rng(
        base_seed + trial_index * 1_000_003 + 23
    )
    training_noise_rng = np.random.default_rng(
        base_seed + trial_index * 1_000_003 + 37
    )
    holdout_noise_rng = np.random.default_rng(
        base_seed + trial_index * 1_000_003 + 53
    )
    conditions = {
        name: {} for name in (*POWER_CONDITIONS, "zero_action")
    }
    correlations = sorted({episode.correlation for episode in episodes})
    tasks = sorted({episode.task_id for episode in episodes})
    state_dimensions = {
        task_id: next(
            episode.state.shape[1]
            for episode in episodes
            if episode.task_id == task_id
        )
        for task_id in tasks
    }
    task_state_weights = {
        task_id: _unit_columns(
            state_rng.standard_normal(
                (state_dimensions[task_id], POWER_OUTPUT_DIMENSION)
            )
        )
        for task_id in tasks
    }
    task_action_weights = {
        task_id: _unit_columns(
            action_rng.standard_normal((16, POWER_OUTPUT_DIMENSION))
        )
        for task_id in tasks
    }
    for correlation in correlations:
        for task_id in tasks:
            selected = [
                episode
                for episode in episodes
                if episode.correlation == correlation
                and episode.task_id == task_id
            ]
            state_weights = task_state_weights[task_id]
            action_weights = task_action_weights[task_id]
            episode_noise = {
                (episode.split, episode.episode_id): _ar_noise(
                    (
                        training_noise_rng
                        if episode.split == "training"
                        else holdout_noise_rng
                    ),
                    POWER_TRANSITIONS,
                    POWER_OUTPUT_DIMENSION,
                    POWER_NOISE_CORRELATION,
                )
                for episode in selected
            }
            training_noise_rms = _rms(
                np.concatenate(
                    [
                        episode_noise[(episode.split, episode.episode_id)]
                        for episode in selected
                        if episode.split == "training"
                    ]
                )
            )
            episode_noise = {
                key: value * (POWER_NOISE_RMS / training_noise_rms)
                for key, value in episode_noise.items()
            }
            for horizon in ACTION_PROBE_HORIZONS:
                full = _canonical_rows(selected, horizon)
                training = [value for value in full if value[0].split == "training"]
                state_values = np.concatenate([value[1] for value in training])
                action_values = np.concatenate([value[2] for value in training])
                state_mean, state_scale = _mean_scale(state_values)
                action_mean, action_scale = _mean_scale(action_values)
                signals = {}
                permutation_signals = {}
                noises = {}
                for episode, state, action in full:
                    key = (
                        correlation,
                        task_id,
                        horizon,
                        episode.split,
                        episode.episode_id,
                    )
                    normalized_state = (state - state_mean) / state_scale
                    normalized_action = (action - action_mean) / action_scale
                    signals[key] = (
                        normalized_state @ state_weights,
                        normalized_action @ action_weights,
                    )
                    permutation = action_rng.permutation(len(normalized_action))
                    permutation_signals[key] = (
                        normalized_action[permutation] @ action_weights
                    )
                    indices = arm_indices("continuous_all_starts", horizon)
                    noises[key] = episode_noise[
                        (episode.split, episode.episode_id)
                    ][indices]
                training_keys = [
                    key for key in signals if key[3] == "training"
                ]
                state_rms = _rms(
                    np.concatenate([signals[key][0] for key in training_keys])
                )
                action_rms = _rms(
                    np.concatenate([signals[key][1] for key in training_keys])
                )
                permutation_rms = _rms(
                    np.concatenate(
                        [permutation_signals[key] for key in training_keys]
                    )
                )
                for key, (state_signal, action_signal) in signals.items():
                    state_signal = state_signal / state_rms
                    action_signal = action_signal / action_rms
                    permuted = permutation_signals[key] / permutation_rms
                    noise = noises[key]
                    null = state_signal + noise
                    conditions["null"][key] = null
                    conditions["zero_action"][key] = (
                        state_signal + 0.0 * action_signal + noise
                    )
                    conditions["planted"][key] = (
                        state_signal
                        + POWER_EFFECT_SCALE * action_signal
                        + noise
                    )
                    conditions["permutation"][key] = (
                        state_signal + POWER_EFFECT_SCALE * permuted + noise
                    )
    return conditions

def _canonical_rows(
    episodes: Sequence[PowerEpisode], horizon: int
) -> list[tuple[PowerEpisode, np.ndarray, np.ndarray]]:
    indices = arm_indices("continuous_all_starts", horizon)
    return [
        (
            episode,
            episode.state[indices],
            np.stack(
                [
                    episode.action[index : index + horizon].mean(axis=0)
                    for index in indices
                ]
            ),
        )
        for episode in episodes
    ]

def _evaluate_trial(
    designs: Mapping[tuple[str, float, str, int], _ProbeDesign],
    targets: Mapping[
        str, Mapping[tuple[float, str, int, str, str], np.ndarray]
    ],
    *,
    trial_index: int,
    bootstrap_samples: int,
    base_seed: int,
    counters: dict[str, object],
) -> dict[str, object]:
    report = {"trial_index": trial_index, "arms": {}}
    for arm_index, arm in enumerate(POWER_ARMS):
        report["arms"][arm] = {}
        for condition_index, condition in enumerate(POWER_CONDITIONS):
            condition_result = _evaluate_condition(
                designs,
                targets[condition],
                arm=arm,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=(
                    base_seed
                    + trial_index * 1_000_003
                    + 71
                    + arm_index * 10_000_019
                    + condition_index * 100_003
                ),
            )
            report["arms"][arm][condition] = condition_result["trial"]
            _record_counter(
                counters,
                arm,
                condition,
                condition_result,
            )
    return report

def _evaluate_condition(
    designs: Mapping[tuple[str, float, str, int], _ProbeDesign],
    targets: Mapping[tuple[float, str, int, str, str], np.ndarray],
    *,
    arm: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    correlations = sorted({key[1] for key in designs if key[0] == arm})
    tasks = sorted({key[2] for key in designs if key[0] == arm})
    rho_results = {}
    horizon_pass: dict[tuple[float, str, int], bool] = {}
    task_pass: dict[tuple[float, str], bool] = {}
    for rho_index, correlation in enumerate(correlations):
        conservative_by_task = []
        task_results = {}
        for task_index, task_id in enumerate(tasks):
            errors = {}
            horizons = {}
            for horizon in ACTION_PROBE_HORIZONS:
                design = designs[(arm, correlation, task_id, horizon)]
                training_target = _select_targets(
                    targets, correlation, task_id, horizon, design.training_refs
                )
                holdout_target = _select_targets(
                    targets, correlation, task_id, horizon, design.holdout_refs
                )
                state_prediction = design.state_model.predict(training_target)
                action_prediction = design.action_model.predict(training_target)
                state_episode = _episode_mse(
                    state_prediction,
                    holdout_target,
                    design.holdout_lengths,
                )
                action_episode = _episode_mse(
                    action_prediction,
                    holdout_target,
                    design.holdout_lengths,
                )
                state_mse = float(state_episode.mean())
                action_mse = float(action_episode.mean())
                ratio = state_mse / max(action_mse, 1.0e-12)
                errors[horizon] = (state_episode, action_episode)
                horizons[str(horizon)] = {
                    "ratio": ratio,
                    "state_only_mse": state_mse,
                    "state_action_mse": action_mse,
                }
            ratios, conservative = _synchronized_horizon_bootstrap(
                errors,
                samples=bootstrap_samples,
                seed=(
                    bootstrap_seed
                    + rho_index * 1_000_003
                    + task_index * 104_729
                ),
            )
            for horizon, values in ratios.items():
                p05 = float(np.quantile(values, 0.05))
                horizons[str(horizon)]["p05"] = p05
                horizon_pass[(correlation, task_id, horizon)] = (
                    horizons[str(horizon)]["ratio"] >= 1.05
                    and p05 >= 1.01
                )
            task_p05 = float(np.quantile(conservative, 0.05))
            passed = (
                all(
                    horizon_pass[(correlation, task_id, horizon)]
                    for horizon in ACTION_PROBE_HORIZONS
                )
                and task_p05 >= 1.01
            )
            task_pass[(correlation, task_id)] = passed
            conservative_by_task.append(conservative)
            task_results[task_id] = {
                "passed": passed,
                "synchronized_p05": task_p05,
                "horizons": horizons,
            }
        aggregate = np.stack(conservative_by_task).min(axis=0)
        aggregate_p05 = float(np.quantile(aggregate, 0.05))
        rho_results[str(correlation)] = {
            "passed": (
                all(task_pass[(correlation, task_id)] for task_id in tasks)
                and aggregate_p05 >= 1.01
            ),
            "aggregate_p05": aggregate_p05,
            "tasks": task_results,
        }
    passed = all(value["passed"] for value in rho_results.values())
    return {
        "trial": {
            "passed": passed,
            "rho_pass": {
                rho: value["passed"] for rho, value in rho_results.items()
            },
        },
        "passed": passed,
        "horizon_pass": horizon_pass,
        "task_pass": task_pass,
    }

def _select_targets(
    targets: Mapping[tuple[float, str, int, str, str], np.ndarray],
    correlation: float,
    task_id: str,
    horizon: int,
    references: Sequence[tuple[str, str, np.ndarray]],
) -> np.ndarray:
    return np.concatenate(
        [
            targets[(correlation, task_id, horizon, split, episode_id)][indices]
            for split, episode_id, indices in references
        ]
    )

def _episode_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    lengths: Sequence[int],
) -> np.ndarray:
    errors = np.square(prediction - target).mean(axis=1)
    offsets = np.cumsum((0, *lengths))
    return np.asarray(
        [
            errors[offsets[index] : offsets[index + 1]].mean()
            for index in range(len(lengths))
        ],
        np.float64,
    )

def _new_counters(
    designs: Mapping[tuple[str, float, str, int], _ProbeDesign]
) -> dict[str, object]:
    correlations = sorted({key[1] for key in designs})
    tasks = sorted({key[2] for key in designs})
    return {
        arm: {
            condition: {
                "passed": 0,
                "task_passed": {
                    (correlation, task): 0
                    for correlation in correlations
                    for task in tasks
                },
                "horizon_passed": {
                    (correlation, task, horizon): 0
                    for correlation in correlations
                    for task in tasks
                    for horizon in ACTION_PROBE_HORIZONS
                },
            }
            for condition in POWER_CONDITIONS
        }
        for arm in POWER_ARMS
    }

def _record_counter(
    counters: dict[str, object],
    arm: str,
    condition: str,
    result: Mapping[str, object],
) -> None:
    counter = counters[arm][condition]
    counter["passed"] += int(result["passed"])
    for key, passed in result["task_pass"].items():
        counter["task_passed"][key] += int(passed)
    for key, passed in result["horizon_pass"].items():
        counter["horizon_passed"][key] += int(passed)

def _counter_summaries(
    counters: Mapping[str, object], trials: int
) -> dict[str, object]:
    summaries = {}
    for arm in POWER_ARMS:
        null = counters[arm]["null"]
        planted = counters[arm]["planted"]
        permutation = counters[arm]["permutation"]
        summaries[arm] = {
            "null_false_positive_rate": null["passed"] / trials,
            "planted_detection_power": planted["passed"] / trials,
            "permutation_false_positive_rate": permutation["passed"] / trials,
            "by_condition": {
                condition: {
                    "task_pass_rate": _named_rates(
                        counters[arm][condition]["task_passed"], trials
                    ),
                    "horizon_pass_rate": _named_rates(
                        counters[arm][condition]["horizon_passed"], trials
                    ),
                }
                for condition in POWER_CONDITIONS
            },
        }
    return summaries

def _named_rates(
    values: Mapping[tuple[object, ...], int], trials: int
) -> dict[str, float]:
    return {
        "|".join(str(part) for part in key): count / trials
        for key, count in sorted(values.items(), key=lambda item: str(item[0]))
    }

def _design_report(
    designs: Mapping[tuple[str, float, str, int], _ProbeDesign]
) -> list[dict[str, object]]:
    return [
        {
            "arm": key[0],
            "correlation": key[1],
            "task_id": key[2],
            "horizon": key[3],
            "training_rows": design.state_model.rows,
            "holdout_rows": sum(design.holdout_lengths),
            "state_columns": design.state_model.columns,
            "state_rank": design.state_model.rank,
            "state_condition_number": design.state_model.condition_number,
            "state_action_columns": design.action_model.columns,
            "state_action_rank": design.action_model.rank,
            "state_action_condition_number": (
                design.action_model.condition_number
            ),
        }
        for key, design in sorted(designs.items())
    ]

def _content_check(episodes: Sequence[PowerEpisode]) -> dict[str, object]:
    hashes = {episode.episode_id: episode.content_sha256 for episode in episodes}
    if len(hashes) != len(episodes):
        raise ValueError("action probe power Episode identities are duplicated")
    return {
        "same_112_transition_content_across_arms": True,
        "episode_count": len(episodes),
        "episode_content_sha256": hashes,
    }

def _require_episode_plan(episodes: Sequence[PowerEpisode]) -> None:
    if len(episodes) != 96:
        raise ValueError("action probe power requires 96 P09 Episodes")
    correlations = sorted({episode.correlation for episode in episodes})
    tasks = sorted({episode.task_id for episode in episodes})
    if correlations != [0.5, 0.96] or len(tasks) != 3:
        raise ValueError("action probe power task or rho coverage differs")
    for correlation in correlations:
        for task_id in tasks:
            for split in ("training", "holdout"):
                selected = [
                    episode
                    for episode in episodes
                    if episode.correlation == correlation
                    and episode.task_id == task_id
                    and episode.split == split
                ]
                if len(selected) != 8:
                    raise ValueError("action probe power split coverage differs")

def _controllable_state(proprioception: np.ndarray) -> np.ndarray:
    if proprioception.shape[1] < 31:
        return proprioception
    indices = (*range(6, 12), *range(18, 26), *range(29, 31))
    return proprioception[:, indices]

def _mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    return mean, scale


def _unit_columns(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    if bool((norms <= 1.0e-12).any()):
        raise ValueError("synthetic coefficient column is degenerate")
    return values / norms


def _ar_noise(
    rng: np.random.Generator,
    length: int,
    dimension: int,
    correlation: float,
) -> np.ndarray:
    values = np.empty((length, dimension), np.float64)
    values[0] = rng.standard_normal(dimension)
    innovation_scale = math.sqrt(1.0 - correlation * correlation)
    for index in range(1, length):
        values[index] = (
            correlation * values[index - 1]
            + innovation_scale * rng.standard_normal(dimension)
        )
    return values


def _rms(values: np.ndarray) -> float:
    result = float(np.sqrt(np.mean(np.square(values))))
    if not math.isfinite(result) or result <= 1.0e-12:
        raise ValueError("synthetic signal RMS is invalid")
    return result
