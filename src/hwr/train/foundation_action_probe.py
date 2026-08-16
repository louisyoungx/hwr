"""Per-task, Episode-clustered probe for action identifiability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.train.foundation_sequence_reservoir import source_episode_id


ACTION_PROBE_SCHEMA = "hwr.foundation-data-action-probe/v4"
ACTION_PROBE_HORIZONS = (1, 4, 8, 16)
ACTION_PROBE_BOOTSTRAP_CONTRACT = (
    "shared-holdout-episode-multiplicity-across-horizons/v1"
)


@dataclass(frozen=True)
class _EpisodeTransitions:
    task_id: str
    episode_id: str
    state: np.ndarray
    action: np.ndarray
    target: np.ndarray


def evaluate_foundation_data_action_probe(
    training_path: Path,
    training_manifest: Mapping[str, object],
    holdout_path: Path,
    holdout_manifest: Mapping[str, object],
    *,
    ridge: float = 1.0e-3,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 0,
    maximum_training_transitions: int = 20_000,
) -> dict[str, object]:
    """Compare state-only and state+action models independently for every task."""
    if (
        ridge <= 0.0
        or bootstrap_samples <= 0
        or bootstrap_seed < 0
        or maximum_training_transitions <= 0
    ):
        raise ValueError("foundation action probe configuration is invalid")
    training_by_horizon = {
        horizon: _load_episodes(training_path, training_manifest, horizon=horizon)
        for horizon in ACTION_PROBE_HORIZONS
    }
    holdout_by_horizon = {
        horizon: _load_episodes(holdout_path, holdout_manifest, horizon=horizon)
        for horizon in ACTION_PROBE_HORIZONS
    }
    training = training_by_horizon[1]
    holdout = holdout_by_horizon[1]
    task_ids = tuple(sorted({item.task_id for item in training}))
    if not task_ids or {item.task_id for item in holdout} != set(task_ids):
        raise ValueError("foundation action probe task coverage differs")
    quota = maximum_training_transitions // len(task_ids)
    if quota <= 0:
        raise ValueError("foundation action probe training limit is too small")
    partitions: dict[str, dict[str, object]] = {}
    bootstrap_by_task: list[np.ndarray] = []
    for task_index, task_id in enumerate(task_ids):
        horizon_reports: dict[str, dict[str, object]] = {}
        horizon_episode_errors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        holdout_episode_ids: tuple[str, ...] | None = None
        for horizon in ACTION_PROBE_HORIZONS:
            train_horizon = training_by_horizon[horizon]
            holdout_horizon = holdout_by_horizon[horizon]
            holdout_task = [
                item for item in holdout_horizon if item.task_id == task_id
            ]
            episode_ids = tuple(item.episode_id for item in holdout_task)
            if holdout_episode_ids is None:
                holdout_episode_ids = episode_ids
            elif episode_ids != holdout_episode_ids:
                raise ValueError(
                    "foundation action probe holdout Episodes differ across horizons"
                )
            report, state_episode, action_episode = _evaluate_partition(
                _limit_episodes(
                    [item for item in train_horizon if item.task_id == task_id],
                    quota,
                ),
                holdout_task,
                ridge=ridge,
            )
            horizon_reports[str(horizon)] = report
            horizon_episode_errors[horizon] = (state_episode, action_episode)
        task_bootstrap_seed = bootstrap_seed + task_index * 104_729
        horizon_ratios, conservative = _synchronized_horizon_bootstrap(
            horizon_episode_errors,
            samples=bootstrap_samples,
            seed=task_bootstrap_seed,
        )
        for horizon, ratios in horizon_ratios.items():
            horizon_reports[str(horizon)]["bootstrap"] = _bootstrap_summary(
                ratios,
                bootstrap_samples,
                task_bootstrap_seed,
                reduction="none",
            )
        one_step = horizon_reports["1"]
        partitions[task_id] = {
            **one_step,
            "state_only_to_state_action_ratio": float(
                min(
                    value["state_only_to_state_action_ratio"]
                    for value in horizon_reports.values()
                )
            ),
            "bootstrap": _bootstrap_summary(
                conservative,
                bootstrap_samples,
                task_bootstrap_seed,
                reduction="minimum_across_horizons_within_replicate",
            ),
            "horizons": horizon_reports,
        }
        bootstrap_by_task.append(conservative)
    aggregate_ratios = np.stack(bootstrap_by_task).min(axis=0)
    state_mse = float(np.mean([value["state_only_mse"] for value in partitions.values()]))
    action_mse = float(np.mean([value["state_action_mse"] for value in partitions.values()]))
    return {
        "schema_version": ACTION_PROBE_SCHEMA,
        "partition_key": "task_id",
        "inputs": {
            "state_only": ["proprioception"],
            "state_action": ["proprioception", "actual_executed_action"],
            "target": "controllable_state_change",
            "controllable_state": [
                "joint_velocity",
                "gripper_position",
                "base_twist",
            ],
            "horizons": list(ACTION_PROBE_HORIZONS),
            "task_semantic_fields": [],
        },
        "training_transition_count": sum(
            int(value["training_transition_count"]) for value in partitions.values()
        ),
        "holdout_transition_count": sum(
            int(value["holdout_transition_count"]) for value in partitions.values()
        ),
        "state_only_mse": state_mse,
        "state_action_mse": action_mse,
        "state_only_to_state_action_ratio": state_mse / max(action_mse, 1.0e-12),
        "bootstrap": _bootstrap_summary(
            aggregate_ratios,
            bootstrap_samples,
            bootstrap_seed,
            reduction="minimum_across_tasks_and_horizons_within_replicate",
        ),
        "bootstrap_provenance": {
            "contract": ACTION_PROBE_BOOTSTRAP_CONTRACT,
            "episode_alignment_key": ["task_id", "holdout_episode_id"],
            "resampling_unit": "holdout_episode",
            "within_task_horizon_coupling": (
                "shared_episode_multiplicity_per_replicate"
            ),
            "within_task_reduction": "minimum_horizon_ratio_per_replicate",
            "task_seed_derivation": "base_seed + sorted_task_index * 104729",
            "across_task_coupling": "independent_episode_resampling",
            "across_task_reduction": "minimum_task_ratio_per_replicate",
        },
        "partitions": partitions,
    }


def _evaluate_partition(
    training: Sequence[_EpisodeTransitions],
    holdout: Sequence[_EpisodeTransitions],
    *,
    ridge: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    if not training or not holdout:
        raise ValueError("foundation action probe partition is empty")
    train_state = np.concatenate([item.state for item in training])
    train_action = np.concatenate([item.action for item in training])
    train_target = np.concatenate([item.target for item in training])
    test_state = np.concatenate([item.state for item in holdout])
    test_action = np.concatenate([item.action for item in holdout])
    test_target = np.concatenate([item.target for item in holdout])
    state_only = _fit_predict_ridge(train_state, train_target, test_state, ridge=ridge)
    state_action = _fit_predict_ridge(
        np.concatenate((train_state, train_action), axis=1),
        train_target,
        np.concatenate((test_state, test_action), axis=1),
        ridge=ridge,
    )
    state_errors = np.square(state_only - test_target).mean(axis=1)
    action_errors = np.square(state_action - test_target).mean(axis=1)
    episode_lengths = tuple(len(item.state) for item in holdout)
    state_episode = _episode_means(state_errors, episode_lengths)
    action_episode = _episode_means(action_errors, episode_lengths)
    state_mse = float(state_episode.mean())
    action_mse = float(action_episode.mean())
    return {
        "training_episode_count": len(training),
        "training_transition_count": len(train_state),
        "holdout_episode_count": len(holdout),
        "holdout_transition_count": len(test_state),
        "holdout_episode_ids": [item.episode_id for item in holdout],
        "episode_weighting": "uniform",
        "state_only_mse": state_mse,
        "state_action_mse": action_mse,
        "state_only_to_state_action_ratio": state_mse / max(action_mse, 1.0e-12),
    }, state_episode, action_episode


def _load_episodes(
    root: Path, manifest: Mapping[str, object], *, horizon: int
) -> list[_EpisodeTransitions]:
    if horizon <= 0:
        raise ValueError("foundation action probe horizon is invalid")
    grouped: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for shard in manifest.get("shards", ()):
        task_id = str(shard.get("task_id", ""))
        episode_id = source_episode_id(shard)
        if not task_id:
            raise ValueError("foundation action probe shard has no task identity")
        with np.load(root / str(shard["path"]), allow_pickle=False) as arrays:
            proprioception = arrays["proprioception"].astype(np.float64)
            executed = arrays["executed_action"].astype(np.float64)
        if len(executed) < horizon:
            continue
        controllable = _controllable_state(proprioception)
        item = (
            proprioception[:-horizon],
            np.stack(
                [executed[index : index + horizon].mean(axis=0)
                 for index in range(len(executed) - horizon + 1)]
            ),
            controllable[horizon:] - controllable[:-horizon],
        )
        grouped.setdefault((task_id, episode_id), []).append(item)
    result = [
        _EpisodeTransitions(
            task_id,
            episode_id,
            *(np.concatenate([value[index] for value in values]) for index in range(3)),
        )
        for (task_id, episode_id), values in sorted(grouped.items())
    ]
    for item in result:
        if len(item.state) < 2 or not all(
            np.isfinite(value).all() for value in (item.state, item.action, item.target)
        ):
            raise ValueError("foundation action probe transitions are invalid")
    if not result:
        raise ValueError("foundation action probe replay is empty")
    return result


def _controllable_state(proprioception: np.ndarray) -> np.ndarray:
    if proprioception.shape[1] < 31:
        return proprioception
    indices = (*range(6, 12), *range(18, 26), *range(29, 31))
    return proprioception[:, indices]


def _limit_episodes(
    episodes: Sequence[_EpisodeTransitions], maximum: int
) -> list[_EpisodeTransitions]:
    if not episodes:
        return []
    quota, remainder = divmod(maximum, len(episodes))
    result = []
    for index, item in enumerate(episodes):
        count = min(len(item.state), quota + int(index < remainder))
        if count < 2:
            raise ValueError("foundation action probe episode quota is too small")
        indices = np.linspace(0, len(item.state) - 1, count).round().astype(np.int64)
        result.append(
            _EpisodeTransitions(
                item.task_id,
                item.episode_id,
                item.state[indices],
                item.action[indices],
                item.target[indices],
            )
        )
    return result


def _fit_predict_ridge(
    training: np.ndarray,
    target: np.ndarray,
    evaluation: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    fitted = (training - mean) / scale
    evaluated = (evaluation - mean) / scale
    fitted = np.concatenate((fitted, np.ones((len(fitted), 1))), axis=1)
    evaluated = np.concatenate((evaluated, np.ones((len(evaluated), 1))), axis=1)
    penalty = np.eye(fitted.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(fitted.T @ fitted + penalty, fitted.T @ target)
    prediction = evaluated @ weights
    if not np.isfinite(prediction).all():
        raise ValueError("foundation action probe prediction is non-finite")
    return prediction


def _episode_means(errors: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    offsets = np.cumsum((0, *lengths))
    return np.asarray(
        [errors[offsets[index] : offsets[index + 1]].mean() for index in range(len(lengths))],
        np.float64,
    )


def _bootstrap_episode_ratios(
    state_errors: np.ndarray,
    action_errors: np.ndarray,
    *,
    selected: np.ndarray,
) -> np.ndarray:
    if (
        state_errors.shape != action_errors.shape
        or state_errors.ndim != 1
        or selected.ndim != 2
        or selected.shape[1] != len(state_errors)
    ):
        raise ValueError("foundation action probe bootstrap inputs are invalid")
    numerator = state_errors[selected].mean(axis=1)
    denominator = action_errors[selected].mean(axis=1)
    result = numerator / np.maximum(denominator, 1.0e-12)
    if not all(math.isfinite(value) and value >= 0.0 for value in result):
        raise ValueError("foundation action probe bootstrap is invalid")
    return result


def _synchronized_horizon_bootstrap(
    episode_errors: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    samples: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if not episode_errors or samples <= 0 or seed < 0:
        raise ValueError("foundation action probe bootstrap configuration is invalid")
    episode_counts = {
        len(state_errors) for state_errors, _ in episode_errors.values()
    }
    if len(episode_counts) != 1 or not episode_counts or min(episode_counts) <= 0:
        raise ValueError(
            "foundation action probe bootstrap Episodes differ across horizons"
        )
    episode_count = episode_counts.pop()
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, episode_count, size=(samples, episode_count))
    ratios = {
        horizon: _bootstrap_episode_ratios(
            state_errors,
            action_errors,
            selected=selected,
        )
        for horizon, (state_errors, action_errors) in episode_errors.items()
    }
    conservative = np.stack(tuple(ratios.values())).min(axis=0)
    return ratios, conservative


def _bootstrap_summary(
    ratios: np.ndarray,
    samples: int,
    seed: int,
    *,
    reduction: str,
) -> dict[str, object]:
    return {
        "unit": "episode_cluster",
        "samples": samples,
        "seed": seed,
        "resampling_contract": ACTION_PROBE_BOOTSTRAP_CONTRACT,
        "replicate_reduction": reduction,
        "ratio_p05": float(np.quantile(ratios, 0.05)),
        "ratio_median": float(np.median(ratios)),
        "ratio_p95": float(np.quantile(ratios, 0.95)),
    }
