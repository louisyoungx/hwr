"""Task-blind linear probe for action identifiability in autonomous replay."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np


ACTION_PROBE_SCHEMA = "hwr.foundation-data-action-probe/v1"


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
    """Compare state-only and state+action prediction of proprioceptive deltas."""
    if (
        ridge <= 0.0
        or bootstrap_samples <= 0
        or bootstrap_seed < 0
        or maximum_training_transitions <= 0
    ):
        raise ValueError("foundation action probe configuration is invalid")
    train_state, train_action, train_target = _load_transitions(
        training_path, training_manifest, maximum_training_transitions
    )
    test_state, test_action, test_target = _load_transitions(
        holdout_path, holdout_manifest, None
    )
    state_only = _fit_predict_ridge(
        train_state, train_target, test_state, ridge=ridge
    )
    state_action = _fit_predict_ridge(
        np.concatenate((train_state, train_action), axis=1),
        train_target,
        np.concatenate((test_state, test_action), axis=1),
        ridge=ridge,
    )
    state_errors = np.square(state_only - test_target).mean(axis=1)
    action_errors = np.square(state_action - test_target).mean(axis=1)
    state_mse = float(state_errors.mean())
    action_mse = float(action_errors.mean())
    ratios = _bootstrap_ratios(
        state_errors,
        action_errors,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "schema_version": ACTION_PROBE_SCHEMA,
        "inputs": {
            "state_only": ["proprioception"],
            "state_action": ["proprioception", "actual_executed_action"],
            "target": "next_proprioception_delta",
            "task_semantic_fields": [],
        },
        "training_transition_count": int(len(train_state)),
        "holdout_transition_count": int(len(test_state)),
        "state_only_mse": state_mse,
        "state_action_mse": action_mse,
        "state_only_to_state_action_ratio": state_mse / max(action_mse, 1.0e-12),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "ratio_p05": float(np.quantile(ratios, 0.05)),
            "ratio_median": float(np.median(ratios)),
            "ratio_p95": float(np.quantile(ratios, 0.95)),
        },
    }


def _load_transitions(
    root: Path,
    manifest: Mapping[str, object],
    maximum: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = []
    actions = []
    targets = []
    for shard in manifest.get("shards", ()):
        path = root / str(shard["path"])
        with np.load(path, allow_pickle=False) as arrays:
            proprioception = arrays["proprioception"].astype(np.float64)
            executed = arrays["executed_action"].astype(np.float64)
        states.append(proprioception[:-1])
        actions.append(executed)
        targets.append(np.diff(proprioception, axis=0))
    if not states:
        raise ValueError("foundation action probe replay is empty")
    state = np.concatenate(states)
    action = np.concatenate(actions)
    target = np.concatenate(targets)
    if maximum is not None and len(state) > maximum:
        indices = np.linspace(0, len(state) - 1, maximum).round().astype(np.int64)
        state, action, target = state[indices], action[indices], target[indices]
    if len(state) < 2 or not all(
        np.isfinite(value).all() for value in (state, action, target)
    ):
        raise ValueError("foundation action probe transitions are invalid")
    return state, action, target


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


def _bootstrap_ratios(
    state_errors: np.ndarray,
    action_errors: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(samples, np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(state_errors), len(state_errors))
        numerator = float(state_errors[selected].mean())
        denominator = float(action_errors[selected].mean())
        result[index] = numerator / max(denominator, 1.0e-12)
    if not all(math.isfinite(value) and value >= 0.0 for value in result):
        raise ValueError("foundation action probe bootstrap is invalid")
    return result
