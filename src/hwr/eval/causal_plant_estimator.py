"""Causal fixed-lag plant estimator for the R0001-P11 head-only gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hwr.core.embodied import (
    DUAL_ARM_ACTION_MAXIMUM,
    DUAL_ARM_ACTION_MINIMUM,
)


PLANT_LAG_CANDIDATES = (0, 1, 2, 3)
PLANT_FEEDBACK_WINDOW = 16
PLANT_STABLE_START = max(PLANT_LAG_CANDIDATES) + PLANT_FEEDBACK_WINDOW


@dataclass(frozen=True)
class CausalPlantPrediction:
    predicted_action: np.ndarray
    selected_lag: np.ndarray
    selected_gain: np.ndarray
    stable: np.ndarray
    feedback_count: np.ndarray

    def __post_init__(self) -> None:
        predicted = np.asarray(self.predicted_action, np.float64)
        selected_lag = np.asarray(self.selected_lag, np.int64)
        selected_gain = np.asarray(self.selected_gain, np.float64)
        stable = np.asarray(self.stable, np.bool_)
        feedback_count = np.asarray(self.feedback_count, np.int64)
        transitions = len(predicted)
        if (
            predicted.shape != (transitions, 16)
            or selected_lag.shape != (transitions,)
            or selected_gain.shape != (transitions,)
            or stable.shape != (transitions,)
            or feedback_count.shape != (transitions,)
            or not np.isfinite(predicted).all()
            or not np.isfinite(selected_gain).all()
        ):
            raise ValueError("causal plant prediction is invalid")
        object.__setattr__(self, "predicted_action", predicted)
        object.__setattr__(self, "selected_lag", selected_lag)
        object.__setattr__(self, "selected_gain", selected_gain)
        object.__setattr__(self, "stable", stable)
        object.__setattr__(self, "feedback_count", feedback_count)


def estimate_causal_plant_actions(
    proposals: np.ndarray,
    applied_actions: np.ndarray,
    safety_interventions: np.ndarray,
) -> CausalPlantPrediction:
    proposal, applied, safety = _validated_episode(
        proposals, applied_actions, safety_interventions
    )
    lower, upper, width = _action_bounds()
    predicted = np.clip(proposal, lower, upper)
    selected_lag = np.full(len(proposal), -1, np.int64)
    selected_gain = np.ones(len(proposal), np.float64)
    stable = np.zeros(len(proposal), np.bool_)
    feedback_count = np.zeros(len(proposal), np.int64)
    for step in range(len(proposal)):
        eligible = np.flatnonzero(
            (~safety[:step])
            & (np.arange(step, dtype=np.int64) >= max(PLANT_LAG_CANDIDATES))
        )
        history = eligible[-PLANT_FEEDBACK_WINDOW:]
        feedback_count[step] = len(history)
        if len(history) < PLANT_FEEDBACK_WINDOW:
            continue
        lag, gain = _select_lag(
            proposal, applied, history, lower, upper, width
        )
        source = proposal[step - lag]
        candidate = source.copy()
        candidate[:14] *= gain
        predicted[step] = np.clip(candidate, lower, upper)
        selected_lag[step] = lag
        selected_gain[step] = gain
        stable[step] = True
    return CausalPlantPrediction(
        predicted, selected_lag, selected_gain, stable, feedback_count
    )


def current_proposal_baseline(proposals: np.ndarray) -> np.ndarray:
    proposal = np.asarray(proposals, np.float64)
    if proposal.ndim != 2 or proposal.shape[1] != 16 or not np.isfinite(proposal).all():
        raise ValueError("current proposal baseline requires finite 16-D actions")
    lower, upper, _ = _action_bounds()
    return np.clip(proposal, lower, upper)


def deterministic_proposal_derangement(
    proposals: np.ndarray, *, seed: int
) -> np.ndarray:
    proposal = np.asarray(proposals, np.float64)
    if (
        seed < 0
        or proposal.ndim != 2
        or proposal.shape[1] != 16
        or len(proposal) < 2
        or not np.isfinite(proposal).all()
    ):
        raise ValueError("proposal derangement input is invalid")
    cycle = np.random.default_rng(seed).permutation(len(proposal))
    sources = np.empty_like(cycle)
    sources[cycle] = np.roll(cycle, 1)
    if np.any(sources == np.arange(len(proposal))):
        raise RuntimeError("proposal derangement contains a fixed point")
    return proposal[sources]


def normalized_action_rmse(
    predicted: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    prediction = np.asarray(predicted, np.float64)
    actual = np.asarray(target, np.float64)
    selected = np.asarray(mask, np.bool_)
    if (
        prediction.shape != actual.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 16
        or selected.shape != (len(prediction),)
        or not np.isfinite(prediction).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("normalized action RMSE input is invalid")
    if not selected.any():
        return float("nan")
    _, _, width = _action_bounds()
    error = (prediction[selected] - actual[selected]) / width
    return float(np.sqrt(np.mean(np.square(error))))


def action_out_of_bounds_rate(actions: np.ndarray) -> float:
    action = np.asarray(actions, np.float64)
    if action.ndim != 2 or action.shape[1] != 16 or not np.isfinite(action).all():
        return 1.0
    lower, upper, _ = _action_bounds()
    return float(np.mean((action < lower) | (action > upper)))


def _select_lag(
    proposal: np.ndarray,
    applied: np.ndarray,
    history: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    width: np.ndarray,
) -> tuple[int, float]:
    candidates = []
    for lag in PLANT_LAG_CANDIDATES:
        source = proposal[history - lag]
        source_motion = source[:, :14] / width[:14]
        target_motion = applied[history, :14] / width[:14]
        denominator = float(np.sum(np.square(source_motion)))
        gain = (
            float(np.sum(source_motion * target_motion) / denominator)
            if denominator > 1.0e-12
            else 1.0
        )
        fitted = source.copy()
        fitted[:, :14] *= gain
        fitted = np.clip(fitted, lower, upper)
        error = (fitted - applied[history]) / width
        candidates.append((float(np.mean(np.square(error))), lag, gain))
    _, lag, gain = min(candidates, key=lambda value: (value[0], value[1]))
    return lag, gain


def _validated_episode(
    proposals: np.ndarray,
    applied_actions: np.ndarray,
    safety_interventions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proposal = np.asarray(proposals, np.float64)
    applied = np.asarray(applied_actions, np.float64)
    safety = np.asarray(safety_interventions, np.bool_)
    if (
        proposal.ndim != 2
        or proposal.shape[1] != 16
        or applied.shape != proposal.shape
        or safety.shape != (len(proposal),)
        or not np.isfinite(proposal).all()
        or not np.isfinite(applied).all()
    ):
        raise ValueError("causal plant Episode is invalid")
    return proposal, applied, safety


def _action_bounds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(DUAL_ARM_ACTION_MINIMUM, np.float64)
    upper = np.asarray(DUAL_ARM_ACTION_MAXIMUM, np.float64)
    return lower, upper, upper - lower
