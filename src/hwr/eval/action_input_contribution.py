"""RSSM transition-input contribution diagnostic for R0001-P20."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch

from hwr.world_model.config import WorldModelConfig
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMSequence


def canonical_normalize_actions(
    actions: torch.Tensor, config: WorldModelConfig
) -> torch.Tensor:
    if actions.shape[-1] != config.action_dimension:
        raise ValueError("canonical action normalization dimension differs")
    lower = actions.new_tensor(config.action_minimum)
    upper = actions.new_tensor(config.action_maximum)
    normalized = 2.0 * (actions - lower) / (upper - lower) - 1.0
    if not torch.isfinite(normalized).all():
        raise ValueError("canonical normalized actions are non-finite")
    return normalized


def evaluate_action_input_contribution(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    executed_actions: torch.Tensor,
) -> dict[str, object]:
    stochastic = sequence.stochastic[:, :-1].detach()
    actions = executed_actions.detach()
    if (
        stochastic.shape[:2] != actions.shape[:2]
        or stochastic.shape[-1] != model.config.stochastic_dimension
        or actions.shape[-1] != model.config.action_dimension
    ):
        raise ValueError("action contribution input shapes are invalid")
    normalized = canonical_normalize_actions(actions, model.config)
    layer = model.rssm.transition_input[0]
    weight = layer.weight.detach()
    stochastic_weight = weight[:, : model.config.stochastic_dimension]
    action_weight = weight[:, model.config.stochastic_dimension :]
    stochastic_contribution = stochastic @ stochastic_weight.T
    raw_contribution = actions @ action_weight.T
    normalized_contribution = normalized @ action_weight.T
    stochastic_rms = _rms(stochastic_contribution)
    raw_rms = _rms(raw_contribution)
    normalized_rms = _rms(normalized_contribution)
    report = {
        "schema_version": "hwr.action-input-contribution/v1",
        "transition_count": int(actions.shape[0] * actions.shape[1]),
        "stochastic_contribution_rms": stochastic_rms,
        "raw_action_contribution_rms": raw_rms,
        "canonical_action_contribution_rms": normalized_rms,
        "raw_action_to_stochastic_ratio": raw_rms / max(stochastic_rms, 1.0e-12),
        "canonical_action_to_stochastic_ratio": (
            normalized_rms / max(stochastic_rms, 1.0e-12)
        ),
        "canonical_to_raw_contribution_gain": (
            normalized_rms / max(raw_rms, 1.0e-12)
        ),
        "raw_action_dimension_rms": _dimension_rms(actions),
        "canonical_action_dimension_rms": _dimension_rms(normalized),
        "canonical_actions_finite": bool(torch.isfinite(normalized).all()),
        "canonical_actions_in_bounds": bool(
            ((normalized >= -1.0 - 1.0e-6) & (normalized <= 1.0 + 1.0e-6)).all()
        ),
        "weights": {
            "stochastic_element_rms": _rms(stochastic_weight),
            "action_element_rms": _rms(action_weight),
            "stochastic_frobenius_norm": float(stochastic_weight.norm().cpu()),
            "action_frobenius_norm": float(action_weight.norm().cpu()),
            "action_column_norms": [
                float(value)
                for value in action_weight.norm(dim=0).cpu()
            ],
        },
    }
    report["assessment"] = assess_action_input_contribution(report)
    return report


def aggregate_action_input_contribution(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(reports) != 24:
        raise ValueError("action contribution diagnostic requires 24 reports")
    names = (
        "stochastic_contribution_rms",
        "raw_action_contribution_rms",
        "canonical_action_contribution_rms",
    )
    values = {
        name: _weighted_mean(reports, name) for name in names
    }
    stochastic = values["stochastic_contribution_rms"]
    raw = values["raw_action_contribution_rms"]
    canonical = values["canonical_action_contribution_rms"]
    aggregate = {
        "schema_version": "hwr.action-input-contribution-aggregate/v1",
        "episode_count": len(reports),
        "transition_count": sum(int(report["transition_count"]) for report in reports),
        **values,
        "raw_action_to_stochastic_ratio": raw / max(stochastic, 1.0e-12),
        "canonical_action_to_stochastic_ratio": canonical / max(stochastic, 1.0e-12),
        "canonical_to_raw_contribution_gain": canonical / max(raw, 1.0e-12),
        "canonical_actions_finite": all(
            report["canonical_actions_finite"] is True for report in reports
        ),
        "canonical_actions_in_bounds": all(
            report["canonical_actions_in_bounds"] is True for report in reports
        ),
        "episodes_passing_contribution_conditions": sum(
            float(report["raw_action_to_stochastic_ratio"]) < 0.20
            and float(report["canonical_to_raw_contribution_gain"]) >= 1.50
            for report in reports
        ),
        "weights": reports[0]["weights"],
        "raw_action_dimension_rms": _mean_dimension_values(
            reports, "raw_action_dimension_rms"
        ),
        "canonical_action_dimension_rms": _mean_dimension_values(
            reports, "canonical_action_dimension_rms"
        ),
    }
    aggregate["assessment"] = assess_action_input_contribution(aggregate)
    return aggregate


def assess_action_input_contribution(
    report: Mapping[str, object],
) -> dict[str, object]:
    checks = {
        "raw_action_to_stochastic_ratio_below_0_20": (
            float(report["raw_action_to_stochastic_ratio"]) < 0.20
        ),
        "canonical_to_raw_gain_at_least_1_50": (
            float(report["canonical_to_raw_contribution_gain"]) >= 1.50
        ),
        "canonical_actions_finite": report["canonical_actions_finite"] is True,
        "canonical_actions_in_bounds": report["canonical_actions_in_bounds"] is True,
    }
    if "episodes_passing_contribution_conditions" in report:
        checks["at_least_20_of_24_episodes_pass"] = (
            int(report["episodes_passing_contribution_conditions"]) >= 20
        )
    return {
        "decision": "diagnostic_passed" if all(checks.values()) else "diagnostic_failed",
        "passed": all(checks.values()),
        "checks": checks,
    }


def _rms(value: torch.Tensor) -> float:
    result = float(value.double().square().mean().sqrt().cpu())
    if not math.isfinite(result):
        raise ValueError("action contribution RMS is non-finite")
    return result


def _dimension_rms(value: torch.Tensor) -> list[float]:
    result = value.double().square().mean(dim=(0, 1)).sqrt().cpu().tolist()
    if any(not math.isfinite(item) for item in result):
        raise ValueError("action dimension RMS is non-finite")
    return [float(item) for item in result]


def _weighted_mean(
    reports: Sequence[Mapping[str, object]], name: str
) -> float:
    counts = np.asarray([int(report["transition_count"]) for report in reports])
    values = np.asarray([float(report[name]) for report in reports])
    return float(np.average(values, weights=counts))


def _mean_dimension_values(
    reports: Sequence[Mapping[str, object]], name: str
) -> list[float]:
    values = np.asarray([report[name] for report in reports], np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("action dimension aggregate is invalid")
    return values.mean(axis=0).tolist()
