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
    return 2.0 * (actions - lower) / (upper - lower) - 1.0


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
    contributions = {
        "stochastic": _contribution_metrics(stochastic_contribution),
        "raw_action": _contribution_metrics(raw_contribution),
        "canonical_action": _contribution_metrics(normalized_contribution),
    }
    stochastic_rms = contributions["stochastic"]["variation_rms"]
    raw_rms = contributions["raw_action"]["variation_rms"]
    normalized_rms = contributions["canonical_action"]["variation_rms"]
    bounds = _canonical_bounds(normalized)
    report = {
        "schema_version": "hwr.action-input-contribution/v2",
        "transition_count": int(actions.shape[0] * actions.shape[1]),
        **_flatten_contributions(contributions),
        "raw_action_variation_to_stochastic_ratio": _ratio(raw_rms, stochastic_rms),
        "canonical_action_variation_to_stochastic_ratio": _ratio(
            normalized_rms, stochastic_rms
        ),
        "canonical_to_raw_variation_contribution_gain": _ratio(
            normalized_rms, raw_rms
        ),
        "raw_action_dimension_rms": _dimension_rms(actions),
        "canonical_action_dimension_rms": _dimension_rms(normalized),
        **bounds,
        "weights": {
            "stochastic_element_rms": _rms(stochastic_weight),
            "action_element_rms": _rms(action_weight),
            "stochastic_frobenius_norm": _norm(stochastic_weight),
            "action_frobenius_norm": _norm(action_weight),
            "stochastic_column_norms": _column_norms(stochastic_weight),
            "action_column_norms": _column_norms(action_weight),
        },
        "bias": {
            "element_rms": _rms(layer.bias),
            "norm": _norm(layer.bias),
        },
    }
    report["assessment"] = assess_action_input_contribution(report)
    return report


def aggregate_action_input_contribution(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(reports) != 24:
        raise ValueError("action contribution diagnostic requires 24 reports")
    _validate_episode_identities(reports)
    names = (
        "absolute_stochastic_contribution_rms",
        "absolute_raw_action_contribution_rms",
        "absolute_canonical_action_contribution_rms",
        "stochastic_dc_contribution_rms",
        "raw_action_dc_contribution_rms",
        "canonical_action_dc_contribution_rms",
        "stochastic_variation_contribution_rms",
        "raw_action_variation_contribution_rms",
        "canonical_action_variation_contribution_rms",
    )
    values = {name: _pooled_rms(reports, name) for name in names}
    stochastic = values["stochastic_variation_contribution_rms"]
    raw = values["raw_action_variation_contribution_rms"]
    canonical = values["canonical_action_variation_contribution_rms"]
    dimension_count = len(reports[0]["canonical_action_dimension_rms"])
    finite_counts = _sum_dimension_counts(
        reports, "canonical_action_dimension_finite_count", dimension_count
    )
    in_bounds_counts = _sum_dimension_counts(
        reports, "canonical_action_dimension_in_bounds_count", dimension_count
    )
    out_of_bounds_counts = _sum_dimension_counts(
        reports, "canonical_action_dimension_out_of_bounds_count", dimension_count
    )
    nonfinite_counts = _sum_dimension_counts(
        reports, "canonical_action_dimension_nonfinite_count", dimension_count
    )
    transition_count = sum(int(report["transition_count"]) for report in reports)
    value_count = transition_count * dimension_count
    if any(
        report["weights"] != reports[0]["weights"]
        or report["bias"] != reports[0]["bias"]
        for report in reports
    ):
        raise ValueError("action contribution parameter audit differs across reports")
    aggregate = {
        "schema_version": "hwr.action-input-contribution-aggregate/v2",
        "episode_count": len(reports),
        "transition_count": transition_count,
        **values,
        "raw_action_variation_to_stochastic_ratio": _ratio(raw, stochastic),
        "canonical_action_variation_to_stochastic_ratio": _ratio(
            canonical, stochastic
        ),
        "canonical_to_raw_variation_contribution_gain": _ratio(canonical, raw),
        "canonical_actions_finite": all(
            report["canonical_actions_finite"] is True for report in reports
        ),
        "canonical_actions_in_bounds": all(
            report["canonical_actions_in_bounds"] is True for report in reports
        ),
        "episodes_passing_contribution_conditions": sum(
            _below(
                report["raw_action_variation_to_stochastic_ratio"], 0.20
            )
            and _at_least(
                report["canonical_to_raw_variation_contribution_gain"], 1.50
            )
            for report in reports
        ),
        "canonical_action_value_count": value_count,
        "canonical_action_finite_count": sum(finite_counts),
        "canonical_action_nonfinite_count": sum(nonfinite_counts),
        "canonical_action_in_bounds_count": sum(in_bounds_counts),
        "canonical_action_out_of_bounds_count": sum(out_of_bounds_counts),
        "canonical_actions_in_bounds_fraction": (
            sum(in_bounds_counts) / max(value_count, 1)
        ),
        "canonical_action_dimension_finite_count": finite_counts,
        "canonical_action_dimension_nonfinite_count": nonfinite_counts,
        "canonical_action_dimension_in_bounds_count": in_bounds_counts,
        "canonical_action_dimension_out_of_bounds_count": out_of_bounds_counts,
        "canonical_action_dimension_in_bounds_fraction": [
            count / max(transition_count, 1) for count in in_bounds_counts
        ],
        "weights": reports[0]["weights"],
        "bias": reports[0]["bias"],
        "raw_action_dimension_rms": _pooled_dimension_rms(
            reports, "raw_action_dimension_rms"
        ),
        "canonical_action_dimension_rms": _pooled_dimension_rms(
            reports, "canonical_action_dimension_rms"
        ),
    }
    aggregate["assessment"] = assess_action_input_contribution(aggregate)
    return aggregate


def assess_action_input_contribution(
    report: Mapping[str, object],
) -> dict[str, object]:
    checks = {
        "raw_action_variation_to_stochastic_ratio_below_0_20": (
            _below(report["raw_action_variation_to_stochastic_ratio"], 0.20)
        ),
        "canonical_to_raw_variation_gain_at_least_1_50": (
            _at_least(
                report["canonical_to_raw_variation_contribution_gain"], 1.50
            )
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
    result = float(value.detach().cpu().double().square().mean().sqrt())
    if not math.isfinite(result):
        raise ValueError("action contribution RMS is non-finite")
    return result


def _contribution_metrics(value: torch.Tensor) -> dict[str, float | None]:
    cpu = value.detach().cpu().double()
    if not torch.isfinite(cpu).all():
        return {"absolute_rms": None, "dc_rms": None, "variation_rms": None}
    mean = cpu.mean(dim=1, keepdim=True)
    return {
        "absolute_rms": _rms(cpu),
        "dc_rms": _rms(mean),
        "variation_rms": _rms(cpu - mean),
    }


def _flatten_contributions(
    values: Mapping[str, Mapping[str, float | None]],
) -> dict[str, float | None]:
    return {
        "absolute_stochastic_contribution_rms": values["stochastic"]["absolute_rms"],
        "absolute_raw_action_contribution_rms": values["raw_action"]["absolute_rms"],
        "absolute_canonical_action_contribution_rms": (
            values["canonical_action"]["absolute_rms"]
        ),
        "stochastic_dc_contribution_rms": values["stochastic"]["dc_rms"],
        "raw_action_dc_contribution_rms": values["raw_action"]["dc_rms"],
        "canonical_action_dc_contribution_rms": values["canonical_action"]["dc_rms"],
        "stochastic_variation_contribution_rms": (
            values["stochastic"]["variation_rms"]
        ),
        "raw_action_variation_contribution_rms": (
            values["raw_action"]["variation_rms"]
        ),
        "canonical_action_variation_contribution_rms": (
            values["canonical_action"]["variation_rms"]
        ),
    }


def _dimension_rms(value: torch.Tensor) -> list[float | None]:
    cpu = value.detach().cpu().double()
    finite = torch.isfinite(cpu).all(dim=(0, 1))
    safe = torch.where(torch.isfinite(cpu), cpu, torch.zeros_like(cpu))
    result = safe.square().mean(dim=(0, 1)).sqrt().tolist()
    return [
        float(item) if bool(valid) else None
        for item, valid in zip(result, finite.tolist(), strict=True)
    ]


def _canonical_bounds(value: torch.Tensor) -> dict[str, object]:
    cpu = value.detach().cpu()
    finite = torch.isfinite(cpu)
    in_bounds = finite & (cpu >= -1.0 - 1.0e-6) & (cpu <= 1.0 + 1.0e-6)
    out_of_bounds = finite & ~in_bounds
    dimensions = value.shape[-1]
    count = int(value.numel())
    per_dimension = int(value.numel() // dimensions)
    finite_counts = finite.sum(dim=(0, 1)).tolist()
    in_bounds_counts = in_bounds.sum(dim=(0, 1)).tolist()
    out_of_bounds_counts = out_of_bounds.sum(dim=(0, 1)).tolist()
    nonfinite_counts = (~finite).sum(dim=(0, 1)).tolist()
    return {
        "canonical_action_value_count": count,
        "canonical_action_finite_count": int(finite.sum()),
        "canonical_action_nonfinite_count": int((~finite).sum()),
        "canonical_action_in_bounds_count": int(in_bounds.sum()),
        "canonical_action_out_of_bounds_count": int(out_of_bounds.sum()),
        "canonical_actions_in_bounds_fraction": float(in_bounds.sum()) / max(count, 1),
        "canonical_action_dimension_finite_count": finite_counts,
        "canonical_action_dimension_nonfinite_count": nonfinite_counts,
        "canonical_action_dimension_in_bounds_count": in_bounds_counts,
        "canonical_action_dimension_out_of_bounds_count": out_of_bounds_counts,
        "canonical_action_dimension_in_bounds_fraction": [
            count / max(per_dimension, 1) for count in in_bounds_counts
        ],
        "canonical_actions_finite": not any(nonfinite_counts),
        "canonical_actions_in_bounds": (
            not any(nonfinite_counts) and not any(out_of_bounds_counts)
        ),
    }


def _column_norms(value: torch.Tensor) -> list[float]:
    if value.ndim != 2:
        raise ValueError("action contribution weight matrix is not two-dimensional")
    return [
        float(item)
        for item in value.detach().cpu().double().norm(dim=0)
    ]


def _norm(value: torch.Tensor) -> float:
    result = float(value.detach().cpu().double().norm())
    if not math.isfinite(result):
        raise ValueError("action contribution norm is non-finite")
    return result


def _ratio(
    numerator: object, denominator: object
) -> float | None:
    if numerator is None or denominator is None:
        return None
    result = float(numerator) / max(float(denominator), 1.0e-12)
    return result if math.isfinite(result) else None


def _below(value: object, threshold: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) < threshold


def _at_least(value: object, threshold: float) -> bool:
    return (
        value is not None
        and math.isfinite(float(value))
        and float(value) >= threshold
    )


def _pooled_rms(
    reports: Sequence[Mapping[str, object]], name: str
) -> float | None:
    counts = np.asarray([int(report["transition_count"]) for report in reports])
    values = [report[name] for report in reports]
    if any(value is None for value in values):
        return None
    array = np.asarray(values, np.float64)
    if not np.isfinite(array).all():
        return None
    return float(np.sqrt(np.average(np.square(array), weights=counts)))


def _pooled_dimension_rms(
    reports: Sequence[Mapping[str, object]], name: str
) -> list[float | None]:
    raw_values = [report[name] for report in reports]
    dimension_count = len(raw_values[0])
    if any(len(value) != dimension_count for value in raw_values):
        raise ValueError("action dimension aggregate is invalid")
    counts = np.asarray([int(report["transition_count"]) for report in reports])
    result: list[float | None] = []
    for index in range(dimension_count):
        values = [value[index] for value in raw_values]
        if any(value is None for value in values):
            result.append(None)
            continue
        array = np.asarray(values, np.float64)
        result.append(float(np.sqrt(np.average(np.square(array), weights=counts))))
    return result


def _sum_dimension_counts(
    reports: Sequence[Mapping[str, object]], name: str, dimension_count: int
) -> list[int]:
    values = np.asarray([report[name] for report in reports], np.int64)
    if values.shape != (len(reports), dimension_count) or (values < 0).any():
        raise ValueError("action dimension counts are invalid")
    return values.sum(axis=0).tolist()


def _validate_episode_identities(
    reports: Sequence[Mapping[str, object]],
) -> None:
    sources = [str(report["source_episode_id"]) for report in reports]
    windows = [
        (
            str(report["window"]["episode_id"]),
            int(report["window"]["transition_start"]),
            int(report["window"]["transition_stop"]),
        )
        for report in reports
    ]
    if len(set(sources)) != len(reports) or len(set(windows)) != len(reports):
        raise ValueError("action contribution Episode identities are not unique")
