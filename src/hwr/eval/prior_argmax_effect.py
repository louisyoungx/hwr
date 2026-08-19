"""Prior-probability to argmax-code diagnostic for R0001-P23."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as nn_functional

from hwr.eval.layerwise_action_effect import (
    ACTION_SHIFTS,
    _forward_stages,
    _require_model_structure,
)
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMSequence


EXPECTED_TRANSITION_COUNT = 16
EXPECTED_STOCHASTIC_VARIABLES = 32
EXPECTED_STOCHASTIC_CLASSES = 32
EXPECTED_PROBABILITY_DIMENSION = 1024
MINIMUM_ACTIVE_DIMENSIONS = 256
ACTIVE_SCALE_MINIMUM = 1.0e-4
ACTIVE_FRACTION_MINIMUM = 0.25
EFFECT_DENOMINATOR_MINIMUM = 1.0e-6
PROBABILITY_EFFECT_MINIMUM = 0.05
FLIP_FRACTION_MAXIMUM = 0.10
RETENTION_MAXIMUM = 0.50
NEAR_TIE_MAXIMUM = 1.0e-8
HARD_FEATURE_EFFECT_MINIMUM = 0.05


def evaluate_prior_argmax_effect(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    executed_actions: torch.Tensor,
) -> dict[str, object]:
    deterministic = sequence.deterministic[:, :-1].detach()
    stochastic = sequence.stochastic[:, :-1].detach()
    actions = executed_actions.detach()
    if (
        actions.ndim != 3
        or actions.shape[0] != 1
        or actions.shape[1] != EXPECTED_TRANSITION_COUNT
        or deterministic.shape[:2] != actions.shape[:2]
        or stochastic.shape[:2] != actions.shape[:2]
        or deterministic.shape[-1] != model.config.deterministic_dimension
        or stochastic.shape[-1] != model.config.stochastic_dimension
        or actions.shape[-1] != model.config.action_dimension
        or model.config.stochastic_variables != EXPECTED_STOCHASTIC_VARIABLES
        or model.config.stochastic_classes != EXPECTED_STOCHASTIC_CLASSES
        or model.config.stochastic_dimension != EXPECTED_PROBABILITY_DIMENSION
    ):
        raise ValueError("prior argmax input shapes are invalid")
    _require_model_structure(model)
    with torch.inference_mode():
        true_stages = _forward_stages(
            model, deterministic, stochastic, actions
        )
        true_probability = _categorical(
            true_stages["prior_probability"], model
        )
        true_logits = _categorical(true_stages["prior_logits"], model)
        true_code = _hard_code(true_probability)
        active = _active_scale(true_probability.flatten(-2))
        true_feature = torch.cat(
            (true_stages["next_deterministic"], true_code.flatten(-2)),
            dim=-1,
        )
        shifts = {
            str(shift): _evaluate_shift(
                model,
                deterministic,
                stochastic,
                actions,
                true_stages,
                true_logits,
                true_probability,
                true_code,
                true_feature,
                active,
                shift,
            )
            for shift in ACTION_SHIFTS
        }
    report = {
        "schema_version": "hwr.prior-argmax-effect/v1",
        "transition_count": EXPECTED_TRANSITION_COUNT,
        "criteria": _criteria(),
        "probability_active_dimension_count": active["active_dimension_count"],
        "probability_active_fraction": active["active_fraction"],
        "probability_active_scale": active["scale_summary"],
        "shifts": shifts,
    }
    report["assessment"] = assess_prior_argmax_episode(report)
    return report


def aggregate_prior_argmax_effect(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(reports) != 24:
        raise ValueError("prior argmax diagnostic requires 24 reports")
    if any(
        int(report["transition_count"]) != EXPECTED_TRANSITION_COUNT
        for report in reports
    ):
        raise ValueError("prior argmax transition count differs")
    _validate_episode_identities(reports)
    expected_tasks = {
        "clear_dining_table_3d/v1": 6,
        "store_kitchen_items_3d/v1": 6,
        "tidy_living_room_3d/v1": 12,
    }
    task_episode_counts = {task: 0 for task in expected_tasks}
    task_pass_counts = {task: 0 for task in expected_tasks}
    hard_guard_task_counts = {task: 0 for task in expected_tasks}
    shift_pass_counts = {str(shift): 0 for shift in ACTION_SHIFTS}
    episode_pass_count = 0
    hard_guard_episode_count = 0
    valid = True
    for report in reports:
        task = str(report["window"]["task_id"])
        if task not in expected_tasks:
            raise ValueError("prior argmax task identity differs")
        task_episode_counts[task] += 1
        passed = report["assessment"]["passed"] is True
        hard_guard = report["assessment"]["hard_feature_guard_passed"] is True
        valid = valid and report["assessment"]["valid"] is True
        episode_pass_count += passed
        hard_guard_episode_count += hard_guard
        task_pass_counts[task] += passed
        hard_guard_task_counts[task] += hard_guard
        for shift in ACTION_SHIFTS:
            shift_pass_counts[str(shift)] += (
                report["shifts"][str(shift)]["assessment"]["passed"] is True
            )
    if task_episode_counts != expected_tasks:
        raise ValueError("prior argmax task coverage differs")
    task_requirements = {
        "clear_dining_table_3d/v1": 5,
        "store_kitchen_items_3d/v1": 5,
        "tidy_living_room_3d/v1": 10,
    }
    checks = {
        "at_least_20_of_24_episodes_pass": episode_pass_count >= 20,
        "task_pass_quotas_met": all(
            task_pass_counts[task] >= required
            for task, required in task_requirements.items()
        ),
        "each_shift_passes_at_least_18_of_24": all(
            count >= 18 for count in shift_pass_counts.values()
        ),
    }
    hard_guard_checks = {
        "at_least_20_of_24_episodes_pass": hard_guard_episode_count >= 20,
        "task_pass_quotas_met": all(
            hard_guard_task_counts[task] >= required
            for task, required in task_requirements.items()
        ),
    }
    passed = valid and all(checks.values())
    decision = (
        "diagnostic_invalid"
        if not valid
        else "diagnostic_passed"
        if passed
        else "diagnostic_failed"
    )
    return {
        "schema_version": "hwr.prior-argmax-effect-aggregate/v1",
        "episode_count": len(reports),
        "transition_count": sum(
            int(report["transition_count"]) for report in reports
        ),
        "criteria": _criteria(),
        "episode_pass_count": episode_pass_count,
        "shift_pass_counts": shift_pass_counts,
        "task_episode_counts": task_episode_counts,
        "task_pass_counts": task_pass_counts,
        "task_pass_requirements": task_requirements,
        "hard_feature_guard_episode_count": hard_guard_episode_count,
        "hard_feature_guard_task_counts": hard_guard_task_counts,
        "hard_feature_guard_assessment": {
            "passed": valid and all(hard_guard_checks.values()),
            "checks": hard_guard_checks,
        },
        "assessment": {
            "decision": decision,
            "passed": passed,
            "valid": valid,
            "checks": checks,
        },
    }


def assess_prior_argmax_episode(
    report: Mapping[str, object],
) -> dict[str, object]:
    shift_pass_count = sum(
        report["shifts"][str(shift)]["assessment"]["passed"] is True
        for shift in ACTION_SHIFTS
    )
    hard_guard_shift_count = sum(
        report["shifts"][str(shift)]["hard_feature"]["guard_passed"] is True
        for shift in ACTION_SHIFTS
    )
    valid = all(
        report["shifts"][str(shift)]["assessment"]["implementation_valid"] is True
        for shift in ACTION_SHIFTS
    )
    passed = valid and shift_pass_count >= 2
    return {
        "decision": (
            "episode_invalid"
            if not valid
            else "episode_passed"
            if passed
            else "episode_failed"
        ),
        "passed": passed,
        "valid": valid,
        "shift_pass_count": shift_pass_count,
        "hard_feature_guard_passed": hard_guard_shift_count >= 2,
        "hard_feature_guard_shift_count": hard_guard_shift_count,
    }


def _evaluate_shift(
    model: ActionConditionedWorldModel,
    deterministic: torch.Tensor,
    stochastic: torch.Tensor,
    actions: torch.Tensor,
    true_stages: Mapping[str, object],
    true_logits: torch.Tensor,
    true_probability: torch.Tensor,
    true_code: torch.Tensor,
    true_feature: torch.Tensor,
    active: Mapping[str, object],
    shift: int,
) -> dict[str, object]:
    shifted_actions = torch.roll(actions, shifts=shift, dims=1)
    shifted_stages = _forward_stages(
        model, deterministic, stochastic, shifted_actions
    )
    shifted_probability = _categorical(
        shifted_stages["prior_probability"], model
    )
    shifted_logits = _categorical(shifted_stages["prior_logits"], model)
    shifted_code = _hard_code(shifted_probability)
    probability_effect = _common_scale_effect(
        true_probability.flatten(-2),
        shifted_probability.flatten(-2),
        active,
    )
    hard_effect = _common_scale_effect(
        true_code.flatten(-2),
        shifted_code.flatten(-2),
        active,
    )
    retention = _ratio(
        hard_effect["standardized_effect"],
        probability_effect["standardized_effect"],
    )
    margin = _margin_report(true_probability, shifted_probability)
    flip = (true_code.argmax(dim=-1) != shifted_code.argmax(dim=-1))
    flip_fraction = float(flip.detach().cpu().double().mean())
    crossing_matches = bool(
        torch.equal(flip.detach().cpu(), margin["crossing_mask"])
    )
    official_true = model.rssm._sample(true_logits, sample=False)
    official_shifted = model.rssm._sample(shifted_logits, sample=False)
    implementation_matches = bool(
        torch.equal(official_true, true_code.flatten(-2))
        and torch.equal(official_shifted, shifted_code.flatten(-2))
    )
    shifted_feature = torch.cat(
        (
            shifted_stages["next_deterministic"],
            shifted_code.flatten(-2),
        ),
        dim=-1,
    )
    hard_feature = _own_scale_effect(true_feature, shifted_feature)
    assessment = _assess_shift(
        probability_effect,
        hard_effect,
        retention,
        flip_fraction,
        margin,
        float(active["active_fraction"]),
        crossing_matches,
        implementation_matches,
    )
    return {
        "shift": shift,
        "probability_effect": probability_effect,
        "hard_code_effect": hard_effect,
        "probability_to_code_retention": retention,
        "argmax_flip_fraction": flip_fraction,
        "margin": {
            key: value
            for key, value in margin.items()
            if key != "crossing_mask"
        },
        "hard_feature": hard_feature,
        "implementation": {
            "flip_matches_margin_crossing": crossing_matches,
            "hard_code_matches_sample_false": implementation_matches,
        },
        "assessment": assessment,
    }


def _assess_shift(
    probability_effect: Mapping[str, object],
    hard_effect: Mapping[str, object],
    retention: object,
    flip_fraction: float,
    margin: Mapping[str, object],
    active_fraction: float,
    crossing_matches: bool,
    implementation_matches: bool,
) -> dict[str, object]:
    implementation_valid = crossing_matches and implementation_matches
    checks = {
        "probability_effect_at_least_0_05": _at_least(
            probability_effect["standardized_effect"],
            PROBABILITY_EFFECT_MINIMUM,
        ),
        "probability_active_fraction_at_least_0_25": (
            active_fraction >= ACTIVE_FRACTION_MINIMUM
        ),
        "argmax_flip_fraction_at_most_0_10": (
            flip_fraction <= FLIP_FRACTION_MAXIMUM
        ),
        "probability_to_code_retention_below_0_50": _below(
            retention, RETENTION_MAXIMUM
        ),
        "near_tie_count_is_zero": margin["near_tie_count"] == 0,
        "flip_matches_margin_crossing": crossing_matches,
        "hard_code_matches_sample_false": implementation_matches,
        "all_values_finite": (
            probability_effect["finite"] is True
            and hard_effect["finite"] is True
            and margin["finite"] is True
        ),
    }
    passed = implementation_valid and all(checks.values())
    return {
        "decision": (
            "shift_invalid"
            if not implementation_valid
            else "shift_passed"
            if passed
            else "shift_failed"
        ),
        "passed": passed,
        "implementation_valid": implementation_valid,
        "checks": checks,
    }


def _active_scale(true_probability: torch.Tensor) -> dict[str, object]:
    cpu = true_probability.detach().cpu().double()
    if not torch.isfinite(cpu).all():
        return {
            "finite": False,
            "scale": None,
            "active_mask": None,
            "active_dimension_count": 0,
            "active_fraction": 0.0,
            "scale_summary": None,
        }
    centered = cpu - cpu.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1).sqrt()[0]
    active = scale >= ACTIVE_SCALE_MINIMUM
    active_count = int(active.sum())
    active_scale = scale[active]
    return {
        "finite": True,
        "scale": scale,
        "active_mask": active,
        "active_dimension_count": active_count,
        "active_fraction": active_count / scale.numel(),
        "scale_summary": (
            {
                "minimum": float(active_scale.min()),
                "median": float(active_scale.median()),
                "maximum": float(active_scale.max()),
            }
            if active_count
            else None
        ),
    }


def _common_scale_effect(
    true: torch.Tensor,
    shifted: torch.Tensor,
    active: Mapping[str, object],
) -> dict[str, object]:
    true_cpu = true.detach().cpu().double()
    shifted_cpu = shifted.detach().cpu().double()
    finite = bool(
        active["finite"] is True
        and torch.isfinite(true_cpu).all()
        and torch.isfinite(shifted_cpu).all()
    )
    mask = active["active_mask"]
    scale = active["scale"]
    active_count = int(active["active_dimension_count"])
    if not finite or mask is None or scale is None:
        return {
            "finite": False,
            "raw_rms": None,
            "standardized_effect": None,
        }
    difference = shifted_cpu - true_cpu
    raw = float(difference.square().mean().sqrt())
    standardized = (
        float(
            (difference[0, :, mask] / scale[mask])
            .square()
            .mean()
            .sqrt()
        )
        if active_count
        else None
    )
    return {
        "finite": True,
        "raw_rms": raw,
        "standardized_effect": standardized,
    }


def _own_scale_effect(
    true: torch.Tensor,
    shifted: torch.Tensor,
) -> dict[str, object]:
    active = _active_scale(true)
    effect = _common_scale_effect(true, shifted, active)
    guard_passed = (
        effect["finite"] is True
        and float(active["active_fraction"]) >= ACTIVE_FRACTION_MINIMUM
        and _at_least(
            effect["standardized_effect"], HARD_FEATURE_EFFECT_MINIMUM
        )
    )
    return {
        **effect,
        "active_dimension_count": active["active_dimension_count"],
        "active_fraction": active["active_fraction"],
        "guard_passed": guard_passed,
    }


def _margin_report(
    true_probability: torch.Tensor,
    shifted_probability: torch.Tensor,
) -> dict[str, object]:
    true_cpu = true_probability.detach().cpu().double()
    shifted_cpu = shifted_probability.detach().cpu().double()
    finite = bool(
        torch.isfinite(true_cpu).all()
        and torch.isfinite(shifted_cpu).all()
    )
    if not finite:
        return {
            "finite": False,
            "near_tie_count": 0,
            "crossing_fraction": None,
            "crossing_mask": torch.zeros(0, dtype=torch.bool),
        }
    winner = true_cpu.argmax(dim=-1, keepdim=True)
    true_winner = true_cpu.gather(-1, winner).squeeze(-1)
    shifted_winner = shifted_cpu.gather(-1, winner).squeeze(-1)
    classes = true_cpu.shape[-1]
    winner_mask = nn_functional.one_hot(
        winner.squeeze(-1), classes
    ).bool()
    true_other = true_cpu.masked_fill(winner_mask, -torch.inf).max(dim=-1).values
    shifted_other = (
        shifted_cpu.masked_fill(winner_mask, -torch.inf)
        .max(dim=-1)
        .values
    )
    true_margin = true_winner - true_other
    shifted_signed_margin = shifted_winner - shifted_other
    consumption = true_margin - shifted_signed_margin
    crossing = shifted_signed_margin <= 0.0
    near_tie = true_margin <= NEAR_TIE_MAXIMUM
    return {
        "finite": True,
        "near_tie_count": int(near_tie.sum()),
        "crossing_fraction": float(crossing.double().mean()),
        "crossing_mask": crossing,
        "true_margin": _distribution(true_margin),
        "shift_signed_margin": _distribution(shifted_signed_margin),
        "margin_consumption": _distribution(consumption),
    }


def _categorical(
    value: torch.Tensor, model: ActionConditionedWorldModel
) -> torch.Tensor:
    return value.reshape(
        *value.shape[:-1],
        model.config.stochastic_variables,
        model.config.stochastic_classes,
    )


def _hard_code(probability: torch.Tensor) -> torch.Tensor:
    indices = probability.argmax(dim=-1)
    return nn_functional.one_hot(
        indices, probability.shape[-1]
    ).to(probability.dtype)


def _distribution(value: torch.Tensor) -> dict[str, float]:
    array = value.numpy().reshape(-1)
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _criteria() -> dict[str, object]:
    return {
        "expected_transition_count": EXPECTED_TRANSITION_COUNT,
        "action_shifts": list(ACTION_SHIFTS),
        "active_scale_minimum": ACTIVE_SCALE_MINIMUM,
        "active_fraction_minimum": ACTIVE_FRACTION_MINIMUM,
        "minimum_active_dimensions": MINIMUM_ACTIVE_DIMENSIONS,
        "stochastic_variables": EXPECTED_STOCHASTIC_VARIABLES,
        "stochastic_classes": EXPECTED_STOCHASTIC_CLASSES,
        "probability_dimension": EXPECTED_PROBABILITY_DIMENSION,
        "effect_denominator_minimum": EFFECT_DENOMINATOR_MINIMUM,
        "probability_effect_minimum": PROBABILITY_EFFECT_MINIMUM,
        "flip_fraction_maximum": FLIP_FRACTION_MAXIMUM,
        "retention_maximum": RETENTION_MAXIMUM,
        "near_tie_maximum": NEAR_TIE_MAXIMUM,
        "hard_feature_effect_minimum": HARD_FEATURE_EFFECT_MINIMUM,
        "episode_shift_pass_minimum": 2,
        "aggregate_episode_pass_minimum": 20,
        "aggregate_shift_pass_minimum": 18,
        "task_pass_requirements": {
            "clear_dining_table_3d/v1": 5,
            "store_kitchen_items_3d/v1": 5,
            "tidy_living_room_3d/v1": 10,
        },
    }


def _ratio(numerator: object, denominator: object) -> float | None:
    if (
        numerator is None
        or denominator is None
        or float(denominator) < EFFECT_DENOMINATOR_MINIMUM
    ):
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def _below(value: object, threshold: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) < threshold


def _at_least(value: object, threshold: float) -> bool:
    return (
        value is not None
        and math.isfinite(float(value))
        and float(value) >= threshold
    )


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
        raise ValueError("prior argmax Episode identities are not unique")
