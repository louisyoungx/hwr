"""Layerwise RSSM action-effect diagnostic for R0001-P21."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as nn_functional

from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMSequence


ACTION_SHIFTS = (1, 5, 9)
LOCAL_EPSILON = 0.05
ACTIVE_SCALE_MINIMUM = 1.0e-4
EFFECT_DENOMINATOR_MINIMUM = 1.0e-6
TRANSITION_EFFECT_MINIMUM = 0.05
RETENTION_MAXIMUM = 0.50
SENSITIVITY_RATIO_MAXIMUM = 0.50
GRU_MAXIMUM_ABSOLUTE_ERROR = 1.0e-5
MAIN_STAGE_ACTIVE_FRACTION = 0.25
INPUT_ACTIVE_FRACTION = 0.50
EXPECTED_TRANSITION_COUNT = 16

STAGE_NAMES = (
    "transition_preactivation",
    "transition_normalized",
    "transition_activation",
    "gru_reset_gate",
    "gru_update_gate",
    "gru_new_gate",
    "next_deterministic",
    "prior_hidden",
    "prior_logits",
    "prior_probability",
)


def _criteria() -> dict[str, object]:
    return {
        "expected_transition_count": EXPECTED_TRANSITION_COUNT,
        "action_shifts": list(ACTION_SHIFTS),
        "local_epsilon": LOCAL_EPSILON,
        "active_scale_minimum": ACTIVE_SCALE_MINIMUM,
        "effect_denominator_minimum": EFFECT_DENOMINATOR_MINIMUM,
        "transition_effect_minimum": TRANSITION_EFFECT_MINIMUM,
        "retention_maximum": RETENTION_MAXIMUM,
        "sensitivity_ratio_maximum": SENSITIVITY_RATIO_MAXIMUM,
        "gru_maximum_absolute_error": GRU_MAXIMUM_ABSOLUTE_ERROR,
        "main_stage_active_fraction": MAIN_STAGE_ACTIVE_FRACTION,
        "input_active_fraction": INPUT_ACTIVE_FRACTION,
        "episode_shift_pass_minimum": 2,
        "aggregate_episode_pass_minimum": 20,
        "aggregate_shift_pass_minimum": 18,
        "concentration_episode_minimum": 16,
        "task_pass_requirements": {
            "clear_dining_table_3d/v1": 5,
            "store_kitchen_items_3d/v1": 5,
            "tidy_living_room_3d/v1": 10,
        },
    }


def _require_model_structure(model: ActionConditionedWorldModel) -> None:
    transition = model.rssm.transition_input
    prior = model.rssm.prior
    expected_transition_input = (
        model.config.stochastic_dimension + model.config.action_dimension
    )
    if (
        len(transition) != 3
        or not isinstance(transition[0], torch.nn.Linear)
        or transition[0].in_features != expected_transition_input
        or transition[0].out_features != model.config.hidden_dimension
        or not isinstance(transition[1], torch.nn.LayerNorm)
        or not isinstance(transition[2], torch.nn.SiLU)
        or not isinstance(model.rssm.recurrent, torch.nn.GRUCell)
        or model.rssm.recurrent.input_size != model.config.hidden_dimension
        or model.rssm.recurrent.hidden_size != model.config.deterministic_dimension
        or len(prior) != 3
        or not isinstance(prior[0], torch.nn.Linear)
        or not isinstance(prior[1], torch.nn.SiLU)
        or not isinstance(prior[2], torch.nn.Linear)
    ):
        raise ValueError("P21 frozen RSSM structure differs")


def evaluate_layerwise_action_effect(
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
        or deterministic.shape[:2] != actions.shape[:2]
        or stochastic.shape[:2] != actions.shape[:2]
        or deterministic.shape[-1] != model.config.deterministic_dimension
        or stochastic.shape[-1] != model.config.stochastic_dimension
        or actions.shape[-1] != model.config.action_dimension
        or actions.shape[1] != EXPECTED_TRANSITION_COUNT
    ):
        raise ValueError("layerwise action-effect input shapes are invalid")
    _require_model_structure(model)
    with torch.inference_mode():
        true_stages = _forward_stages(
            model, deterministic, stochastic, actions
        )
        shifts = {
            str(shift): _evaluate_shift(
                model,
                deterministic,
                stochastic,
                actions,
                true_stages,
                shift,
            )
            for shift in ACTION_SHIFTS
        }
    report = {
        "schema_version": "hwr.layerwise-action-effect/v1",
        "transition_count": int(actions.shape[1]),
        "action_shifts": list(ACTION_SHIFTS),
        "local_epsilon": LOCAL_EPSILON,
        "active_scale_minimum": ACTIVE_SCALE_MINIMUM,
        "criteria": _criteria(),
        "gru_gate_distributions": {
            "reset": _distribution_report(true_stages["gru_reset_gate"]),
            "update": _distribution_report(true_stages["gru_update_gate"]),
            "new": _distribution_report(true_stages["gru_new_gate"]),
        },
        "shifts": shifts,
    }
    report["assessment"] = assess_layerwise_action_effect_episode(report)
    return report


def aggregate_layerwise_action_effect(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(reports) != 24:
        raise ValueError("layerwise action-effect diagnostic requires 24 reports")
    if any(
        int(report["transition_count"]) != EXPECTED_TRANSITION_COUNT
        for report in reports
    ):
        raise ValueError("layerwise action-effect transition count differs")
    _validate_episode_identities(reports)
    expected_tasks = {
        "clear_dining_table_3d/v1": 6,
        "store_kitchen_items_3d/v1": 6,
        "tidy_living_room_3d/v1": 12,
    }
    task_episode_counts = {task: 0 for task in expected_tasks}
    task_pass_counts = {task: 0 for task in expected_tasks}
    shift_pass_counts = {str(shift): 0 for shift in ACTION_SHIFTS}
    location_counts = {"activation_to_h": 0, "h_to_prior": 0}
    episode_pass_count = 0
    for report in reports:
        task = str(report["window"]["task_id"])
        if task not in expected_tasks:
            raise ValueError("layerwise action-effect task identity differs")
        task_episode_counts[task] += 1
        passed = report["assessment"]["passed"] is True
        episode_pass_count += passed
        task_pass_counts[task] += passed
        for shift in ACTION_SHIFTS:
            shift_pass_counts[str(shift)] += (
                report["shifts"][str(shift)]["assessment"]["passed"] is True
            )
        location = report["assessment"]["consensus_first_low_retention"]
        if location is not None:
            location_counts[str(location)] += 1
    if task_episode_counts != expected_tasks:
        raise ValueError("layerwise action-effect task coverage differs")
    task_requirements = {
        "clear_dining_table_3d/v1": 5,
        "store_kitchen_items_3d/v1": 5,
        "tidy_living_room_3d/v1": 10,
    }
    core_checks = {
        "at_least_20_of_24_episodes_pass": episode_pass_count >= 20,
        "task_pass_quotas_met": all(
            task_pass_counts[task] >= requirement
            for task, requirement in task_requirements.items()
        ),
        "each_shift_passes_at_least_18_of_24": all(
            count >= 18 for count in shift_pass_counts.values()
        ),
    }
    core_passed = all(core_checks.values())
    concentration_location = next(
        (
            location
            for location, count in location_counts.items()
            if count >= 16
        ),
        None,
    )
    concentration_passed = concentration_location is not None
    if core_passed and concentration_passed:
        decision = "diagnostic_passed"
    elif core_passed:
        decision = "diagnostic_inconclusive"
    else:
        decision = "diagnostic_failed"
    return {
        "schema_version": "hwr.layerwise-action-effect-aggregate/v1",
        "episode_count": len(reports),
        "transition_count": sum(int(report["transition_count"]) for report in reports),
        "criteria": _criteria(),
        "episode_pass_count": episode_pass_count,
        "shift_pass_counts": shift_pass_counts,
        "task_episode_counts": task_episode_counts,
        "task_pass_counts": task_pass_counts,
        "task_pass_requirements": task_requirements,
        "consensus_location_episode_counts": location_counts,
        "gru_gate_distributions": {
            gate: _aggregate_distributions(reports, gate)
            for gate in ("reset", "update", "new")
        },
        "assessment": {
            "decision": decision,
            "passed": decision == "diagnostic_passed",
            "core_passed": core_passed,
            "concentration_passed": concentration_passed,
            "concentration_location": concentration_location,
            "checks": core_checks,
        },
    }


def assess_layerwise_action_effect_episode(
    report: Mapping[str, object],
) -> dict[str, object]:
    shift_pass_count = sum(
        report["shifts"][str(shift)]["assessment"]["passed"] is True
        for shift in ACTION_SHIFTS
    )
    locations = [
        report["shifts"][str(shift)]["assessment"]["first_low_retention"]
        for shift in ACTION_SHIFTS
        if report["shifts"][str(shift)]["assessment"]["passed"] is True
    ]
    consensus = next(
        (
            location
            for location in ("activation_to_h", "h_to_prior")
            if locations.count(location) >= 2
        ),
        None,
    )
    passed = shift_pass_count >= 2
    return {
        "decision": "episode_passed" if passed else "episode_failed",
        "passed": passed,
        "shift_pass_count": shift_pass_count,
        "consensus_first_low_retention": consensus,
    }


def _evaluate_shift(
    model: ActionConditionedWorldModel,
    deterministic: torch.Tensor,
    stochastic: torch.Tensor,
    actions: torch.Tensor,
    true_stages: Mapping[str, object],
    shift: int,
) -> dict[str, object]:
    shifted_actions = torch.roll(actions, shifts=shift, dims=1)
    shifted_deterministic = torch.roll(deterministic, shifts=shift, dims=1)
    shifted_stages = _forward_stages(
        model, deterministic, stochastic, shifted_actions
    )
    stage_effects = {
        name: _effect_report(true_stages[name], shifted_stages[name])
        for name in STAGE_NAMES
    }
    action_epsilon = (
        (1.0 - LOCAL_EPSILON) * actions
        + LOCAL_EPSILON * shifted_actions
    )
    deterministic_epsilon = (
        (1.0 - LOCAL_EPSILON) * deterministic
        + LOCAL_EPSILON * shifted_deterministic
    )
    action_stages = _forward_stages(
        model, deterministic, stochastic, action_epsilon
    )
    deterministic_stages = _forward_stages(
        model, deterministic_epsilon, stochastic, actions
    )
    local = _local_sensitivity_report(
        true_stages,
        actions,
        action_epsilon,
        deterministic,
        deterministic_epsilon,
        action_stages,
        deterministic_stages,
    )
    gru_errors = (
        float(true_stages["gru_maximum_absolute_error"]),
        float(shifted_stages["gru_maximum_absolute_error"]),
        float(action_stages["gru_maximum_absolute_error"]),
        float(deterministic_stages["gru_maximum_absolute_error"]),
    )
    retention, gru_error, assessment = _assess_shift(
        stage_effects, local, gru_errors
    )
    return {
        "shift": shift,
        "stage_effects": stage_effects,
        "retention": retention,
        "local_sensitivity": local,
        "gru_maximum_absolute_error": gru_error,
        "assessment": assessment,
    }


def _assess_shift(
    stage_effects: Mapping[str, Mapping[str, object]],
    local: Mapping[str, object],
    gru_errors: Sequence[float],
) -> tuple[dict[str, float | None], float | None, dict[str, object]]:
    activation_effect = stage_effects["transition_activation"][
        "standardized_effect"
    ]
    next_effect = stage_effects["next_deterministic"]["standardized_effect"]
    prior_effect = stage_effects["prior_probability"]["standardized_effect"]
    activation_to_h = _ratio(next_effect, activation_effect)
    h_to_prior = _ratio(prior_effect, next_effect)
    retention_denominators_valid = (
        _at_least(activation_effect, EFFECT_DENOMINATOR_MINIMUM)
        and _at_least(next_effect, EFFECT_DENOMINATOR_MINIMUM)
    )
    if retention_denominators_valid and _below(
        activation_to_h, RETENTION_MAXIMUM
    ):
        first_low = "activation_to_h"
        sensitivity_ratio = local["next_deterministic"][
            "action_to_deterministic_gain_ratio"
        ]
    elif retention_denominators_valid and _below(
        h_to_prior, RETENTION_MAXIMUM
    ):
        first_low = "h_to_prior"
        sensitivity_ratio = local["prior_probability"][
            "action_to_deterministic_gain_ratio"
        ]
    else:
        first_low = None
        sensitivity_ratio = None
    main_active = all(
        stage_effects[name]["active_fraction"] >= MAIN_STAGE_ACTIVE_FRACTION
        for name in (
            "transition_activation",
            "next_deterministic",
            "prior_probability",
        )
    )
    all_stage_finite = all(
        stage_effects[name]["finite"] is True for name in STAGE_NAMES
    )
    gru_finite = all(math.isfinite(error) for error in gru_errors)
    gru_error = max(gru_errors) if gru_finite else None
    checks = {
        "transition_effect_at_least_0_05": _at_least(
            activation_effect, TRANSITION_EFFECT_MINIMUM
        ),
        "main_stages_have_active_dimensions": main_active,
        "retention_denominators_valid": retention_denominators_valid,
        "first_low_retention_exists": first_low is not None,
        "corresponding_sensitivity_ratio_below_0_50": _below(
            sensitivity_ratio, SENSITIVITY_RATIO_MAXIMUM
        ),
        "local_inputs_have_active_dimensions": (
            local["action_input"]["active_fraction"] >= INPUT_ACTIVE_FRACTION
            and local["deterministic_input"]["active_fraction"]
            >= INPUT_ACTIVE_FRACTION
        ),
        "all_stage_effects_finite": all_stage_finite,
        "local_sensitivity_valid": local["valid"] is True,
        "gru_errors_finite": gru_finite,
        "gru_matches_torch": (
            gru_error is not None
            and gru_error <= GRU_MAXIMUM_ABSOLUTE_ERROR
        ),
    }
    passed = all(checks.values())
    return (
        {
            "activation_to_h": activation_to_h,
            "h_to_prior": h_to_prior,
        },
        gru_error,
        {
            "decision": "shift_passed" if passed else "shift_failed",
            "passed": passed,
            "first_low_retention": first_low,
            "corresponding_sensitivity_ratio": sensitivity_ratio,
            "checks": checks,
        },
    )


def _forward_stages(
    model: ActionConditionedWorldModel,
    deterministic: torch.Tensor,
    stochastic: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    layer = model.rssm.transition_input
    transition_input = torch.cat((stochastic, actions), dim=-1)
    preactivation = layer[0](transition_input)
    normalized = layer[1](preactivation)
    activation = layer[2](normalized)
    gates = _gru_stages(model.rssm.recurrent, activation, deterministic)
    prior_hidden = model.rssm.prior[1](
        model.rssm.prior[0](gates["output"])
    )
    prior_logits = model.rssm.prior[2](prior_hidden).reshape(
        *prior_hidden.shape[:-1],
        model.config.stochastic_variables,
        model.config.stochastic_classes,
    )
    probabilities = prior_logits.softmax(dim=-1)
    if model.config.categorical_unimix:
        probabilities = (
            (1.0 - model.config.categorical_unimix) * probabilities
            + model.config.categorical_unimix
            / model.config.stochastic_classes
        )
    return {
        "transition_preactivation": preactivation,
        "transition_normalized": normalized,
        "transition_activation": activation,
        "gru_reset_gate": gates["reset"],
        "gru_update_gate": gates["update"],
        "gru_new_gate": gates["new"],
        "next_deterministic": gates["output"],
        "prior_hidden": prior_hidden,
        "prior_logits": prior_logits.flatten(-2),
        "prior_probability": probabilities.flatten(-2),
        "gru_maximum_absolute_error": gates["maximum_absolute_error"],
    }


def _gru_stages(
    cell: torch.nn.GRUCell,
    inputs: torch.Tensor,
    hidden: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    input_gates = nn_functional.linear(
        inputs, cell.weight_ih, cell.bias_ih
    )
    hidden_gates = nn_functional.linear(
        hidden, cell.weight_hh, cell.bias_hh
    )
    input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
    hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
    reset = torch.sigmoid(input_reset + hidden_reset)
    update = torch.sigmoid(input_update + hidden_update)
    new = torch.tanh(input_new + reset * hidden_new)
    output = (1.0 - update) * new + update * hidden
    reference = cell(
        inputs.flatten(0, 1), hidden.flatten(0, 1)
    ).reshape_as(output)
    maximum_error = float(
        (output - reference).detach().abs().max().cpu()
    )
    return {
        "reset": reset,
        "update": update,
        "new": new,
        "output": output,
        "maximum_absolute_error": maximum_error,
    }


def _effect_report(
    true: torch.Tensor,
    changed: torch.Tensor,
) -> dict[str, object]:
    if true.shape != changed.shape or true.ndim != 3:
        raise ValueError("layerwise action-effect tensor shapes differ")
    true_cpu = true.detach().cpu().double()
    changed_cpu = changed.detach().cpu().double()
    finite = bool(
        torch.isfinite(true_cpu).all()
        and torch.isfinite(changed_cpu).all()
    )
    dimension_count = true.shape[-1]
    if not finite:
        return {
            "finite": False,
            "dimension_count": dimension_count,
            "active_dimension_count": 0,
            "active_fraction": 0.0,
            "raw_paired_rms": None,
            "standardized_effect": None,
        }
    centered = true_cpu - true_cpu.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1).sqrt()[0]
    active = scale >= ACTIVE_SCALE_MINIMUM
    active_count = int(active.sum())
    difference = changed_cpu - true_cpu
    raw = float(difference.square().mean().sqrt())
    standardized = (
        float(
            (difference[0, :, active] / scale[active])
            .square()
            .mean()
            .sqrt()
        )
        if active_count
        else None
    )
    active_scale = scale[active]
    return {
        "finite": True,
        "dimension_count": dimension_count,
        "active_dimension_count": active_count,
        "active_fraction": active_count / dimension_count,
        "raw_paired_rms": raw,
        "standardized_effect": standardized,
        "active_scale_minimum": (
            float(active_scale.min()) if active_count else None
        ),
        "active_scale_median": (
            float(active_scale.median()) if active_count else None
        ),
        "active_scale_maximum": (
            float(active_scale.max()) if active_count else None
        ),
    }


def _distribution_report(value: torch.Tensor) -> dict[str, object]:
    array = value.detach().cpu().double().numpy().reshape(-1)
    if not len(array):
        raise ValueError("layerwise action-effect gate values are empty")
    if not np.isfinite(array).all():
        return {
            "finite": False,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "finite": True,
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _aggregate_distributions(
    reports: Sequence[Mapping[str, object]], gate: str
) -> dict[str, object]:
    finite = all(
        report["gru_gate_distributions"][gate]["finite"] is True
        for report in reports
    )
    if not finite:
        return {
            "finite": False,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    names = ("minimum", "p05", "median", "p95", "maximum")
    values = {
        name: [
            float(report["gru_gate_distributions"][gate][name])
            for report in reports
        ]
        for name in names
    }
    if any(not math.isfinite(item) for items in values.values() for item in items):
        return {
            "finite": False,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    result: dict[str, object] = {"finite": True}
    result.update(
        {
            name: {
            "mean": sum(items) / len(items),
            "minimum": min(items),
            "maximum": max(items),
        }
        for name, items in values.items()
        }
    )
    return result


def _local_sensitivity_report(
    true_stages: Mapping[str, object],
    actions: torch.Tensor,
    action_epsilon: torch.Tensor,
    deterministic: torch.Tensor,
    deterministic_epsilon: torch.Tensor,
    action_stages: Mapping[str, object],
    deterministic_stages: Mapping[str, object],
) -> dict[str, object]:
    action_input = _effect_report(actions, action_epsilon)
    deterministic_input = _effect_report(
        deterministic, deterministic_epsilon
    )
    outputs = {}
    for name in ("next_deterministic", "prior_probability"):
        action_output = _effect_report(
            true_stages[name], action_stages[name]
        )
        deterministic_output = _effect_report(
            true_stages[name], deterministic_stages[name]
        )
        action_gain = _gain(
            action_output["standardized_effect"],
            action_input["standardized_effect"],
        )
        deterministic_gain = _gain(
            deterministic_output["standardized_effect"],
            deterministic_input["standardized_effect"],
        )
        outputs[name] = {
            "action_output": action_output,
            "deterministic_output": deterministic_output,
            "action_local_gain": action_gain,
            "deterministic_local_gain": deterministic_gain,
            "action_to_deterministic_gain_ratio": _ratio(
                action_gain, deterministic_gain
            ),
        }
    valid = (
        action_input["finite"] is True
        and deterministic_input["finite"] is True
        and action_input["active_fraction"] >= INPUT_ACTIVE_FRACTION
        and deterministic_input["active_fraction"]
        >= INPUT_ACTIVE_FRACTION
        and _at_least(
            action_input["standardized_effect"],
            EFFECT_DENOMINATOR_MINIMUM,
        )
        and _at_least(
            deterministic_input["standardized_effect"],
            EFFECT_DENOMINATOR_MINIMUM,
        )
        and all(
            value["action_output"]["finite"] is True
            and value["deterministic_output"]["finite"] is True
            and value["action_output"]["active_fraction"]
            >= MAIN_STAGE_ACTIVE_FRACTION
            and value["deterministic_output"]["active_fraction"]
            >= MAIN_STAGE_ACTIVE_FRACTION
            and value["action_local_gain"] is not None
            and value["deterministic_local_gain"] is not None
            and value["action_to_deterministic_gain_ratio"] is not None
            for value in outputs.values()
        )
    )
    return {
        "epsilon": LOCAL_EPSILON,
        "action_input": action_input,
        "deterministic_input": deterministic_input,
        **outputs,
        "valid": valid,
    }


def _gain(output_effect: object, input_effect: object) -> float | None:
    if (
        output_effect is None
        or input_effect is None
        or float(input_effect) < EFFECT_DENOMINATOR_MINIMUM
    ):
        return None
    value = float(output_effect) / float(input_effect)
    return value if math.isfinite(value) else None


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
        raise ValueError("layerwise action-effect Episode identities are not unique")
