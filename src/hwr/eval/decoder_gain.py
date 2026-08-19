"""Decoder layerwise gain diagnostic for R0001-P24."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from hwr.eval.decoder_gain_calibration import (
    ACTIVE_FRACTION_MINIMUM,
    ACTIVE_SCALE_MINIMUM,
    EFFECT_DENOMINATOR_MINIMUM,
    PATH_SEGMENTS,
    _calibrated_effect,
    _decoder_stages,
    _edge_functions,
    _head,
    _path_report,
    build_decoder_calibration,
    deserialize_decoder_calibration,
    serialize_decoder_calibration,
)
from hwr.eval.layerwise_action_effect import ACTION_SHIFTS, _forward_stages
from hwr.eval.prior_argmax_effect import _categorical, _hard_code, _own_scale_effect
from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMSequence


HEAD_NAMES = ("visual", "proprioception")
STAGE_NAMES = (
    "feature",
    "linear_preactivation",
    "layer_norm_normalized",
    "layer_norm_affine",
    "hidden",
    "output",
)
EDGE_NAMES = (
    "feature_to_linear",
    "linear_to_norm",
    "norm_to_affine",
    "affine_to_hidden",
    "hidden_to_output",
)
EXPECTED_TRANSITION_COUNT = 16
FEATURE_EFFECT_MINIMUM = 0.05
RETENTION_MAXIMUM = 0.50
RECONSTRUCTION_COSINE_MINIMUM = 0.90
RECONSTRUCTION_RELATIVE_ERROR_MAXIMUM = 0.10


@dataclass(frozen=True)
class DecoderBranches:
    true_feature: torch.Tensor
    shifted_features: Mapping[int, torch.Tensor]
    p23_guard: Mapping[int, Mapping[str, object]]
    p23_endpoint_valid: Mapping[int, bool]


def build_true_decoder_feature(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    executed_actions: torch.Tensor,
) -> torch.Tensor:
    deterministic, stochastic, actions = _branch_inputs(
        model, sequence, executed_actions
    )
    with torch.inference_mode():
        stages = _forward_stages(model, deterministic, stochastic, actions)
        code = _independent_hard_code(stages["prior_probability"], model)
        feature = torch.cat((stages["next_deterministic"], code), dim=-1)
    return feature.clone()


def build_decoder_branches(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    executed_actions: torch.Tensor,
) -> DecoderBranches:
    deterministic, stochastic, actions = _branch_inputs(
        model, sequence, executed_actions
    )
    with torch.inference_mode():
        true_stages = _forward_stages(model, deterministic, stochastic, actions)
        true_code = _independent_hard_code(
            true_stages["prior_probability"], model
        )
        true_feature = torch.cat(
            (true_stages["next_deterministic"], true_code), dim=-1
        )
        shifted_features = {}
        guards = {}
        endpoint_valid = {}
        for shift in ACTION_SHIFTS:
            shifted_stages = _forward_stages(
                model,
                deterministic,
                stochastic,
                torch.roll(actions, shifts=shift, dims=1),
            )
            shifted_code = _independent_hard_code(
                shifted_stages["prior_probability"], model
            )
            shifted = torch.cat(
                (shifted_stages["next_deterministic"], shifted_code), dim=-1
            )
            shifted_features[shift] = shifted
            guards[shift] = _own_scale_effect(true_feature, shifted)
            endpoint_valid[shift] = (
                _official_code_matches(model, true_stages, true_code)
                and _official_code_matches(
                    model, shifted_stages, shifted_code
                )
            )
    return DecoderBranches(
        true_feature.clone(),
        {shift: value.clone() for shift, value in shifted_features.items()},
        guards,
        endpoint_valid,
    )


def _branch_inputs(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    executed_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    deterministic = sequence.deterministic[:, :-1].detach()
    stochastic = sequence.stochastic[:, :-1].detach()
    actions = executed_actions.detach()
    if (
        actions.shape[0] != 1
        or actions.shape[1] != EXPECTED_TRANSITION_COUNT
        or deterministic.shape[:2] != actions.shape[:2]
        or stochastic.shape[:2] != actions.shape[:2]
        or deterministic.shape[-1] != model.config.deterministic_dimension
        or stochastic.shape[-1] != model.config.stochastic_dimension
        or actions.shape[-1] != model.config.action_dimension
    ):
        raise ValueError("decoder gain branch inputs are invalid")
    return deterministic, stochastic, actions


def _independent_hard_code(
    probability: torch.Tensor,
    model: ActionConditionedWorldModel,
) -> torch.Tensor:
    categorical = _categorical(probability, model)
    return torch.nn.functional.one_hot(
        categorical.argmax(dim=-1), model.config.stochastic_classes
    ).to(categorical.dtype).flatten(-2)


def _official_code_matches(
    model: ActionConditionedWorldModel,
    stages: Mapping[str, object],
    independent_code: torch.Tensor,
) -> bool:
    logits = _categorical(stages["prior_logits"], model)
    official = model.rssm._sample(logits, sample=False)
    return bool(torch.equal(official, independent_code))


def evaluate_decoder_gain(
    model: ActionConditionedWorldModel,
    branches: DecoderBranches,
    calibration: Mapping[str, object],
) -> dict[str, object]:
    official_true = model.decode_features(branches.true_feature)
    heads = {}
    for head_index, head_name in enumerate(HEAD_NAMES):
        head = _head(model, head_name)
        true_stages, _ = _decoder_stages(head, branches.true_feature)
        true_endpoint_valid = bool(
            torch.allclose(
                official_true[head_index],
                true_stages["output"],
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        )
        shift_reports = {
            str(shift): _evaluate_branch(
                model,
                head,
                head_index,
                true_stages,
                branches.shifted_features[shift],
                calibration["heads"][head_name]["stages"],
                branches.p23_guard[shift],
                branches.p23_endpoint_valid[shift],
                true_endpoint_valid,
                shift,
            )
            for shift in ACTION_SHIFTS
        }
        heads[head_name] = {
            "shifts": shift_reports,
            "assessment": assess_decoder_head_episode(
                {"shifts": shift_reports}
            ),
        }
    report = {
        "schema_version": "hwr.decoder-gain/v1",
        "transition_count": EXPECTED_TRANSITION_COUNT,
        "criteria": _criteria(),
        "heads": heads,
    }
    report["assessment"] = {
        "valid": all(head["assessment"]["valid"] for head in heads.values())
    }
    return report


def aggregate_decoder_gain(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(reports) != 24:
        raise ValueError("decoder gain diagnostic requires 24 reports")
    _validate_episode_identities(reports)
    if any(
        int(report["transition_count"]) != EXPECTED_TRANSITION_COUNT
        for report in reports
    ):
        raise ValueError("decoder gain transition count differs")
    expected_tasks = {
        "clear_dining_table_3d/v1": 6,
        "store_kitchen_items_3d/v1": 6,
        "tidy_living_room_3d/v1": 12,
    }
    task_counts = {task: 0 for task in expected_tasks}
    for report in reports:
        task = str(report["window"]["task_id"])
        if task not in task_counts:
            raise ValueError("decoder gain task identity differs")
        task_counts[task] += 1
    if task_counts != expected_tasks:
        raise ValueError("decoder gain task coverage differs")
    heads = {
        head_name: _aggregate_head(reports, head_name, expected_tasks)
        for head_name in HEAD_NAMES
    }
    valid = all(head["valid"] for head in heads.values())
    return {
        "schema_version": "hwr.decoder-gain-aggregate/v1",
        "episode_count": len(reports),
        "transition_count": sum(
            int(report["transition_count"]) for report in reports
        ),
        "criteria": _criteria(),
        "task_episode_counts": task_counts,
        "heads": heads,
        "assessment": {
            "decision": (
                "diagnostic_invalid" if not valid else "diagnostic_complete"
            ),
            "valid": valid,
        },
    }


def assess_decoder_head_episode(
    report: Mapping[str, object],
) -> dict[str, object]:
    shifts = report["shifts"]
    valid_shift_count = sum(
        shifts[str(shift)]["assessment"]["valid"] is True
        for shift in ACTION_SHIFTS
    )
    valid = valid_shift_count >= 2
    passed_edges = [
        shifts[str(shift)]["assessment"]["localized_edge"]
        for shift in ACTION_SHIFTS
        if shifts[str(shift)]["assessment"]["passed"] is True
    ]
    localized_edge = next(
        (
            edge
            for edge in EDGE_NAMES
            if passed_edges.count(edge) >= 2
        ),
        None,
    )
    passed = valid and localized_edge is not None
    output_guard_count = sum(
        _at_least(
            shifts[str(shift)]["stage_effects"]["output"][
                "standardized_effect"
            ],
            FEATURE_EFFECT_MINIMUM,
        )
        for shift in ACTION_SHIFTS
        if shifts[str(shift)]["assessment"]["valid"] is True
    )
    return {
        "valid": valid,
        "valid_shift_count": valid_shift_count,
        "passed": passed,
        "localized_edge": localized_edge,
        "passed_shift_count": len(passed_edges),
        "output_guard_passed": valid and output_guard_count >= 2,
        "output_guard_shift_count": output_guard_count,
    }


def _evaluate_branch(
    model: ActionConditionedWorldModel,
    head: torch.nn.Sequential,
    head_index: int,
    true_stages: Mapping[str, torch.Tensor],
    shifted_feature: torch.Tensor,
    calibration: Mapping[str, Mapping[str, object]],
    p23_guard: Mapping[str, object],
    p23_endpoint_valid: bool,
    true_decoder_endpoint_valid: bool,
    shift: int,
) -> dict[str, object]:
    shifted_stages, _ = _decoder_stages(head, shifted_feature)
    official_shifted = model.decode_features(shifted_feature)[head_index]
    shifted_decoder_endpoint_valid = bool(
        torch.allclose(
            official_shifted,
            shifted_stages["output"],
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    )
    stage_effects = {
        name: _calibrated_effect(
            true_stages[name], shifted_stages[name], calibration[name]
        )
        for name in STAGE_NAMES
    }
    actual_retentions, first_low, retention_invalid = _scan_retentions(
        stage_effects
    )
    path_reports = {}
    selected_path = None
    if first_low is not None:
        edge = next(
            value
            for value in _edge_functions(head)
            if value[0] == first_low
        )
        edge_name, input_name, output_name, function = edge
        selected_path = _path_report(
            function,
            true_stages[input_name],
            shifted_stages[input_name],
            true_stages[output_name],
            shifted_stages[output_name],
            calibration[input_name],
            calibration[output_name],
        )
        path_reports[edge_name] = selected_path
    endpoint_valid = (
        p23_endpoint_valid
        and true_decoder_endpoint_valid
        and shifted_decoder_endpoint_valid
    )
    stage_valid = (
        p23_guard["guard_passed"] is True
        and all(effect["valid"] for effect in stage_effects.values())
        and _at_least(
            stage_effects["feature"]["standardized_effect"],
            FEATURE_EFFECT_MINIMUM,
        )
        and endpoint_valid
        and not retention_invalid
    )
    assessment = _assess_branch(
        first_low, selected_path, stage_valid
    )
    return {
        "shift": shift,
        "p23_hard_feature_guard": dict(p23_guard),
        "stage_effects": stage_effects,
        "actual_retentions": actual_retentions,
        "first_low_retention_edge": first_low,
        "retention_invalid": retention_invalid,
        "path_reports": path_reports,
        "endpoint_validation": {
            "p23_hard_code_matches_sample_false": p23_endpoint_valid,
            "true_decoder_matches_decode_features": true_decoder_endpoint_valid,
            "shifted_decoder_matches_decode_features": (
                shifted_decoder_endpoint_valid
            ),
        },
        "assessment": assessment,
    }


def _scan_retentions(
    stage_effects: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, float | None], str | None, bool]:
    actual_retentions = {}
    for edge_name, input_name, output_name in _edges():
        retention = _ratio(
            stage_effects[output_name]["standardized_effect"],
            stage_effects[input_name]["standardized_effect"],
        )
        actual_retentions[edge_name] = retention
        if retention is None:
            return actual_retentions, None, True
        if _below(retention, RETENTION_MAXIMUM):
            return actual_retentions, edge_name, False
    return actual_retentions, None, False


def _assess_branch(
    first_low: str | None,
    selected_path: Mapping[str, object] | None,
    stage_valid: bool,
) -> dict[str, object]:
    if first_low is None:
        valid = stage_valid
        state = "not_localized" if valid else "jvp_invalid"
        passed = False
    else:
        passed = (
            stage_valid
            and selected_path is not None
            and selected_path["valid"] is True
            and _below(
                selected_path["path_retention"], RETENTION_MAXIMUM
            )
            and _at_least(
                selected_path["reconstruction_cosine"],
                RECONSTRUCTION_COSINE_MINIMUM,
            )
            and _at_most(
                selected_path["relative_error"],
                RECONSTRUCTION_RELATIVE_ERROR_MAXIMUM,
            )
        )
        valid = passed
        state = "localized" if passed else "jvp_invalid"
    return {
        "state": state,
        "valid": valid,
        "passed": passed,
        "localized_edge": first_low if passed else None,
    }


def _aggregate_head(
    reports: Sequence[Mapping[str, object]],
    head_name: str,
    expected_tasks: Mapping[str, int],
) -> dict[str, object]:
    episode_measurements_valid = all(
        report["heads"][head_name]["assessment"]["valid"] is True
        for report in reports
    )
    all_branches_valid = all(
        report["heads"][head_name]["shifts"][str(shift)][
            "assessment"
        ]["valid"]
        is True
        for report in reports
        for shift in ACTION_SHIFTS
    )
    episode_pass_count = 0
    output_guard_episode_count = 0
    task_pass_counts = {task: 0 for task in expected_tasks}
    output_guard_task_counts = {task: 0 for task in expected_tasks}
    shift_pass_counts = {str(shift): 0 for shift in ACTION_SHIFTS}
    edge_episode_counts = {edge: 0 for edge in EDGE_NAMES}
    for report in reports:
        task = str(report["window"]["task_id"])
        assessment = report["heads"][head_name]["assessment"]
        episode_pass_count += assessment["passed"] is True
        output_guard_episode_count += assessment["output_guard_passed"] is True
        task_pass_counts[task] += assessment["passed"] is True
        output_guard_task_counts[task] += (
            assessment["output_guard_passed"] is True
        )
        edge = assessment["localized_edge"]
        if edge is not None:
            edge_episode_counts[str(edge)] += 1
        for shift in ACTION_SHIFTS:
            shift_pass_counts[str(shift)] += (
                report["heads"][head_name]["shifts"][str(shift)][
                    "assessment"
                ]["passed"]
                is True
            )
    task_requirements = _criteria()["task_pass_requirements"]
    core_passed = (
        episode_pass_count >= 20
        and all(
            task_pass_counts[task] >= required
            for task, required in task_requirements.items()
        )
        and all(count >= 18 for count in shift_pass_counts.values())
    )
    concentrated_edge = next(
        (
            edge
            for edge in EDGE_NAMES
            if edge_episode_counts[edge] >= 16
        ),
        None,
    )
    output_guard_passed = (
        output_guard_episode_count >= 20
        and all(
            output_guard_task_counts[task] >= required
            for task, required in task_requirements.items()
        )
    )
    if core_passed and concentrated_edge is not None:
        state = f"passed({concentrated_edge})"
    elif not episode_measurements_valid or not all_branches_valid:
        state = "jvp_invalid"
    elif output_guard_passed:
        state = "not_localized"
    else:
        state = "output_guard_failed"
    return {
        "valid": state != "jvp_invalid",
        "all_branches_valid": all_branches_valid,
        "episode_measurements_valid": episode_measurements_valid,
        "state": state,
        "localized_edge": concentrated_edge,
        "episode_pass_count": episode_pass_count,
        "shift_pass_counts": shift_pass_counts,
        "task_pass_counts": task_pass_counts,
        "edge_episode_counts": edge_episode_counts,
        "output_guard_episode_count": output_guard_episode_count,
        "output_guard_task_counts": output_guard_task_counts,
    }


def _edges() -> tuple[tuple[str, str, str], ...]:
    return (
        ("feature_to_linear", "feature", "linear_preactivation"),
        (
            "linear_to_norm",
            "linear_preactivation",
            "layer_norm_normalized",
        ),
        (
            "norm_to_affine",
            "layer_norm_normalized",
            "layer_norm_affine",
        ),
        ("affine_to_hidden", "layer_norm_affine", "hidden"),
        ("hidden_to_output", "hidden", "output"),
    )


def _criteria() -> dict[str, object]:
    return {
        "expected_transition_count": EXPECTED_TRANSITION_COUNT,
        "action_shifts": list(ACTION_SHIFTS),
        "path_segments": PATH_SEGMENTS,
        "active_scale_minimum": ACTIVE_SCALE_MINIMUM,
        "active_fraction_minimum": ACTIVE_FRACTION_MINIMUM,
        "effect_denominator_minimum": EFFECT_DENOMINATOR_MINIMUM,
        "feature_effect_minimum": FEATURE_EFFECT_MINIMUM,
        "retention_maximum": RETENTION_MAXIMUM,
        "reconstruction_cosine_minimum": RECONSTRUCTION_COSINE_MINIMUM,
        "reconstruction_relative_error_maximum": (
            RECONSTRUCTION_RELATIVE_ERROR_MAXIMUM
        ),
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


def _at_most(value: object, threshold: float) -> bool:
    return (
        value is not None
        and math.isfinite(float(value))
        and float(value) <= threshold
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
        raise ValueError("decoder gain Episode identities are not unique")
