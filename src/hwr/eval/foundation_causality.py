"""Independent validation of serialized foundation action-causality evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from hwr.train.foundation_registry import ACTION_CAUSALITY_SCHEMA
from hwr.world_model import (
    ACTION_CAUSALITY_COMPONENTS,
    ActionCausalityCriteria,
    assess_action_causality,
    counterfactual_report_from_dict,
)


def require_foundation_causality_structure(
    report: Mapping[str, Any], run_manifest: Mapping[str, Any]
) -> None:
    expected_tasks = {
        str(task["task_id"]) for task in run_manifest.get("tasks", ())
    }
    partitions = report.get("partitions", {})
    if not expected_tasks or not isinstance(partitions, Mapping) or set(partitions) != expected_tasks:
        raise ValueError("action causality partition evidence is incomplete")
    assessment = report.get("assessment", {})
    training = run_manifest["training_config"]
    criteria = ActionCausalityCriteria(
        float(training["minimum_action_causality_ratio"]),
        float(training["minimum_action_causality_horizon_fraction"]),
    )
    _require_raw_assessments(report, partitions, criteria)
    physical = report.get("one_step_action_utilization", {})
    required_components = set(ACTION_CAUSALITY_COMPONENTS)
    component_assessments = assessment.get("components", {})
    physical_components = {"visual_latent", "proprioception"}
    physical_assessment = physical.get("assessment", {})
    shuffle_repeats = int(report.get("shuffle_repeats", 0))
    _require_all_shuffle_statistics(
        report, partitions, physical, criteria, shuffle_repeats
    )
    if (
        report.get("schema_version") != ACTION_CAUSALITY_SCHEMA
        or report.get("action_source") != "actual_executed_action"
        or report.get("safety_action_source") != "actor_proposal"
        or report.get("counterfactual_pairing") != "proposal-executed-pair/v1"
        or report.get("counterfactual_transform")
        != "deterministic-global-derangement/v1"
        or report.get("partition_key") != "task_id"
        or physical.get("conditioning") != "teacher-forced-posterior-state/v1"
        or set(physical.get("physical_components", ())) != physical_components
        or physical_assessment.get("passed") is not True
        or not _all_lower_bounds_pass(report, partitions, physical)
        or set(physical_assessment.get("components", {})) != physical_components
        or not expected_tasks
        or set(partitions) != expected_tasks
        or assessment.get("aggregate_passed") is not True
        or assessment.get("required_components_passed") is not True
        or set(component_assessments) != required_components
        or any(
            component_assessments[name].get("passed") is not True
            for name in physical_components
        )
        or assessment.get("all_partitions_passed") is not True
        or not _partition_assessments_pass(
            partitions, required_components, physical_components
        )
    ):
        raise ValueError("action causality partition evidence is incomplete")
    _require_window_selection(report, run_manifest, expected_tasks)


def _require_raw_assessments(
    report: Mapping[str, Any],
    partitions: Mapping[str, Any],
    criteria: ActionCausalityCriteria,
) -> None:
    _require_assessment_matches(
        report.get("report"), report.get("assessment", {}), criteria, "aggregate"
    )
    physical = report.get("one_step_action_utilization", {})
    _require_assessment_matches(
        physical.get("report"),
        physical.get("assessment", {}),
        criteria,
        "aggregate one-step physical",
    )
    for task_id, partition in partitions.items():
        _require_assessment_matches(
            partition.get("report"), partition.get("assessment", {}), criteria, task_id
        )
        one_step = partition.get("one_step_action_utilization", {})
        _require_assessment_matches(
            one_step.get("report"),
            one_step.get("assessment", {}),
            criteria,
            f"{task_id} one-step physical",
        )


def _require_all_shuffle_statistics(
    report: Mapping[str, Any],
    partitions: Mapping[str, Any],
    physical: Mapping[str, Any],
    criteria: ActionCausalityCriteria,
    repeats: int,
) -> None:
    aggregate_count = repeats * len(partitions)
    _require_shuffle_statistics(
        report.get("shuffle_statistics", {}), criteria, aggregate_count, "aggregate"
    )
    _require_shuffle_statistics(
        physical.get("shuffle_statistics", {}),
        criteria,
        aggregate_count,
        "aggregate one-step physical",
    )
    for task_id, partition in partitions.items():
        _require_shuffle_statistics(
            partition.get("shuffle_statistics", {}), criteria, repeats, task_id
        )
        _require_shuffle_statistics(
            partition.get("one_step_action_utilization", {}).get(
                "shuffle_statistics", {}
            ),
            criteria,
            repeats,
            f"{task_id} one-step physical",
        )


def _partition_assessments_pass(
    partitions: Mapping[str, Any],
    required_components: set[str],
    physical_components: set[str],
) -> bool:
    return all(
        value.get("assessment", {}).get("passed") is True
        and value.get("assessment", {}).get("required_components_passed") is True
        and set(value.get("assessment", {}).get("components", {}))
        == required_components
        and value.get("one_step_action_utilization", {})
        .get("assessment", {})
        .get("passed")
        is True
        and set(
            value.get("one_step_action_utilization", {})
            .get("assessment", {})
            .get("components", {})
        )
        == physical_components
        for value in partitions.values()
    )


def _all_lower_bounds_pass(
    report: Mapping[str, Any],
    partitions: Mapping[str, Any],
    physical: Mapping[str, Any],
) -> bool:
    return (
        report.get("shuffle_statistics", {}).get("robust_passed") is True
        and physical.get("shuffle_statistics", {}).get("robust_passed") is True
        and all(
            value.get("shuffle_statistics", {}).get("robust_passed") is True
            and value.get("one_step_action_utilization", {})
            .get("shuffle_statistics", {})
            .get("robust_passed")
            is True
            for value in partitions.values()
        )
    )


def _require_assessment_matches(
    raw_report: object,
    claimed: Mapping[str, Any],
    criteria: ActionCausalityCriteria,
    label: str,
) -> None:
    if not isinstance(raw_report, Mapping):
        raise ValueError(f"action causality raw evidence is missing: {label}")
    expected = assess_action_causality(
        counterfactual_report_from_dict(raw_report), criteria
    )
    if any(claimed.get(name) != value for name, value in expected.items()):
        raise ValueError(f"action causality assessment differs from evidence: {label}")


def _require_shuffle_statistics(
    value: object,
    criteria: ActionCausalityCriteria,
    expected_count: int,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or expected_count <= 0:
        raise ValueError(f"action shuffle statistics are missing: {label}")
    ratios = [
        float(
            assess_action_causality(
                counterfactual_report_from_dict(report), criteria
            )["shuffled_to_true_ratio"]
        )
        for report in value.get("reports", ())
    ]
    if len(ratios) != expected_count:
        raise ValueError(f"action shuffle repeat count differs: {label}")
    p05 = float(np.quantile(np.asarray(ratios, np.float64), 0.05))
    report_passes = [
        assess_action_causality(
            counterfactual_report_from_dict(report), criteria
        )["passed"]
        is True
        for report in value.get("reports", ())
    ]
    lower_bound_passed = p05 >= criteria.minimum_shuffled_to_true_ratio
    expected = {
        "count": expected_count,
        "shuffled_to_true_ratios": ratios,
        "ratio_p05": p05,
        "ratio_median": float(np.median(ratios)),
        "ratio_p95": float(np.quantile(ratios, 0.95)),
        "lower_bound_passed": lower_bound_passed,
        "passed_fraction": sum(report_passes) / len(report_passes),
        "all_reports_passed": all(report_passes),
        "robust_passed": lower_bound_passed and all(report_passes),
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise ValueError(f"action shuffle statistics differ from evidence: {label}")


def _require_window_selection(
    report: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    expected_tasks: set[str],
) -> None:
    configured = int(
        run_manifest["training_config"]["causality_audit_windows_per_task"]
    )
    counts = {task_id: 0 for task_id in expected_tasks}
    intervals: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for window in report.get("window_selection", ()):
        task_id = str(window.get("task_id"))
        start = int(window.get("transition_start", -1))
        stop = int(window.get("transition_stop", -1))
        if task_id not in counts or start < 0 or stop <= start:
            raise ValueError("action causality window selection is invalid")
        counts[task_id] += 1
        intervals.setdefault((task_id, str(window.get("episode_id"))), []).append(
            (start, stop)
        )
    if any(value != configured for value in counts.values()):
        raise ValueError("action causality window coverage differs from training")
    if any(_overlaps(values) for values in intervals.values()):
        raise ValueError("action causality windows overlap")


def _overlaps(intervals: Sequence[tuple[int, int]]) -> bool:
    ordered = sorted(intervals)
    return any(
        current[0] < previous[1]
        for previous, current in zip(ordered, ordered[1:])
    )
