"""Aggregate the frozen R0001-P05 batch-arm formal runs."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Mapping, Sequence


REQUIRED_ARMS = ("duplicate", "same_source", "cross_source")
PHYSICAL_COMPONENTS = ("visual_latent", "proprioception")


def aggregate_foundation_batch_arms(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped = _group_reports(reports)
    seed_reports = {}
    differences = []
    for seed, arms in sorted(grouped.items()):
        baseline = arms["same_source"]
        candidate = arms["cross_source"]
        duplicate = arms["duplicate"]
        _require_comparable(seed, duplicate, baseline, candidate)
        baseline_metrics = _physical_metrics(baseline)
        candidate_metrics = _physical_metrics(candidate)
        difference = (
            min(candidate_metrics["ratios"].values())
            - min(baseline_metrics["ratios"].values())
        )
        differences.append(difference)
        checks = {
            "candidate_all_physical_ratios_at_least_1_05": all(
                value >= 1.05 for value in candidate_metrics["ratios"].values()
            ),
            "candidate_all_shuffle_families_pass": all(
                candidate_metrics["shuffle_passes"].values()
            ),
            "candidate_worst_ratio_above_same_source": difference > 0.0,
            "candidate_true_errors_no_higher": all(
                candidate_metrics["true_errors"][name]
                <= baseline_metrics["true_errors"][name]
                for name in baseline_metrics["true_errors"]
            ),
            "action_execution_no_regression": _action_execution_no_regression(
                baseline, candidate
            ),
            "collision_no_regression": _collision_no_regression(
                baseline, candidate
            ),
            "wall_clock_increase_at_most_20_percent": (
                float(candidate["elapsed_seconds"])
                <= 1.20 * float(baseline["elapsed_seconds"])
            ),
            "all_frozen_components_unchanged": all(
                value.get("frozen_components_unchanged") is True
                for value in arms.values()
            ),
        }
        seed_reports[str(seed)] = {
            "passed": all(checks.values()),
            "checks": checks,
            "candidate_minus_same_source_worst_ratio": difference,
            "duplicate": _arm_summary(duplicate),
            "same_source": _arm_summary(baseline),
            "cross_source": _arm_summary(candidate),
        }
    median_difference = statistics.median(differences)
    cross_seed_checks = {
        "three_frozen_seeds": len(seed_reports) == 3,
        "all_seed_differences_positive": all(value > 0.0 for value in differences),
        "median_difference_at_least_0_02": median_difference >= 0.02,
        "all_seed_checks_pass": all(
            value["passed"] for value in seed_reports.values()
        ),
    }
    accepted = all(cross_seed_checks.values())
    return {
        "schema_version": "hwr.foundation-batch-arm-aggregate/v1",
        "proposal_id": "R0001-P05",
        "decision": "accepted" if accepted else "rejected",
        "cross_seed_checks": cross_seed_checks,
        "candidate_minus_same_source_differences": differences,
        "median_candidate_minus_same_source_difference": median_difference,
        "seeds": seed_reports,
    }


def load_batch_arm_report(path: Path) -> dict[str, object]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != "hwr.foundation-batch-arm-run/v1"
        or value.get("mode") != "formal"
        or value.get("decision") != "completed"
    ):
        raise ValueError(f"batch-arm report is not a completed formal run: {path}")
    return value


def _group_reports(
    reports: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, Mapping[str, object]]]:
    grouped: dict[int, dict[str, Mapping[str, object]]] = {}
    for report in reports:
        seed = int(report.get("seed", -1))
        arm = str(report.get("arm", ""))
        if seed < 0 or arm not in REQUIRED_ARMS or arm in grouped.setdefault(seed, {}):
            raise ValueError("batch-arm report identities are incomplete or duplicated")
        grouped[seed][arm] = report
    if len(grouped) != 3 or any(set(value) != set(REQUIRED_ARMS) for value in grouped.values()):
        raise ValueError("batch-arm aggregate requires three seeds and all arms")
    return grouped


def _require_comparable(
    seed: int,
    *reports: Mapping[str, object],
) -> None:
    first = reports[0]
    names = (
        "source_commit",
        "seed",
        "input_identity",
        "schedule_sha256",
        "updates",
        "audit_updates",
    )
    if int(first["seed"]) != seed or any(
        report.get(name) != first.get(name)
        for report in reports[1:]
        for name in names
    ):
        raise ValueError(f"batch-arm seed {seed} reports are not comparable")
    if any(report.get("schedule_audit", {}).get("passed") is not True for report in reports):
        raise ValueError(f"batch-arm seed {seed} schedule audit failed")


def _physical_metrics(report: Mapping[str, object]) -> dict[str, dict[str, object]]:
    diagnostic = report["final_audit"]["diagnostic"]
    ratios = {}
    errors = {}
    shuffle_passes = {}
    groups = {"aggregate": diagnostic}
    groups.update(
        {
            f"task:{task_id}": value
            for task_id, value in diagnostic["partitions"].items()
        }
    )
    for group_name, value in groups.items():
        physical = value["one_step_action_utilization"]
        for component in PHYSICAL_COMPONENTS:
            component_report = physical["report"]["component_reports"][component]
            key = f"{group_name}|{component}"
            ratios[key] = float(component_report["shuffled_to_true_ratio"])
            errors[key] = float(component_report["true_error"])
        shuffle_passes[group_name] = (
            physical["shuffle_statistics"]["robust_passed"] is True
            and float(physical["shuffle_statistics"]["ratio_p05"]) >= 1.05
        )
    return {
        "ratios": ratios,
        "true_errors": errors,
        "shuffle_passes": shuffle_passes,
    }


def _action_execution_no_regression(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> bool:
    left = baseline["final_audit"]["diagnostic"]["action_execution_validation"]
    right = candidate["final_audit"]["diagnostic"]["action_execution_validation"]
    lower_better = (
        "brier_score",
        "intervention_action_normalized_rmse",
        "identity_action_normalized_rmse",
        "out_of_bounds_rate",
    )
    higher_better = ("recall", "pr_auc")
    return _partition_metrics_no_regression(
        left["partitions"], right["partitions"], lower_better, higher_better
    )


def _collision_no_regression(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> bool:
    left = baseline["final_audit"]["diagnostic"]["collision_validation"]
    right = candidate["final_audit"]["diagnostic"]["collision_validation"]
    lower_better = ("brier_score", "false_positive_rate")
    higher_better = (
        "recall",
        "pr_auc",
        "terminal_alignment_rate",
        "shuffled_to_true_brier_ratio",
    )
    return _partition_metrics_no_regression(
        left["partitions"], right["partitions"], lower_better, higher_better
    )


def _partition_metrics_no_regression(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    lower_better: tuple[str, ...],
    higher_better: tuple[str, ...],
) -> bool:
    if set(baseline) != set(candidate):
        return False
    return all(
        all(float(candidate[task][name]) <= float(baseline[task][name]) for name in lower_better)
        and all(float(candidate[task][name]) >= float(baseline[task][name]) for name in higher_better)
        for task in baseline
    )


def _arm_summary(report: Mapping[str, object]) -> dict[str, object]:
    metrics = _physical_metrics(report)
    return {
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "minimum_physical_ratio": min(metrics["ratios"].values()),
        "maximum_true_error": max(metrics["true_errors"].values()),
        "all_shuffle_families_pass": all(metrics["shuffle_passes"].values()),
        "physical_ratios": metrics["ratios"],
        "true_errors": metrics["true_errors"],
    }
