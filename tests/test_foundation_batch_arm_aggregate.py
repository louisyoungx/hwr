from __future__ import annotations

from copy import deepcopy

from hwr.eval.foundation_batch_arm_aggregate import (
    aggregate_foundation_batch_arms,
)


TASKS = ("a", "b", "c")


def _validation(collision: bool = False):
    values = {}
    for task in TASKS:
        if collision:
            values[task] = {
                "recall": 0.9,
                "pr_auc": 0.8,
                "brier_score": 0.05,
                "false_positive_rate": 0.02,
                "terminal_alignment_rate": 0.9,
                "shuffled_to_true_brier_ratio": 1.2,
            }
        else:
            values[task] = {
                "recall": 0.9,
                "pr_auc": 0.8,
                "brier_score": 0.05,
                "intervention_action_normalized_rmse": 0.05,
                "identity_action_normalized_rmse": 0.02,
                "out_of_bounds_rate": 0.0,
            }
    return {"partitions": values}


def _physical(ratio: float, true_error: float):
    components = {
        name: {
            "shuffled_to_true_ratio": ratio,
            "true_error": true_error,
        }
        for name in ("visual_latent", "proprioception")
    }
    return {
        "report": {"component_reports": components},
        "shuffle_statistics": {"robust_passed": True, "ratio_p05": ratio},
    }


def _report(seed: int, arm: str, ratio: float, *, elapsed: float = 100.0):
    physical = _physical(ratio, 0.5)
    diagnostic = {
        "one_step_action_utilization": deepcopy(physical),
        "partitions": {
            task: {"one_step_action_utilization": deepcopy(physical)}
            for task in TASKS
        },
        "action_execution_validation": _validation(),
        "collision_validation": _validation(collision=True),
    }
    return {
        "source_commit": "a" * 40,
        "seed": seed,
        "arm": arm,
        "input_identity": {"replay": "x"},
        "schedule_sha256": f"{seed:064x}",
        "updates": 1600,
        "audit_updates": list(range(200, 1601, 200)),
        "schedule_audit": {"passed": True},
        "frozen_components_unchanged": True,
        "elapsed_seconds": elapsed,
        "final_audit": {"diagnostic": diagnostic},
    }


def _accepted_reports():
    reports = []
    for seed in (1, 2, 3):
        reports.extend(
            (
                _report(seed, "duplicate", 1.01),
                _report(seed, "same_source", 1.06),
                _report(seed, "cross_source", 1.09),
            )
        )
    return reports


def test_batch_arm_aggregate_accepts_consistent_cross_source_improvement() -> None:
    report = aggregate_foundation_batch_arms(_accepted_reports())

    assert report["decision"] == "accepted"
    assert report["cross_seed_checks"]["all_seed_differences_positive"]
    assert report["median_candidate_minus_same_source_difference"] > 0.02


def test_batch_arm_aggregate_rejects_one_seed_regression() -> None:
    reports = _accepted_reports()
    reports[-1] = _report(3, "cross_source", 1.04)

    report = aggregate_foundation_batch_arms(reports)

    assert report["decision"] == "rejected"
    assert not report["cross_seed_checks"]["all_seed_differences_positive"]
