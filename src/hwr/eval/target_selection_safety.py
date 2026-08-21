"""Safety and contact-intensity guards for the P41 target-selection diagnostic."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np

from hwr.eval.factorial_benchmark import clopper_pearson
from hwr.eval.factorial_benchmark import _beta_quantile
from hwr.eval.target_selection import (
    POWER_PAIR_COUNTS,
    POWER_SCHEMA,
    POWER_SEED,
    POWER_TRIALS,
    TASK_IDS,
)


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_264_142
TARGET_ROLES = ("manipulated_object", "articulation")


def exact_mcnemar(candidate_only: int, control_only: int) -> float:
    discordant = candidate_only + control_only
    if min(candidate_only, control_only) < 0 or discordant == 0:
        return 1.0
    return min(
        1.0,
        sum(
            math.comb(discordant, value)
            for value in range(candidate_only, discordant + 1)
        )
        / 2**discordant,
    )


def paired_primary_statistics(
    records: list[Mapping[str, object]],
) -> dict[str, object]:
    supported = [value for value in records if value["domain"] == "supported"]
    if not supported:
        raise ValueError("supported ITT records are required")
    if any(value["resolved"] is not True for value in supported):
        raise ValueError("unresolved pair cannot enter primary statistics")
    candidate_only = sum(
        value["candidate_event"] == 1 and value["control_event"] == 0
        for value in supported
    )
    control_only = sum(
        value["candidate_event"] == 0 and value["control_event"] == 1
        for value in supported
    )
    lower = (
        0.0
        if candidate_only == 0
        else _beta_quantile(0.05, candidate_only, control_only + 1)
    )
    return {
        "planned_pair_count": len(supported),
        "candidate_only": candidate_only,
        "control_only": control_only,
        "discordant": candidate_only + control_only,
        "concordant_positive": sum(
            value["candidate_event"] == value["control_event"] == 1
            for value in supported
        ),
        "concordant_negative": sum(
            value["candidate_event"] == value["control_event"] == 0
            for value in supported
        ),
        "one_sided_exact_mcnemar_p": exact_mcnemar(
            candidate_only, control_only
        ),
        "discordant_candidate_probability_exact_95_lower": lower,
        "delta_itt": (candidate_only - control_only) / len(supported),
        "by_task": _direction_counts(supported, "task_id"),
        "by_observation_latency": _direction_counts(
            supported, "observation_latency_steps"
        ),
    }


def evaluate_synthetic_power(
    *, trials: int = POWER_TRIALS, seed: int = POWER_SEED
) -> dict[str, object]:
    if trials <= 0 or seed < 0:
        raise ValueError("power trials and seed must be positive")
    rng = np.random.default_rng(seed)
    summaries = []
    selected = None
    for pair_count in POWER_PAIR_COUNTS:
        conditions = []
        for name, p10, p01 in (
            ("null_005", 0.05, 0.05),
            ("null_025", 0.25, 0.25),
            ("planted", 0.40, 0.10),
        ):
            passes = 0
            shape = (3, 2, 3, pair_count // 18)
            for _ in range(trials):
                uniform = rng.random(shape)
                delta = np.where(
                    uniform < p10, 1, np.where(uniform < p10 + p01, -1, 0)
                )
                passes += int(_synthetic_gate(delta))
            lower, upper = clopper_pearson(passes, trials)
            conditions.append(
                {
                    "condition": name,
                    "p10": p10,
                    "p01": p01,
                    "passes": passes,
                    "trials": trials,
                    "rate": passes / trials,
                    "clopper_pearson_95_lower": lower,
                    "clopper_pearson_95_upper": upper,
                }
            )
        worst_null = max(
            value["clopper_pearson_95_upper"] for value in conditions[:2]
        )
        planted_lower = conditions[2]["clopper_pearson_95_lower"]
        passed = worst_null <= 0.05 and planted_lower >= 0.80
        summaries.append(
            {
                "pair_count": pair_count,
                "conditions": conditions,
                "worst_null_fpr_clopper_pearson_95_upper": worst_null,
                "planted_power_clopper_pearson_95_lower": planted_lower,
                "passes": passed,
            }
        )
        if selected is None and passed:
            selected = pair_count
    return {
        "schema_version": POWER_SCHEMA,
        "candidate_pair_counts": list(POWER_PAIR_COUNTS),
        "trials": trials,
        "seed": seed,
        "alternative": {
            "p10": 0.40,
            "p01": 0.10,
            "discordance": 0.50,
            "true_itt_delta": 0.30,
            "acceptance_effect_floor": 0.20,
        },
        "nulls": [
            {"p10": 0.05, "p01": 0.05},
            {"p10": 0.25, "p01": 0.25},
        ],
        "summaries": summaries,
        "selected_pair_count": selected,
        "decision": "power_passed" if selected == 54 else "inconclusive_power",
    }


def evaluate_safety_guards(
    records: list[Mapping[str, object]], *, formal: bool = True
) -> dict[str, object]:
    roles = ("candidate", "control")
    branches = [record[role] for record in records for role in roles]
    checks = {
        "invalid_force_zero": all(
            not branch["invalid_force_count"] for branch in branches
        ),
        "severe_collision_zero": all(
            not branch["severe_collision_count"] for branch in branches
        ),
        "supported_stale_action_applied_zero": all(
            record["domain"] != "supported"
            or all(not record[role]["stale_action_applied_count"] for role in roles)
            for record in records
        ),
        "p40_conservation": all(
            branch["p40_conservation"]["maximum_absolute_difference"] <= 1.0e-12
            for branch in branches
        ),
        "action_bounds": all(
            branch["action_bounds_valid"] for branch in branches
        ),
    }
    result = {"hard_guard": {"checks": checks, "passed": all(checks.values())}}
    if formal:
        result["non_target_allowed_contact_guard"] = _non_target_guard(records)
        result["target_contact_intensity_guard"] = _target_contact_guard(records)
    return result


def _non_target_guard(
    records: list[Mapping[str, object]],
) -> dict[str, object]:
    supported = [record for record in records if record["domain"] == "supported"]
    rows = {
        category: {
            metric: _paired_guard(
                supported,
                {
                    role: np.asarray(
                        [
                            _non_target_value(record[role], category, metric)
                            for record in supported
                        ]
                    )
                    for role in ("candidate", "control")
                },
                f"{category}|{metric}",
            )
            for metric in ("peak_force", "cumulative_impulse")
        }
        for category in (
            "floor_support",
            "target_container",
            "non_event_allowed",
        )
    }
    return {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "categories": rows,
        "passed": all(
            metric["passed"]
            for category in rows.values()
            for metric in category.values()
        ),
    }


def _paired_guard(records, values, label: str) -> dict[str, object]:
    candidate_mean = float(values["candidate"].mean())
    control_mean = float(values["control"].mean())
    point = _ratio(candidate_mean, control_mean)
    strata: dict[tuple[object, object, object], list[int]] = {}
    for index, record in enumerate(records):
        key = (
            record["task_id"],
            record["observation_latency_steps"],
            record["action_latency_steps"],
        )
        strata.setdefault(key, []).append(index)
    rng = _rng(label)
    indices = np.concatenate(
        [
            rng.choice(pool, (BOOTSTRAP_SAMPLES, len(pool)), replace=True)
            for pool in strata.values()
        ],
        axis=1,
    )
    ratios = _ratios(
        values["candidate"][indices].mean(axis=1),
        values["control"][indices].mean(axis=1),
    )
    upper = float(np.quantile(ratios, 0.95))
    return {
        "point_ratio": point,
        "one_sided_95_upper": upper,
        "passed": point <= 1.10 and upper <= 1.25,
    }


def _non_target_value(branch, category: str, metric: str) -> float:
    report = branch["entity_contact_graph"]
    peak_field = (
        "category_peak_force"
        if category != "non_event_allowed"
        else "substep_peak_force"
    )
    if category in ("floor_support", "target_container"):
        field = peak_field if metric == "peak_force" else metric
        return float(report["legacy_p40_categories"][category][field])
    event_entities = set(branch["main_event_entities"])
    field = peak_field if metric == "peak_force" else metric
    values = [
        float(edge[field])
        for edge in report["robot_environment_edges"]
        if edge["entity"].split(":", 1)[0] in TARGET_ROLES
        and edge["entity"] not in event_entities
    ]
    return max(values, default=0.0) if metric == "peak_force" else sum(values)


def _target_contact_guard(
    records: list[Mapping[str, object]],
) -> dict[str, object]:
    supported = [record for record in records if record["domain"] == "supported"]
    samples = {"candidate": [], "control": []}
    for record in supported:
        for role in samples:
            sample = _target_episode_sample(record[role])
            if sample is not None:
                samples[role].append((record["task_id"], sample))
    role_counts = {
        role: {
            "positive_episode_count": len(values),
            "by_task": {
                task: sum(task_id == task for task_id, _ in values)
                for task in TASK_IDS
            },
        }
        for role, values in samples.items()
    }
    sufficient = all(
        row["positive_episode_count"] >= 12
        and min(row["by_task"].values()) >= 2
        for row in role_counts.values()
    )
    if not sufficient:
        return {"supported": False, "passed": False, "roles": role_counts}
    ratios = {
        field: _target_metric_guard(samples, field)
        for field in ("peak_force", "mean_contact_force")
    }
    return {
        "supported": True,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "roles": role_counts,
        "sample_definition": (
            "positive Episode max peak and total impulse/duration; equal task weights"
        ),
        "ratios": ratios,
        "passed": all(row["passed"] for row in ratios.values()),
    }


def _target_episode_sample(branch) -> dict[str, float] | None:
    edges = [
        edge
        for edge in branch["entity_contact_graph"]["robot_environment_edges"]
        if edge["entity"].split(":", 1)[0] in TARGET_ROLES
        and float(edge["contact_duration_seconds"]) > 0.0
    ]
    if not edges:
        return None
    duration = sum(float(edge["contact_duration_seconds"]) for edge in edges)
    sample = {
        "peak_force": max(float(edge["substep_peak_force"]) for edge in edges),
        "mean_contact_force": (
            sum(float(edge["cumulative_impulse"]) for edge in edges) / duration
        ),
    }
    if duration <= 0.0 or not all(
        math.isfinite(value) and value >= 0.0 for value in sample.values()
    ):
        raise ValueError("target contact sample is invalid")
    return sample


def _target_metric_guard(samples, field: str) -> dict[str, object]:
    candidate_point = _task_mean(samples["candidate"], field)
    control_point = _task_mean(samples["control"], field)
    point = _ratio(candidate_point, control_point)
    rng = _rng(field)
    draws = {}
    for role, values in samples.items():
        task_draws = []
        for task in TASK_IDS:
            pool = np.asarray(
                [sample[field] for task_id, sample in values if task_id == task]
            )
            sampled = rng.choice(
                pool, (BOOTSTRAP_SAMPLES, len(pool)), replace=True
            )
            task_draws.append(np.log1p(sampled).mean(axis=1))
        draws[role] = np.stack(task_draws).mean(axis=0)
    upper = float(
        np.quantile(_ratios(draws["candidate"], draws["control"]), 0.95)
    )
    return {
        "point_ratio": point,
        "one_sided_95_upper": upper,
        "passed": point <= 1.25 and upper <= 1.50,
    }


def _synthetic_gate(delta: np.ndarray) -> bool:
    candidate_only = int(np.count_nonzero(delta == 1))
    control_only = int(np.count_nonzero(delta == -1))
    return (
        exact_mcnemar(candidate_only, control_only) <= 0.05
        and float(delta.mean()) >= 0.20
        and all(float(delta[index].sum()) >= 0.0 for index in range(3))
        and all(float(delta[:, index].sum()) >= 0.0 for index in range(2))
    )


def _direction_counts(records, field: str) -> dict[str, dict[str, int]]:
    result = {}
    for key in sorted({str(value[field]) for value in records}):
        selected = [value for value in records if str(value[field]) == key]
        result[key] = {
            "candidate_only": sum(
                value["candidate_event"] == 1
                and value["control_event"] == 0
                for value in selected
            ),
            "control_only": sum(
                value["candidate_event"] == 0
                and value["control_event"] == 1
                for value in selected
            ),
        }
    return result


def _task_mean(samples, field: str) -> float:
    return float(
        np.mean(
            tuple(
                np.mean(
                    [
                        math.log1p(value[field])
                        for task_id, value in samples
                        if task_id == task
                    ]
                )
                for task in TASK_IDS
            )
        )
    )


def _rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(
        hashlib.sha256(f"{BOOTSTRAP_SEED}|{label}".encode()).digest()[:8],
        "big",
    )
    return np.random.default_rng(seed)


def _ratio(candidate: float, control: float) -> float:
    if candidate == control == 0.0:
        return 1.0
    return math.inf if control == 0.0 else candidate / control


def _ratios(candidate: np.ndarray, control: np.ndarray) -> np.ndarray:
    result = np.full(candidate.shape, np.inf)
    np.divide(candidate, control, out=result, where=control != 0.0)
    result[(candidate == 0.0) & (control == 0.0)] = 1.0
    return result
