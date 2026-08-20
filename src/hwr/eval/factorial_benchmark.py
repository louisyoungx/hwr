"""Frozen balanced-factorial benchmark planning and integrity contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hwr.eval.seed_contract import (
    SEED_SCHEMA, derive_domain_seed, plan_episode_seeds, planned_episode_id,
    seed_commitment,
)


EXPERIMENT_ID = "R0001-P36-E2"
PLAN_ID = "R0001-P36-E2-diagnostic-plan"
PLAN_SCHEMA = "hwr.factorial-benchmark-plan/v1"
TERMINAL_SCHEMA = "hwr.factorial-terminal-record/v1"
POWER_SCHEMA = "hwr.factorial-synthetic-power/v1"
TASKS = (
    "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
    "tidy_living_room_3d/v1",
)
OBSERVATION_LATENCIES = (1, 2, 3)
ACTION_LATENCIES = (1, 2, 3)
TRAINING_SEED_SLOTS = (0, 1, 2)
ROLES = ("baseline", "candidate")
SOURCE_AGE_NS = {1: 50_000_000, 2: 100_000_000, 3: 150_000_000}
ZERO_HASH = "0" * 64
VALID_FAILURE_REASONS = frozenset({
    "task_timeout", "task_failure", "safety_rejection", "safety_termination",
    "severe_collision", "policy_invalid", "policy_nan", "policy_exception",
})
POLICY_FAILURE_REASONS = frozenset({"policy_invalid", "policy_nan", "policy_exception"})
INFRASTRUCTURE_REASONS = frozenset(
    {"host_kill", "power_loss", "artifact_corruption", "unattributed_exception"}
)
CELL_KEYS = tuple((task, observation, action) for task in TASKS
                  for observation in OBSERVATION_LATENCIES
                  for action in ACTION_LATENCIES)
SUPPORTED_INDICES = np.asarray([
    index for index, (_, observation, _) in enumerate(CELL_KEYS)
    if observation in (1, 2)
], dtype=np.int64)


class FactorialBenchmarkContractError(ValueError):
    """Raised when a planned or terminal ledger violates the frozen contract."""
@dataclass(frozen=True)
class PowerDesign:
    candidate_n: tuple[int, ...] = (4, 8, 12, 16, 24, 32)
    probabilities: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70)
    shared_fractions: tuple[float, ...] = (0.0, 0.5, 0.9)
    trials: int = 500
    bootstrap_samples: int = 1_000
    base_seed: int = 20_263_602
    workers: int = 0

    def __post_init__(self) -> None:
        if (
            not self.candidate_n
            or any(value <= 0 for value in self.candidate_n)
            or len(set(self.candidate_n)) != len(self.candidate_n)
        ):
            raise ValueError("candidate n values must be unique and positive")
        if not self.probabilities or any(
            not 0.0 < value <= 0.9 for value in self.probabilities
        ):
            raise ValueError("baseline probabilities must be in (0, 0.9]")
        if not self.shared_fractions or any(
            value < 0.0 or value > 1.0 for value in self.shared_fractions
        ):
            raise ValueError("shared-randomness fractions must be in [0, 1]")
        if self.trials <= 0 or self.bootstrap_samples <= 0 or self.base_seed < 0:
            raise ValueError("power simulation counts and seed must be positive")
        if self.workers < 0:
            raise ValueError("power workers cannot be negative")


def benchmark_cells() -> list[dict[str, object]]:
    return [
        {
            "cell_ordinal": index,
            "task_id": task,
            "observation_latency_steps": observation,
            "action_latency_steps": action,
            "maximum_source_age_ns": SOURCE_AGE_NS[observation],
            "domain": "supported" if observation in (1, 2) else "challenge",
        }
        for index, (task, observation, action) in enumerate(CELL_KEYS)
    ]


def ledger_contract() -> dict[str, object]:
    return {
        "primary_ledger": "complete_challenge",
        "full_profile_supported": False,
        "complete_challenge": {
            "cell_count": 27,
            "cell_weight": 1.0 / 27.0,
            "observation_latency_3_weight": 1.0 / 3.0,
            "valid_failures_retained": True,
        },
        "supported_conditional": {
            "cell_count": 18,
            "cell_weight": 1.0 / 18.0,
            "observation_latencies": [1, 2],
            "may_replace_primary": False,
        },
        "future_joint_primary_gate": {
            "delta_complete_macro_percentile_95_lower_strictly_above": 0.0,
            "delta_supported_macro_point_at_least": 0.10,
            "delta_supported_macro_percentile_95_lower_strictly_above": 0.0,
            "per_task_supported_regression_floor": -0.05,
        },
        "guardrails": {
            "severe_collision_count": 0,
            "stale_action_applied_rate": 0.0,
            "existing_safety_and_ablation_gates_unchanged": True,
        },
        "runner_integrity": {
            "planned_manifest_atomic_before_terminal": True,
            "terminal_records_append_only": True,
            "terminal_records_hash_chained": True,
            "missing_terminal_is_unresolved": True,
            "replacement_seed_allowed": False, "complete_case_deletion_allowed": False,
        },
        "source_age_contract": {
            str(latency): {"maximum_source_age_ns": age,
                           "domain": "supported" if latency in (1, 2) else "challenge"}
            for latency, age in SOURCE_AGE_NS.items()},
    }


def build_planned_ledger(
    replicate_count: int, salt: str, *, available: bool = True
) -> dict[str, object]:
    if replicate_count <= 0:
        raise ValueError("planned replicate count must be positive")
    commitment = seed_commitment(salt)
    pairs: list[dict[str, object]] = []
    for slot in TRAINING_SEED_SLOTS:
        for cell_ordinal, (task, observation, action) in enumerate(CELL_KEYS):
            ablation = _cell_ablation(slot, observation, action)
            seeds = plan_episode_seeds(
                PLAN_ID, task, ablation, replicate_count, salt
            )
            for episode in seeds:
                pairs.append(
                    {
                        "pair_id": episode.planned_episode_id,
                        "task_id": task,
                        "cell_ordinal": cell_ordinal,
                        "training_seed_slot": slot,
                        "observation_latency_steps": observation,
                        "action_latency_steps": action,
                        "replicate_ordinal": episode.episode_ordinal,
                        "environment_seed": episode.environment_seed,
                        "policy_rng_seed": episode.policy_rng_seed,
                        "seed_commitment": commitment,
                        "execution_roles": list(ROLES),
                    }
                )
    result = {
        "schema_version": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "plan_id": PLAN_ID,
        "diagnostic_plan_available": available,
        "formal_capability_plan_usable": False,
        "replicate_count_per_slot_cell": replicate_count,
        "training_seed_slots": list(TRAINING_SEED_SLOTS),
        "cells": benchmark_cells(),
        "cell_count": len(CELL_KEYS),
        "pair_count": len(pairs),
        "execution_count": len(pairs) * len(ROLES),
        "roles": list(ROLES),
        "role_enters_seed_derivation": False,
        "cell_label_policy_visible": False,
        "policy_visible_fields": [],
        "replacement_seed_allowed": False,
        "complete_case_deletion_allowed": False,
        "formal_seed_bank": False,
        "capability_claim_allowed": False,
        "closed_loop_success_available": False,
        "seed_commitment": commitment, "seed_schema": SEED_SCHEMA,
        "environment_seed_mode": "derived",
        "pairs": pairs,
        "ledger_contract": ledger_contract(),
    }
    validate_planned_ledger(result, salt)
    return result


def validate_planned_ledger(
    ledger: Mapping[str, object], salt: str
) -> None:
    if (
        ledger.get("schema_version") != PLAN_SCHEMA
        or ledger.get("experiment_id") != EXPERIMENT_ID
        or ledger.get("plan_id") != PLAN_ID
    ):
        raise FactorialBenchmarkContractError("planned ledger identity differs")
    replicate_count = _strict_int(
        ledger.get("replicate_count_per_slot_cell"), "planned replicate count"
    )
    expected_pairs = len(TRAINING_SEED_SLOTS) * len(CELL_KEYS) * replicate_count
    pairs = ledger.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != expected_pairs:
        raise FactorialBenchmarkContractError("planned pair count differs")
    if (
        ledger.get("training_seed_slots") != list(TRAINING_SEED_SLOTS)
        or ledger.get("cells") != benchmark_cells()
        or ledger.get("cell_count") != 27
        or ledger.get("pair_count") != expected_pairs
        or ledger.get("execution_count") != expected_pairs * 2
        or ledger.get("roles") != list(ROLES)
        or ledger.get("role_enters_seed_derivation") is not False
        or ledger.get("cell_label_policy_visible") is not False
        or ledger.get("policy_visible_fields") != []
        or ledger.get("replacement_seed_allowed") is not False
        or ledger.get("complete_case_deletion_allowed") is not False
        or type(ledger.get("diagnostic_plan_available")) is not bool
        or ledger.get("formal_capability_plan_usable") is not False
        or any(ledger.get(field) is not False for field in
               ("formal_seed_bank", "capability_claim_allowed",
                "closed_loop_success_available"))
        or ledger.get("ledger_contract") != ledger_contract()
        or ledger.get("seed_schema") != SEED_SCHEMA
        or ledger.get("environment_seed_mode") != "derived"
        or ledger.get("seed_commitment") != seed_commitment(salt)
    ):
        raise FactorialBenchmarkContractError("planned ledger contract differs")
    seen: set[str] = set()
    environment_seeds: set[int] = set()
    policy_seeds: set[int] = set()
    expected_keys: set[tuple[int, int, int]] = set()
    for value in pairs:
        pair = _mapping(value, "planned pair")
        slot = _strict_int(pair.get("training_seed_slot"), "training seed slot")
        cell = _strict_int(pair.get("cell_ordinal"), "cell ordinal")
        replicate = _strict_int(pair.get("replicate_ordinal"), "replicate ordinal")
        if (
            slot not in TRAINING_SEED_SLOTS
            or cell not in range(len(CELL_KEYS))
            or replicate not in range(replicate_count)
        ):
            raise FactorialBenchmarkContractError("planned pair stratum is invalid")
        task, observation, action = CELL_KEYS[cell]
        ablation = _cell_ablation(slot, observation, action)
        expected_id = planned_episode_id(PLAN_ID, task, ablation, replicate)
        expected_environment = derive_domain_seed(salt, "environment", expected_id)
        expected_policy = derive_domain_seed(salt, "policy", expected_id)
        if (
            pair.get("pair_id") != expected_id
            or pair.get("task_id") != task
            or pair.get("observation_latency_steps") != observation
            or pair.get("action_latency_steps") != action
            or pair.get("environment_seed") != expected_environment
            or pair.get("policy_rng_seed") != expected_policy
            or pair.get("seed_commitment") != seed_commitment(salt)
            or pair.get("execution_roles") != list(ROLES)
        ):
            raise FactorialBenchmarkContractError("planned pair lineage differs")
        if expected_id in seen or expected_environment == expected_policy:
            raise FactorialBenchmarkContractError("planned pair identity or seed collided")
        seen.add(expected_id)
        environment_seeds.add(expected_environment)
        policy_seeds.add(expected_policy)
        expected_keys.add((slot, cell, replicate))
    if (len(expected_keys) != expected_pairs
            or len(environment_seeds) != expected_pairs
            or len(policy_seeds) != expected_pairs
            or environment_seeds & policy_seeds):
        raise FactorialBenchmarkContractError("planned pair coverage or seed domains differ")


def make_terminal_record(
    pair: Mapping[str, object],
    role: str,
    status: str,
    reason: str,
    previous_record_sha256: str,
    *,
    policy_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if role not in ROLES or not _is_sha256(previous_record_sha256):
        raise FactorialBenchmarkContractError("terminal role or chain identity differs")
    _validate_terminal_classification(status, reason, policy_provenance)
    record: dict[str, object] = {
        "schema_version": TERMINAL_SCHEMA,
        "pair_id": pair["pair_id"],
        "role": role,
        "task_id": pair["task_id"],
        "cell_ordinal": pair["cell_ordinal"],
        "training_seed_slot": pair["training_seed_slot"],
        "observation_latency_steps": pair["observation_latency_steps"],
        "action_latency_steps": pair["action_latency_steps"],
        "replicate_ordinal": pair["replicate_ordinal"],
        "environment_seed": pair["environment_seed"],
        "policy_rng_seed": pair["policy_rng_seed"],
        "status": status,
        "reason": reason,
        "policy_provenance": (
            dict(policy_provenance) if policy_provenance is not None else None
        ),
        "previous_record_sha256": previous_record_sha256,
    }
    record["record_sha256"] = _record_sha256(record)
    return record


def aggregate_terminal_ledger(
    planned: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    salt: str,
) -> dict[str, object]:
    validate_planned_ledger(planned, salt)
    pairs = {
        str(value["pair_id"]): _mapping(value, "planned pair")
        for value in planned["pairs"]  # type: ignore[index]
    }
    seen: set[tuple[str, str]] = set()
    previous = ZERO_HASH
    counts = {
        "valid_success": 0,
        "valid_failure": 0,
        "unresolved_infrastructure": 0,
    }
    successes: dict[tuple[str, str], int] = {}
    reasons: dict[str, int] = {}
    for value in records:
        record = _mapping(value, "terminal record")
        pair_id = str(record.get("pair_id", ""))
        role = str(record.get("role", ""))
        pair = pairs.get(pair_id)
        key = (pair_id, role)
        if pair is None or role not in ROLES or key in seen:
            raise FactorialBenchmarkContractError("terminal execution identity differs")
        if (
            record.get("schema_version") != TERMINAL_SCHEMA
            or record.get("previous_record_sha256") != previous
            or record.get("record_sha256") != _record_sha256(record)
        ):
            raise FactorialBenchmarkContractError("terminal hash chain differs")
        for field in (
            "task_id",
            "cell_ordinal",
            "training_seed_slot",
            "observation_latency_steps",
            "action_latency_steps",
            "replicate_ordinal",
            "environment_seed",
            "policy_rng_seed",
        ):
            if record.get(field) != pair.get(field):
                raise FactorialBenchmarkContractError(
                    f"terminal {field} differs from plan"
                )
        status = str(record.get("status", ""))
        reason = str(record.get("reason", ""))
        _validate_terminal_classification(
            status, reason, record.get("policy_provenance")
        )
        counts[status] += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        successes[key] = int(status == "valid_success")
        seen.add(key)
        previous = str(record["record_sha256"])
    planned_count = int(planned["execution_count"])
    missing = planned_count - len(seen)
    counts["unresolved_infrastructure"] += missing
    unresolved = counts["unresolved_infrastructure"]
    report: dict[str, object] = {
        "planned": planned_count,
        **counts,
        "terminal_records": len(records),
        "missing_terminal_records": missing,
        "identity_holds": planned_count == sum(counts.values()),
        "reason_counts": reasons,
        "last_record_sha256": previous,
        "decision": "inconclusive" if unresolved else "complete",
        "complete_case_deletion_used": False,
        "replacement_seed_used": False,
    }
    if unresolved == 0:
        report["ledgers"] = _success_ledgers(planned, successes)
    return report


def evaluate_synthetic_power(
    design: PowerDesign = PowerDesign(),
) -> dict[str, object]:
    jobs = [
        (
            n,
            probability_index,
            shared_index,
            probability,
            shared,
            condition,
            design.trials,
            design.bootstrap_samples,
            design.base_seed,
        )
        for n in design.candidate_n
        for probability_index, probability in enumerate(design.probabilities)
        for shared_index, shared in enumerate(design.shared_fractions)
        for condition in ("null", "planted")
    ]
    workers = design.workers or min(8, os.cpu_count() or 1)
    if workers == 1:
        raw = [_simulate_stratum(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            raw = list(pool.map(_simulate_stratum, jobs, chunksize=1))
    by_key = {
        (value["n"], value["probability_index"], value["shared_index"],
         value["condition"]): value for value in raw
    }
    strata: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    selected_n: int | None = None
    for n in design.candidate_n:
        null_uppers: list[float] = []
        power_lowers: list[float] = []
        n_strata: list[dict[str, object]] = []
        for probability_index, probability in enumerate(design.probabilities):
            for shared_index, shared in enumerate(design.shared_fractions):
                null = by_key[(n, probability_index, shared_index, "null")]
                planted = by_key[(n, probability_index, shared_index, "planted")]
                _, null_upper = clopper_pearson(
                    int(null["passes"]), design.trials
                )
                power_lower, _ = clopper_pearson(
                    int(planted["passes"]), design.trials
                )
                row = {
                    "n": n,
                    "baseline_probability": probability,
                    "shared_randomness_fraction": shared,
                    "null": {
                        "passes": null["passes"],
                        "trials": design.trials,
                        "empirical_fpr": int(null["passes"]) / design.trials,
                        "clopper_pearson_95_upper": null_upper,
                    },
                    "planted": {
                        "passes": planted["passes"],
                        "trials": design.trials,
                        "empirical_power": int(planted["passes"]) / design.trials,
                        "clopper_pearson_95_lower": power_lower,
                        "mean_point_estimate_complete": planted["mean_complete"],
                        "mean_point_estimate_supported": planted["mean_supported"],
                        "point_estimate_threshold_is_acceptance_gate": False,
                    },
                }
                n_strata.append(row)
                null_uppers.append(null_upper)
                power_lowers.append(power_lower)
        passes = max(null_uppers) <= 0.05 and min(power_lowers) >= 0.80
        strata.extend(n_strata)
        summaries.append(
            {
                "n": n,
                "worst_null_fpr_clopper_pearson_95_upper": max(null_uppers),
                "worst_planted_power_clopper_pearson_95_lower": min(power_lowers),
                "passes_all_frozen_strata": passes,
            }
        )
        if selected_n is None and passes:
            selected_n = n
    return {
        "schema_version": POWER_SCHEMA,
        "design": asdict(design),
        "training_seed_slots": 3,
        "cell_count": 27,
        "conditions": {
            "null_candidate_increment": 0.0,
            "planted_candidate_increment": 0.10,
        },
        "paired_generation": (
            "Bernoulli mixture: shared fraction uses one uniform for both roles; "
            "otherwise roles use independent uniforms"
        ),
        "bootstrap": {
            "cell_unit": "synchronized_paired_episode",
            "outer_unit": "synchronized_training_seed_slot",
            "cell_weights": "equal",
            "percentile_lower_quantile": 0.025,
            "simultaneous_gates": [
                "delta_complete_macro_lower>0",
                "delta_supported_macro_lower>0",
            ],
            "seed_derivation": (
                "SHA256(domain || base_seed || n || probability_index || "
                "shared_index || condition || trial_index)"
            ),
        },
        "clopper_pearson_confidence": 0.95,
        "strata": strata,
        "n_summaries": summaries,
        "selected_n": selected_n,
        "decision": (
            "power_passed" if selected_n is not None else "inconclusive_power"
        ),
    }


def clopper_pearson(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or successes < 0
        or trials <= 0
        or successes > trials
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("invalid binomial interval inputs")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else _beta_quantile(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = (
        1.0
        if successes == trials
        else _beta_quantile(
            1.0 - alpha / 2.0, successes + 1, trials - successes
        )
    )
    return lower, upper


def _simulate_stratum(job: tuple[Any, ...]) -> dict[str, object]:
    (
        n,
        probability_index,
        shared_index,
        probability,
        shared,
        condition,
        trials,
        bootstrap_samples,
        base_seed,
    ) = job
    candidate_probability = probability + (0.10 if condition == "planted" else 0.0)
    passes = 0
    complete_sum = 0.0
    supported_sum = 0.0
    for trial in range(trials):
        outcome_rng = np.random.default_rng(
            _derived_seed(
                "synthetic-outcomes",
                base_seed,
                n,
                probability_index,
                shared_index,
                condition,
                trial,
            )
        )
        selector = outcome_rng.random((3, 27, n)) < shared
        shared_uniform = outcome_rng.random((3, 27, n))
        baseline_uniform = np.where(
            selector, shared_uniform, outcome_rng.random((3, 27, n))
        )
        candidate_uniform = np.where(
            selector, shared_uniform, outcome_rng.random((3, 27, n))
        )
        delta = (
            (candidate_uniform < candidate_probability).astype(np.int8)
            - (baseline_uniform < probability).astype(np.int8)
        )
        complete_sum += float(delta.mean())
        supported_sum += float(delta[:, SUPPORTED_INDICES, :].mean())
        bootstrap_rng = np.random.default_rng(
            _derived_seed(
                "hierarchical-paired-bootstrap",
                base_seed,
                n,
                probability_index,
                shared_index,
                condition,
                trial,
            )
        )
        complete_lower, supported_lower = _bootstrap_lowers(
            delta, bootstrap_samples, bootstrap_rng
        )
        passes += int(complete_lower > 0.0 and supported_lower > 0.0)
    return {
        "n": n,
        "probability_index": probability_index,
        "shared_index": shared_index,
        "condition": condition,
        "passes": passes,
        "mean_complete": complete_sum / trials,
        "mean_supported": supported_sum / trials,
    }


def _bootstrap_lowers(
    delta: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    n = delta.shape[2]
    positive_probability = (delta == 1).sum(axis=2) / n
    negative_probability = (delta == -1).sum(axis=2) / n
    outer = rng.integers(0, 3, size=(samples, 3))
    positive_probability = positive_probability[outer]
    negative_probability = negative_probability[outer]
    positive = rng.binomial(n, positive_probability)
    remaining_probability = np.divide(
        negative_probability,
        1.0 - positive_probability,
        out=np.zeros_like(negative_probability),
        where=positive_probability < 1.0,
    )
    remaining_probability = np.clip(remaining_probability, 0.0, 1.0)
    negative = rng.binomial(n - positive, remaining_probability)
    sampled = (positive - negative) / n
    complete = sampled.mean(axis=(1, 2))
    supported = sampled[:, :, SUPPORTED_INDICES].mean(axis=(1, 2))
    return float(np.quantile(complete, 0.025)), float(np.quantile(supported, 0.025))


def _success_ledgers(
    planned: Mapping[str, object], successes: Mapping[tuple[str, str], int]
) -> dict[str, object]:
    deltas: list[tuple[str, int, int, float]] = []
    for pair in planned["pairs"]:  # type: ignore[assignment]
        pair_id = str(pair["pair_id"])
        deltas.append(
            (
                str(pair["task_id"]),
                int(pair["observation_latency_steps"]),
                int(pair["action_latency_steps"]),
                successes[(pair_id, "candidate")]
                - successes[(pair_id, "baseline")],
            )
        )
    cell_means = {
        (task, observation, action): float(
            np.mean(
                [
                    delta
                    for row_task, row_observation, row_action, delta in deltas
                    if (row_task, row_observation, row_action)
                    == (task, observation, action)
                ]
            )
        )
        for task, observation, action in CELL_KEYS
    }
    supported_keys = [
        key for key in CELL_KEYS if key[1] in (1, 2)
    ]
    return {
        "complete_challenge": {
            "delta_macro": float(np.mean(list(cell_means.values()))),
            "cell_count": 27,
            "equal_cell_weight": 1.0 / 27.0,
        },
        "supported_conditional": {
            "delta_macro": float(np.mean([cell_means[key] for key in supported_keys])),
            "cell_count": 18,
            "equal_cell_weight": 1.0 / 18.0,
            "per_task_delta": {
                task: float(
                    np.mean([cell_means[key] for key in supported_keys if key[0] == task])
                )
                for task in TASKS
            },
        },
    }


def _validate_terminal_classification(
    status: str, reason: str, policy_provenance: object
) -> None:
    valid = (
        status == "valid_success"
        and reason == "task_success"
        and policy_provenance is None
    )
    if status == "valid_failure" and reason in VALID_FAILURE_REASONS:
        valid = reason not in POLICY_FAILURE_REASONS or _valid_policy_provenance(
            policy_provenance
        )
    if status == "unresolved_infrastructure" and reason in INFRASTRUCTURE_REASONS:
        valid = policy_provenance is None
    if not valid:
        raise FactorialBenchmarkContractError(
            "terminal status, reason, or provenance differs"
        )


def _valid_policy_provenance(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    source = value.get("source_commit")
    return _is_sha256(value.get("deployment_sha256")) and isinstance(
        source, str
    ) and len(source) == 40 and all(character in "0123456789abcdef" for character in source)


def _record_sha256(record: Mapping[str, object]) -> str:
    value = dict(record)
    value.pop("record_sha256", None)
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()
def _cell_ablation(slot: int, observation: int, action: int) -> str:
    return f"slot={slot}|observation_latency={observation}|action_latency={action}"
def _derived_seed(domain: str, *values: object) -> int:
    payload = "||".join([domain, *(str(value) for value in values)]).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
def _beta_quantile(probability: float, alpha: float, beta: float) -> float:
    low = 0.0
    high = 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _regularized_beta(middle, alpha, beta) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0
def _regularized_beta(value: float, alpha: float, beta: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + alpha * math.log(value)
        + beta * math.log1p(-value)
    )
    boundary = (alpha + 1.0) / (alpha + beta + 2.0)
    if value < boundary:
        return factor * _beta_fraction(alpha, beta, value) / alpha
    return 1.0 - factor * _beta_fraction(beta, alpha, 1.0 - value) / beta
def _beta_fraction(alpha: float, beta: float, value: float) -> float:
    qab = alpha + beta
    qap = alpha + 1.0
    qam = alpha - 1.0
    c = 1.0
    d = 1.0 - qab * value / qap
    d = 1.0 / max(abs(d), 1.0e-300) * (1.0 if d >= 0 else -1.0)
    result = d
    for iteration in range(1, 201):
        even = 2 * iteration
        coefficient = (
            iteration * (beta - iteration) * value
            / ((qam + even) * (alpha + even))
        )
        d = 1.0 + coefficient * d
        d = 1.0 / (d if abs(d) > 1.0e-300 else 1.0e-300)
        c = 1.0 + coefficient / c
        c = c if abs(c) > 1.0e-300 else 1.0e-300
        result *= d * c
        coefficient = -(
            (alpha + iteration)
            * (qab + iteration)
            * value
            / ((alpha + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        d = 1.0 / (d if abs(d) > 1.0e-300 else 1.0e-300)
        c = 1.0 + coefficient / c
        c = c if abs(c) > 1.0e-300 else 1.0e-300
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3.0e-14:
            break
    return result


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")
def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorialBenchmarkContractError(f"{label} must be a mapping")
    return value
def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FactorialBenchmarkContractError(f"{label} must be non-negative int")
    return value
def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))
