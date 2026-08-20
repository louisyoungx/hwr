from __future__ import annotations

import copy

import numpy as np
import pytest

from hwr.eval import factorial_benchmark
from hwr.eval.factorial_benchmark import (
    ACTION_LATENCIES,
    CELL_KEYS,
    OBSERVATION_LATENCIES,
    SOURCE_AGE_NS,
    TASKS,
    ZERO_HASH,
    FactorialBenchmarkContractError,
    PowerDesign,
    aggregate_terminal_ledger,
    benchmark_cells,
    build_planned_ledger,
    clopper_pearson,
    evaluate_synthetic_power,
    ledger_contract,
    make_terminal_record,
    validate_planned_ledger,
)


SALT = "R0001-P36-E2-s20263602"


def test_frozen_cells_and_dual_ledgers_are_balanced() -> None:
    cells = benchmark_cells()
    contract = ledger_contract()

    assert len(cells) == 27
    assert {
        (
            value["task_id"],
            value["observation_latency_steps"],
            value["action_latency_steps"],
        )
        for value in cells
    } == set(CELL_KEYS)
    assert {value["task_id"] for value in cells} == set(TASKS)
    assert {
        value["observation_latency_steps"] for value in cells
    } == set(OBSERVATION_LATENCIES)
    assert {
        value["action_latency_steps"] for value in cells
    } == set(ACTION_LATENCIES)
    assert {
        value["observation_latency_steps"]: value["maximum_source_age_ns"]
        for value in cells
    } == SOURCE_AGE_NS
    assert sum(value["domain"] == "supported" for value in cells) == 18
    assert sum(value["domain"] == "challenge" for value in cells) == 9
    assert contract["primary_ledger"] == "complete_challenge"
    assert contract["full_profile_supported"] is False
    assert contract["complete_challenge"]["cell_weight"] == pytest.approx(1 / 27)
    assert contract["complete_challenge"]["observation_latency_3_weight"] == (
        pytest.approx(1 / 3)
    )
    assert contract["supported_conditional"]["cell_weight"] == pytest.approx(1 / 18)
    assert contract["future_joint_primary_gate"]["per_task_supported_regression_floor"] == (
        -0.05
    )
    assert contract["guardrails"]["severe_collision_count"] == 0
    assert contract["guardrails"]["stale_action_applied_rate"] == 0.0


def test_planned_ledger_has_three_slots_balanced_pairs_and_domain_seeds() -> None:
    ledger = build_planned_ledger(4, SALT)
    pairs = ledger["pairs"]

    validate_planned_ledger(ledger, SALT)
    assert ledger["pair_count"] == 3 * 27 * 4
    assert ledger["execution_count"] == 2 * 3 * 27 * 4
    assert ledger["seed_commitment"] == (
        "f094032ccc029cc15979be8ffd636d956"
        "6500398f356256bc23efc5d8f88cdc9"
    )
    assert ledger["seed_schema"] == "hwr.opaque-episode-seeds/v1"
    assert ledger["environment_seed_mode"] == "derived"
    assert ledger["role_enters_seed_derivation"] is False
    assert ledger["cell_label_policy_visible"] is False
    assert ledger["policy_visible_fields"] == []
    assert ledger["replacement_seed_allowed"] is False
    assert ledger["complete_case_deletion_allowed"] is False
    assert {
        (value["training_seed_slot"], value["cell_ordinal"], value["replicate_ordinal"])
        for value in pairs
    } == {
        (slot, cell, replicate)
        for slot in range(3)
        for cell in range(27)
        for replicate in range(4)
    }
    pair_ids = {value["pair_id"] for value in pairs}
    environment = {value["environment_seed"] for value in pairs}
    policy = {value["policy_rng_seed"] for value in pairs}
    assert len(pair_ids) == len(pairs)
    assert len(environment) == len(pairs)
    assert len(policy) == len(pairs)
    assert environment.isdisjoint(policy)
    assert all(value["execution_roles"] == ["baseline", "candidate"] for value in pairs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cell_ordinal", 27, "stratum"),
        ("policy_rng_seed", 0, "lineage"),
        ("observation_latency_steps", 4, "lineage"),
    ],
)
def test_planned_ledger_rejects_cell_or_seed_mutation(
    field: str, value: int, message: str
) -> None:
    ledger = build_planned_ledger(1, SALT)
    ledger["pairs"][0][field] = value

    with pytest.raises(FactorialBenchmarkContractError, match=message):
        validate_planned_ledger(ledger, SALT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("formal_seed_bank", True),
        ("capability_claim_allowed", True),
        ("closed_loop_success_available", True),
        ("diagnostic_plan_available", "yes"),
        ("formal_capability_plan_usable", True),
        ("ledger_contract", {}),
    ],
)
def test_planned_ledger_rejects_fixed_contract_mutation(field, value) -> None:
    ledger = build_planned_ledger(1, SALT)
    ledger[field] = value

    with pytest.raises(FactorialBenchmarkContractError, match="contract"):
        validate_planned_ledger(ledger, SALT)


def test_terminal_ledger_is_hash_bound_complete_and_cell_weighted() -> None:
    planned = build_planned_ledger(1, SALT)
    records = []
    previous = ZERO_HASH
    for pair in planned["pairs"]:
        baseline = make_terminal_record(
            pair, "baseline", "valid_failure", "task_failure", previous
        )
        records.append(baseline)
        previous = baseline["record_sha256"]
        candidate = make_terminal_record(
            pair, "candidate", "valid_success", "task_success", previous
        )
        records.append(candidate)
        previous = candidate["record_sha256"]

    report = aggregate_terminal_ledger(planned, records, SALT)

    assert report["planned"] == 162
    assert report["valid_success"] == 81
    assert report["valid_failure"] == 81
    assert report["unresolved_infrastructure"] == 0
    assert report["identity_holds"] is True
    assert report["decision"] == "complete"
    assert report["ledgers"]["complete_challenge"]["delta_macro"] == 1.0
    assert report["ledgers"]["supported_conditional"]["delta_macro"] == 1.0
    assert report["ledgers"]["supported_conditional"]["per_task_delta"] == {
        task: 1.0 for task in TASKS
    }


def test_missing_terminal_is_unresolved_and_policy_failure_stays_valid() -> None:
    planned = build_planned_ledger(1, SALT)
    pair = planned["pairs"][0]
    policy_failure = make_terminal_record(
        pair,
        "candidate",
        "valid_failure",
        "policy_exception",
        ZERO_HASH,
        policy_provenance={
            "deployment_sha256": "a" * 64,
            "source_commit": "b" * 40,
        },
    )

    report = aggregate_terminal_ledger(planned, [policy_failure], SALT)

    assert report["valid_failure"] == 1
    assert report["unresolved_infrastructure"] == 161
    assert report["planned"] == (
        report["valid_success"]
        + report["valid_failure"]
        + report["unresolved_infrastructure"]
    )
    assert report["decision"] == "inconclusive"
    assert "ledgers" not in report

    with pytest.raises(FactorialBenchmarkContractError, match="provenance"):
        make_terminal_record(
            pair,
            "candidate",
            "valid_failure",
            "policy_exception",
            ZERO_HASH,
        )


def test_terminal_ledger_rejects_duplicate_corruption_and_unknown_reason() -> None:
    planned = build_planned_ledger(1, SALT)
    pair = planned["pairs"][0]
    record = make_terminal_record(
        pair, "baseline", "valid_success", "task_success", ZERO_HASH
    )

    with pytest.raises(FactorialBenchmarkContractError):
        aggregate_terminal_ledger(planned, [record, record], SALT)

    corrupt = copy.deepcopy(record)
    corrupt["environment_seed"] += 1
    with pytest.raises(FactorialBenchmarkContractError):
        aggregate_terminal_ledger(planned, [corrupt], SALT)

    with pytest.raises(FactorialBenchmarkContractError):
        make_terminal_record(
            pair,
            "baseline",
            "unresolved_infrastructure",
            "unknown",
            ZERO_HASH,
        )


def test_clopper_pearson_is_exact_at_binomial_boundaries() -> None:
    zero_lower, zero_upper = clopper_pearson(0, 500)
    full_lower, full_upper = clopper_pearson(500, 500)

    assert zero_lower == 0.0
    assert zero_upper == pytest.approx(1.0 - 0.025 ** (1.0 / 500), rel=1e-10)
    assert full_lower == pytest.approx(0.025 ** (1.0 / 500), rel=1e-10)
    assert full_upper == 1.0


def test_synthetic_power_is_replayable_and_reports_all_frozen_gates() -> None:
    design = PowerDesign(
        candidate_n=(4,),
        probabilities=(0.30,),
        shared_fractions=(0.5,),
        trials=8,
        bootstrap_samples=30,
        base_seed=20_263_602,
        workers=1,
    )

    first = evaluate_synthetic_power(design)
    second = evaluate_synthetic_power(design)

    assert first == second
    assert first["decision"] == "inconclusive_power"
    assert first["selected_n"] is None
    assert len(first["strata"]) == 1
    assert first["design"] == {
        "candidate_n": (4,),
        "probabilities": (0.3,),
        "shared_fractions": (0.5,),
        "trials": 8,
        "bootstrap_samples": 30,
        "base_seed": 20_263_602,
        "workers": 1,
    }
    assert first["bootstrap"]["cell_unit"] == "synchronized_paired_episode"
    assert first["bootstrap"]["outer_unit"] == "synchronized_training_seed_slot"
    assert first["bootstrap"]["cell_weights"] == "equal"
    assert first["strata"][0]["planted"][
        "point_estimate_threshold_is_acceptance_gate"
    ] is False


@pytest.mark.parametrize("value", (-1, 1))
def test_hierarchical_bootstrap_handles_empirical_probability_boundaries(
    value: int,
) -> None:
    delta = np.full((3, 27, 4), value, dtype=np.int8)

    complete, supported = factorial_benchmark._bootstrap_lowers(
        delta, 20, np.random.default_rng(7)
    )

    assert complete == float(value)
    assert supported == float(value)


def test_frozen_power_design_has_no_post_result_budget_freedom() -> None:
    design = PowerDesign()

    assert design.candidate_n == (4, 8, 12, 16, 24, 32)
    assert design.probabilities == (0.10, 0.30, 0.50, 0.70)
    assert design.shared_fractions == (0.0, 0.5, 0.9)
    assert design.trials == 500
    assert design.bootstrap_samples == 1_000
    assert design.base_seed == 20_263_602
