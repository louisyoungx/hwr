from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hwr.eval import experiment_contract_oracle as oracle


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads(
    (ROOT / "configs/eval/r0017_experiment_contracts.json").read_text()
)
BANK = json.loads(
    (
        ROOT
        / "runs/research-loop/0014/"
        "r0014-p79-candidate-bank-s20267901/bank.json"
    ).read_text()
)


def _capsules(bank: dict[str, object] = BANK) -> dict[str, object]:
    episodes = []
    for row in bank["episodes"]:
        cell = row["cell_id"]
        observation, action = cell.split("-obs-")[1].split("-action-")
        episodes.append(
            {
                "planned_episode_id": row["planned_episode_id"],
                "task_id": row["task_id"],
                "cell_id": cell,
                "planned_latency": {
                    "observation_steps": int(observation),
                    "action_steps": int(action),
                },
                "candidate_set": copy.deepcopy(row["candidate_set"]),
            }
        )
    return {
        "schema_version": oracle.P50_SCHEMA,
        "capsule_count": len(episodes),
        "episodes": episodes,
    }


def _cohort() -> dict[str, object]:
    return oracle.build_cohort(_capsules(), BANK, REGISTRY)


def _contracts() -> dict[str, dict[str, object]]:
    return {row["contract_id"]: row for row in REGISTRY["contracts"]}


def test_frozen_join_derives_episode_cohort_from_p50_and_p79() -> None:
    cohort = _cohort()
    expected = REGISTRY["expected_cohort"]

    assert cohort["sample_unit"] == "Episode"
    assert cohort["summary"]["episode_count"] == expected["episode_count"]
    assert cohort["summary"]["nonempty_count"] == expected["nonempty_count"]
    assert cohort["summary"]["empty_count"] == expected["empty_count"]
    assert cohort["summary"]["choice_opportunity_count"] == (
        expected["choice_opportunity_count"]
    )
    assert cohort["summary"]["task_nonempty_counts"] == (
        expected["task_nonempty_counts"]
    )
    assert cohort["summary"]["latency_pair_nonempty_counts"] == (
        expected["latency_pair_nonempty_counts"]
    )
    assert all(
        set(row)
        == {
            "episode_id",
            "task_id",
            "cell_id",
            "observation_latency_steps",
            "action_latency_steps",
            "candidate_count",
            "denominators",
        }
        for row in cohort["episodes"]
    )


def test_all_registry_contracts_agree_and_have_independent_evidence() -> None:
    analysis = oracle.analyze_registry(REGISTRY, _cohort())
    metrics = analysis["metrics"]

    assert metrics["contract_count"] == len(REGISTRY["contracts"])
    assert metrics["solver_agreement_count"] == metrics["contract_count"]
    assert metrics["valid_accepted_witness_count"] == (
        metrics["reachable_contract_count"]
    )
    assert metrics["valid_contradiction_count"] == (
        metrics["rejected_contract_count"]
    )
    assert metrics["denominator_conservation_count"] == metrics["contract_count"]
    assert metrics["exposure_policy_valid_count"] == metrics["contract_count"]
    assert metrics["private_outcome_read_count"] == 0
    assert metrics["sample_unit_violation_count"] == 0
    for result in analysis["contracts"]:
        if result["reachable"]:
            assert result["solver_b"]["accepted_witness"]
            assert result["witness_verification"]["passed"] is True
        else:
            assert result["solver_a"]["contradictions"]
            assert result["solver_b"]["accepted_witness"] is None
            assert result["contradiction_verification"]["passed"] is True


def test_thresholds_are_consumed_from_registry_not_formal_verdicts() -> None:
    cohort = _cohort()
    contract = copy.deepcopy(REGISTRY["contracts"][0])
    original = oracle.solve_analytic(contract, REGISTRY, cohort)
    contract["target_minimum"] = original["target_eligibility_count"]
    contract["claim_scope"] = ["overall"]
    contract["stratum_minimums"] = {}

    analytic = oracle.solve_analytic(contract, REGISTRY, cohort)
    enumerated = oracle.solve_enumeration(contract, REGISTRY, cohort)

    assert analytic["reachable"] is True
    assert enumerated["reachable"] is True
    assert oracle.verify_assignment(
        contract, REGISTRY, cohort, enumerated["accepted_witness"]
    )["passed"]


def test_witness_verifier_recomputes_total_strata_and_partition() -> None:
    cohort = _cohort()
    analysis = oracle.analyze_registry(REGISTRY, cohort)
    result = next(row for row in analysis["contracts"] if row["reachable"])
    contract = _contracts()[result["contract_id"]]
    witness = copy.deepcopy(result["solver_b"]["accepted_witness"])
    witness["positive_episode_ids"] = witness["positive_episode_ids"][:1]

    verification = oracle.verify_assignment(contract, REGISTRY, cohort, witness)

    assert verification["passed"] is False
    assert {"witness_total", "witness_negative_partition"} <= set(
        verification["errors"]
    )


def test_exposure_exclusion_recomputes_denominator_and_strata() -> None:
    cohort = _cohort()
    contract = next(
        row
        for row in REGISTRY["contracts"]
        if row["exposure_policy"] == "exclude_matching_outcome_fields"
        and row["outcome_field"] == "safe_b2_entry"
    )
    analytic = oracle.solve_analytic(contract, REGISTRY, cohort)
    denominator = analytic["denominator"]

    assert denominator["base_count"] == (
        denominator["excluded_count"] + denominator["effective_count"]
    )
    assert denominator["excluded_episode_ids"] == sorted(
        row["episode_id"] for row in REGISTRY["result_exposure_ledger"]
    )
    assert any(
        row["category"] == "stratum_required_gt_eligible"
        for row in analytic["contradictions"]
    )


@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        (lambda value: value.update(sample_unit="Frame"), "sample_unit"),
        (
            lambda value: value["denominators"]["empty"].update(
                candidate_count_maximum=1
            ),
            "denominator_semantics",
        ),
        (
            lambda value: value["result_exposure_ledger"][0]["fields"].append(
                "private_truth"
            ),
            "exposure_fields",
        ),
        (
            lambda value: value["contracts"][0].update(
                exposure_policy="unknown"
            ),
            "exposure_policy",
        ),
    ),
)
def test_registry_mutations_report_semantic_category(mutation, category) -> None:
    registry = copy.deepcopy(REGISTRY)
    mutation(registry)

    with pytest.raises(oracle.ContractOracleError, match=category):
        oracle.validate_registry(registry)


def test_join_rejects_duplicate_missing_and_identity_drift() -> None:
    duplicate = _capsules()
    duplicate["episodes"].append(copy.deepcopy(duplicate["episodes"][0]))
    duplicate["capsule_count"] += 1
    with pytest.raises(oracle.ContractOracleError, match="episode_duplicate"):
        oracle.build_cohort(duplicate, BANK, REGISTRY)

    missing = _capsules()
    missing["episodes"].pop()
    missing["capsule_count"] -= 1
    with pytest.raises(
        oracle.ContractOracleError, match="episode_identity_mismatch"
    ):
        oracle.build_cohort(missing, BANK, REGISTRY)

    identity = copy.deepcopy(BANK)
    identity["episodes"][0]["task_id"] = "unknown"
    with pytest.raises(oracle.ContractOracleError, match="task_identity"):
        oracle.build_cohort(_capsules(), identity, REGISTRY)


def test_solver_disagreement_is_invalid() -> None:
    with pytest.raises(
        oracle.ContractOracleError, match="invalid_solver_disagreement"
    ):
        oracle.combine_solver_results(
            {"reachable": True}, {"reachable": False}
        )


def test_enumerator_and_contradiction_verifier_do_not_call_solver_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = _cohort()
    contract = _contracts()["r0016-p68-e3-selector-negative"]
    monkeypatch.setattr(
        oracle,
        "solve_analytic",
        lambda *args: (_ for _ in ()).throw(AssertionError("solver A called")),
    )
    monkeypatch.setattr(
        oracle,
        "_contract_context",
        lambda *args: (_ for _ in ()).throw(AssertionError("shared context called")),
    )
    monkeypatch.setattr(
        oracle,
        "verify_assignment",
        lambda *args: (_ for _ in ()).throw(AssertionError("verifier called")),
    )

    result = oracle.solve_enumeration(contract, REGISTRY, cohort)
    contradiction = result["independent_rejections"][0]
    monkeypatch.setattr(
        oracle,
        "_independent_context",
        lambda *args: (_ for _ in ()).throw(AssertionError("solver B called")),
    )

    assert result["reachable"] is False
    assert result["enumerated_assignment_count"] > 0
    assert result["exhaustive"] is True
    assert oracle.verify_contradiction(
        contract, REGISTRY, cohort, contradiction
    )["passed"]
