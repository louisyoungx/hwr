from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from hwr.eval import cartesian_convergence as convergence
from hwr.eval import cartesian_convergence_validation as validation
from hwr.eval.seed_contract import seed_commitment
from hwr.eval.target_selection import Candidate


def test_frozen_plan_has_twelve_cells_and_domain_separated_seeds(
    monkeypatch,
) -> None:
    salt = "ab" * 32
    cells = convergence.frozen_cells()
    records = [
        convergence.raw_seed_record(salt, cell, ordinal)
        for cell in cells
        for ordinal in range(4)
    ]

    assert len(cells) == 12
    assert [cell.ordinal for cell in cells] == list(range(12))
    assert {
        (cell.task_id, cell.observation_latency_steps, cell.action_latency_steps)
        for cell in cells
    } == {
        (task, observation, action)
        for task in convergence.TASK_IDS
        for observation in (1, 2)
        for action in (1, 2)
    }
    assert len({record["planned_episode_id"] for record in records}) == 48
    assert len({record["environment_seed"] for record in records}) == 48
    assert len({record["policy_rng_seed"] for record in records}) == 48
    assert not (
        {record["environment_seed"] for record in records}
        & {record["policy_rng_seed"] for record in records}
    )

    pair = convergence.pair_identity(records[0]["planned_episode_id"])
    first = convergence.role_order(salt, pair)
    second = convergence.role_order(salt, pair)
    assert first == second
    assert set(first[1]) == set(convergence.ROLES)


def test_targets_and_legacy_treatment_match_frozen_formula() -> None:
    candidate = Candidate(
        center=(1.0, 0.0, 0.7),
        normal=(-1.0, 0.0, 0.0),
        width=0.12,
        prominence=0.1,
        support_count=30,
        view_count=2,
        first_frame=0,
        first_row=20,
        first_column=30,
    )
    targets = convergence.preposition_targets(
        candidate, (0.0, 0.0, 0.0), (0.1, 0.2, 0.4)
    )
    forward = np.asarray((0.9, -0.2)) / np.linalg.norm((0.9, -0.2))
    normal = np.asarray((-forward[0], -forward[1], 0.0))
    lateral = np.asarray((-forward[1], forward[0], 0.0))
    assert targets["left"] == pytest.approx(
        np.asarray(candidate.center)
        + 0.18 * normal
        + 0.12 * lateral
        + (0.0, 0.0, 0.05)
    )
    assert targets["right"] == pytest.approx(
        np.asarray(candidate.center)
        + 0.18 * normal
        - 0.12 * lateral
        + (0.0, 0.0, 0.05)
    )

    legacy = convergence.legacy_transform(
        (1.0, 1.0, 0.0),
        0.08,
        acquisition_yaw=1.2,
        current_base_yaw=-0.7,
    )
    assert np.linalg.norm(legacy) == pytest.approx(0.08)
    assert legacy[0] == pytest.approx(legacy[1])


def test_first_treatment_guard_allows_only_arm_linear_xy() -> None:
    legacy = np.zeros(16, dtype="<f8")
    fixed = legacy.copy()
    legacy[[2, 8]] = 0.2
    fixed[[3, 9]] = 0.2

    guard = convergence.first_treatment_guard(legacy, fixed)

    assert guard["different_bytes"] is True
    assert guard["only_arm_linear_xy_differs"] is True
    assert guard["arm_action_noncollapsed"] is True
    fixed[14] = 0.1
    assert convergence.first_treatment_guard(
        legacy, fixed
    )["only_arm_linear_xy_differs"] is False


def test_symmetric_carry_forward_and_normalized_auc() -> None:
    outcome = convergence.arm_outcome((0.20, 0.10, 0.05))

    assert len(outcome["distances_m"]) == 101
    assert outcome["d_100_m"] == pytest.approx(0.05)
    assert outcome["minimum_m"] == pytest.approx(0.05)
    assert outcome["normalized_auc"] == pytest.approx(
        (0.10 + 99 * 0.05) / 100 / 0.20
    )
    assert outcome["symmetric_terminal_carry_forward"] is True
    with pytest.raises(convergence.CartesianConvergenceContractError):
        convergence.arm_outcome((None,))


def test_equal_cell_bootstrap_and_binary_guards_accept_frozen_effect(
    monkeypatch,
) -> None:
    bank = _local_bank(monkeypatch)
    terminals = _terminal_document(bank, delta=0.20)

    first = validation.analyze_terminals(terminals, bank)
    second = validation.analyze_terminals(terminals, bank)

    assert first == second
    assert first["decision"] == (
        "accepted as paired physical Cartesian convergence evidence"
    )
    assert first["continuous"]["point_estimate"] == pytest.approx(0.20)
    assert first["continuous"]["one_sided_95_lower"] == pytest.approx(0.20)
    assert first["continuous"]["bootstrap_replicates"] == 10_000
    assert first["continuous"]["bootstrap_seed"] == 20_265_102
    assert first["binary"]["frame_fixed_win_count"] == 36
    assert set(first["binary"]["by_latency_combination"].values()) == {9}


def test_analysis_rejects_duplicate_identity_and_hard_safety(
    monkeypatch,
) -> None:
    bank = _local_bank(monkeypatch)
    records = _terminal_records(bank, delta=0.20)
    records[1]["pair_id"] = records[0]["pair_id"]
    invalid = validation.analyze_terminals(_terminal_document(bank, records=records), bank)
    assert invalid["decision"] == "invalid"

    records = _terminal_records(bank, delta=0.20)
    records[-1]["arms"]["frame_fixed"]["severe_collision_count"] = 1
    records[-1]["arms"]["frame_fixed"]["hard_guard_passed"] = False
    records[-1]["arms"]["frame_fixed"]["hard_failure_reason"] = "severe_collision"
    records[-1]["hard_safety_stop"] = True
    rejected = validation.analyze_terminals(_terminal_document(bank, records=records), bank)
    assert rejected["decision"] == "rejected"
    assert rejected["hard_guard"]["passed"] is False

    records = _terminal_records(bank, delta=0.20)
    role = records[0]["role_order"][0]
    records[0]["arms"] = {role: records[0]["arms"][role]}
    records[0]["arms"][role]["hard_guard_passed"] = False
    records[0]["arms"][role]["hard_failure_reason"] = "severe_collision"
    records[0]["arms"][role]["severe_collision_count"] = 1
    records[0]["continuation_replay_identities"] = {
        role: records[0]["continuation_identity"]
    }
    records[0]["hard_safety_stop"] = True
    records[0]["delta_i"] = None
    convergence.attach_pair_invariants(records[0], complete=False)
    partial = validation.analyze_terminals(
        _terminal_document(bank, records=records[:1]), bank
    )
    assert partial["decision"] == "rejected"
    assert partial["identity_guard"]["terminated_by_hard_safety"] is True


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("arms", "frame_fixed", "normalized_auc"), 9.0),
        (("arms", "frame_fixed", "d_100_m"), 9.0),
        (("arms", "frame_fixed", "minimum_m"), 9.0),
        (("arms", "frame_fixed", "proposed_action_sha256"), "0" * 64),
        (("planned_episode_id",), "0" * 64),
        (("continuation_identity_equal",), False),
    ),
)
def test_analysis_rejects_terminal_tampering(monkeypatch, path, value) -> None:
    bank = _local_bank(monkeypatch)
    terminals = _terminal_document(bank, delta=0.20)
    target = terminals["records"][0]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value

    assert validation.analyze_terminals(terminals, bank)["decision"] == "invalid"


def test_validate_bank_rejects_candidate_and_seed_tampering(
    monkeypatch,
) -> None:
    salt = "cd" * 32
    monkeypatch.setattr(convergence, "SALT_COMMITMENT", seed_commitment(salt))
    bank = _bank(salt)

    validation.validate_bank(bank)

    tampered = copy.deepcopy(bank)
    tampered["pairs"][0]["selected_index"] = 1
    with pytest.raises(
        convergence.CartesianConvergenceContractError,
        match="pair differs|selected index",
    ):
        validation.validate_bank(tampered)

    tampered = copy.deepcopy(bank)
    tampered["seed_audit"][1]["candidate_ordinal"] = 2
    with pytest.raises(
        convergence.CartesianConvergenceContractError,
        match="contiguous",
    ):
        validation.validate_bank(tampered)


def test_bank_state_machine_rejects_mismatch_prefix_and_post_accept_audit(
    monkeypatch,
) -> None:
    bank = _local_bank(monkeypatch)
    mismatch = copy.deepcopy(bank)
    row = mismatch["seed_audit"][0]
    row["sampled_observation_latency_steps"] = 3
    row["latency_matched"] = row["acquisition_executed"] = False
    row["eligibility_reason"] = "natural_latency_mismatch"
    with pytest.raises(
        convergence.CartesianConvergenceContractError,
        match="mismatch contains prefix",
    ):
        validation.validate_bank(mismatch)

    extra = copy.deepcopy(bank)
    cell = convergence.frozen_cells()[0]
    seed = convergence.raw_seed_record(bank["salt_reveal"], cell, 3)
    extra["seed_audit"].insert(
        3,
        {
            **cell.to_dict(),
            **seed,
            "sampled_observation_latency_steps": 3,
            "sampled_action_latency_steps": 3,
            "latency_matched": False,
            "acquisition_executed": False,
            "eligibility_reason": "natural_latency_mismatch",
        },
    )
    with pytest.raises(
        convergence.CartesianConvergenceContractError,
        match="after third eligible",
    ):
        validation.validate_bank(extra)


def test_seed_state_machine_rejects_sixty_fifth_matched_prefix(monkeypatch) -> None:
    salt = "34" * 32
    monkeypatch.setattr(convergence, "SALT_COMMITMENT", seed_commitment(salt))
    cell = convergence.frozen_cells()[0]
    records = []
    for ordinal in range(65):
        seed = convergence.raw_seed_record(salt, cell, ordinal)
        records.append(
            {
                **cell.to_dict(),
                **seed,
                "sampled_observation_latency_steps": cell.observation_latency_steps,
                "sampled_action_latency_steps": cell.action_latency_steps,
                "latency_matched": True,
                "acquisition_executed": True,
                **_ineligible_prefix(),
            }
        )
    with pytest.raises(
        convergence.CartesianConvergenceContractError,
        match="budget exceeded",
    ):
        validation._validate_seed_records(records, salt)


def _terminal_document(bank, delta=0.20, records=None) -> dict[str, object]:
    values = _terminal_records(bank, delta) if records is None else records
    return {
        "schema_version": convergence.TERMINAL_SCHEMA,
        "bank_source_commit": bank["source_commit"],
        "planned_pair_count": len(bank["pairs"]),
        "terminal_pair_count": len(values),
        "records": values,
    }


def _terminal_records(bank, delta: float) -> list[dict[str, object]]:
    records = []
    for pair in bank["pairs"]:
        arms = {
            role: _arm(0.50 if role == "frame_fixed" else 0.50 + delta, pair, role)
            for role in pair["role_order"]
        }
        fields = (
            "pair_id", "planned_episode_id", "task_id", "cell_id",
            "replicate_ordinal", "observation_latency_steps",
            "action_latency_steps", "environment_seed", "policy_rng_seed",
            "role_order", "candidate_set_sha256", "selected_index",
            "continuation_identity", "prefix_trace_sha256",
            "first_treatment_guard",
        )
        record = {
            **{name: pair[name] for name in fields},
            "continuation_replay_identities": {
                role: pair["continuation_identity"] for role in pair["role_order"]
            },
            "continuation_identity_equal": True,
            "pair_identity_valid": True,
            "resolved": True,
            "hard_safety_stop": False,
            "arms": arms,
            "delta_i": (
                arms["frame_legacy"]["normalized_auc"]
                - arms["frame_fixed"]["normalized_auc"]
            ),
        }
        convergence.attach_pair_invariants(record, complete=True)
        records.append(record)
    return records


def _arm(auc: float, pair, role: str) -> dict[str, object]:
    endpoint = 0.20 * auc
    distances = [0.20] + [endpoint] * 100
    outcome = convergence.arm_outcome(distances)
    actions = [pair["first_treatment_actions"][role]] * 100
    tool = [
        {"left_m": value, "right_m": value, "mean_m": value}
        for value in distances
    ]
    invariants = {
        name: {"sha256": name, "bytes": len(name)}
        for name in ("safety", "cap", "gripper", "phase", "target", "fk", "backend")
    }
    return {
        "role": role,
        "b2_control_step_limit": 100,
        "executed_b2_steps": 100,
        **outcome,
        "tool_distances": tool,
        "carried_forward_step_count": 0,
        "first_treatment_action": pair["first_treatment_actions"][role],
        "proposed_actions": actions,
        "applied_actions": actions,
        "proposed_action_sha256": convergence.canonical_sha256(actions),
        "applied_action_sha256": convergence.canonical_sha256(actions),
        "action_summary": convergence.action_summary(actions, actions),
        "first_10_applied_nonzero_arm_signed_derivatives": (
            convergence.signed_derivatives(tool, actions)
        ),
        "severe_collision_count": 0,
        "invalid_force_count": 0,
        "stale_action_applied_count": 0,
        "p40_conservation_maximum_absolute_difference": 0.0,
        "action_bounds_valid": True,
        "hard_guard_passed": True,
        "hard_failure_reason": None,
        "invariant_identities": invariants,
    }


def _bank(salt: str) -> dict[str, object]:
    candidate = Candidate(
        (1.0, 0.0, 0.7),
        (-1.0, 0.0, 0.0),
        0.12,
        0.1,
        30,
        2,
        0,
        20,
        30,
    )
    document = {"candidates": [list(candidate.canonical_record())]}
    candidate_bytes = json.dumps(
        document, separators=(",", ":"), sort_keys=True
    ).encode()
    audit = []
    pairs = []
    treatment_actions = _treatment_actions()
    treatment_guard = convergence.first_treatment_guard(
        treatment_actions["frame_legacy"],
        treatment_actions["frame_fixed"],
    )
    for cell in convergence.frozen_cells():
        for ordinal in range(3):
            seed = convergence.raw_seed_record(salt, cell, ordinal)
            audit_record = {
                **cell.to_dict(),
                **seed,
                "sampled_observation_latency_steps": (
                    cell.observation_latency_steps
                ),
                "sampled_action_latency_steps": cell.action_latency_steps,
                "latency_matched": True,
                "acquisition_executed": True,
                **_audit_prefix(candidate, candidate_bytes),
            }
            audit.append(audit_record)
            pair_id = convergence.pair_identity(seed["planned_episode_id"])
            order_seed, order = convergence.role_order(salt, pair_id)
            pairs.append(
                {
                    **audit_record,
                    "pair_id": pair_id,
                    "replicate_ordinal": ordinal,
                    "role_order_domain_seed": order_seed,
                    "role_order": list(order),
                }
            )
    return {
        "schema_version": convergence.BANK_SCHEMA,
        "plan_id": convergence.PLAN_ID,
        "source_commit": "a" * 40,
        "salt_commitment": convergence.SALT_COMMITMENT,
        "salt_reveal": salt,
        "seed_schema": convergence.SEED_SCHEMA,
        "cells": [cell.to_dict() for cell in convergence.frozen_cells()],
        "seed_audit": audit,
        "pairs": pairs,
    }


def _audit_prefix(candidate, candidate_bytes) -> dict[str, object]:
    actions = _treatment_actions()
    return {
        "eligible": True,
        "eligibility_reason": "eligible",
        "candidate_count": 1,
        "candidate_set_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes_hex": candidate_bytes.hex(),
        "selected_index": 0,
        "selected_record": candidate.__dict__,
        "continuation_identity": {"identity": {"sha256": "a" * 64}},
        "prefix_trace_sha256": "b" * 64,
        "relative_yaw_at_b2": np.pi / 2.0,
        "acquisition_input_hashes": ["c" * 64],
        "acquisition_input_sequence_sha256": "d" * 64,
        "b0_b1_proposed_action_sha256": "e" * 64,
        "b0_b1_applied_action_sha256": "f" * 64,
        "acquisition_base_pose": [0.0, 0.0, 0.0],
        "acquisition_world_origin": [0.0, 0.0, 0.22],
        "first_treatment_actions": actions,
        "preposition_targets": {
            "left": [0.82, 0.12, 0.75],
            "right": [0.82, -0.12, 0.75],
        },
        "primitive_target_crosscheck": {"passed": True},
        "first_treatment_guard": convergence.first_treatment_guard(
            actions["frame_legacy"], actions["frame_fixed"]
        ),
    }


def _treatment_actions() -> dict[str, list[float]]:
    legacy = [0.0] * 16
    fixed = [0.0] * 16
    legacy[2] = legacy[8] = 0.2
    fixed[3] = fixed[9] = 0.2
    return {"frame_legacy": legacy, "frame_fixed": fixed}


def _ineligible_prefix() -> dict[str, object]:
    value = _audit_prefix(
        Candidate((1.0, 0.0, 0.7), (-1.0, 0.0, 0.0), 0.12, 0.1, 30, 2, 0, 20, 30),
        json.dumps(
            {"candidates": [[1000, 0, 700, -10000, 0, 0, 120, 0, 20, 30, 100, 30, 2]]},
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
    )
    value["eligible"] = False
    value["eligibility_reason"] = "candidate_set_empty"
    return value


def _local_bank(monkeypatch) -> dict[str, object]:
    salt = "cd" * 32
    monkeypatch.setattr(convergence, "SALT_COMMITMENT", seed_commitment(salt))
    bank = _bank(salt)
    bank["source_commit"] = "a" * 40
    return bank
