from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from hwr.eval import cartesian_convergence as convergence
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
    targets = convergence.preposition_targets(candidate)
    assert targets["left"] == pytest.approx((0.82, 0.12, 0.75))
    assert targets["right"] == pytest.approx((0.82, -0.12, 0.75))

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


def test_equal_cell_bootstrap_and_binary_guards_accept_frozen_effect() -> None:
    records = _terminal_records(delta=0.20)
    terminals = {
        "schema_version": convergence.TERMINAL_SCHEMA,
        "records": records,
    }

    first = convergence.analyze_terminals(terminals)
    second = convergence.analyze_terminals(terminals)

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


def test_analysis_rejects_duplicate_identity_and_hard_safety() -> None:
    records = _terminal_records(delta=0.20)
    records[1]["pair_id"] = records[0]["pair_id"]
    invalid = convergence.analyze_terminals(
        {"schema_version": convergence.TERMINAL_SCHEMA, "records": records}
    )
    assert invalid["decision"] == "invalid"

    records = _terminal_records(delta=0.20)
    records[0]["arms"]["frame_fixed"]["severe_collision_count"] = 1
    rejected = convergence.analyze_terminals(
        {"schema_version": convergence.TERMINAL_SCHEMA, "records": records}
    )
    assert rejected["decision"] == "rejected"
    assert rejected["hard_guard"]["passed"] is False

    records[0]["hard_safety_stop"] = True
    partial = convergence.analyze_terminals(
        {
            "schema_version": convergence.TERMINAL_SCHEMA,
            "records": records[:1],
        }
    )
    assert partial["decision"] == "rejected"
    assert partial["identity_guard"]["terminated_by_hard_safety"] is True


def test_validate_bank_rejects_candidate_and_seed_tampering(
    monkeypatch,
) -> None:
    salt = "cd" * 32
    monkeypatch.setattr(convergence, "SALT_COMMITMENT", seed_commitment(salt))
    bank = _bank(salt)

    convergence.validate_bank(bank)

    tampered = copy.deepcopy(bank)
    tampered["pairs"][0]["selected_index"] = 1
    with pytest.raises(
        convergence.CartesianConvergenceContractError,
        match="pair differs|selected index",
    ):
        convergence.validate_bank(tampered)

    tampered = copy.deepcopy(bank)
    tampered["seed_audit"][1]["candidate_ordinal"] = 2
    with pytest.raises(
        convergence.CartesianConvergenceContractError,
        match="contiguous",
    ):
        convergence.validate_bank(tampered)


def _terminal_records(delta: float) -> list[dict[str, object]]:
    records = []
    for cell in convergence.frozen_cells():
        for replicate in range(3):
            fixed_auc = 0.50
            legacy_auc = fixed_auc + delta
            arms = {
                "frame_fixed": _arm(fixed_auc, 0.10),
                "frame_legacy": _arm(legacy_auc, 0.20),
            }
            records.append(
                {
                    **cell.to_dict(),
                    "pair_id": hashlib.sha256(
                        f"{cell.cell_id}-{replicate}".encode()
                    ).hexdigest(),
                    "pair_identity_valid": True,
                    "continuation_identity_equal": True,
                    "first_treatment_guard": {
                        "finite": True,
                        "only_arm_linear_xy_differs": True,
                        "different_bytes": True,
                        "arm_action_noncollapsed": True,
                    },
                    "resolved": True,
                    "arms": arms,
                    "safety_identity_equal": True,
                    "cap_identity_equal": True,
                    "gripper_identity_equal": True,
                    "phase_identity_equal": True,
                    "target_identity_equal": True,
                    "fk_identity_equal": True,
                    "backend_identity_equal": True,
                }
            )
    return records


def _arm(auc: float, endpoint: float) -> dict[str, object]:
    return {
        "distances_m": [0.20] + [endpoint] * 100,
        "normalized_auc": auc,
        "d_100_m": endpoint,
        "severe_collision_count": 0,
        "invalid_force_count": 0,
        "stale_action_applied_count": 0,
        "p40_conservation_maximum_absolute_difference": 0.0,
        "action_bounds_valid": True,
        "hard_guard_passed": True,
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
            audit.append(
                {
                    **seed,
                    "cell_id": cell.cell_id,
                    "task_id": cell.task_id,
                    "sampled_observation_latency_steps": (
                        cell.observation_latency_steps
                    ),
                    "sampled_action_latency_steps": (
                        cell.action_latency_steps
                    ),
                    "latency_matched": True,
                    "acquisition_executed": True,
                    **_audit_prefix(candidate, candidate_bytes),
                }
            )
            pair_id = convergence.pair_identity(seed["planned_episode_id"])
            order_seed, order = convergence.role_order(salt, pair_id)
            pairs.append(
                {
                    **cell.to_dict(),
                    **seed,
                    "pair_id": pair_id,
                    "replicate_ordinal": ordinal,
                    "eligible": True,
                    "candidate_bytes_hex": candidate_bytes.hex(),
                    "candidate_set_sha256": hashlib.sha256(
                        candidate_bytes
                    ).hexdigest(),
                    "selected_index": 0,
                    "candidate_count": 1,
                    "selected_record": candidate.__dict__,
                    "relative_yaw_at_b2": np.pi / 2.0,
                    "continuation_identity": {"identity": {"sha256": "a" * 64}},
                    "prefix_trace_sha256": "b" * 64,
                    "first_treatment_actions": treatment_actions,
                    "role_order_domain_seed": order_seed,
                    "role_order": list(order),
                    "first_treatment_guard": treatment_guard,
                }
            )
    return {
        "schema_version": convergence.BANK_SCHEMA,
        "plan_id": convergence.PLAN_ID,
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
        "selected_index": 0,
        "continuation_identity": {"identity": {"sha256": "a" * 64}},
        "prefix_trace_sha256": "b" * 64,
        "relative_yaw_at_b2": np.pi / 2.0,
        "first_treatment_actions": actions,
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
