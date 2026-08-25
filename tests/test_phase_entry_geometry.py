from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from hwr.adapters.mujoco import phase_entry_geometry as bridge
from hwr.eval import phase_entry_geometry as geometry
from hwr.eval.seed_contract import seed_commitment
from hwr.eval.target_selection import Candidate, CandidateSet


def test_frozen_cohort_and_seed_derivation_are_exact() -> None:
    salt = "ab" * 32
    cells = geometry.frozen_cells()
    rows = [
        geometry.raw_seed_record(salt, cell, ordinal)
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
        for task in geometry.TASK_IDS
        for observation in geometry.LATENCY_VALUES
        for action in geometry.LATENCY_VALUES
    }
    assert len({row["planned_episode_id"] for row in rows}) == 48
    assert len({row["environment_seed"] for row in rows}) == 48
    assert len({row["policy_rng_seed"] for row in rows}) == 48
    assert not (
        {row["environment_seed"] for row in rows}
        & {row["policy_rng_seed"] for row in rows}
    )
    assert seed_commitment("p60-test") != geometry.SALT_COMMITMENT
    with pytest.raises(geometry.PhaseEntryGeometryContractError):
        geometry.raw_seed_record(salt, cells[0], geometry.RAW_SEED_LIMIT)


def test_geometry_separates_any_arm_outer_impossible_from_nominal_support(
    monkeypatch,
) -> None:
    candidate = _candidate()
    targets = {
        "left": (0.04 + geometry.ARM_OUTER_LENGTH_M, 0.31, 0.82),
        "right": (0.02, -0.31, 0.82),
    }
    tools = {
        "left": np.asarray(targets["left"], np.float64),
        "right": np.asarray((1.0, -0.31, 0.82), np.float64),
    }
    monkeypatch.setattr(
        geometry,
        "preposition_targets",
        lambda *args: targets,
    )
    monkeypatch.setattr(
        geometry,
        "independent_preposition_targets",
        lambda *args: targets,
    )
    monkeypatch.setattr(
        geometry,
        "policy_tool_position",
        lambda joints, arm: tuple(tools[arm]),
    )

    result = geometry.measure_phase_entry_geometry(
        candidate,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0,) * 6,
        (0.0,) * 6,
    )

    assert result["arms"]["left"]["strict_outer_impossible"] is True
    assert result["arms"]["right"]["strict_outer_status"] == "not_disproven"
    assert result["hard_bilateral_impossible"] is True
    assert result["both_arms_strict_outer_impossible"] is False
    assert result["arms"]["left"]["nominal_b2_support_deficit"] is False
    assert result["arms"]["right"]["nominal_b2_support_deficit"] is True
    assert result["nominal_bilateral_support_deficit"] is True
    assert result["nominal_b2_command_m"] == pytest.approx(0.40)


def test_real_target_fk_and_shoulder_recompute_in_acquisition_frame() -> None:
    candidate = _candidate()
    acquisition = (0.4, -0.2, 0.7)
    current = (0.9, 0.1, -0.3)

    result = geometry.measure_phase_entry_geometry(
        candidate,
        acquisition,
        current,
        (0.1, -0.2, 0.3, -0.1, 0.2, -0.3),
        (-0.1, 0.2, -0.3, 0.1, -0.2, 0.3),
    )

    assert result["target_formula_crosscheck"]["passed"] is True
    assert result["target_formula_crosscheck"]["maximum_error_m"] <= 1.0e-12
    assert result["arm_outer_length_m"] == pytest.approx(
        0.13 + 0.31 + 0.27 + 0.09 + 0.08 + np.hypot(0.255, 0.045)
    )
    assert result["frame"] == "acquisition"
    for arm in geometry.ARM_ORDER:
        assert len(result["arms"][arm]["shoulder_acquisition_m"]) == 3
        assert len(result["arms"][arm]["tool_acquisition_m"]) == 3
        assert len(result["arms"][arm]["preposition_target_acquisition_m"]) == 3


def test_threshold_decisions_are_independent_and_task_gated() -> None:
    rows = _summary_rows(strict_count=30, nominal_count=12)
    summary = geometry.aggregate_episode_rows(rows)

    assert (
        geometry.strict_diagnostic_decision(summary)
        == "strict_phase_entry_deficit_supported"
    )
    assert (
        geometry.nominal_diagnostic_decision(summary)
        == "nominal_b2_support_deficit_rejected"
    )

    rows[0]["geometry"]["hard_bilateral_impossible"] = False
    rows[1]["geometry"]["hard_bilateral_impossible"] = False
    rows[2]["geometry"]["hard_bilateral_impossible"] = False
    summary = geometry.aggregate_episode_rows(rows)
    assert (
        geometry.strict_diagnostic_decision(summary)
        == "strict_phase_entry_diagnostic_inconclusive"
    )


def test_analyze_evidence_recomputes_geometry_and_rejects_b2_or_nonprefix() -> None:
    salt = "cd" * 32
    plan, seed_audit, episodes = _evidence(salt)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(geometry, "SALT_COMMITMENT", seed_commitment(salt))
        analysis = geometry.analyze_evidence(plan, seed_audit, episodes)
        assert analysis["decision"].startswith("accepted")
        assert analysis["summary"]["episode_count"] == 36
        assert all(
            value["episode_count"] == 3
            for value in analysis["summary"]["by_cell"].values()
        )

        tampered = copy.deepcopy(episodes)
        tampered["records"][0]["b2_action_generated"] = True
        invalid = geometry.analyze_evidence(plan, seed_audit, tampered)
        assert invalid["decision"] == "invalid"
        assert "prefix-only" in invalid["validation_error"]

        tampered = copy.deepcopy(seed_audit)
        tampered["records"][0], tampered["records"][1] = (
            tampered["records"][1],
            tampered["records"][0],
        )
        invalid = geometry.analyze_evidence(plan, tampered, episodes)
        assert invalid["decision"] == "invalid"
        assert "raw seed prefix" in invalid["validation_error"]

        tampered = copy.deepcopy(episodes)
        tampered["records"][0]["selected_record"]["center"][0] += 0.01
        invalid = geometry.analyze_evidence(plan, seed_audit, tampered)
        assert invalid["decision"] == "invalid"
        assert "selected candidate identity" in invalid["validation_error"]


def test_adapter_b0_b1_stops_before_b2_without_gui(monkeypatch) -> None:
    candidate = _candidate()
    candidate_set = CandidateSet(
        ("a" * 64,),
        (candidate,),
        b"candidate",
        hashlib.sha256(b"candidate").hexdigest(),
    )
    observation = SimpleNamespace(timestamp_ns=1, sequence_id=1)
    history = []
    available = []
    trace = []
    input_hashes = []
    phase_indices = []
    post_steps = []

    def payload(*args, phase_index, phase_step):
        del args, phase_step
        phase_indices.append(phase_index)
        return b"input"

    def action(payload, selected, acquisition_pose, post_step):
        del payload, selected, acquisition_pose
        post_steps.append(post_step)
        return (0.0,) * 16

    def advance(backend, graph, current, vector, step):
        del backend, graph, vector
        row = {
            "step": step,
            "terminal": False,
            "action_bounds_valid": True,
            "outside_validity_window": False,
            "applied_action": [0.0] * 16,
            "hold_action": [0.0] * 16,
            "safety_intervened": False,
            "_motion_start": object(),
            "_motion_end": object(),
        }
        return current, row, None

    monkeypatch.setattr(bridge, "policy_input_bytes", payload)
    monkeypatch.setattr(bridge, "_input_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge.target_selection, "primitive_action", action)
    monkeypatch.setattr(bridge, "_advance", advance)
    monkeypatch.setattr(bridge, "_prefix_step_failure", lambda *args: None)

    result = bridge._run_b0_b1(
        SimpleNamespace(),
        SimpleNamespace(),
        observation,
        history,
        available,
        candidate_set,
        0,
        (0.0, 0.0, 0.0),
        7,
        trace,
        input_hashes,
        (0, 0),
    )

    assert result[1:4] == (None, None, False)
    assert len(trace) == geometry.B0_STEPS + geometry.B1_STEPS
    assert phase_indices == [5] * geometry.B0_STEPS + [6] * geometry.B1_STEPS
    assert post_steps == list(range(geometry.B0_STEPS + geometry.B1_STEPS))
    assert max(phase_indices) < bridge.B2_PHASE_INDEX
    assert max(post_steps) == 399


def _candidate() -> Candidate:
    return Candidate(
        center=(1.7, 0.2, 0.75),
        normal=(-1.0, 0.0, 0.0),
        width=0.12,
        prominence=0.1,
        support_count=30,
        view_count=2,
        first_frame=0,
        first_row=20,
        first_column=30,
    )


def _summary_rows(strict_count: int, nominal_count: int):
    rows = []
    for index in range(36):
        task = geometry.TASK_IDS[index // 12]
        cell = geometry.frozen_cells()[index // 3]
        task_index = index % 12
        arms = {
            arm: {
                "shoulder_to_preposition_m": 1.0,
                "strict_outer_margin_m": -0.1,
                "tool_to_preposition_d0_m": 1.0,
                "nominal_b2_support_margin_m": -0.5,
            }
            for arm in geometry.ARM_ORDER
        }
        rows.append(
            {
                "task_id": task,
                "cell_id": cell.cell_id,
                "observation_latency_steps": cell.observation_latency_steps,
                "action_latency_steps": cell.action_latency_steps,
                "geometry": {
                    "hard_bilateral_impossible": (
                        task_index < strict_count // len(geometry.TASK_IDS)
                    ),
                    "both_arms_strict_outer_impossible": False,
                    "nominal_bilateral_support_deficit": (
                        task_index < nominal_count // len(geometry.TASK_IDS)
                    ),
                    "candidate_base_horizontal_range_m": 1.0,
                    "candidate_heading_error_rad": 0.0,
                    "arms": arms,
                },
            }
        )
    return rows


def _evidence(salt: str):
    plan_rows = []
    audits = []
    records = []
    candidate = _candidate()
    candidate_document = {
        "schema_version": geometry.target_selection.CANDIDATE_SCHEMA,
        "acquisition_input_sha256": ["a" * 64],
        "candidate_count": 1,
        "candidates": [list(candidate.canonical_record())],
    }
    candidate_bytes = geometry.canonical_bytes(candidate_document)
    for cell in geometry.frozen_cells():
        for episode_ordinal in range(geometry.EPISODES_PER_CELL):
            seed = geometry.raw_seed_record(salt, cell, episode_ordinal)
            geometry_row = geometry.measure_phase_entry_geometry(
                candidate,
                (0.0, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (0.0,) * 6,
                (0.0,) * 6,
            )
            trace = [
                {
                    "step": step,
                    "terminal": False,
                    "action_bounds_valid": True,
                    "outside_validity_window": False,
                    "applied_action": [0.0] * 16,
                    "hold_action": [0.0] * 16,
                    "safety_intervened": False,
                }
                for step in range(geometry.PREFIX_STEPS)
            ]
            shared = {
                **cell.to_dict(),
                **seed,
                "sampled_observation_latency_steps": cell.observation_latency_steps,
                "sampled_action_latency_steps": cell.action_latency_steps,
                "latency_matched": True,
            }
            plan_rows.append({**shared, "episode_ordinal": episode_ordinal})
            audits.append(
                {
                    **shared,
                    "physical_prefix_executed": True,
                    "eligible": True,
                    "eligibility_reason": "eligible",
                    "raw_prefix_trace_sha256": geometry.canonical_sha256(trace),
                }
            )
            records.append(
                {
                    **shared,
                    "episode_ordinal": episode_ordinal,
                    "eligible": True,
                    "candidate_count": 1,
                    "candidate_set_sha256": hashlib.sha256(
                        candidate_bytes
                    ).hexdigest(),
                    "candidate_bytes_hex": candidate_bytes.hex(),
                    "selected_index": 0,
                    "selected_record": {
                        "center": list(candidate.center),
                        "normal": list(candidate.normal),
                        "width": candidate.width,
                        "prominence": candidate.prominence,
                        "support_count": candidate.support_count,
                        "view_count": candidate.view_count,
                        "first_frame": candidate.first_frame,
                        "first_row": candidate.first_row,
                        "first_column": candidate.first_column,
                    },
                    "hard_safety_failure": False,
                    "runtime_observation_latency_steps": (
                        cell.observation_latency_steps
                    ),
                    "runtime_action_latency_steps": cell.action_latency_steps,
                    "latency_override_inactive": True,
                    "runtime_randomization_sha256": "9" * 64,
                    "input_failure_reason": None,
                    "prefix_failure_reason": None,
                    "prefix_step_count": geometry.PREFIX_STEPS,
                    "prefix_complete": True,
                    "prefix_terminal_observed": False,
                    "prefix_action_bounds_valid": True,
                    "prefix_stale_action_applied_count": 0,
                    "prefix_safety_intervention_count": 0,
                    "prefix_severe_collision_count": 0,
                    "prefix_invalid_force_count": 0,
                    "prefix_p40_conservation_maximum_absolute_difference": 0.0,
                    "policy_input_count": geometry.PREFIX_STEPS,
                    "policy_input_sha256": ["a" * 64] * geometry.PREFIX_STEPS,
                    "policy_input_sequence_sha256": geometry.canonical_sha256(
                        ["a" * 64] * geometry.PREFIX_STEPS
                    ),
                    "candidate_final_policy_input_sha256": "b" * 64,
                    "b2_entry_policy_input_sha256": "c" * 64,
                    "raw_prefix_trace": trace,
                    "raw_prefix_trace_sha256": geometry.canonical_sha256(trace),
                    "acquisition_base_pose": [0.0, 0.0, 0.0],
                    "b2_policy_base_pose": [0.2, 0.0, 0.0],
                    "geometry": geometry_row,
                    "fk_crosscheck_max_error_m": 0.0,
                    "b2_action_generated": False,
                    "b2_action_executed": False,
                    "post_prefix_action_count": 0,
                }
            )
    plan = {
        "schema_version": geometry.PLAN_SCHEMA,
        "proposal_id": geometry.PROPOSAL_ID,
        "plan_id": geometry.PLAN_ID,
        "source_commit": "e" * 40,
        "frozen_document_commit": "f" * 40,
        "salt_commitment": seed_commitment(salt),
        "seed_schema": geometry.SEED_SCHEMA,
        "natural_evaluation_latency_rejection": True,
        "reset_latency_override_used": False,
        "prefix_only": True,
        "b2_action_allowed": False,
        "complete_case_deletion_allowed": False,
        "raw_seed_limit_per_cell": geometry.RAW_SEED_LIMIT,
        "latency_match_limit_per_cell": geometry.LATENCY_MATCH_LIMIT,
        "episodes_per_cell": geometry.EPISODES_PER_CELL,
        "prefix_steps_per_episode": geometry.PREFIX_STEPS,
        "cells": [cell.to_dict() for cell in geometry.frozen_cells()],
        "planned_episode_count": 36,
        "episodes": plan_rows,
        "infeasible_cells": [],
        "hard_stop": None,
    }
    seed_audit = {
        "schema_version": geometry.SEED_AUDIT_SCHEMA,
        "proposal_id": geometry.PROPOSAL_ID,
        "plan_id": geometry.PLAN_ID,
        "source_commit": "e" * 40,
        "salt_commitment": seed_commitment(salt),
        "salt_reveal": salt,
        "records": audits,
    }
    episodes = {
        "schema_version": geometry.EPISODES_SCHEMA,
        "proposal_id": geometry.PROPOSAL_ID,
        "plan_id": geometry.PLAN_ID,
        "source_commit": "e" * 40,
        "records": records,
    }
    return plan, seed_audit, episodes
