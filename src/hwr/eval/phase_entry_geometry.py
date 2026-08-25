"""Frozen geometry and cohort contracts for R0001-P60."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from hwr.eval import target_selection
from hwr.eval.cartesian_convergence import preposition_targets
from hwr.eval.seed_contract import (
    SEED_SCHEMA,
    derive_domain_seed,
    planned_episode_id,
    require_seed_reveal,
)
from hwr.eval.target_selection import Candidate
from hwr.eval.tool_kinematics import policy_tool_position


PROPOSAL_ID = "R0001-P60"
PLAN_ID = "R0001-P60-E1-formal"
SALT_COMMITMENT = "263e9f85e32f4a3f5f1560ba82cd820a558cc0aad9a5710bbdf6a3306e3f9c55"
TASK_IDS = (
    "tidy_living_room_3d/v1",
    "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
)
LATENCY_VALUES = (1, 2)
EPISODES_PER_CELL = 3
LATENCY_MATCH_LIMIT = 16
RAW_SEED_LIMIT = 768
ACQUISITION_STEPS = 995
B0_STEPS = 100
B1_STEPS = 300
PREFIX_STEPS = ACQUISITION_STEPS + B0_STEPS + B1_STEPS
B2_STEPS = 100
CONTROL_HZ = 20.0
B2_VELOCITY_M_PER_S = 0.08
READINESS_ALLOWANCE_M = 0.10
NOMINAL_B2_COMMAND_M = B2_STEPS * B2_VELOCITY_M_PER_S / CONTROL_HZ
NOMINAL_B2_SUPPORT_M = NOMINAL_B2_COMMAND_M + READINESS_ALLOWANCE_M
FLOAT_TOLERANCE_M = 1.0e-12
ARM_ORDER = ("left", "right")
SHOULDER_LOCAL_M = {
    "left": (0.02, 0.31, 0.82),
    "right": (0.02, -0.31, 0.82),
}
ARM_OUTER_SEGMENTS_M = (
    0.13,
    0.31,
    0.27,
    0.09,
    0.08,
    math.sqrt(0.255**2 + 0.045**2),
)
ARM_OUTER_LENGTH_M = sum(ARM_OUTER_SEGMENTS_M)
PLAN_SCHEMA = "hwr.p60-phase-entry-plan/v1"
SEED_AUDIT_SCHEMA = "hwr.p60-phase-entry-seed-audit/v1"
EPISODES_SCHEMA = "hwr.p60-phase-entry-episodes/v1"


class PhaseEntryGeometryContractError(ValueError):
    """Raised when P60 evidence violates its frozen contract."""


@dataclass(frozen=True)
class Cell:
    ordinal: int
    task_id: str
    observation_latency_steps: int
    action_latency_steps: int

    @property
    def cell_id(self) -> str:
        return (
            f"cell-{self.ordinal:02d}-"
            f"o{self.observation_latency_steps}-a{self.action_latency_steps}"
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "cell_id": self.cell_id}


def frozen_cells() -> tuple[Cell, ...]:
    return tuple(
        Cell(ordinal, task, observation, action)
        for ordinal, (task, observation, action) in enumerate(
            (task, observation, action)
            for task in TASK_IDS
            for observation in LATENCY_VALUES
            for action in LATENCY_VALUES
        )
    )


def raw_seed_record(
    salt: str,
    cell: Cell,
    raw_ordinal: int,
) -> dict[str, object]:
    if not 0 <= raw_ordinal < RAW_SEED_LIMIT:
        raise PhaseEntryGeometryContractError("raw seed ordinal is outside plan")
    episode_id = planned_episode_id(
        PLAN_ID,
        cell.task_id,
        cell.cell_id,
        raw_ordinal,
    )
    return {
        "raw_seed_ordinal": raw_ordinal,
        "planned_episode_id": episode_id,
        "environment_seed": derive_domain_seed(salt, "environment", episode_id),
        "policy_rng_seed": derive_domain_seed(salt, "policy", episode_id),
    }


def independent_preposition_targets(
    candidate: Candidate,
    acquisition_base_pose: Sequence[float],
    b2_policy_base_pose: Sequence[float],
) -> dict[str, tuple[float, float, float]]:
    point = _vector(candidate.center, 3, "candidate center")
    transform = target_selection._acquisition_from_robot(
        acquisition_base_pose,
        b2_policy_base_pose,
    )
    base = transform[:3, 3]
    forward = point[:2] - base[:2]
    horizontal = float(np.linalg.norm(forward))
    if horizontal < 0.35:
        raise PhaseEntryGeometryContractError(
            "selected candidate cannot define B2 targets"
        )
    forward /= horizontal
    normal = np.asarray((-forward[0], -forward[1], 0.0), np.float64)
    lateral = np.asarray((-forward[1], forward[0], 0.0), np.float64)
    spacing = float(np.clip(candidate.width + 0.12, 0.18, 0.34))
    vertical = np.asarray((0.0, 0.0, 1.0), np.float64)
    return {
        "left": tuple(
            point + 0.18 * normal + 0.5 * spacing * lateral + 0.05 * vertical
        ),
        "right": tuple(
            point + 0.18 * normal - 0.5 * spacing * lateral + 0.05 * vertical
        ),
    }


def measure_phase_entry_geometry(
    candidate: Candidate,
    acquisition_base_pose: Sequence[float],
    b2_policy_base_pose: Sequence[float],
    left_joint_position: Sequence[float],
    right_joint_position: Sequence[float],
) -> dict[str, object]:
    acquisition = _vector(acquisition_base_pose, 3, "acquisition base pose")
    current = _vector(b2_policy_base_pose, 3, "B2 policy base pose")
    transform = target_selection._acquisition_from_robot(acquisition, current)
    base = transform[:3, 3]
    point = _vector(candidate.center, 3, "candidate center")
    horizontal = float(np.linalg.norm(point[:2] - base[:2]))
    if horizontal < 0.35:
        raise PhaseEntryGeometryContractError(
            "selected candidate is inside the frozen minimum range"
        )
    heading = math.atan2(point[1] - base[1], point[0] - base[0])
    base_yaw = _wrap(float(current[2] - acquisition[2]))
    heading_error = _wrap(heading - base_yaw)
    reused_targets = preposition_targets(candidate, acquisition, current)
    independent_targets = independent_preposition_targets(
        candidate,
        acquisition,
        current,
    )
    target_error = max(
        float(
            np.linalg.norm(
                np.asarray(reused_targets[arm], np.float64)
                - np.asarray(independent_targets[arm], np.float64)
            )
        )
        for arm in ARM_ORDER
    )
    joints = {
        "left": _vector(left_joint_position, 6, "left joint position"),
        "right": _vector(right_joint_position, 6, "right joint position"),
    }
    arms = {}
    for arm in ARM_ORDER:
        shoulder = _transform_point(transform, SHOULDER_LOCAL_M[arm])
        local_tool = policy_tool_position(joints[arm], arm)
        tool = _transform_point(transform, local_tool)
        target = np.asarray(reused_targets[arm], np.float64)
        outer_distance = float(np.linalg.norm(target - shoulder))
        tool_distance = float(np.linalg.norm(target - tool))
        strict_margin = ARM_OUTER_LENGTH_M - outer_distance
        nominal_margin = NOMINAL_B2_SUPPORT_M - tool_distance
        arms[arm] = {
            "shoulder_acquisition_m": shoulder.tolist(),
            "tool_acquisition_m": tool.tolist(),
            "preposition_target_acquisition_m": target.tolist(),
            "shoulder_to_preposition_m": outer_distance,
            "strict_outer_margin_m": strict_margin,
            "strict_outer_impossible": strict_margin < -FLOAT_TOLERANCE_M,
            "strict_outer_status": (
                "strict_outer_impossible"
                if strict_margin < -FLOAT_TOLERANCE_M
                else "not_disproven"
            ),
            "tool_to_preposition_d0_m": tool_distance,
            "nominal_b2_support_margin_m": nominal_margin,
            "nominal_b2_support_deficit": nominal_margin < -FLOAT_TOLERANCE_M,
        }
    result = {
        "frame": "acquisition",
        "candidate_base_horizontal_range_m": horizontal,
        "candidate_heading_error_rad": heading_error,
        "b1_residual_command": {
            "base_linear_m_per_s": (
                0.0
                if abs(heading_error) > 0.35
                else float(np.clip(0.6 * (horizontal - 0.85), 0.0, 0.12))
            ),
            "base_angular_rad_per_s": float(
                np.clip(heading_error, -0.25, 0.25)
            ),
        },
        "acquisition_base_pose": acquisition.tolist(),
        "b2_policy_base_pose": current.tolist(),
        "joint_position": {
            arm: joints[arm].tolist() for arm in ARM_ORDER
        },
        "preposition_targets": {
            arm: list(reused_targets[arm]) for arm in ARM_ORDER
        },
        "target_formula_crosscheck": {
            "independent_targets": {
                arm: list(independent_targets[arm]) for arm in ARM_ORDER
            },
            "maximum_error_m": target_error,
            "passed": target_error <= FLOAT_TOLERANCE_M,
        },
        "arm_outer_segments_m": list(ARM_OUTER_SEGMENTS_M),
        "arm_outer_length_m": ARM_OUTER_LENGTH_M,
        "nominal_b2_command_m": NOMINAL_B2_COMMAND_M,
        "readiness_allowance_m": READINESS_ALLOWANCE_M,
        "arms": arms,
        "hard_bilateral_impossible": any(
            bool(value["strict_outer_impossible"]) for value in arms.values()
        ),
        "both_arms_strict_outer_impossible": all(
            bool(value["strict_outer_impossible"]) for value in arms.values()
        ),
        "nominal_bilateral_support_deficit": any(
            bool(value["nominal_b2_support_deficit"]) for value in arms.values()
        ),
    }
    if not _all_finite(result):
        raise PhaseEntryGeometryContractError("phase-entry geometry is nonfinite")
    return result


def analyze_evidence(
    plan: Mapping[str, object],
    seed_audit: Mapping[str, object],
    episodes: Mapping[str, object],
) -> dict[str, object]:
    try:
        rows = _validate_evidence(plan, seed_audit, episodes)
        summary = aggregate_episode_rows(rows)
        strict = strict_diagnostic_decision(summary)
        nominal = nominal_diagnostic_decision(summary)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "decision": "invalid",
            "strict_diagnostic": None,
            "nominal_diagnostic": None,
            "validation_error": str(error),
            "checks": {"passed": False},
            "summary": None,
        }
    checks = {
        "episode_count_36": len(rows) == 36,
        "cell_count_12": len(summary["by_cell"]) == 12,
        "three_episodes_per_cell": all(
            value["episode_count"] == 3
            for value in summary["by_cell"].values()
        ),
        "twelve_episodes_per_task": all(
            value["episode_count"] == 12
            for value in summary["by_task"].values()
        ),
        "natural_latency_prefix_cohort": True,
        "seed_derivation_recomputed": True,
        "candidate_identity_recomputed": True,
        "prefix_guards_recomputed": True,
        "b2_not_generated_or_executed": True,
        "target_formula_recomputed": True,
        "shoulder_and_outer_length_recomputed": True,
        "p52_fk_recomputed": True,
        "finite_tolerance_checked": True,
        "geometry_bit_identical_on_recompute": True,
        "strict_nominal_estimands_separate": True,
        "full_task_cell_latency_accounting": True,
    }
    return {
        "decision": "accepted as phase-entry necessary-geometry measurement evidence",
        "strict_diagnostic": strict,
        "nominal_diagnostic": nominal,
        "validation_error": None,
        "checks": {**checks, "passed": all(checks.values())},
        "summary": summary,
    }


def aggregate_episode_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    values = list(rows)
    return {
        "episode_count": len(values),
        "arm_count": 2 * len(values),
        "overall": _group_summary(values),
        "by_task": {
            task: _group_summary(
                [row for row in values if row["task_id"] == task]
            )
            for task in TASK_IDS
        },
        "by_cell": {
            cell.cell_id: _group_summary(
                [row for row in values if row["cell_id"] == cell.cell_id]
            )
            for cell in frozen_cells()
        },
        "by_observation_latency": {
            str(latency): _group_summary(
                [
                    row
                    for row in values
                    if row["observation_latency_steps"] == latency
                ]
            )
            for latency in LATENCY_VALUES
        },
        "by_action_latency": {
            str(latency): _group_summary(
                [
                    row
                    for row in values
                    if row["action_latency_steps"] == latency
                ]
            )
            for latency in LATENCY_VALUES
        },
    }


def strict_diagnostic_decision(summary: Mapping[str, object]) -> str:
    overall = int(summary["overall"]["hard_bilateral_impossible_count"])
    by_task = [
        int(value["hard_bilateral_impossible_count"])
        for value in summary["by_task"].values()
    ]
    if overall >= 30 and all(value >= 8 for value in by_task):
        return "strict_phase_entry_deficit_supported"
    if overall <= 12 and all(value <= 6 for value in by_task):
        return "strict_phase_entry_deficit_rejected"
    return "strict_phase_entry_diagnostic_inconclusive"


def nominal_diagnostic_decision(summary: Mapping[str, object]) -> str:
    overall = int(summary["overall"]["nominal_bilateral_support_deficit_count"])
    by_task = [
        int(value["nominal_bilateral_support_deficit_count"])
        for value in summary["by_task"].values()
    ]
    if overall >= 30 and all(value >= 8 for value in by_task):
        return "nominal_b2_support_deficit_supported"
    if overall <= 12 and all(value <= 6 for value in by_task):
        return "nominal_b2_support_deficit_rejected"
    return "nominal_b2_support_diagnostic_inconclusive"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _validate_evidence(plan, seed_audit, episodes):
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or seed_audit.get("schema_version") != SEED_AUDIT_SCHEMA
        or episodes.get("schema_version") != EPISODES_SCHEMA
    ):
        raise PhaseEntryGeometryContractError("artifact schema differs")
    if any(
        value.get("proposal_id") != PROPOSAL_ID
        for value in (plan, seed_audit, episodes)
    ):
        raise PhaseEntryGeometryContractError("artifact proposal differs")
    if any(
        value.get("plan_id") != PLAN_ID for value in (plan, seed_audit, episodes)
    ):
        raise PhaseEntryGeometryContractError("artifact plan differs")
    source_commits = {
        value.get("source_commit") for value in (plan, seed_audit, episodes)
    }
    if len(source_commits) != 1 or not _is_commit(next(iter(source_commits))):
        raise PhaseEntryGeometryContractError("artifact source commit differs")
    salt = str(seed_audit.get("salt_reveal", ""))
    require_seed_reveal(SALT_COMMITMENT, salt)
    if seed_audit.get("salt_commitment") != SALT_COMMITMENT:
        raise PhaseEntryGeometryContractError("salt commitment differs")
    cells = [cell.to_dict() for cell in frozen_cells()]
    frozen_plan = (
        plan.get("cells") == cells,
        plan.get("seed_schema") == SEED_SCHEMA,
        plan.get("salt_commitment") == SALT_COMMITMENT,
        plan.get("natural_evaluation_latency_rejection") is True,
        plan.get("reset_latency_override_used") is False,
        plan.get("prefix_only") is True,
        plan.get("b2_action_allowed") is False,
        plan.get("complete_case_deletion_allowed") is False,
        plan.get("raw_seed_limit_per_cell") == RAW_SEED_LIMIT,
        plan.get("latency_match_limit_per_cell") == LATENCY_MATCH_LIMIT,
        plan.get("episodes_per_cell") == EPISODES_PER_CELL,
        plan.get("prefix_steps_per_episode") == PREFIX_STEPS,
        plan.get("infeasible_cells") == [],
        plan.get("hard_stop") is None,
    )
    if not all(frozen_plan):
        raise PhaseEntryGeometryContractError("frozen cell plan differs")
    audits = seed_audit.get("records")
    physical = episodes.get("records")
    selected = plan.get("episodes")
    if not all(isinstance(value, list) for value in (audits, physical, selected)):
        raise PhaseEntryGeometryContractError("artifact records are missing")
    if plan.get("planned_episode_count") != 36 or len(selected) != 36:
        raise PhaseEntryGeometryContractError("cohort is not complete")
    physical_by_id = _unique_by_id(physical, "physical prefix")
    _unique_by_id(audits, "seed audit")
    selected_ids = [str(value["planned_episode_id"]) for value in selected]
    if len(set(selected_ids)) != 36:
        raise PhaseEntryGeometryContractError("cohort identities are duplicated")
    expected_selected = []
    matched_ids = []
    partitioned_audit_count = 0
    for cell in frozen_cells():
        group = [row for row in audits if row.get("cell_id") == cell.cell_id]
        partitioned_audit_count += len(group)
        ordinals = [int(row["raw_seed_ordinal"]) for row in group]
        if ordinals != list(range(len(group))) or len(group) > RAW_SEED_LIMIT:
            raise PhaseEntryGeometryContractError("raw seed prefix is not frozen")
        matched_count = 0
        eligible_ids = []
        for row in group:
            rebuilt = raw_seed_record(salt, cell, int(row["raw_seed_ordinal"]))
            if any(row.get(name) != value for name, value in rebuilt.items()):
                raise PhaseEntryGeometryContractError("seed derivation differs")
            sampled = (
                row.get("sampled_observation_latency_steps"),
                row.get("sampled_action_latency_steps"),
            )
            matched = sampled == (
                cell.observation_latency_steps,
                cell.action_latency_steps,
            )
            if row.get("latency_matched") is not matched:
                raise PhaseEntryGeometryContractError("natural latency audit differs")
            identity = str(row["planned_episode_id"])
            if matched:
                matched_count += 1
                matched_ids.append(identity)
                record = physical_by_id.get(identity)
                if record is None or row.get("physical_prefix_executed") is not True:
                    raise PhaseEntryGeometryContractError(
                        "latency-matched physical prefix is missing"
                    )
                _validate_episode(record, row)
                if record.get("eligible") is True:
                    eligible_ids.append(identity)
            elif (
                row.get("physical_prefix_executed") is not False
                or identity in physical_by_id
            ):
                raise PhaseEntryGeometryContractError(
                    "latency mismatch executed a physical prefix"
                )
        if matched_count > LATENCY_MATCH_LIMIT:
            raise PhaseEntryGeometryContractError("latency-matched budget exceeded")
        if (
            len(eligible_ids) != EPISODES_PER_CELL
            or not group
            or str(group[-1]["planned_episode_id"]) != eligible_ids[-1]
        ):
            raise PhaseEntryGeometryContractError("cell eligible count differs")
        expected_selected.extend(eligible_ids)
    if partitioned_audit_count != len(audits):
        raise PhaseEntryGeometryContractError("seed audit contains an unknown cell")
    if set(physical_by_id) != set(matched_ids):
        raise PhaseEntryGeometryContractError("physical prefix ledger differs")
    if selected_ids != expected_selected:
        raise PhaseEntryGeometryContractError("selected cohort is not prefix-only")
    for episode_ordinal, (planned, identity) in enumerate(
        zip(selected, selected_ids, strict=True)
    ):
        audit = next(row for row in audits if row["planned_episode_id"] == identity)
        physical_row = physical_by_id[identity]
        for field in (
            "task_id",
            "cell_id",
            "observation_latency_steps",
            "action_latency_steps",
            "raw_seed_ordinal",
            "environment_seed",
            "policy_rng_seed",
            "sampled_observation_latency_steps",
            "sampled_action_latency_steps",
        ):
            if planned.get(field) != audit.get(field):
                raise PhaseEntryGeometryContractError("planned Episode ledger differs")
        if (
            planned.get("episode_ordinal") != physical_row.get("episode_ordinal")
            or planned.get("episode_ordinal") != episode_ordinal % EPISODES_PER_CELL
        ):
            raise PhaseEntryGeometryContractError("Episode ordinal differs")
    if set(selected_ids) - set(physical_by_id):
        raise PhaseEntryGeometryContractError("selected physical evidence is missing")
    environment_seeds = [int(row["environment_seed"]) for row in selected]
    policy_seeds = [int(row["policy_rng_seed"]) for row in selected]
    if (
        len(set(environment_seeds)) != 36
        or len(set(policy_seeds)) != 36
        or set(environment_seeds) & set(policy_seeds)
    ):
        raise PhaseEntryGeometryContractError("Episode seed domains are not independent")
    return [physical_by_id[identity] for identity in selected_ids]


def _validate_episode(record, audit):
    identity_fields = (
        "planned_episode_id",
        "task_id",
        "cell_id",
        "observation_latency_steps",
        "action_latency_steps",
        "environment_seed",
        "policy_rng_seed",
        "raw_seed_ordinal",
    )
    if any(record.get(name) != audit.get(name) for name in identity_fields):
        raise PhaseEntryGeometryContractError("physical prefix identity differs")
    if record.get("latency_matched") is not True:
        raise PhaseEntryGeometryContractError("physical prefix latency differs")
    trace = record.get("raw_prefix_trace")
    if not isinstance(trace, list):
        raise PhaseEntryGeometryContractError("raw prefix trace is missing")
    if (
        record.get("prefix_step_count") != len(trace)
        or len(trace) > PREFIX_STEPS
        or record.get("b2_action_generated") is not False
        or record.get("b2_action_executed") is not False
        or record.get("post_prefix_action_count") != 0
        or record.get("raw_prefix_trace_sha256") != canonical_sha256(trace)
    ):
        raise PhaseEntryGeometryContractError("prefix-only execution differs")
    steps = [row.get("step") for row in trace]
    if steps != list(range(len(trace))):
        raise PhaseEntryGeometryContractError("raw prefix step sequence differs")
    recomputed_trace = {
        "action_bounds_valid": all(
            row.get("action_bounds_valid") is True for row in trace
        ),
        "stale_action_applied_count": sum(
            bool(row.get("outside_validity_window"))
            and row.get("applied_action") != row.get("hold_action")
            for row in trace
        ),
        "safety_intervention_count": sum(
            bool(row.get("safety_intervened")) for row in trace
        ),
        "terminal_observed": any(bool(row.get("terminal")) for row in trace),
    }
    if recomputed_trace != {
        "action_bounds_valid": record.get("prefix_action_bounds_valid"),
        "stale_action_applied_count": record.get(
            "prefix_stale_action_applied_count"
        ),
        "safety_intervention_count": record.get(
            "prefix_safety_intervention_count"
        ),
        "terminal_observed": record.get("prefix_terminal_observed"),
    }:
        raise PhaseEntryGeometryContractError("raw prefix guards differ")
    if (
        record.get("hard_safety_failure") is not False
        or record.get("prefix_action_bounds_valid") is not True
        or record.get("prefix_stale_action_applied_count") != 0
        or record.get("prefix_safety_intervention_count") != 0
        or record.get("prefix_severe_collision_count") != 0
        or record.get("prefix_invalid_force_count") != 0
        or record.get("prefix_p40_conservation_maximum_absolute_difference") != 0.0
    ):
        raise PhaseEntryGeometryContractError("physical prefix hard guard failed")
    if record.get("eligible") is not True:
        return
    guards = (
        len(trace) == PREFIX_STEPS,
        record.get("prefix_complete") is True,
        not record.get("prefix_terminal_observed"),
        record.get("runtime_observation_latency_steps")
        == record.get("observation_latency_steps"),
        record.get("runtime_action_latency_steps")
        == record.get("action_latency_steps"),
        record.get("latency_override_inactive") is True,
        _is_sha256(record.get("runtime_randomization_sha256")),
        record.get("candidate_count", 0) > 0,
        0 <= record.get("selected_index", -1) < record.get("candidate_count", 0),
        record.get("input_failure_reason") is None,
        record.get("prefix_failure_reason") is None,
    )
    if not all(guards):
        raise PhaseEntryGeometryContractError("eligible prefix guard differs")
    _validate_candidate_identity(record)
    candidate = Candidate(**record["selected_record"])
    geometry = measure_phase_entry_geometry(
        candidate,
        record["acquisition_base_pose"],
        record["b2_policy_base_pose"],
        record["geometry"]["joint_position"]["left"],
        record["geometry"]["joint_position"]["right"],
    )
    if canonical_bytes(geometry) != canonical_bytes(record.get("geometry")):
        raise PhaseEntryGeometryContractError("geometry recomputation differs")
    if not geometry["target_formula_crosscheck"]["passed"]:
        raise PhaseEntryGeometryContractError("target formula crosscheck failed")
    if float(record.get("fk_crosscheck_max_error_m", math.inf)) > FLOAT_TOLERANCE_M:
        raise PhaseEntryGeometryContractError("FK crosscheck failed")
    if (
        record.get("policy_input_count") != PREFIX_STEPS
        or not isinstance(record.get("policy_input_sha256"), list)
        or len(record["policy_input_sha256"]) != PREFIX_STEPS
        or not all(_is_sha256(value) for value in record["policy_input_sha256"])
        or record.get("policy_input_sequence_sha256")
        != canonical_sha256(record["policy_input_sha256"])
        or not _is_sha256(record.get("candidate_final_policy_input_sha256"))
        or not _is_sha256(record.get("b2_entry_policy_input_sha256"))
    ):
        raise PhaseEntryGeometryContractError("policy input identity is incomplete")


def _validate_candidate_identity(record):
    try:
        payload = bytes.fromhex(str(record["candidate_bytes_hex"]))
        document = json.loads(payload.decode("ascii"))
    except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseEntryGeometryContractError(
            "candidate bytes are invalid"
        ) from error
    candidate = Candidate(**record["selected_record"])
    selected_index = int(record["selected_index"])
    if (
        hashlib.sha256(payload).hexdigest() != record["candidate_set_sha256"]
        or not isinstance(document, Mapping)
        or document.get("schema_version") != target_selection.CANDIDATE_SCHEMA
        or document.get("candidate_count") != record["candidate_count"]
        or not isinstance(document.get("candidates"), list)
        or len(document["candidates"]) != record["candidate_count"]
        or document["candidates"][selected_index] != list(candidate.canonical_record())
    ):
        raise PhaseEntryGeometryContractError(
            "selected candidate identity differs"
        )


def _group_summary(rows):
    values = list(rows)
    arms = [arm for row in values for arm in row["geometry"]["arms"].values()]
    hard = sum(bool(row["geometry"]["hard_bilateral_impossible"]) for row in values)
    both = sum(
        bool(row["geometry"]["both_arms_strict_outer_impossible"])
        for row in values
    )
    nominal = sum(
        bool(row["geometry"]["nominal_bilateral_support_deficit"])
        for row in values
    )
    return {
        "episode_count": len(values),
        "arm_count": len(arms),
        "hard_bilateral_impossible_count": hard,
        "hard_bilateral_impossible_rate": hard / len(values) if values else None,
        "both_arms_strict_outer_impossible_count": both,
        "nominal_bilateral_support_deficit_count": nominal,
        "nominal_bilateral_support_deficit_rate": (
            nominal / len(values) if values else None
        ),
        "descriptive": {
            name: _numeric_summary(
                [float(row["geometry"][name]) for row in values]
            )
            for name in (
                "candidate_base_horizontal_range_m",
                "candidate_heading_error_rad",
            )
        },
        "arm_metrics": {
            name: _numeric_summary([float(arm[name]) for arm in arms])
            for name in (
                "shoulder_to_preposition_m",
                "strict_outer_margin_m",
                "tool_to_preposition_d0_m",
                "nominal_b2_support_margin_m",
            )
        },
    }


def _numeric_summary(values):
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None}
    array = np.asarray(values, np.float64)
    if not np.isfinite(array).all():
        raise PhaseEntryGeometryContractError("aggregate metric is nonfinite")
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _unique_by_id(rows, label):
    identities = [str(row.get("planned_episode_id")) for row in rows]
    if len(identities) != len(set(identities)):
        raise PhaseEntryGeometryContractError(f"{label} identities are duplicated")
    return dict(zip(identities, rows, strict=True))


def _vector(values, size, name):
    result = np.asarray(values, np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise PhaseEntryGeometryContractError(f"{name} must be finite length {size}")
    return result


def _transform_point(transform, point):
    value = _vector(point, 3, "local point")
    return transform[:3, :3] @ value + transform[:3, 3]


def _wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def _all_finite(value):
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value):
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )
