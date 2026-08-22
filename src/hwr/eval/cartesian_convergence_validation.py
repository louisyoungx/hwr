"""Fail-closed artifact validation for R0001-P51-E1."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from hwr.eval import cartesian_convergence as contract
from hwr.eval.cartesian_convergence import (
    AUDIT_BASE_FIELDS,
    B2_STEPS,
    BANK_SCHEMA,
    BINARY_WIN_TARGET,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONTINUOUS_MDE,
    ELIGIBILITY_REASONS,
    INPUT_FAILURE_REASONS,
    LATENCY_VALUES,
    LATENCY_MATCH_LIMIT,
    ORDINARY_TERMINAL_REASONS,
    PAIR_COUNT_PER_CELL,
    PLAN_ID,
    PREFIX_FIELDS,
    RAW_SEED_LIMIT,
    ROLES,
    SEED_SCHEMA,
    TASK_IDS,
    TERMINAL_SCHEMA,
    CartesianConvergenceContractError,
    action_summary,
    arm_outcome,
    attach_pair_invariants,
    canonical_sha256,
    first_treatment_guard,
    frozen_cells,
    guard_summary_matches,
    identity,
    pair_identity,
    preposition_targets,
    raw_seed_record,
    role_order,
    signed_derivatives,
    treatment_guard_passes,
    validate_runtime_step_trace,
)
from hwr.eval.seed_contract import require_seed_reveal
from hwr.eval.target_selection import ACTION_MAXIMUM, ACTION_MINIMUM, Candidate

def analyze_terminals(
    terminals: Mapping[str, object], bank: Mapping[str, object]
) -> dict[str, object]:
    try:
        records, identity_guard = _validated_terminal_records(terminals, bank)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "decision": "invalid",
            "validation_error": str(error),
            "identity_guard": {"passed": False, "invalid_count": 1},
            "unresolved_infrastructure": 0,
            "hard_guard": None,
            "continuous": None,
            "binary": None,
        }
    hard_guard = _hard_safety_guards(records)
    unresolved = sum(not bool(record.get("resolved", False)) for record in records)
    if unresolved:
        return _analysis(identity_guard, hard_guard, unresolved, None, None, "inconclusive")
    if not hard_guard["passed"]:
        return _analysis(identity_guard, hard_guard, 0, None, None, "rejected")
    continuous = _continuous_analysis(records)
    binary = _binary_analysis(records)
    decision = (
        "accepted as paired physical Cartesian convergence evidence"
        if continuous["passed"] and binary["passed"]
        else "rejected"
    )
    return _analysis(identity_guard, hard_guard, 0, continuous, binary, decision)


def _validated_terminal_records(terminals, bank):
    validate_bank(bank)
    if terminals.get("schema_version") != TERMINAL_SCHEMA:
        raise CartesianConvergenceContractError("terminal schema differs")
    records = terminals.get("records")
    if not isinstance(records, list):
        raise CartesianConvergenceContractError("terminal records are missing")
    planned = bank["pairs"]
    if terminals.get("planned_pair_count") != len(planned):
        raise CartesianConvergenceContractError("terminal planned count differs")
    if terminals.get("terminal_pair_count") != len(records):
        raise CartesianConvergenceContractError("terminal count differs")
    if terminals.get("bank_source_commit") != bank.get("source_commit"):
        raise CartesianConvergenceContractError("terminal bank source differs")
    if len(records) > len(planned):
        raise CartesianConvergenceContractError("terminal count exceeds bank")
    pair_ids = [str(record.get("pair_id", "")) for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise CartesianConvergenceContractError("terminal pair identity is duplicate")
    normalized = [
        _validate_terminal_pair(record, planned[index])
        for index, record in enumerate(records)
    ]
    hard_stop = any(bool(record.get("hard_safety_stop")) for record in records)
    unresolved_indices = [
        index for index, record in enumerate(records)
        if not record.get("resolved")
    ]
    unresolved = bool(unresolved_indices)
    if hard_stop and not records[-1].get("hard_safety_stop"):
        raise CartesianConvergenceContractError("hard safety stop is not terminal")
    if unresolved_indices and unresolved_indices != [len(records) - 1]:
        raise CartesianConvergenceContractError("unresolved record is not terminal")
    if not hard_stop and not unresolved and len(records) != len(planned):
        raise CartesianConvergenceContractError("terminal plan is incomplete")
    counts = {
        cell.cell_id: sum(record["cell_id"] == cell.cell_id for record in records)
        for cell in frozen_cells()
    }
    return normalized, {
        "expected_pair_count": len(planned),
        "published_pair_count": len(records),
        "duplicate_pair_count": 0,
        "cell_pair_counts": counts,
        "terminated_by_hard_safety": hard_stop,
        "invalid_count": 0,
        "passed": True,
    }


def _validate_terminal_pair(record, pair) -> dict[str, object]:
    fields = (
        "pair_id", "planned_episode_id", "task_id", "cell_id",
        "replicate_ordinal", "observation_latency_steps", "action_latency_steps",
        "environment_seed", "policy_rng_seed", "role_order",
        "candidate_set_sha256", "selected_index", "continuation_identity",
        "prefix_trace_sha256", "first_treatment_guard", "preposition_targets",
        "preposition_target_identity", "preposition_target_identities",
    )
    if any(record.get(name) != pair.get(name) for name in fields):
        raise CartesianConvergenceContractError("terminal pair differs from bank")
    if record.get("bank_pair_sha256") != canonical_sha256(pair):
        raise CartesianConvergenceContractError("terminal bank identity differs")
    actions = pair["first_treatment_actions"]
    expected_guard = first_treatment_guard(
        actions["frame_legacy"], actions["frame_fixed"]
    )
    if record["first_treatment_guard"] != expected_guard:
        raise CartesianConvergenceContractError("terminal treatment guard differs")
    arms = record.get("arms")
    if not isinstance(arms, Mapping) or not set(arms) <= set(ROLES):
        raise CartesianConvergenceContractError("terminal role set differs")
    if not record.get("resolved"):
        if (
            arms
            or record.get("hard_safety_stop")
            or record.get("pair_identity_valid") is not True
            or not isinstance(record.get("infrastructure_failure"), Mapping)
        ):
            raise CartesianConvergenceContractError("unresolved terminal payload differs")
        return dict(record)
    if not record.get("pair_identity_valid"):
        raise CartesianConvergenceContractError("terminal pair identity is invalid")
    expected_roles = list(pair["role_order"])[: len(arms)]
    if list(arms) != expected_roles:
        raise CartesianConvergenceContractError("terminal role order differs")
    normalized_arms = {
        role: _validated_arm(role, arms[role], pair) for role in expected_roles
    }
    hard_stop = any(not arm["hard_guard_passed"] for arm in normalized_arms.values())
    if bool(record.get("hard_safety_stop")) != hard_stop:
        raise CartesianConvergenceContractError("terminal hard-stop flag differs")
    if not hard_stop and set(normalized_arms) != set(ROLES):
        raise CartesianConvergenceContractError("resolved pair omitted a role")
    replay = record.get("continuation_replay_identities")
    if not isinstance(replay, Mapping) or set(replay) != set(normalized_arms):
        raise CartesianConvergenceContractError("continuation replay roles differ")
    continuation_equal = all(
        replay[role] == pair["continuation_identity"] for role in normalized_arms
    )
    if not continuation_equal or record.get("continuation_identity_equal") is not True:
        raise CartesianConvergenceContractError("continuation equality flag differs")
    normalized = {**record, "arms": normalized_arms}
    expected_flags = dict(record)
    attach_pair_invariants(expected_flags, complete=set(normalized_arms) == set(ROLES))
    invariant_names = (
        "safety_identity_equal", "cap_identity_equal", "gripper_identity_equal",
        "phase_identity_equal", "target_identity_equal", "fk_identity_equal",
        "backend_identity_equal",
    )
    if any(record.get(name) != expected_flags[name] for name in invariant_names):
        raise CartesianConvergenceContractError("terminal invariant identity differs")
    if set(normalized_arms) == set(ROLES):
        delta = (
            normalized_arms["frame_legacy"]["normalized_auc"]
            - normalized_arms["frame_fixed"]["normalized_auc"]
        )
        if not _same_number(record.get("delta_i"), delta):
            raise CartesianConvergenceContractError("terminal pair delta differs")
        normalized["delta_i"] = delta
    return normalized


def _validated_arm(role, arm, pair) -> dict[str, object]:
    if arm.get("role") != role or arm.get("b2_control_step_limit") != B2_STEPS:
        raise CartesianConvergenceContractError("terminal arm identity differs")
    distances = arm.get("distances_m")
    proposed, applied = arm.get("proposed_actions"), arm.get("applied_actions")
    if not all(isinstance(value, list) for value in (distances, proposed, applied)):
        raise CartesianConvergenceContractError("terminal raw arrays are missing")
    runtime = validate_runtime_step_trace(arm)
    executed = runtime["executed_b2_steps"]
    if (
        len(distances) != B2_STEPS + 1
        or arm.get("executed_b2_steps") != executed
        or len(proposed) != executed
        or len(applied) != executed
        or not 0 < executed <= B2_STEPS
    ):
        raise CartesianConvergenceContractError("terminal raw array length differs")
    if (
        proposed != runtime["proposed_actions"]
        or applied != runtime["applied_actions"]
    ):
        raise CartesianConvergenceContractError("terminal actions differ from raw trace")
    arrays = [np.asarray(value, np.float64) for value in (*proposed, *applied)]
    if any(value.shape != (16,) or not np.isfinite(value).all() for value in arrays):
        raise CartesianConvergenceContractError("terminal action array differs")
    computed_bounds = all(
        np.all(value >= ACTION_MINIMUM) and np.all(value <= ACTION_MAXIMUM)
        for value in arrays
    )
    if bool(arm.get("action_bounds_valid")) != computed_bounds:
        raise CartesianConvergenceContractError("terminal action bounds flag differs")
    guard = runtime["guard_summary"]
    guard_fields = (
        "safety_intervention_count",
        "stale_action_applied_count",
        "action_bounds_valid",
        "severe_collision_count",
        "invalid_force_count",
        "p40_conservation_maximum_absolute_difference",
        "nonfinite_runtime_value_count",
    )
    if not guard_summary_matches(arm, guard, guard_fields):
        raise CartesianConvergenceContractError("terminal guard summary differs")
    hard_reason = arm.get("hard_failure_reason")
    observed_hard = any((
        guard["safety_intervention_count"],
        guard["stale_action_applied_count"],
        not guard["action_bounds_valid"],
        guard["severe_collision_count"],
        guard["invalid_force_count"],
        guard["p40_conservation_maximum_absolute_difference"] != 0.0,
        guard["nonfinite_runtime_value_count"],
        hard_reason is not None,
    ))
    if bool(arm.get("hard_guard_passed")) == bool(observed_hard):
        raise CartesianConvergenceContractError("terminal hard guard differs")
    if canonical_sha256(proposed) != arm.get("proposed_action_sha256"):
        raise CartesianConvergenceContractError("proposed action hash differs")
    if canonical_sha256(applied) != arm.get("applied_action_sha256"):
        raise CartesianConvergenceContractError("applied action hash differs")
    if action_summary(proposed, applied) != arm.get("action_summary"):
        raise CartesianConvergenceContractError("action summary differs")
    expected_first = np.asarray(pair["first_treatment_actions"][role], dtype="<f8")
    if not proposed or np.asarray(proposed[0], dtype="<f8").tobytes() != expected_first.tobytes():
        raise CartesianConvergenceContractError("first treatment action differs")
    outcome = arm_outcome(distances)
    for name in ("d_0_m", "d_100_m", "minimum_m", "normalized_auc"):
        if not _same_number(arm.get(name), outcome[name]):
            raise CartesianConvergenceContractError(f"terminal {name} differs")
    carry = B2_STEPS - executed
    ordinary_reason = arm.get("ordinary_runtime_terminal")
    if ordinary_reason != runtime["ordinary_terminal_reason"]:
        raise CartesianConvergenceContractError("terminal reason differs from raw trace")
    if hard_reason != runtime["hard_failure_reason"]:
        raise CartesianConvergenceContractError("hard reason differs from raw trace")
    if executed < B2_STEPS:
        if runtime["terminal_step"] != executed:
            raise CartesianConvergenceContractError("truncation lacks terminal evidence")
        if (
            arm["hard_guard_passed"]
            and runtime["ordinary_terminal_reason"] not in ORDINARY_TERMINAL_REASONS
        ):
            raise CartesianConvergenceContractError("ordinary terminal reason differs")
        if not arm["hard_guard_passed"] and hard_reason is None:
            raise CartesianConvergenceContractError("hard terminal reason differs")
    if arm.get("carried_forward_step_count") != carry:
        raise CartesianConvergenceContractError("terminal carry-forward count differs")
    if bool(arm.get("symmetric_terminal_carry_forward")) != bool(carry):
        raise CartesianConvergenceContractError("terminal carry-forward flag differs")
    if carry and distances[executed:] != [distances[executed]] * (carry + 1):
        raise CartesianConvergenceContractError("terminal carry-forward values differ")
    tool = arm.get("tool_distances")
    if not isinstance(tool, list) or len(tool) != B2_STEPS + 1:
        raise CartesianConvergenceContractError("tool distance trace differs")
    for index, row in enumerate(tool):
        values = tuple(float(row[name]) for name in ("left_m", "right_m", "mean_m"))
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise CartesianConvergenceContractError("tool distance value differs")
        if not _same_number(values[2], (values[0] + values[1]) / 2.0):
            raise CartesianConvergenceContractError("tool distance mean differs")
        if not _same_number(values[2], distances[index]):
            raise CartesianConvergenceContractError("distance trace differs")
    target = arm.get("preposition_targets")
    target_identity = arm.get("preposition_target_identity")
    if target != pair.get("preposition_targets") or target_identity != pair.get(
        "preposition_target_identity"
    ):
        raise CartesianConvergenceContractError("terminal target differs from bank")
    if identity(target) != target_identity:
        raise CartesianConvergenceContractError("terminal target identity differs")
    target_identities = arm.get("preposition_target_identities")
    if target_identities != pair.get("preposition_target_identities") or any(
        identity(target[name]) != target_identities.get(name)
        for name in ("left", "right")
    ):
        raise CartesianConvergenceContractError("terminal arm target identity differs")
    if carry and tool[executed:] != [tool[executed]] * (carry + 1):
        raise CartesianConvergenceContractError("tool carry-forward values differ")
    if runtime["tool_distances"] != tool[1 : executed + 1]:
        raise CartesianConvergenceContractError("tool distances differ from raw trace")
    if signed_derivatives(tool, applied) != arm.get(
        "first_10_applied_nonzero_arm_signed_derivatives"
    ):
        raise CartesianConvergenceContractError("signed derivatives differ")
    return {**arm, **outcome}
def _same_number(left, right) -> bool:
    try:
        return float(left) == float(right) and math.isfinite(float(right))
    except (TypeError, ValueError):
        return False


def _hard_safety_guards(
    records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    totals = {
        "severe_collision_count": 0,
        "invalid_force_count": 0,
        "safety_intervention_count": 0,
        "stale_action_applied_count": 0,
        "p40_conservation_violation_count": 0,
        "action_bounds_violation_count": 0,
        "nonfinite_runtime_value_count": 0,
        "nonfinite_metric_count": 0,
        "reported_hard_failure_count": 0,
    }
    invariants = (
        "safety_identity_equal", "cap_identity_equal", "gripper_identity_equal",
        "phase_identity_equal", "target_identity_equal", "fk_identity_equal",
        "backend_identity_equal",
    )
    invariant_failures = dict.fromkeys(invariants, 0)
    for record in records:
        for arm in record.get("arms", {}).values():
            for name in (
                "severe_collision_count",
                "invalid_force_count",
                "safety_intervention_count",
                "stale_action_applied_count",
                "nonfinite_runtime_value_count",
            ):
                totals[name] += int(arm.get(name, 0))
            totals["p40_conservation_violation_count"] += int(
                float(arm.get("p40_conservation_maximum_absolute_difference", math.inf))
                != 0.0
            )
            totals["action_bounds_violation_count"] += int(
                not bool(arm.get("action_bounds_valid", False))
            )
            distances = arm.get("distances_m", ())
            totals["nonfinite_metric_count"] += int(
                len(distances) != B2_STEPS + 1
                or any(not math.isfinite(float(value)) for value in distances)
                or not math.isfinite(float(arm.get("normalized_auc", math.nan)))
            )
            totals["reported_hard_failure_count"] += int(
                not bool(arm.get("hard_guard_passed", False))
            )
        for name in invariants:
            invariant_failures[name] += int(not bool(record.get(name, False)))
    return {
        **totals,
        "invariant_failures": invariant_failures,
        "passed": not any(totals.values()) and not any(invariant_failures.values()),
    }


def _continuous_analysis(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = [_delta_row(record) for record in records]
    cells = frozen_cells()
    by_cell = {
        cell.cell_id: _mean(
            [row["delta"] for row in rows if row["cell_id"] == cell.cell_id]
        )
        for cell in cells
    }
    point = _mean(list(by_cell.values()))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values_by_cell = {
        cell.cell_id: np.asarray(
            [row["delta"] for row in rows if row["cell_id"] == cell.cell_id],
            np.float64,
        )
        for cell in cells
    }
    distribution = np.asarray([
        np.mean([
            np.mean(values[rng.integers(0, PAIR_COUNT_PER_CELL, PAIR_COUNT_PER_CELL)])
            for values in values_by_cell.values()
        ])
        for _ in range(BOOTSTRAP_REPLICATES)
    ])
    if not np.isfinite(distribution).all():
        raise CartesianConvergenceContractError("bootstrap produced nonfinite replicate")
    lower = float(np.quantile(distribution, 0.05, method="linear"))
    by_task = {
        task: _mean([
            by_cell[cell.cell_id] for cell in cells if cell.task_id == task
        ])
        for task in TASK_IDS
    }
    by_observation = _latency_means(rows, "observation_latency_steps")
    by_action = _latency_means(rows, "action_latency_steps")
    checks = {
        "point_estimate_at_least_mde": point >= CONTINUOUS_MDE,
        "one_sided_95_lower_positive": lower > 0.0,
        "each_task_positive": all(value > 0.0 for value in by_task.values()),
        "each_observation_latency_positive": all(
            value > 0.0 for value in by_observation.values()
        ),
        "each_action_latency_positive": all(value > 0.0 for value in by_action.values()),
    }
    return {
        "point_estimate": point,
        "one_sided_95_lower": lower,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "quantile_method": "linear",
        "by_cell": by_cell,
        "by_task": by_task,
        "by_observation_latency": by_observation,
        "by_action_latency": by_action,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _binary_analysis(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = [_delta_row(record) for record in records]
    for row, record in zip(rows, records, strict=True):
        fixed = record["arms"]["frame_fixed"]
        legacy = record["arms"]["frame_legacy"]
        row["win"] = bool(
            row["delta"] >= BINARY_WIN_TARGET
            and float(fixed["d_100_m"]) <= float(legacy["d_100_m"])
            and record["pair_identity_valid"]
        )
    by_task = {
        task: sum(row["win"] for row in rows if row["task_id"] == task)
        for task in TASK_IDS
    }
    by_latency = {
        f"o{observation}-a{action}": sum(
            row["win"] for row in rows
            if row["observation_latency_steps"] == observation
            and row["action_latency_steps"] == action
        )
        for observation in LATENCY_VALUES for action in LATENCY_VALUES
    }
    wins = sum(row["win"] for row in rows)
    checks = {
        "total_wins_at_least_24": wins >= 24,
        "each_task_wins_at_least_6": all(value >= 6 for value in by_task.values()),
        "each_latency_combination_wins_at_least_4": all(
            value >= 4 for value in by_latency.values()
        ),
    }
    return {
        "frame_fixed_win_count": wins,
        "by_task": by_task,
        "by_observation_latency": _win_counts(rows, "observation_latency_steps"),
        "by_action_latency": _win_counts(rows, "action_latency_steps"),
        "by_latency_combination": by_latency,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _delta_row(record: Mapping[str, object]) -> dict[str, object]:
    fixed, legacy = record["arms"]["frame_fixed"], record["arms"]["frame_legacy"]
    delta = float(legacy["normalized_auc"]) - float(fixed["normalized_auc"])
    if not math.isfinite(delta):
        raise CartesianConvergenceContractError("pair delta is nonfinite")
    return {
        "cell_id": record["cell_id"],
        "task_id": record["task_id"],
        "observation_latency_steps": record["observation_latency_steps"],
        "action_latency_steps": record["action_latency_steps"],
        "delta": delta,
    }

def _latency_means(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    result = {}
    for latency in LATENCY_VALUES:
        cell_means = [
            _mean([row["delta"] for row in rows if row["cell_id"] == cell.cell_id])
            for cell in frozen_cells() if getattr(cell, field) == latency
        ]
        result[str(latency)] = _mean(cell_means)
    return result

def _win_counts(rows, field):
    return {
        str(value): sum(row["win"] for row in rows if row[field] == value)
        for value in LATENCY_VALUES
    }

def _analysis(identity, hard_safety, unresolved, continuous, binary, decision):
    return {
        "decision": decision,
        "identity_guard": identity,
        "unresolved_infrastructure": unresolved,
        "hard_guard": hard_safety,
        "continuous": continuous,
        "binary": binary,
    }


def validate_bank(bank: Mapping[str, object]) -> None:
    if bank.get("schema_version") != BANK_SCHEMA or bank.get("plan_id") != PLAN_ID:
        raise CartesianConvergenceContractError("bank identity differs")
    if bank.get("salt_commitment") != contract.SALT_COMMITMENT:
        raise CartesianConvergenceContractError("bank salt commitment differs")
    reveal = bank.get("salt_reveal")
    if not isinstance(reveal, str):
        raise CartesianConvergenceContractError("bank salt reveal is missing")
    require_seed_reveal(contract.SALT_COMMITMENT, reveal)
    if bank.get("seed_schema") != SEED_SCHEMA:
        raise CartesianConvergenceContractError("bank seed schema differs")
    cells, pairs, audit = bank.get("cells"), bank.get("pairs"), bank.get("seed_audit")
    if not all(isinstance(value, list) for value in (cells, pairs, audit)):
        raise CartesianConvergenceContractError("bank plan sections are missing")
    if cells != [cell.to_dict() for cell in frozen_cells()]:
        raise CartesianConvergenceContractError("bank cells differ")
    if len(pairs) != len(frozen_cells()) * PAIR_COUNT_PER_CELL:
        raise CartesianConvergenceContractError("bank does not contain 36 pairs")
    eligible = _validate_seed_records(audit, reveal)
    expected = [
        (cell, replicate, eligible[cell.cell_id][replicate])
        for cell in frozen_cells() for replicate in range(PAIR_COUNT_PER_CELL)
    ]
    for pair, (cell, replicate, source) in zip(pairs, expected, strict=True):
        if pair.get("cell_id") != cell.cell_id or pair.get("replicate_ordinal") != replicate:
            raise CartesianConvergenceContractError("bank pair order differs")
        if any(pair.get(name) != source.get(name) for name in AUDIT_BASE_FIELDS | PREFIX_FIELDS):
            raise CartesianConvergenceContractError("pair differs from eligible audit")
        _validate_pair_record(pair, reveal)


def _validate_pair_record(pair, reveal):
    if pair.get("pair_id") != pair_identity(str(pair.get("planned_episode_id", ""))):
        raise CartesianConvergenceContractError("bank pair identity differs")
    order_seed, order = role_order(reveal, str(pair["pair_id"]))
    if (
        pair.get("role_order_domain_seed") != order_seed
        or pair.get("role_order") != list(order)
    ):
        raise CartesianConvergenceContractError("bank role order differs")
    candidate_bytes = bytes.fromhex(str(pair.get("candidate_bytes_hex", "")))
    if hashlib.sha256(candidate_bytes).hexdigest() != pair.get("candidate_set_sha256"):
        raise CartesianConvergenceContractError("bank candidate bytes differ")
    document = json.loads(candidate_bytes)
    index = int(pair["selected_index"])
    if (
        not 0 <= index < len(document["candidates"])
        or int(pair.get("candidate_count", 0)) != len(document["candidates"])
    ):
        raise CartesianConvergenceContractError("bank selected index differs")
    candidate = Candidate(**pair["selected_record"])
    if list(candidate.canonical_record()) != document["candidates"][index]:
        raise CartesianConvergenceContractError("bank selected record differs")
    rebuilt_targets = preposition_targets(
        candidate, pair["acquisition_base_pose"], pair["b2_policy_base_pose"]
    )
    if {
        name: list(value) for name, value in rebuilt_targets.items()
    } != pair.get("preposition_targets"):
        raise CartesianConvergenceContractError("bank target reconstruction differs")
    guard, actions = pair["first_treatment_guard"], pair["first_treatment_actions"]
    targets = pair.get("preposition_targets")
    if not isinstance(targets, Mapping) or set(targets) != {"left", "right"}:
        raise CartesianConvergenceContractError("bank preposition targets differ")
    if identity(targets) != pair.get("preposition_target_identity"):
        raise CartesianConvergenceContractError("bank target identity differs")
    target_identities = pair.get("preposition_target_identities")
    if not isinstance(target_identities, Mapping) or any(
        identity(targets[name]) != target_identities.get(name)
        for name in ("left", "right")
    ):
        raise CartesianConvergenceContractError("bank arm target identity differs")
    if (
        set(actions) != set(ROLES)
        or first_treatment_guard(actions["frame_legacy"], actions["frame_fixed"]) != guard
        or not treatment_guard_passes(guard)
        or abs(float(pair.get("relative_yaw_at_b2", 0.0))) < math.pi / 6.0
        or not pair["primitive_target_crosscheck"].get("passed")
    ):
        raise CartesianConvergenceContractError("bank eligibility evidence differs")


def _validate_seed_records(
    audit, reveal: str
) -> dict[str, list[Mapping[str, object]]]:
    cells = frozen_cells()
    by_id = {cell.cell_id: cell for cell in cells}
    state = {
        cell.cell_id: {"next": 0, "matched": 0, "eligible": []}
        for cell in cells
    }
    environment, policy, identities, previous_cell = [], [], [], 0
    for record in audit:
        cell_id, ordinal = record.get("cell_id"), int(record.get("candidate_ordinal", -1))
        if cell_id not in by_id or not 0 <= ordinal < RAW_SEED_LIMIT:
            raise CartesianConvergenceContractError("seed audit cell or ordinal differs")
        cell = by_id[cell_id]
        if cell.ordinal < previous_cell:
            raise CartesianConvergenceContractError("seed audit cell order differs")
        previous_cell = cell.ordinal
        current = state[cell_id]
        if len(current["eligible"]) == PAIR_COUNT_PER_CELL:
            raise CartesianConvergenceContractError("audit continues after third eligible")
        if current["matched"] == LATENCY_MATCH_LIMIT:
            raise CartesianConvergenceContractError("audit continues after matched budget")
        if ordinal != current["next"]:
            raise CartesianConvergenceContractError("seed audit ordinals are not contiguous")
        current["next"] += 1
        planned = raw_seed_record(reveal, cell, ordinal)
        if any(record.get(name) != planned[name] for name in planned):
            raise CartesianConvergenceContractError("seed derivation differs")
        environment.append(int(record["environment_seed"]))
        policy.append(int(record["policy_rng_seed"]))
        identities.append(str(record["planned_episode_id"]))
        matched = (
            record.get("sampled_observation_latency_steps")
            == cell.observation_latency_steps
            and record.get("sampled_action_latency_steps")
            == cell.action_latency_steps
        )
        if bool(record.get("latency_matched")) != matched:
            raise CartesianConvergenceContractError("natural latency audit differs")
        if bool(record.get("acquisition_executed")) != matched:
            raise CartesianConvergenceContractError("acquisition execution audit differs")
        if not matched:
            if (
                record.get("eligibility_reason") != "natural_latency_mismatch"
                or set(record) & PREFIX_FIELDS
            ):
                raise CartesianConvergenceContractError("latency mismatch contains prefix")
            continue
        current["matched"] += 1
        if current["matched"] > LATENCY_MATCH_LIMIT:
            raise CartesianConvergenceContractError("latency-matched budget exceeded")
        _validate_prefix_record(record)
        if record["eligible"]:
            current["eligible"].append(record)
    if len(set(environment)) != len(environment) or len(set(policy)) != len(policy):
        raise CartesianConvergenceContractError("seed collision in checked plan")
    if set(environment) & set(policy) or len(set(identities)) != len(identities):
        raise CartesianConvergenceContractError("seed domains or identities collided")
    if any(len(state[cell.cell_id]["eligible"]) != PAIR_COUNT_PER_CELL for cell in cells):
        raise CartesianConvergenceContractError("audit does not complete every cell")
    return {cell.cell_id: list(state[cell.cell_id]["eligible"]) for cell in cells}


def _validate_prefix_record(record: Mapping[str, object]) -> None:
    missing = PREFIX_FIELDS - set(record)
    if missing:
        raise CartesianConvergenceContractError(
            f"matched prefix fields missing: {sorted(missing)}"
        )
    eligible, reason = bool(record["eligible"]), record.get("eligibility_reason")
    if reason not in ELIGIBILITY_REASONS:
        raise CartesianConvergenceContractError("prefix eligibility reason is unknown")
    candidate_bytes = bytes.fromhex(str(record["candidate_bytes_hex"]))
    if hashlib.sha256(candidate_bytes).hexdigest() != record["candidate_set_sha256"]:
        raise CartesianConvergenceContractError("prefix candidate bytes differ")
    if candidate_bytes:
        document = json.loads(candidate_bytes)
        candidates = document.get("candidates")
        if not isinstance(candidates, list):
            raise CartesianConvergenceContractError("prefix candidate document differs")
    else:
        candidates = []
    if int(record["candidate_count"]) != len(candidates):
        raise CartesianConvergenceContractError("prefix candidate count differs")
    selected = int(record["selected_index"])
    selected_valid = 0 <= selected < len(candidates)
    selected_record = record["selected_record"]
    if selected_valid:
        candidate = Candidate(**selected_record)
        if list(candidate.canonical_record()) != candidates[selected]:
            raise CartesianConvergenceContractError("prefix selected record differs")
    elif selected_record is not None:
        raise CartesianConvergenceContractError("invalid selection has a record")
    hashes = record["acquisition_input_hashes"]
    if (
        not isinstance(hashes, list)
        or canonical_sha256(hashes) != record["acquisition_input_sequence_sha256"]
    ):
        raise CartesianConvergenceContractError("acquisition input hashes differ")
    step_count = int(record["prefix_step_count"])
    if (
        not 0 <= step_count <= 1395
        or bool(record["prefix_complete"])
        != (step_count == 1395 and not record["prefix_terminal_observed"])
    ):
        raise CartesianConvergenceContractError("prefix completion evidence differs")
    if any(
        int(record[name]) < 0
        for name in (
            "prefix_safety_intervention_count",
            "prefix_stale_action_applied_count",
            "prefix_severe_collision_count",
            "prefix_invalid_force_count",
        )
    ):
        raise CartesianConvergenceContractError("prefix counter is negative")
    actions = record["first_treatment_actions"]
    if set(actions) not in (set(), set(ROLES)):
        raise CartesianConvergenceContractError("prefix treatment actions differ")
    if actions and first_treatment_guard(
        actions["frame_legacy"], actions["frame_fixed"]
    ) != record["first_treatment_guard"]:
        raise CartesianConvergenceContractError("prefix treatment guard differs")
    expected_reason = _prefix_failure_reason(record)
    if reason != expected_reason or eligible != (expected_reason == "eligible"):
        raise CartesianConvergenceContractError("prefix eligibility reason differs")
    if record.get("prefix_failure_reason") != (
        None if eligible else expected_reason
    ):
        raise CartesianConvergenceContractError("prefix failure reason differs")
    if eligible and (
        int(record["candidate_count"]) <= 0
        or not record["selected_record"]
        or not record["continuation_identity"]
        or not record["acquisition_input_hashes"]
        or not record["primitive_target_crosscheck"].get("passed")
    ):
        raise CartesianConvergenceContractError("eligible prefix is incomplete")


def _prefix_failure_reason(record: Mapping[str, object]) -> str:
    input_failure = record.get("input_failure_reason")
    if input_failure is not None:
        if input_failure not in INPUT_FAILURE_REASONS:
            raise CartesianConvergenceContractError("input failure reason is unknown")
        return str(input_failure)
    checks = (
        ("acquisition_main_event", "main_event_during_acquisition", True),
        ("prefix_action_bounds_valid", "action_bounds_violation", False),
        ("prefix_stale_action_applied_count", "stale_action_applied", "positive"),
        ("prefix_severe_collision_count", "severe_collision", "positive"),
        ("prefix_safety_intervention_count", "safety_intervention_during_prefix", "positive"),
        ("prefix_terminal_observed", "runtime_terminal_during_prefix", True),
        ("prefix_invalid_force_count", "invalid_force", "positive"),
        ("prefix_p40_conservation_maximum_absolute_difference",
         "p40_conservation_violation", "positive"),
    )
    for field, reason, failure_value in checks:
        value = record.get(field)
        failed = value == failure_value if failure_value != "positive" else float(value) > 0.0
        if failed:
            return reason
    if int(record["candidate_count"]) <= 0:
        return "candidate_set_empty"
    selected = int(record["selected_index"])
    if not 0 <= selected < int(record["candidate_count"]):
        return "selected_index_out_of_range"
    if not record.get("prefix_complete"):
        raise CartesianConvergenceContractError("incomplete prefix lacks failure evidence")
    if abs(float(record["relative_yaw_at_b2"])) < math.pi / 6.0:
        return "relative_yaw_below_pi_over_6"
    if not record["primitive_target_crosscheck"].get("passed"):
        return "primitive_target_crosscheck_failed"
    if not treatment_guard_passes(record["first_treatment_guard"]):
        return "first_treatment_action_ineligible"
    if record.get("prefix_failure_reason") is not None:
        raise CartesianConvergenceContractError("prefix failure reason lacks evidence")
    return "eligible"
def _mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise CartesianConvergenceContractError("analysis group is empty or nonfinite")
    return float(np.mean(np.asarray(values, np.float64)))
