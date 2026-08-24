"""Offline measurement contract for R0001-P57 pre-contact reachability."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from hwr.eval.cartesian_convergence import (
    B2_STEPS,
    LATENCY_VALUES,
    PAIR_COUNT_PER_CELL,
    TASK_IDS,
    canonical_sha256,
    frozen_cells,
    identity,
    preposition_targets,
)
from hwr.eval.cartesian_convergence_validation import (
    analyze_terminals,
    validate_bank,
)
from hwr.eval.target_selection import ACTION_MAXIMUM, ACTION_MINIMUM, Candidate

PROPOSAL_ID = "R0001-P57"
PAIR_SCHEMA = "hwr.p57-precontact-reachability-pairs/v1"
READY_DISTANCE_M = 0.10
ACTION_SCALE_M_PER_S = 0.30
CONTROL_HZ = 20.0
B3_STEPS = 50
B3_VELOCITY_M_PER_S = 0.03
B4_STEPS = 20
B4_VELOCITY_M_PER_S = 0.02
CONTACT_TRANSITION_NOMINAL_M = (
    B3_STEPS * B3_VELOCITY_M_PER_S / CONTROL_HZ
    + B4_STEPS * B4_VELOCITY_M_PER_S / CONTROL_HZ
)
ARM_FIELDS = {"left": (2, 5), "right": (8, 11)}


class PrecontactReachabilityContractError(ValueError):
    """Raised when frozen P51 evidence cannot support the P57 measurement."""


def analyze_precontact_reachability(
    bank: Mapping[str, object],
    terminals: Mapping[str, object],
) -> dict[str, object]:
    """Validate frozen P51 evidence and recompute every P57 metric."""
    try:
        validate_bank(bank)
        p51_analysis = analyze_terminals(terminals, bank)
        if p51_analysis.get("decision") == "invalid":
            raise PrecontactReachabilityContractError(
                f"P51 terminal validation failed: {p51_analysis.get('validation_error')}"
            )
        if not bool((p51_analysis.get("hard_guard") or {}).get("passed")):
            raise PrecontactReachabilityContractError("P51 hard guards did not pass")
        rows = _pair_rows(bank, terminals)
        summary = aggregate_pair_rows(rows)
        checks = _contract_checks(bank, terminals, rows, summary)
        if not all(checks.values()):
            raise PrecontactReachabilityContractError(
                f"P57 contract checks failed: "
                f"{sorted(name for name, passed in checks.items() if not passed)}"
            )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "decision": "invalid",
            "diagnostic": None,
            "validation_error": str(error),
            "checks": {"passed": False},
            "summary": None,
            "pairs": [],
        }
    diagnostic = diagnostic_decision(summary)
    return {
        "decision": "accepted as bilateral pre-contact reachability measurement evidence",
        "diagnostic": diagnostic,
        "validation_error": None,
        "checks": {**checks, "passed": True},
        "summary": summary,
        "pairs": rows,
    }


def analyze_pair(
    pair: Mapping[str, object],
    terminal: Mapping[str, object],
) -> dict[str, object]:
    """Recompute one frame-fixed pair from raw distance and applied-action arrays."""
    _require_pair_identity(pair, terminal)
    fixed = terminal["arms"]["frame_fixed"]
    tool = fixed.get("tool_distances")
    applied = fixed.get("applied_actions")
    if not isinstance(tool, list) or len(tool) != B2_STEPS + 1:
        raise PrecontactReachabilityContractError("pair must have 101 tool distances")
    if not isinstance(applied, list) or len(applied) != B2_STEPS:
        raise PrecontactReachabilityContractError("pair must have 100 applied actions")
    distances = _arm_distances(tool)
    actions = _applied_actions(applied)
    mean_distances = [
        (distances["left"][index] + distances["right"][index]) / 2.0
        for index in range(B2_STEPS + 1)
    ]
    reported = {
        "d_0_m": float(fixed["d_0_m"]),
        "d_100_m": float(fixed["d_100_m"]),
        "minimum_m": float(fixed["minimum_m"]),
        "normalized_auc": float(fixed["normalized_auc"]),
    }
    recomputed = _distance_metrics(mean_distances)
    if reported != recomputed or fixed.get("distances_m") != mean_distances:
        raise PrecontactReachabilityContractError(
            "reported P51 distance metrics differ from raw tool distances"
        )
    targets = contact_targets(pair)
    arm_rows = {
        arm: _arm_row(
            distances[arm],
            actions,
            arm,
            targets["preposition"][arm],
            targets["contact"][arm],
        )
        for arm in ARM_FIELDS
    }
    bilateral = [
        distances["left"][index] <= READY_DISTANCE_M
        and distances["right"][index] <= READY_DISTANCE_M
        for index in range(B2_STEPS + 1)
    ]
    return {
        "pair_id": pair["pair_id"],
        "planned_episode_id": pair["planned_episode_id"],
        "task_id": pair["task_id"],
        "cell_id": pair["cell_id"],
        "observation_latency_steps": pair["observation_latency_steps"],
        "action_latency_steps": pair["action_latency_steps"],
        "replicate_ordinal": pair["replicate_ordinal"],
        "environment_seed": pair["environment_seed"],
        "policy_rng_seed": pair["policy_rng_seed"],
        "selected_index": pair["selected_index"],
        "candidate_set_sha256": pair["candidate_set_sha256"],
        "bank_pair_sha256": canonical_sha256(pair),
        "frame_fixed_applied_action_sha256": fixed["applied_action_sha256"],
        "frame_fixed_applied_actions": actions.tolist(),
        "p51_mean_distance_recomputation": recomputed,
        "arms": arm_rows,
        "both_arms_improved": all(
            row["d_100_m"] < row["d_0_m"] for row in arm_rows.values()
        ),
        "both_initial_command_margins_negative": all(
            row["initial_command_margin_m"] < 0.0 for row in arm_rows.values()
        ),
        "max_arm_endpoint_distance_m": max(
            row["d_100_m"] for row in arm_rows.values()
        ),
        "bilateral_ready_by_step": bilateral,
        "ever_bilateral_ready": any(bilateral),
        "endpoint_bilateral_ready": bilateral[-1],
        "first_bilateral_ready_step": (
            bilateral.index(True) if any(bilateral) else None
        ),
        "target_reconstruction": targets,
    }


def contact_targets(pair: Mapping[str, object]) -> dict[str, object]:
    """Rebuild B2 preposition and B3/B4 nominal contact targets."""
    candidate = Candidate(**pair["selected_record"])
    preposition = preposition_targets(
        candidate,
        pair["acquisition_base_pose"],
        pair["b2_policy_base_pose"],
    )
    expected_preposition = {
        arm: tuple(float(value) for value in pair["preposition_targets"][arm])
        for arm in ARM_FIELDS
    }
    if preposition != expected_preposition:
        raise PrecontactReachabilityContractError(
            "preposition target reconstruction differs from bank"
        )
    point = np.asarray(candidate.center, np.float64)
    base = _base_in_acquisition(
        pair["acquisition_base_pose"], pair["b2_policy_base_pose"]
    )
    forward = point[:2] - base[:2]
    horizontal = float(np.linalg.norm(forward))
    if point.shape != (3,) or not np.isfinite(point).all() or horizontal < 0.35:
        raise PrecontactReachabilityContractError(
            "selected candidate cannot define contact targets"
        )
    forward /= horizontal
    normal = np.asarray((-forward[0], -forward[1], 0.0))
    lateral = np.asarray((-forward[1], forward[0], 0.0))
    spacing = float(np.clip(candidate.width + 0.04, 0.10, 0.24))
    contact = {
        "left": tuple(point + 0.015 * normal + 0.5 * spacing * lateral),
        "right": tuple(point + 0.015 * normal - 0.5 * spacing * lateral),
    }
    return {
        "preposition": {arm: list(preposition[arm]) for arm in ARM_FIELDS},
        "preposition_identity": identity(
            {arm: list(preposition[arm]) for arm in ARM_FIELDS}
        ),
        "contact": {arm: list(contact[arm]) for arm in ARM_FIELDS},
        "contact_identity": identity(
            {arm: list(contact[arm]) for arm in ARM_FIELDS}
        ),
        "b3_nominal_maximum_m": B3_STEPS * B3_VELOCITY_M_PER_S / CONTROL_HZ,
        "b4_nominal_maximum_m": B4_STEPS * B4_VELOCITY_M_PER_S / CONTROL_HZ,
        "total_nominal_maximum_m": CONTACT_TRANSITION_NOMINAL_M,
    }


def aggregate_pair_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Produce complete task, cell, and latency partitions."""
    rows = list(rows)
    return {
        "pair_count": len(rows),
        "arm_count": sum(len(row["arms"]) for row in rows),
        "overall": _group_summary(rows),
        "by_task": {
            task: _group_summary([row for row in rows if row["task_id"] == task])
            for task in TASK_IDS
        },
        "by_cell": {
            cell.cell_id: _group_summary(
                [row for row in rows if row["cell_id"] == cell.cell_id]
            )
            for cell in frozen_cells()
        },
        "by_observation_latency": {
            str(value): _group_summary(
                [
                    row
                    for row in rows
                    if row["observation_latency_steps"] == value
                ]
            )
            for value in LATENCY_VALUES
        },
        "by_action_latency": {
            str(value): _group_summary(
                [row for row in rows if row["action_latency_steps"] == value]
            )
            for value in LATENCY_VALUES
        },
        "by_latency_combination": {
            f"o{observation}-a{action}": _group_summary(
                [
                    row
                    for row in rows
                    if row["observation_latency_steps"] == observation
                    and row["action_latency_steps"] == action
                ]
            )
            for observation in LATENCY_VALUES
            for action in LATENCY_VALUES
        },
    }


def diagnostic_decision(summary: Mapping[str, object]) -> str:
    overall = summary["overall"]
    task_counts = [
        value["ever_bilateral_ready_count"]
        for value in summary["by_task"].values()
    ]
    if (
        overall["ever_bilateral_ready_count"] <= 6
        and overall["both_initial_command_margins_negative_count"] >= 30
        and all(value <= 4 for value in task_counts)
    ):
        return "precontact_support_deficit_supported"
    if (
        overall["ever_bilateral_ready_count"] >= 24
        and all(value >= 6 for value in task_counts)
    ):
        return "precontact_support_deficit_rejected"
    return "diagnostic_inconclusive"


def _pair_rows(bank, terminals) -> list[dict[str, object]]:
    pairs = bank.get("pairs")
    records = terminals.get("records")
    if not isinstance(pairs, list) or not isinstance(records, list):
        raise PrecontactReachabilityContractError("P51 pair records are missing")
    if len(pairs) != 36 or len(records) != 36:
        raise PrecontactReachabilityContractError("P57 requires exactly 36 pairs")
    return [
        analyze_pair(pair, terminal)
        for pair, terminal in zip(pairs, records, strict=True)
    ]


def _require_pair_identity(pair, terminal) -> None:
    fields = (
        "pair_id",
        "planned_episode_id",
        "task_id",
        "cell_id",
        "observation_latency_steps",
        "action_latency_steps",
        "replicate_ordinal",
        "environment_seed",
        "policy_rng_seed",
        "selected_index",
        "candidate_set_sha256",
    )
    if any(pair.get(name) != terminal.get(name) for name in fields):
        raise PrecontactReachabilityContractError("terminal pair differs from bank")
    if terminal.get("bank_pair_sha256") != canonical_sha256(pair):
        raise PrecontactReachabilityContractError("terminal bank-pair hash differs")
    arms = terminal.get("arms")
    if (
        not terminal.get("resolved")
        or not isinstance(arms, Mapping)
        or "frame_fixed" not in arms
    ):
        raise PrecontactReachabilityContractError("frame-fixed terminal is incomplete")


def _arm_distances(tool: Sequence[Mapping[str, object]]) -> dict[str, list[float]]:
    result = {"left": [], "right": []}
    for row in tool:
        left = float(row["left_m"])
        right = float(row["right_m"])
        mean = float(row["mean_m"])
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in (left, right))
            or mean != (left + right) / 2.0
        ):
            raise PrecontactReachabilityContractError("tool distance row is invalid")
        result["left"].append(left)
        result["right"].append(right)
    return result


def _applied_actions(values: Sequence[Sequence[float]]) -> np.ndarray:
    actions = np.asarray(values, np.float64)
    if actions.shape != (B2_STEPS, 16) or not np.isfinite(actions).all():
        raise PrecontactReachabilityContractError("applied action matrix is invalid")
    if np.any(actions < ACTION_MINIMUM) or np.any(actions > ACTION_MAXIMUM):
        raise PrecontactReachabilityContractError("applied action exceeds bounds")
    return actions


def _distance_metrics(distances: Sequence[float]) -> dict[str, float]:
    values = np.asarray(distances, np.float64)
    if (
        values.shape != (B2_STEPS + 1,)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
    ):
        raise PrecontactReachabilityContractError("distance trace is invalid")
    return {
        "d_0_m": float(values[0]),
        "d_100_m": float(values[-1]),
        "minimum_m": float(np.min(values)),
        "normalized_auc": float(np.mean(values[1:]) / max(values[0], 0.05)),
    }


def _arm_row(distances, actions, arm, preposition, contact) -> dict[str, object]:
    metrics = _distance_metrics(distances)
    start, stop = ARM_FIELDS[arm]
    norms = np.linalg.norm(actions[:, start:stop], axis=1)
    budget = float(
        np.sum(norms)
        * ACTION_SCALE_M_PER_S
        / CONTROL_HZ
    )
    transition = float(
        np.linalg.norm(
            np.asarray(contact, np.float64) - np.asarray(preposition, np.float64)
        )
    )
    return {
        "distances_m": list(distances),
        **metrics,
        "applied_command_step_count": len(norms),
        "applied_command_norms": norms.tolist(),
        "actual_applied_command_budget_m": budget,
        "initial_command_margin_m": budget - metrics["d_0_m"],
        "contact_target": list(contact),
        "preposition_target": list(preposition),
        "contact_to_preposition_distance_m": transition,
        "contact_transition_margin_m": CONTACT_TRANSITION_NOMINAL_M - transition,
    }


def _base_in_acquisition(acquisition_pose, robot_pose) -> np.ndarray:
    acquisition = np.asarray(acquisition_pose, np.float64)
    robot = np.asarray(robot_pose, np.float64)
    if (
        acquisition.shape != (3,)
        or robot.shape != (3,)
        or not np.isfinite(acquisition).all()
        or not np.isfinite(robot).all()
    ):
        raise PrecontactReachabilityContractError("base pose is invalid")
    delta = robot[:2] - acquisition[:2]
    cosine, sine = math.cos(acquisition[2]), math.sin(acquisition[2])
    xy = np.asarray(((cosine, sine), (-sine, cosine))) @ delta
    return np.asarray((xy[0], xy[1], robot[2] - acquisition[2]), np.float64)


def _group_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = list(rows)
    pair_count = len(rows)
    arm_rows = [arm for row in rows for arm in row["arms"].values()]
    ready = sum(bool(row["ever_bilateral_ready"]) for row in rows)
    endpoint = sum(bool(row["endpoint_bilateral_ready"]) for row in rows)
    both_improved = sum(bool(row["both_arms_improved"]) for row in rows)
    both_negative = sum(
        bool(row["both_initial_command_margins_negative"]) for row in rows
    )
    return {
        "pair_count": pair_count,
        "arm_count": len(arm_rows),
        "ever_bilateral_ready_count": ready,
        "ever_bilateral_ready_rate": ready / pair_count if pair_count else None,
        "endpoint_bilateral_ready_count": endpoint,
        "endpoint_bilateral_ready_rate": endpoint / pair_count if pair_count else None,
        "both_arms_improved_count": both_improved,
        "both_arms_improved_rate": (
            both_improved / pair_count if pair_count else None
        ),
        "both_initial_command_margins_negative_count": both_negative,
        "both_initial_command_margins_negative_rate": (
            both_negative / pair_count if pair_count else None
        ),
        "arm_metrics": {
            name: _numeric_summary([float(row[name]) for row in arm_rows])
            for name in (
                "d_0_m",
                "d_100_m",
                "minimum_m",
                "normalized_auc",
                "actual_applied_command_budget_m",
                "initial_command_margin_m",
                "contact_to_preposition_distance_m",
                "contact_transition_margin_m",
            )
        },
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None}
    array = np.asarray(values, np.float64)
    if not np.isfinite(array).all():
        raise PrecontactReachabilityContractError("aggregate metric is nonfinite")
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _contract_checks(bank, terminals, rows, summary) -> dict[str, bool]:
    expected_cells = {cell.cell_id for cell in frozen_cells()}
    pair_ids = [row["pair_id"] for row in rows]
    cells = {name: value["pair_count"] for name, value in summary["by_cell"].items()}
    return {
        "bank_pair_count_36": len(bank["pairs"]) == 36,
        "terminal_pair_count_36": terminals.get("terminal_pair_count") == 36,
        "pair_count_36": len(rows) == 36,
        "arm_count_72": summary["arm_count"] == 72,
        "unique_pair_samples": len(set(pair_ids)) == 36,
        "all_distance_traces_101": all(
            len(arm["distances_m"]) == 101
            for row in rows
            for arm in row["arms"].values()
        ),
        "all_bilateral_traces_101": all(
            len(row["bilateral_ready_by_step"]) == 101 for row in rows
        ),
        "all_applied_action_budgets_recomputed": all(
            arm["applied_command_step_count"] == 100
            and len(arm["applied_command_norms"]) == 100
            and arm["actual_applied_command_budget_m"] >= 0.0
            for row in rows
            for arm in row["arms"].values()
        ),
        "target_identity_reconstructed": all(
            row["target_reconstruction"]["preposition_identity"]
            == pair["preposition_target_identity"]
            for row, pair in zip(rows, bank["pairs"], strict=True)
        ),
        "tasks_complete": {
            task: value["pair_count"]
            for task, value in summary["by_task"].items()
        }
        == dict.fromkeys(TASK_IDS, 12),
        "cells_complete": set(cells) == expected_cells
        and all(value == PAIR_COUNT_PER_CELL for value in cells.values()),
        "observation_latency_complete": all(
            value["pair_count"] == 18
            for value in summary["by_observation_latency"].values()
        ),
        "action_latency_complete": all(
            value["pair_count"] == 18
            for value in summary["by_action_latency"].values()
        ),
        "latency_combinations_complete": all(
            value["pair_count"] == 9
            for value in summary["by_latency_combination"].values()
        ),
        "finite_metrics": all(
            math.isfinite(float(arm[name]))
            for row in rows
            for arm in row["arms"].values()
            for name in (
                "d_0_m",
                "d_100_m",
                "minimum_m",
                "normalized_auc",
                "actual_applied_command_budget_m",
                "initial_command_margin_m",
                "contact_to_preposition_distance_m",
                "contact_transition_margin_m",
            )
        ),
        "sample_unit_is_pair": summary["pair_count"] == len(pair_ids),
    }
