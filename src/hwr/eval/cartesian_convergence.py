"""Frozen contracts and statistics for R0001-P51-E1 convergence evidence."""

from __future__ import annotations
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import numpy as np
from hwr.eval.seed_contract import (
    SEED_SCHEMA,
    derive_domain_seed,
    planned_episode_id,
    require_seed_reveal,
)
from hwr.eval.target_selection import Candidate
PROPOSAL_ID = "R0001-P51-E1"
PLAN_ID = "R0001-P51-E1-formal"
SALT_COMMITMENT = "a12c867f79013a830a89ea1e76b7c50c6df260bff2bf9ee502f49a56cc501d2b"
TASK_IDS = ("tidy_living_room_3d/v1", "clear_dining_table_3d/v1",
            "store_kitchen_items_3d/v1")
LATENCY_VALUES = (1, 2)
ROLES = ("frame_legacy", "frame_fixed")
PAIR_COUNT_PER_CELL = 3
LATENCY_MATCH_LIMIT = 64
RAW_SEED_LIMIT = 768
B2_STEPS = 100
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_265_102
CONTINUOUS_MDE = 0.10
BINARY_WIN_TARGET = 0.10
DISTANCE_FLOOR_M = 0.05
BANK_SCHEMA = "hwr.p51-cartesian-convergence-bank/v1"
SEED_AUDIT_SCHEMA = "hwr.p51-cartesian-convergence-seed-audit/v1"
TERMINAL_SCHEMA = "hwr.p51-cartesian-convergence-terminals/v1"
ROLE_ORDER_DOMAIN = b"hwr.p51-cartesian-convergence-role-order/v1"
ALLOWED_TREATMENT_INDICES = frozenset((2, 3, 8, 9))


class CartesianConvergenceContractError(ValueError):
    """Raised when a frozen P51-E1 artifact violates its contract."""
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
            (task, observation, action) for task in TASK_IDS
            for observation in LATENCY_VALUES for action in LATENCY_VALUES
        )
    )
def raw_seed_record(salt: str, cell: Cell, candidate_ordinal: int) -> dict[str, object]:
    if not 0 <= candidate_ordinal < RAW_SEED_LIMIT:
        raise CartesianConvergenceContractError("raw candidate ordinal is outside plan")
    episode_id = planned_episode_id(
        PLAN_ID, cell.task_id, cell.cell_id, candidate_ordinal
    )
    return {
        "candidate_ordinal": candidate_ordinal,
        "planned_episode_id": episode_id,
        "environment_seed": derive_domain_seed(salt, "environment", episode_id),
        "policy_rng_seed": derive_domain_seed(salt, "policy", episode_id),
    }
def pair_identity(planned_id: str) -> str:
    payload = b"hwr.p51-cartesian-convergence-pair/v1" + planned_id.encode("ascii")
    return hashlib.sha256(payload).hexdigest()
def role_order(salt: str, pair_id: str) -> tuple[int, tuple[str, str]]:
    if len(pair_id) != 64:
        raise CartesianConvergenceContractError("pair identity must be SHA-256")
    digest = hashlib.sha256(
        salt.encode("ascii") + ROLE_ORDER_DOMAIN + pair_id.encode("ascii")
    ).digest()
    seed = int.from_bytes(digest, "big") & ((1 << 63) - 1)
    return seed, ROLES if seed % 2 == 0 else tuple(reversed(ROLES))
def preposition_targets(candidate: Candidate) -> dict[str, tuple[float, float, float]]:
    point = np.asarray(candidate.center, np.float64)
    horizontal = float(np.linalg.norm(point[:2]))
    if point.shape != (3,) or not np.isfinite(point).all() or horizontal < 0.35:
        raise CartesianConvergenceContractError("selected candidate cannot define B2 targets")
    forward = point[:2] / horizontal
    normal = np.asarray((-forward[0], -forward[1], 0.0))
    lateral = np.asarray((-forward[1], forward[0], 0.0))
    spacing = float(np.clip(candidate.width + 0.12, 0.18, 0.34))
    vertical = np.asarray((0.0, 0.0, 1.0))
    return {
        "left": tuple(point + 0.18 * normal + 0.5 * spacing * lateral + 0.05 * vertical),
        "right": tuple(point + 0.18 * normal - 0.5 * spacing * lateral + 0.05 * vertical),
    }
def legacy_transform(
    acquisition_error: Sequence[float],
    velocity_max: float,
    **unused_yaws: float,
) -> np.ndarray:
    del unused_yaws
    error = np.asarray(acquisition_error, np.float64)
    if error.shape != (3,) or not np.isfinite(error).all():
        raise CartesianConvergenceContractError("legacy error must be finite xyz")
    if not math.isfinite(velocity_max) or velocity_max < 0.0:
        raise CartesianConvergenceContractError("legacy velocity cap is invalid")
    velocity = 2.0 * error
    norm = float(np.linalg.norm(velocity))
    return velocity if norm <= velocity_max or norm == 0.0 else velocity * velocity_max / norm
def first_treatment_guard(
    legacy: Sequence[float], fixed: Sequence[float]
) -> dict[str, object]:
    legacy_array = np.asarray(legacy, dtype="<f8")
    fixed_array = np.asarray(fixed, dtype="<f8")
    if legacy_array.shape != (16,) or fixed_array.shape != (16,):
        raise CartesianConvergenceContractError("treatment actions must be 16-D")
    finite = bool(np.isfinite(legacy_array).all() and np.isfinite(fixed_array).all())
    differing = {
        index
        for index, (left, right) in enumerate(zip(legacy_array, fixed_array, strict=True))
        if left.tobytes() != right.tobytes()
    }
    arm_norms = {
        role: {
            "left_linear": float(np.linalg.norm(value[2:5])),
            "right_linear": float(np.linalg.norm(value[8:11])),
        }
        for role, value in (("frame_legacy", legacy_array), ("frame_fixed", fixed_array))
    }
    return {
        "finite": finite,
        "different_bytes": legacy_array.tobytes() != fixed_array.tobytes(),
        "differing_indices": sorted(differing),
        "only_arm_linear_xy_differs": bool(differing)
        and differing <= ALLOWED_TREATMENT_INDICES,
        "arm_action_noncollapsed": all(
            values["left_linear"] > 1.0e-12 and values["right_linear"] > 1.0e-12
            for values in arm_norms.values()
        ),
        "arm_linear_norms": arm_norms,
        "legacy_float64_sha256": hashlib.sha256(legacy_array.tobytes()).hexdigest(),
        "fixed_float64_sha256": hashlib.sha256(fixed_array.tobytes()).hexdigest(),
    }
def carry_forward_distances(values: Sequence[float | None]) -> tuple[float, ...]:
    if not values:
        raise CartesianConvergenceContractError("distance trace is empty")
    result: list[float] = []
    last: float | None = None
    for value in values:
        if value is not None:
            candidate = float(value)
            if not math.isfinite(candidate) or candidate < 0.0:
                raise CartesianConvergenceContractError("distance must be finite and nonnegative")
            last = candidate
        if last is None:
            raise CartesianConvergenceContractError("first distance cannot be missing")
        result.append(last)
    if len(result) > B2_STEPS + 1:
        raise CartesianConvergenceContractError("distance trace exceeds B2 horizon")
    result.extend((last,) * (B2_STEPS + 1 - len(result)))
    return tuple(result)
def arm_outcome(distances: Sequence[float | None]) -> dict[str, object]:
    carried = carry_forward_distances(distances)
    normalized_auc = float(np.mean(carried[1:]) / max(carried[0], DISTANCE_FLOOR_M))
    return {
        "distances_m": list(carried),
        "d_0_m": carried[0],
        "d_100_m": carried[-1],
        "minimum_m": min(carried),
        "normalized_auc": normalized_auc,
        "symmetric_terminal_carry_forward": len(distances) < B2_STEPS + 1,
    }
def carry_forward_records(
    values: Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    if not values or len(values) > B2_STEPS + 1:
        raise CartesianConvergenceContractError("distance records violate B2 horizon")
    result = [dict(value) for value in values]
    if any(
        not math.isfinite(float(number)) or float(number) < 0.0
        for value in result
        for number in value.values()
    ):
        raise CartesianConvergenceContractError("distance record is invalid")
    result.extend(dict(result[-1]) for _ in range(B2_STEPS + 1 - len(result)))
    return result
def signed_derivatives(
    distances: Sequence[Mapping[str, float]],
    applied: Sequence[Sequence[float]],
) -> list[dict[str, object]]:
    records = []
    for index, action in enumerate(applied):
        vector = np.asarray(action, np.float64)
        if vector.shape != (16,):
            raise CartesianConvergenceContractError("applied action must be 16-D")
        if np.linalg.norm(np.r_[vector[2:5], vector[8:11]]) <= 1.0e-12:
            continue
        records.append(
            {
                "b2_step": index + 1,
                "signed_derivative_m": float(
                    distances[index + 1]["mean_m"] - distances[index]["mean_m"]
                ),
            }
        )
        if len(records) == 10:
            break
    return records
def action_summary(
    proposed: Sequence[Sequence[float]],
    applied: Sequence[Sequence[float]],
) -> dict[str, object]:
    result = {}
    for name, values in (("proposed", proposed), ("applied", applied)):
        array = np.asarray(values, np.float64).reshape((-1, 16))
        rms = np.sqrt(np.mean(np.square(array), axis=0)) if len(array) else np.zeros(16)
        active = (
            np.flatnonzero(np.any(np.abs(array) > 1.0e-12, axis=0)).tolist()
            if len(array)
            else []
        )
        result[name] = {
            "rms_per_dimension": rms.tolist(),
            "rms": float(np.sqrt(np.mean(np.square(array)))) if len(array) else 0.0,
            "active_dimensions": active,
        }
    return result
def attach_pair_invariants(
    record: dict[str, object], *, complete: bool
) -> None:
    arms = list(record["arms"].values())
    verified = bool(record["pair_identity_valid"])
    same_initial_target = verified and all(
        arm["tool_distances"][0] == arms[0]["tool_distances"][0]
        for arm in arms[1:]
    )
    def same(name: str) -> bool:
        return verified and all(
            arm["invariant_identities"][name]
            == arms[0]["invariant_identities"][name]
            for arm in arms[1:]
        )
    record.update(
        {
            "safety_identity_equal": same("safety"),
            "cap_identity_equal": same("cap"),
            "gripper_identity_equal": same("gripper"),
            "phase_identity_equal": same("phase"),
            "target_identity_equal": same_initial_target and same("target"),
            "fk_identity_equal": same("fk"),
            "backend_identity_equal": same("backend"),
        }
    )
def treatment_guard_passes(guard: Mapping[str, object]) -> bool:
    return all(
        bool(guard.get(name))
        for name in (
            "finite",
            "different_bytes",
            "only_arm_linear_xy_differs",
            "arm_action_noncollapsed",
        )
    )
def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                      sort_keys=True).encode("ascii")
def identity(value: object) -> dict[str, object]:
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
def array_bundle_identity(
    values: Sequence[np.ndarray],
) -> dict[str, object]:
    chunks = []
    for value in values:
        array = np.ascontiguousarray(value)
        chunks.extend((
            array.dtype.str.encode("ascii"),
            str(tuple(array.shape)).encode("ascii"),
            array.tobytes(),
        ))
    return identity(b"".join(chunks))
def observation_identity_record(value) -> dict[str, object]:
    return {
        "timestamp_ns": value.timestamp_ns,
        "sequence_id": value.sequence_id,
        "task_id": value.task_id,
        "instruction": asdict(value.instruction),
        "proprioception": list(value.proprioception.vector()),
        "safety_state": value.safety_state.value,
        "quality": dict(value.quality),
        "cameras": [
            {
                **camera.to_dict(),
                "payload": None if camera.payload is None else identity(camera.payload),
            }
            for camera in value.cameras
        ],
        "camera_calibrations": [asdict(item) for item in value.camera_calibrations],
    }
def runtime_counter_record(backend, observation, graph) -> dict[str, object]:
    result = backend.result()
    return {
        "episode_seed": backend._episode_seed,
        "timestamp_ns": backend._timestamp_ns(),
        "observation_timestamp_ns": observation.timestamp_ns,
        "observation_sequence_id": observation.sequence_id,
        "backend_sequence": backend._sequence,
        "backend_steps": backend._steps,
        "episode_result": None if result is None else asdict(result),
        "placement_stable_steps": backend._placement.stable_steps,
        "left_contact_steps": backend._left_contact_steps,
        "right_contact_steps": backend._right_contact_steps,
        "simultaneous_contact_steps": backend._simultaneous_contact_steps,
        "concurrent_steps": backend._concurrent_steps,
        "maximum_concurrent_steps": backend._maximum_concurrent_steps,
        "severe_collision_count": backend._severe_collision_count,
        "maximum_forbidden_force": backend._maximum_forbidden_force,
        "maximum_forbidden_pair": backend._maximum_forbidden_pair,
        "step_left_contact": backend._step_left_contact,
        "step_right_contact": backend._step_right_contact,
        "initial_target_distance": backend._initial_target_distance,
        "maximum_controlled_target_progress": (
            backend._maximum_controlled_target_progress
        ),
        "maximum_controlled_articulation_progress": (
            backend._maximum_controlled_articulation_progress
        ),
        "previous_potential": backend._previous_potential,
        "randomization": backend._randomization,
        "rng_state": repr(backend._rng.getstate()),
        "camera_rendering_enabled": backend._camera_rendering_enabled,
        "cached_camera_payloads": [
            None if camera.payload is None else identity(camera.payload)
            for camera in backend._cached_cameras
        ],
        "safety_limits": asdict(backend.safety.limits),
        "contact_ledger": identity(backend.contact_ledger.report()),
        "entity_contact_graph": identity(graph.report()),
    }
def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))
def analyze_terminals(terminals: Mapping[str, object]) -> dict[str, object]:
    records = _terminal_records(terminals)
    identity = _identity_guards(records)
    hard_safety = _hard_safety_guards(records)
    unresolved = sum(not bool(record.get("resolved", False)) for record in records)
    if identity["invalid_count"]:
        return _analysis(identity, hard_safety, unresolved, None, None, "invalid")
    if unresolved:
        return _analysis(identity, hard_safety, unresolved, None, None, "inconclusive")
    if not hard_safety["passed"]:
        return _analysis(identity, hard_safety, 0, None, None, "rejected")
    continuous = _continuous_analysis(records)
    binary = _binary_analysis(records)
    accepted = continuous["passed"] and binary["passed"]
    decision = (
        "accepted as paired physical Cartesian convergence evidence"
        if accepted
        else "rejected"
    )
    return _analysis(identity, hard_safety, 0, continuous, binary, decision)
def _terminal_records(terminals: Mapping[str, object]) -> list[Mapping[str, object]]:
    if terminals.get("schema_version") != TERMINAL_SCHEMA:
        raise CartesianConvergenceContractError("terminal schema differs")
    records = terminals.get("records")
    if not isinstance(records, list):
        raise CartesianConvergenceContractError("terminal records are missing")
    return records
def _identity_guards(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    expected_cells = {cell.cell_id: cell for cell in frozen_cells()}
    pair_ids = [str(record.get("pair_id", "")) for record in records]
    counts = {cell_id: 0 for cell_id in expected_cells}
    hard_stop = any(bool(record.get("hard_safety_stop")) for record in records)
    invalid = (
        not hard_stop
        and len(records) != len(expected_cells) * PAIR_COUNT_PER_CELL
    )
    invalid |= len(set(pair_ids)) != len(pair_ids)
    for record in records:
        cell_id = str(record.get("cell_id", ""))
        if cell_id not in expected_cells:
            invalid = True
            continue
        arms = set(record.get("arms", {}))
        invalid |= not arms <= set(ROLES)
        needs_identity = bool(record.get("resolved")) or bool(
            record.get("hard_safety_stop")
        )
        invalid |= needs_identity and not hard_stop and arms != set(ROLES)
        cell = expected_cells[cell_id]
        counts[cell_id] += 1
        invalid |= needs_identity and any(
            (
                record.get("task_id") != cell.task_id,
                record.get("observation_latency_steps")
                != cell.observation_latency_steps,
                record.get("action_latency_steps") != cell.action_latency_steps,
                not bool(record.get("pair_identity_valid", False)),
                not bool(record.get("continuation_identity_equal", False)),
                not bool(record.get("first_treatment_guard", {}).get(
                    "finite", False
                )),
                not bool(record.get("first_treatment_guard", {}).get(
                    "only_arm_linear_xy_differs", False
                )),
                not bool(record.get("first_treatment_guard", {}).get(
                    "different_bytes", False
                )),
                not bool(record.get("first_treatment_guard", {}).get(
                    "arm_action_noncollapsed", False
                )),
            )
        )
    if not hard_stop:
        invalid |= any(value != PAIR_COUNT_PER_CELL for value in counts.values())
    return {
        "expected_pair_count": len(expected_cells) * PAIR_COUNT_PER_CELL,
        "published_pair_count": len(records),
        "duplicate_pair_count": len(pair_ids) - len(set(pair_ids)),
        "cell_pair_counts": counts,
        "terminated_by_hard_safety": hard_stop,
        "invalid_count": int(bool(invalid)),
        "passed": not invalid,
    }
def _hard_safety_guards(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    totals = {
        "severe_collision_count": 0,
        "invalid_force_count": 0,
        "stale_action_applied_count": 0,
        "p40_conservation_violation_count": 0,
        "action_bounds_violation_count": 0,
        "nonfinite_metric_count": 0,
        "reported_hard_failure_count": 0,
    }
    invariants = (
        "safety_identity_equal",
        "cap_identity_equal",
        "gripper_identity_equal",
        "phase_identity_equal",
        "target_identity_equal",
        "fk_identity_equal",
        "backend_identity_equal",
    )
    invariant_failures = {name: 0 for name in invariants}
    for record in records:
        for arm in record.get("arms", {}).values():
            for name in tuple(totals)[:3]:
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
    passed = not any(totals.values()) and not any(invariant_failures.values())
    return {
        **totals,
        "invariant_failures": invariant_failures,
        "passed": passed,
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
    distributions = np.empty(BOOTSTRAP_REPLICATES, np.float64)
    values_by_cell = {
        cell.cell_id: np.asarray(
            [row["delta"] for row in rows if row["cell_id"] == cell.cell_id],
            np.float64,
        )
        for cell in cells
    }
    for replicate in range(BOOTSTRAP_REPLICATES):
        distributions[replicate] = np.mean(
            [
                np.mean(values[rng.integers(0, PAIR_COUNT_PER_CELL, PAIR_COUNT_PER_CELL)])
                for values in values_by_cell.values()
            ]
        )
    if not np.isfinite(distributions).all():
        raise CartesianConvergenceContractError("bootstrap produced nonfinite replicate")
    lower = float(np.quantile(distributions, 0.05, method="linear"))
    by_task = {
        task: _mean(
            [
                value
                for cell, value in (
                    (cell, by_cell[cell.cell_id]) for cell in cells
                )
                if cell.task_id == task
            ]
        )
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
            row["win"]
            for row in rows
            if row["observation_latency_steps"] == observation
            and row["action_latency_steps"] == action
        )
        for observation in LATENCY_VALUES
        for action in LATENCY_VALUES
    }
    by_observation = {
        str(value): sum(
            row["win"] for row in rows
            if row["observation_latency_steps"] == value
        )
        for value in LATENCY_VALUES
    }
    by_action = {
        str(value): sum(
            row["win"] for row in rows
            if row["action_latency_steps"] == value
        )
        for value in LATENCY_VALUES
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
        "by_observation_latency": by_observation,
        "by_action_latency": by_action,
        "by_latency_combination": by_latency,
        "checks": checks,
        "passed": all(checks.values()),
    }
def _delta_row(record: Mapping[str, object]) -> dict[str, object]:
    fixed = record["arms"]["frame_fixed"]
    legacy = record["arms"]["frame_legacy"]
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
def _latency_means(
    rows: Sequence[Mapping[str, object]], field: str
) -> dict[str, float]:
    result = {}
    for latency in LATENCY_VALUES:
        cell_means = []
        for cell in frozen_cells():
            if getattr(cell, field) == latency:
                cell_means.append(
                    _mean(
                        [
                            row["delta"]
                            for row in rows
                            if row["cell_id"] == cell.cell_id
                        ]
                    )
                )
        result[str(latency)] = _mean(cell_means)
    return result
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
    if bank.get("salt_commitment") != SALT_COMMITMENT:
        raise CartesianConvergenceContractError("bank salt commitment differs")
    reveal = bank.get("salt_reveal")
    if not isinstance(reveal, str):
        raise CartesianConvergenceContractError("bank salt reveal is missing")
    require_seed_reveal(SALT_COMMITMENT, reveal)
    if bank.get("seed_schema") != SEED_SCHEMA:
        raise CartesianConvergenceContractError("bank seed schema differs")
    cells = bank.get("cells")
    pairs = bank.get("pairs")
    audit = bank.get("seed_audit")
    if not isinstance(cells, list) or not isinstance(pairs, list) or not isinstance(audit, list):
        raise CartesianConvergenceContractError("bank plan sections are missing")
    if cells != [cell.to_dict() for cell in frozen_cells()]:
        raise CartesianConvergenceContractError("bank cells differ")
    if len(pairs) != len(frozen_cells()) * PAIR_COUNT_PER_CELL:
        raise CartesianConvergenceContractError("bank does not contain 36 pairs")
    _validate_seed_records(audit, pairs, reveal)
    counts = {cell.cell_id: 0 for cell in frozen_cells()}
    for pair in pairs:
        cell_id = pair.get("cell_id")
        if cell_id not in counts or not pair.get("eligible"):
            raise CartesianConvergenceContractError("bank pair eligibility differs")
        counts[cell_id] += 1
        if pair.get("pair_id") != pair_identity(
            str(pair.get("planned_episode_id", ""))
        ):
            raise CartesianConvergenceContractError("bank pair identity differs")
        expected_order_seed, expected_order = role_order(
            reveal, str(pair["pair_id"])
        )
        if pair.get("role_order_domain_seed") != expected_order_seed:
            raise CartesianConvergenceContractError("bank role order seed differs")
        if pair.get("role_order") not in (list(ROLES), list(reversed(ROLES))):
            raise CartesianConvergenceContractError("bank role order differs")
        if pair["role_order"] != list(expected_order):
            raise CartesianConvergenceContractError("bank role order derivation differs")
        if not pair.get("continuation_identity") or not pair.get("selected_record"):
            raise CartesianConvergenceContractError("bank continuation is incomplete")
        if (
            int(pair.get("candidate_count", 0)) <= 0
            or abs(float(pair.get("relative_yaw_at_b2", 0.0))) < math.pi / 6.0
        ):
            raise CartesianConvergenceContractError("bank eligibility fields differ")
        candidate_bytes = bytes.fromhex(str(pair.get("candidate_bytes_hex", "")))
        if hashlib.sha256(candidate_bytes).hexdigest() != pair.get("candidate_set_sha256"):
            raise CartesianConvergenceContractError("bank candidate bytes differ")
        candidate_document = json.loads(candidate_bytes)
        index = int(pair["selected_index"])
        if not 0 <= index < len(candidate_document["candidates"]):
            raise CartesianConvergenceContractError("bank selected index differs")
        candidate = Candidate(**pair["selected_record"])
        if list(candidate.canonical_record()) != candidate_document["candidates"][index]:
            raise CartesianConvergenceContractError("bank selected record differs")
        guard = pair.get("first_treatment_guard", {})
        actions = pair.get("first_treatment_actions", {})
        if set(actions) != set(ROLES) or first_treatment_guard(
            actions["frame_legacy"], actions["frame_fixed"]
        ) != guard:
            raise CartesianConvergenceContractError(
                "bank treatment action evidence differs"
            )
        if not all(
            bool(guard.get(name))
            for name in (
                "finite",
                "different_bytes",
                "only_arm_linear_xy_differs",
                "arm_action_noncollapsed",
            )
        ):
            raise CartesianConvergenceContractError("bank treatment eligibility differs")
    if any(value != PAIR_COUNT_PER_CELL for value in counts.values()):
        raise CartesianConvergenceContractError("bank cell eligibility is incomplete")
def _validate_seed_records(audit, pairs, reveal: str) -> None:
    expected_cells = {cell.cell_id: cell for cell in frozen_cells()}
    environment, policy, identities = [], [], []
    matched = {cell_id: 0 for cell_id in expected_cells}
    prior = {cell_id: -1 for cell_id in expected_cells}
    audit_by_id = {}
    for record in audit:
        cell_id = record.get("cell_id")
        ordinal = int(record.get("candidate_ordinal", -1))
        if cell_id not in expected_cells or not 0 <= ordinal < RAW_SEED_LIMIT:
            raise CartesianConvergenceContractError("seed audit cell or ordinal differs")
        if ordinal != prior[cell_id] + 1:
            raise CartesianConvergenceContractError("seed audit ordinals are not contiguous")
        prior[cell_id] = ordinal
        cell = expected_cells[cell_id]
        planned = raw_seed_record(reveal, cell, ordinal)
        if any(record.get(name) != planned[name] for name in planned):
            raise CartesianConvergenceContractError("seed derivation differs")
        environment.append(int(record["environment_seed"]))
        policy.append(int(record["policy_rng_seed"]))
        identities.append(str(record["planned_episode_id"]))
        expected_match = (
            record.get("sampled_observation_latency_steps")
            == cell.observation_latency_steps
            and record.get("sampled_action_latency_steps")
            == cell.action_latency_steps
        )
        if bool(record.get("latency_matched")) != expected_match:
            raise CartesianConvergenceContractError("natural latency audit differs")
        if bool(record.get("acquisition_executed")) != expected_match:
            raise CartesianConvergenceContractError("acquisition execution audit differs")
        matched[cell_id] += int(expected_match)
        if matched[cell_id] > LATENCY_MATCH_LIMIT:
            raise CartesianConvergenceContractError("latency-matched budget exceeded")
        audit_by_id[record["planned_episode_id"]] = record
    if len(set(environment)) != len(environment) or len(set(policy)) != len(policy):
        raise CartesianConvergenceContractError("seed collision in checked plan")
    if set(environment) & set(policy) or len(set(identities)) != len(identities):
        raise CartesianConvergenceContractError("seed domains or identities collided")
    for pair in pairs:
        audited = audit_by_id.get(pair.get("planned_episode_id"))
        if audited is None:
            raise CartesianConvergenceContractError("pair is absent from seed audit")
        if not audited.get("eligible") or any(
            pair.get(name) != audited.get(name)
            for name in (
                "cell_id",
                "task_id",
                "environment_seed",
                "policy_rng_seed",
                "candidate_set_sha256",
                "selected_index",
                "continuation_identity",
                "prefix_trace_sha256",
                "first_treatment_actions",
                "first_treatment_guard",
            )
        ):
            raise CartesianConvergenceContractError("pair differs from eligible audit")
    for cell_id in expected_cells:
        eligible = [
            record["planned_episode_id"] for record in audit
            if record["cell_id"] == cell_id and record.get("eligible")
        ]
        cell_pairs = sorted(
            (pair for pair in pairs if pair["cell_id"] == cell_id),
            key=lambda value: value["replicate_ordinal"],
        )
        replicates = [int(pair["replicate_ordinal"]) for pair in cell_pairs]
        selected = [pair["planned_episode_id"] for pair in cell_pairs]
        if replicates != list(range(PAIR_COUNT_PER_CELL)):
            raise CartesianConvergenceContractError("bank replicate order differs")
        if selected != eligible[:PAIR_COUNT_PER_CELL]:
            raise CartesianConvergenceContractError(
                "bank did not select first eligible pairs"
            )
def _mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise CartesianConvergenceContractError("analysis group is empty or nonfinite")
    return float(np.mean(np.asarray(values, np.float64)))
