"""Frozen contracts and statistics for R0001-P51-E1 convergence evidence."""

from __future__ import annotations
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import numpy as np
from hwr.eval import target_selection
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
AUDIT_BASE_FIELDS = frozenset((
    "ordinal", "cell_id", "task_id", "observation_latency_steps",
    "action_latency_steps", "candidate_ordinal", "planned_episode_id",
    "environment_seed", "policy_rng_seed",
    "sampled_observation_latency_steps", "sampled_action_latency_steps",
    "latency_matched", "acquisition_executed", "eligibility_reason",
))
PREFIX_FIELDS = frozenset((
    "eligible", "candidate_count", "candidate_set_sha256",
    "candidate_bytes_hex", "selected_index", "selected_record",
    "prefix_failure_reason", "input_failure_reason", "prefix_step_count",
    "prefix_complete", "prefix_terminal_observed",
    "prefix_safety_intervention_count", "prefix_action_bounds_valid",
    "prefix_stale_action_applied_count", "prefix_severe_collision_count",
    "prefix_invalid_force_count",
    "prefix_p40_conservation_maximum_absolute_difference",
    "acquisition_main_event",
    "acquisition_input_hashes", "acquisition_input_sequence_sha256",
    "prefix_trace_sha256", "b0_b1_proposed_action_sha256",
    "b0_b1_applied_action_sha256", "relative_yaw_at_b2",
    "acquisition_base_pose", "acquisition_world_origin",
    "continuation_identity", "first_treatment_actions",
    "first_treatment_guard", "b2_policy_base_pose", "preposition_targets",
    "preposition_target_identity", "preposition_target_identities",
    "primitive_target_crosscheck",
))
INPUT_FAILURE_REASONS = frozenset((
    "invalid_policy_visible_input", "supported_source_age_exceeded",
    "nonmonotonic_observation", "safety_state_not_ok",
))
ELIGIBILITY_REASONS = frozenset((
    "eligible", "natural_latency_mismatch", *INPUT_FAILURE_REASONS,
    "main_event_during_acquisition", "action_bounds_violation",
    "stale_action_applied", "severe_collision",
    "safety_intervention_during_prefix", "runtime_terminal_during_prefix",
    "invalid_force", "p40_conservation_violation", "candidate_set_empty",
    "selected_index_out_of_range", "relative_yaw_below_pi_over_6",
    "primitive_target_crosscheck_failed", "first_treatment_action_ineligible",
))
ORDINARY_TERMINAL_REASONS = frozenset((
    "formal_household_bimanual_success", "formal_household_timeout",
))


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
def preposition_targets(
    candidate: Candidate,
    acquisition_base_pose: Sequence[float],
    current_base_pose: Sequence[float],
) -> dict[str, tuple[float, float, float]]:
    point = np.asarray(candidate.center, np.float64)
    base = target_selection._acquisition_from_robot(
        acquisition_base_pose, current_base_pose
    )[:3, 3]
    forward = point[:2] - base[:2]
    horizontal = float(np.linalg.norm(forward))
    if point.shape != (3,) or not np.isfinite(point).all() or horizontal < 0.35:
        raise CartesianConvergenceContractError("selected candidate cannot define B2 targets")
    forward /= horizontal
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
