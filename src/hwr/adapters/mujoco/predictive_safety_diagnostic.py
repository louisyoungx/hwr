"""Evaluator-private witness for the production predictive-safety clone."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import mujoco
import numpy as np

from hwr.adapters.mujoco.formal_household_backend import (
    MujocoFormalHouseholdDualArmBackend,
)
from hwr.adapters.mujoco.phase_entry_geometry import PhaseEntryGeometryMujoco
from hwr.adapters.mujoco.target_selection_diagnostic import (
    TargetSelectionDiagnostic,
)
from hwr.core.embodied import DualArmAction, DualArmActionFrame, DualArmObservation


@dataclass(frozen=True)
class RawContactPoint:
    contact_index: int
    normal_force: float
    geom_ids: tuple[int, int]
    geom_names: tuple[str, str]
    body_ids: tuple[int, int]
    body_names: tuple[str, str]


def contact_point_witness(
    contacts: Iterable[RawContactPoint],
    *,
    robot_geom_ids: frozenset[int],
    allowed_environment_geom_ids: frozenset[int],
    threshold: float,
) -> dict[str, object]:
    """Recompute the production threshold crossing from raw contact points."""
    forbidden = []
    invalid_force_count = 0
    for point in contacts:
        first_robot = point.geom_ids[0] in robot_geom_ids
        second_robot = point.geom_ids[1] in robot_geom_ids
        if first_robot == second_robot:
            continue
        environment_index = 1 if first_robot else 0
        if point.geom_ids[environment_index] in allowed_environment_geom_ids:
            continue
        observed_force = abs(float(point.normal_force))
        finite = math.isfinite(observed_force)
        invalid_force_count += int(not finite)
        robot_index = 0 if first_robot else 1
        canonical_pair = tuple(sorted(point.geom_names))
        forbidden.append(
            {
                "contact_index": int(point.contact_index),
                "normal_force": observed_force if finite else None,
                "nonfinite_force": None if finite else repr(observed_force),
                "force_finite": finite,
                "geom_ids": list(point.geom_ids),
                "geom_names": list(point.geom_names),
                "body_ids": list(point.body_ids),
                "body_names": list(point.body_names),
                "canonical_geom_pair": list(canonical_pair),
                "robot_side": {
                    "geom_id": point.geom_ids[robot_index],
                    "geom_name": point.geom_names[robot_index],
                    "body_id": point.body_ids[robot_index],
                    "body_name": point.body_names[robot_index],
                },
                "environment_side": {
                    "geom_id": point.geom_ids[environment_index],
                    "geom_name": point.geom_names[environment_index],
                    "body_id": point.body_ids[environment_index],
                    "body_name": point.body_names[environment_index],
                },
            }
        )
    finite_contacts = [point for point in forbidden if point["force_finite"]]
    ordered = sorted(
        finite_contacts,
        key=lambda point: (
            -float(point["normal_force"]),
            tuple(point["canonical_geom_pair"]),
            int(point["contact_index"]),
        ),
    )
    display_maximum = ordered[0] if ordered else None
    maximum_force = (
        0.0 if display_maximum is None else float(display_maximum["normal_force"])
    )
    return {
        "threshold": float(threshold),
        "forbidden_contact_points": forbidden,
        "forbidden_contact_point_count": len(forbidden),
        "invalid_force_count": invalid_force_count,
        "valid": invalid_force_count == 0,
        "display_maximum": display_maximum,
        "maximum_forbidden_contact_point_force": maximum_force,
        "witness_violation": invalid_force_count == 0
        and maximum_force >= threshold,
        "aggregation": "maximum_absolute_normal_force_per_contact_point",
    }


def scan_mujoco_contact_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_geom_ids: frozenset[int],
    allowed_environment_geom_ids: frozenset[int],
    threshold: float,
) -> dict[str, object]:
    points = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        body_ids = tuple(int(model.geom_bodyid[geom]) for geom in geom_ids)
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        points.append(
            RawContactPoint(
                contact_index=contact_index,
                normal_force=float(force[0]),
                geom_ids=geom_ids,
                geom_names=tuple(
                    _object_name(model, mujoco.mjtObj.mjOBJ_GEOM, value, "geom")
                    for value in geom_ids
                ),
                body_ids=body_ids,
                body_names=tuple(
                    _object_name(model, mujoco.mjtObj.mjOBJ_BODY, value, "body")
                    for value in body_ids
                ),
            )
        )
    return contact_point_witness(
        points,
        robot_geom_ids=robot_geom_ids,
        allowed_environment_geom_ids=allowed_environment_geom_ids,
        threshold=threshold,
    )


class PredictiveSafetyDiagnosticBackend(MujocoFormalHouseholdDualArmBackend):
    """Capture sidecar-only action, contact, and authoritative-state evidence."""

    def __init__(self, task, binding, *, observer_enabled: bool) -> None:
        self.observer_enabled = bool(observer_enabled)
        self._source_queue: list[int] = []
        self._diagnostic_steps: list[dict[str, object]] = []
        self._pending_step: dict[str, object] | None = None
        self._predictor_frame: DualArmActionFrame | None = None
        self._predictor_boundary_ordinal = 0
        super().__init__(
            task,
            binding,
            camera_width=256,
            camera_height=192,
            evaluation_profile=True,
        )

    def _reset_evidence(self) -> None:
        super()._reset_evidence()
        self._source_queue = []
        self._diagnostic_steps = []
        self._pending_step = None
        self._predictor_frame = None
        self._predictor_boundary_ordinal = 0

    def apply(self, frame: DualArmActionFrame):
        self._pending_step = {
            "step": self._steps,
            "policy_proposal": _frame_record(frame),
            "pre_authoritative_state": self._authoritative_state(),
            "boundaries": [],
        }
        try:
            outcome = super().apply(frame)
        except BaseException:
            self._pending_step = None
            raise
        pending = self._pending_step
        if pending is None:
            raise RuntimeError("predictive-safety diagnostic step context was lost")
        pending.update(
            {
                "final_applied_action": _frame_record(
                    outcome.info["applied_action"]
                ),
                "events": [event.to_dict() for event in outcome.events],
                "authoritative_physics_advanced": bool(
                    outcome.info["physics_advanced"]
                ),
                "post_authoritative_state": self._authoritative_state(),
            }
        )
        for boundary in pending["boundaries"]:
            boundary["authoritative_physics_advanced"] = bool(
                outcome.info["physics_advanced"]
            )
        self._diagnostic_steps.append(pending)
        self._pending_step = None
        return outcome

    def _delayed_scaled_action(self, action: DualArmAction) -> DualArmAction:
        source_step = self._steps
        self._source_queue.append(source_step)
        delay = int(self._randomization["action_latency_steps"])
        queue_source_step = (
            None if len(self._source_queue) <= delay else self._source_queue.pop(0)
        )
        delayed = super()._delayed_scaled_action(action)
        if self._pending_step is not None:
            self._pending_step.update(
                {
                    "delayed_scaled_plant_action": list(delayed.vector()),
                    "queue_source_step": queue_source_step,
                    "queue_source": (
                        "initial_hold"
                        if queue_source_step is None
                        else "scaled_policy_proposal"
                    ),
                }
            )
        return delayed

    def _predictive_filter(self, frame, hold_grippers):
        self._predictor_frame = frame
        self._predictor_boundary_ordinal = 0
        try:
            return super()._predictive_filter(frame, hold_grippers)
        finally:
            self._predictor_frame = None

    def _predictive_safety_violation(self) -> bool:
        production_violation = super()._predictive_safety_violation()
        if self._pending_step is None:
            raise RuntimeError("predictive-safety observer has no active step")
        if self._predictor_frame is None:
            raise RuntimeError("predictive-safety observer has no predictor frame")
        self._predictor_boundary_ordinal += 1
        boundary = {
            "boundary_ordinal": self._predictor_boundary_ordinal,
            "cumulative_substep": (
                self._predictor_boundary_ordinal * self._substeps
            ),
            "predictor_input": _frame_record(self._predictor_frame),
            "production_violation": bool(production_violation),
            "predictive_trial_physics_advanced": True,
            "authoritative_physics_advanced": False,
            "witness": None,
        }
        if self.observer_enabled:
            boundary["witness"] = scan_mujoco_contact_points(
                self.model,
                self.data,
                robot_geom_ids=self.household_ids.robot_geoms,
                allowed_environment_geom_ids=(
                    self.household_ids.allowed_contact_geoms
                ),
                threshold=self.severe_force_threshold,
            )
        self._pending_step["boundaries"].append(boundary)
        return production_violation

    def diagnostic_record(self) -> dict[str, object]:
        return {
            "schema_version": "hwr.p66-predictive-safety-diagnostic/v1",
            "observer_enabled": self.observer_enabled,
            "steps": self._diagnostic_steps,
            "final_authoritative_state": self._authoritative_state(),
        }

    def _authoritative_state(self) -> dict[str, object]:
        audit = self.task_audit()
        return {
            "qpos": [float(value) for value in self.data.qpos],
            "qvel": [float(value) for value in self.data.qvel],
            "ctrl": [float(value) for value in self.data.ctrl],
            "time": float(self.data.time),
            "action_queue": [
                list(action.vector()) for action in self._action_queue
            ],
            "action_queue_source_steps": list(self._source_queue),
            "observation_queue": [
                _observation_record(value) for value in self._observation_queue
            ],
            "left_arm_targets": [float(value) for value in self._left_targets],
            "right_arm_targets": [float(value) for value in self._right_targets],
            "steps": self._steps,
            "sequence": self._sequence,
            "task_counters": {
                "stable_steps": audit["stable_steps"],
                "concurrent_steps": self._concurrent_steps,
                "maximum_concurrent_steps": audit["maximum_concurrent_steps"],
                "left_contact_steps": audit["left_contact_steps"],
                "right_contact_steps": audit["right_contact_steps"],
                "simultaneous_contact_steps": audit["simultaneous_contact_steps"],
                "severe_collision_count": audit["severe_collision_count"],
                "maximum_forbidden_force": audit["maximum_forbidden_force"],
                "maximum_forbidden_pair": audit["maximum_forbidden_pair"],
                "step_left_contact": self._step_left_contact,
                "step_right_contact": self._step_right_contact,
                "initial_target_distance": self._initial_target_distance,
                "maximum_controlled_target_progress": (
                    self._maximum_controlled_target_progress
                ),
                "maximum_controlled_articulation_progress": (
                    self._maximum_controlled_articulation_progress
                ),
                "previous_potential": self._previous_potential,
                "episode_result": (
                    None if self._result is None else asdict(self._result)
                ),
            },
        }


class _DiagnosticFactory(TargetSelectionDiagnostic):
    def __init__(self, task, binding, observer_enabled: bool) -> None:
        super().__init__(task, binding)
        self.observer_enabled = observer_enabled
        self.backend: PredictiveSafetyDiagnosticBackend | None = None

    def _backend(self) -> PredictiveSafetyDiagnosticBackend:
        self.backend = PredictiveSafetyDiagnosticBackend(
            self.task,
            self.binding,
            observer_enabled=self.observer_enabled,
        )
        return self.backend


class PredictiveSafetyAnchorReplay:
    """Replay the P60 prefix through the unmodified production decision path."""

    def __init__(self, task, binding) -> None:
        self.task = task
        self.binding = binding

    def run(
        self,
        *,
        environment_seed: int,
        policy_rng_seed: int,
        observer_enabled: bool,
    ) -> dict[str, object]:
        bridge = PhaseEntryGeometryMujoco(self.task, self.binding)
        factory = _DiagnosticFactory(
            self.task,
            self.binding,
            observer_enabled,
        )
        bridge._diagnostic = factory
        prefix = bridge.inspect_prefix(environment_seed, policy_rng_seed)
        if factory.backend is None:
            raise RuntimeError("predictive-safety replay did not create a backend")
        return {
            "observer_enabled": observer_enabled,
            "prefix": prefix,
            "diagnostic": factory.backend.diagnostic_record(),
        }


def _frame_record(frame: DualArmActionFrame) -> dict[str, object]:
    return {
        "created_at_ns": frame.created_at_ns,
        "valid_from_ns": frame.valid_from_ns,
        "valid_until_ns": frame.valid_until_ns,
        "source": frame.source,
        "confidence": frame.confidence,
        "policy_version": frame.policy_version,
        "schema_version": frame.schema_version,
        "action": list(frame.action.vector()),
    }


def _observation_record(observation: DualArmObservation) -> dict[str, object]:
    return {
        "timestamp_ns": observation.timestamp_ns,
        "sequence_id": observation.sequence_id,
        "task_id": observation.task_id,
        "instruction": asdict(observation.instruction),
        "proprioception": list(observation.proprioception.vector()),
        "cameras": [
            {
                **camera.to_dict(),
                "payload_sha256": (
                    None
                    if camera.payload is None
                    else hashlib.sha256(camera.payload).hexdigest()
                ),
            }
            for camera in observation.cameras
        ],
        "camera_calibrations": [
            asdict(calibration)
            for calibration in observation.camera_calibrations
        ],
        "safety_state": observation.safety_state.value,
        "quality": dict(observation.quality),
        "schema_version": observation.schema_version,
    }


def _object_name(model, kind, object_id: int, prefix: str) -> str:
    return mujoco.mj_id2name(model, kind, object_id) or f"{prefix}_{object_id}"
