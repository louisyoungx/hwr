"""Collect full-state Episodes for the R0001-P09 alignment diagnostic."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from hwr.core.embodied import DualArmActionFrame, DualArmObservation
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.eval.observation_action_alignment import (
    AlignmentEpisode,
    AlignmentEpisodePlan,
)
from hwr.train.bimanual_runtime import dual_arm_action_frame
from hwr.train.foundation_collection import AutonomousActionSource


class ObservationLatencyDiagnosticBackend(Protocol):
    def reset_for_observation_latency_diagnostic(
        self, *, seed: int, task_id: str, observation_latency_steps: int
    ) -> DualArmObservation: ...

    def set_camera_rendering(self, enabled: bool) -> None: ...

    def apply(self, frame: DualArmActionFrame): ...

    def capture_state_snapshot(self) -> PhysicalStateSnapshot: ...

    def task_audit(self) -> Mapping[str, object]: ...


def collect_alignment_episode(
    backend: ObservationLatencyDiagnosticBackend,
    action_source: AutonomousActionSource,
    plan: AlignmentEpisodePlan,
    output_path: Path,
    *,
    transition_count: int,
    artifact_root: Path | None = None,
) -> AlignmentEpisode:
    """Collect one prefix action plus the frozen number of scored transitions."""
    if transition_count <= 0:
        raise ValueError("alignment transition count must be positive")
    observation = backend.reset_for_observation_latency_diagnostic(
        seed=plan.seed,
        task_id=plan.task_id,
        observation_latency_steps=plan.observation_latency_steps,
    )
    audit = backend.task_audit()
    randomization = _required_mapping(audit, "randomization")
    latency_override = _required_mapping(
        audit, "observation_latency_diagnostic"
    )
    backend.set_camera_rendering(False)
    action_source.reset(task_id=plan.task_id, seed=plan.seed)
    observations = [observation.proprioception.vector()]
    snapshots = [backend.capture_state_snapshot()]
    proposals: list[tuple[float, ...]] = []
    plant_actions: list[tuple[float, ...]] = []
    safety: list[bool] = []
    contacts: list[int] = []
    severe: list[int] = []
    for _ in range(transition_count + 1):
        proposal = action_source.propose(observation)
        frame = dual_arm_action_frame(
            observation.timestamp_ns,
            proposal,
            source=action_source.action_source,
        )
        outcome = backend.apply(frame)
        applied = outcome.info.get("applied_action")
        if not isinstance(applied, DualArmActionFrame):
            raise TypeError("alignment backend omitted actual plant action")
        action_source.record_applied_action(applied.action)
        if outcome.terminated or outcome.truncated:
            raise RuntimeError("alignment Episode ended before its frozen sample count")
        current_audit = backend.task_audit()
        proposals.append(proposal.vector())
        plant_actions.append(applied.action.vector())
        safety.append(bool(outcome.info.get("safety_intervened", False)))
        contacts.append(int(outcome.info.get("physics_contacts", -1)))
        severe.append(int(current_audit.get("severe_collision_count", -1)))
        observation = outcome.observation
        observations.append(observation.proprioception.vector())
        snapshots.append(backend.capture_state_snapshot())
    arrays = _artifact_arrays(
        observations,
        proposals,
        plant_actions,
        safety,
        contacts,
        severe,
        snapshots,
        randomization,
    )
    _write_npz_atomic(output_path, arrays)
    artifact_path = (
        output_path.relative_to(artifact_root)
        if artifact_root is not None
        else Path(output_path.parent.name) / output_path.name
    )
    return AlignmentEpisode(
        plan=plan,
        transition_count=transition_count,
        visible_proprioception=arrays["visible_proprioception"],
        actor_proposal=arrays["actor_proposal_with_prefix"],
        plant_action=arrays["plant_action_with_prefix"],
        safety_intervention=arrays["safety_intervention_with_prefix"],
        physics_contacts=arrays["physics_contacts_with_prefix"],
        severe_collision_count=arrays["severe_collision_count_with_prefix"],
        physical_state_count=len(snapshots),
        randomization=dict(randomization),
        latency_override=dict(latency_override),
        artifact_path=str(artifact_path),
        artifact_sha256=file_sha256(output_path),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_arrays(
    observations,
    proposals,
    plant_actions,
    safety,
    contacts,
    severe,
    snapshots: list[PhysicalStateSnapshot],
    randomization: Mapping[str, object],
) -> dict[str, np.ndarray]:
    state_fields = (
        "generalized_positions",
        "generalized_velocities",
        "generalized_accelerations",
        "actuator_controls",
        "solver_state",
        "runtime_state",
    )
    arrays = {
        "visible_proprioception": np.asarray(observations, np.float64),
        "actor_proposal_with_prefix": np.asarray(proposals, np.float64),
        "plant_action_with_prefix": np.asarray(plant_actions, np.float64),
        "safety_intervention_with_prefix": np.asarray(safety, np.bool_),
        "physics_contacts_with_prefix": np.asarray(contacts, np.int64),
        "severe_collision_count_with_prefix": np.asarray(severe, np.int64),
        "observation_latency_steps": np.asarray(
            int(randomization["observation_latency_steps"]), np.int64
        ),
        "action_latency_steps": np.asarray(
            int(randomization["action_latency_steps"]), np.int64
        ),
        "actuator_scale": np.asarray(
            float(randomization["actuator_scale"]), np.float64
        ),
        "physical_state_backend_fingerprint": np.asarray(
            snapshots[0].backend_fingerprint
        ),
    }
    arrays.update(
        {
            f"physical_state_{name}": np.asarray(
                [getattr(snapshot, name) for snapshot in snapshots],
                np.float64,
            )
            for name in state_fields
        }
    )
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fc"):
        raise ValueError("alignment artifact contains non-finite values")
    return arrays


def _required_mapping(
    value: Mapping[str, object], key: str
) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise TypeError(f"alignment backend omitted {key}")
    return result


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
