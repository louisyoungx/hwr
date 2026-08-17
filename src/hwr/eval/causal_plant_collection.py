"""Controlled physical feedback collection for the R0001-P11 confirmation."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from hwr.core.embodied import DualArmActionFrame, DualArmObservation
from hwr.train.bimanual_runtime import dual_arm_action_frame
from hwr.train.foundation_collection import AutonomousActionSource


class ActionLatencyDiagnosticBackend(Protocol):
    def reset_for_action_latency_diagnostic(
        self, *, seed: int, task_id: str, action_latency_steps: int
    ) -> DualArmObservation: ...

    def set_camera_rendering(self, enabled: bool) -> None: ...

    def apply(self, frame: DualArmActionFrame): ...

    def task_audit(self) -> Mapping[str, object]: ...


def collect_causal_plant_episode(
    backend: ActionLatencyDiagnosticBackend,
    action_source: AutonomousActionSource,
    *,
    task_id: str,
    seed: int,
    correlation: float,
    action_latency_steps: int,
    transition_count: int,
    output_path: Path,
) -> dict[str, object]:
    if transition_count <= 0:
        raise ValueError("causal plant transition count must be positive")
    observation = backend.reset_for_action_latency_diagnostic(
        seed=seed,
        task_id=task_id,
        action_latency_steps=action_latency_steps,
    )
    audit = backend.task_audit()
    randomization = _required_mapping(audit, "randomization")
    provenance = _required_mapping(audit, "action_latency_diagnostic")
    backend.set_camera_rendering(False)
    action_source.reset(task_id=task_id, seed=seed)
    proposals = []
    applied_actions = []
    safety = []
    severe = []
    terminated = False
    for _ in range(transition_count):
        proposal = action_source.propose(observation)
        outcome = backend.apply(
            dual_arm_action_frame(
                observation.timestamp_ns,
                proposal,
                source=action_source.action_source,
            )
        )
        applied = outcome.info.get("applied_action")
        if not isinstance(applied, DualArmActionFrame):
            raise TypeError("causal plant backend omitted actual plant action")
        action_source.record_applied_action(applied.action)
        proposals.append(proposal.vector())
        applied_actions.append(applied.action.vector())
        safety.append(bool(outcome.info.get("safety_intervened", False)))
        current_audit = backend.task_audit()
        severe.append(int(current_audit.get("severe_collision_count", -1)))
        terminated = bool(outcome.terminated or outcome.truncated)
        observation = outcome.observation
        if terminated:
            break
    arrays = {
        "actor_proposal": np.asarray(proposals, np.float64),
        "applied_action": np.asarray(applied_actions, np.float64),
        "safety_intervention": np.asarray(safety, np.bool_),
        "severe_collision_count": np.asarray(severe, np.int64),
        "action_latency_steps": np.asarray(action_latency_steps, np.int64),
        "actuator_scale": np.asarray(
            float(randomization["actuator_scale"]), np.float64
        ),
    }
    if not all(
        np.isfinite(value).all()
        for value in arrays.values()
        if value.dtype.kind in "fc"
    ):
        raise ValueError("causal plant artifact contains non-finite values")
    _write_npz_atomic(output_path, arrays)
    return {
        "task_id": task_id,
        "seed": seed,
        "motion_correlation": correlation,
        "action_latency_steps": action_latency_steps,
        "transition_count": len(proposals),
        "safety_intervention_count": int(np.sum(safety)),
        "severe_collision_count": max(severe, default=0),
        "terminated_early": terminated or len(proposals) != transition_count,
        "actuator_scale": float(randomization["actuator_scale"]),
        "observation_latency_steps": int(
            randomization["observation_latency_steps"]
        ),
        "action_latency_diagnostic": dict(provenance),
        "artifact": {
            "path": output_path.name,
            "sha256": file_sha256(output_path),
            "bytes": output_path.stat().st_size,
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_mapping(
    value: Mapping[str, object], key: str
) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise TypeError(f"causal plant backend omitted {key}")
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
