"""Deterministic MuJoCo branch collection for paired action interventions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from hwr.core.embodied import DualArmAction, DualArmActionFrame, DualArmObservation
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.eval.paired_action_intervention import (
    PAIRED_HORIZONS,
    PairedEpisodeEffect,
)
from hwr.policy.latent_actions import LatentActionScaling
from hwr.train.bimanual_runtime import dual_arm_action_frame
from hwr.train.bimanual_runtime import dual_arm_action_frame


BRANCH_STEPS = 17
NORMALIZED_AMPLITUDE = 0.5 / math.sqrt(14.0)


class PairedBranchBackend(Protocol):
    def reset(
        self, *, seed: int, task_id: str, initial_state: PhysicalStateSnapshot
    ) -> DualArmObservation: ...

    def set_camera_rendering(self, enabled: bool) -> None: ...

    def apply(self, frame: DualArmActionFrame): ...

    def observe(self) -> DualArmObservation: ...

    def capture_state_snapshot(self) -> PhysicalStateSnapshot: ...

    def task_audit(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class BranchTrace:
    proposal: np.ndarray
    actual_action: np.ndarray
    current_proprioception: np.ndarray
    delayed_proprioception: np.ndarray
    rewards: np.ndarray
    safety_intervention: np.ndarray
    severe_collision_count: np.ndarray
    terminated: np.ndarray
    event_json: tuple[str, ...]
    final_runtime_state: np.ndarray

    def __post_init__(self) -> None:
        proposal = np.asarray(self.proposal, np.float64)
        actual = np.asarray(self.actual_action, np.float64)
        current = np.asarray(self.current_proprioception, np.float64)
        delayed = np.asarray(self.delayed_proprioception, np.float64)
        transitions = len(actual)
        if (
            transitions <= 0
            or proposal.shape != (transitions, 16)
            or current.ndim != 2
            or current.shape[0] != transitions
            or delayed.shape != current.shape
            or self.rewards.shape != (transitions,)
            or self.safety_intervention.shape != (transitions,)
            or self.severe_collision_count.shape != (transitions,)
            or self.terminated.shape != (transitions,)
            or len(self.event_json) != transitions
            or not all(
                np.isfinite(value).all()
                for value in (proposal, actual, current, delayed, self.rewards)
            )
        ):
            raise ValueError("paired branch trace is invalid")
        object.__setattr__(self, "proposal", proposal)
        object.__setattr__(self, "actual_action", actual)
        object.__setattr__(self, "current_proprioception", current)
        object.__setattr__(self, "delayed_proprioception", delayed)


def paired_direction(task_index: int, episode_index: int) -> np.ndarray:
    seed = 20_261_017 + task_index * 104_729 + episode_index * 1_000_003
    return np.random.default_rng(seed).choice((-1.0, 1.0), size=14)


def branch_order(task_index: int, episode_index: int) -> tuple[str, ...]:
    seed = 20_261_017 + task_index * 1_000_003 + episode_index * 1_009
    values = np.asarray(("plus", "minus", "sham_a", "sham_b"))
    return tuple(np.random.default_rng(seed).permutation(values).tolist())


def collect_paired_episode(
    backend: PairedBranchBackend,
    *,
    task_id: str,
    task_index: int,
    seed: int,
    episode_index: int,
    snapshot: PhysicalStateSnapshot,
    output_path: Path,
    scaling: LatentActionScaling | None = None,
) -> tuple[PairedEpisodeEffect, dict[str, object]]:
    direction = paired_direction(task_index, episode_index)
    action_scaling = scaling or LatentActionScaling()
    traces = {}
    for name in branch_order(task_index, episode_index):
        sign = -1.0 if name == "minus" else 1.0
        traces[name] = _collect_branch(
            backend,
            task_id=task_id,
            seed=seed,
            snapshot=snapshot,
            direction=direction,
            sign=sign,
            scaling=action_scaling,
            branch_name=name,
        )
    sham_equal = _traces_equal(traces["sham_a"], traces["sham_b"])
    effect, audit = paired_effect_from_traces(
        task_id,
        seed,
        episode_index,
        direction,
        traces,
        scaling=action_scaling,
    )
    arrays = _trace_arrays(traces, direction)
    _write_npz_atomic(output_path, arrays)
    artifact = {
        "path": output_path.name,
        "sha256": file_sha256(output_path),
        "bytes": output_path.stat().st_size,
    }
    total_steps = sum(len(trace.actual_action) for trace in traces.values())
    all_safety = sum(
        int(trace.safety_intervention.sum()) for trace in traces.values()
    )
    all_severe = max(
        int(trace.severe_collision_count.max(initial=0))
        for trace in traces.values()
    )
    any_terminated = any(trace.terminated.any() for trace in traces.values())
    return effect, {
        **audit,
        "sham_equal": sham_equal,
        "branch_order": list(branch_order(task_index, episode_index)),
        "all_branch_steps": total_steps,
        "all_branch_safety_interventions": all_safety,
        "all_branch_safety_rate": all_safety / max(total_steps, 1),
        "all_branch_severe_collisions": all_severe,
        "all_branch_terminated_early": bool(any_terminated),
        "artifact": artifact,
    }


def paired_effect_from_traces(
    task_id: str,
    seed: int,
    episode_index: int,
    direction: np.ndarray,
    traces: Mapping[str, BranchTrace],
    *,
    scaling: LatentActionScaling | None = None,
) -> tuple[PairedEpisodeEffect, dict[str, object]]:
    action_scaling = scaling or LatentActionScaling()
    plus = traces["plus"]
    minus = traces["minus"]
    sham_equal = _traces_equal(traces["sham_a"], traces["sham_b"])
    scales = np.asarray(
        (
            action_scaling.base_linear,
            action_scaling.base_angular,
            *(action_scaling.arm_velocity,) * 12,
            1.0,
            1.0,
        ),
        np.float64,
    )
    action_difference = (plus.actual_action - minus.actual_action) / scales
    norms = np.linalg.norm(action_difference[:, :14], axis=1)
    active = np.flatnonzero(norms > 1.0e-8)
    if not len(active):
        start = len(norms)
        direction_cosines = ()
    else:
        start = int(active[0])
        unit_direction = direction / np.linalg.norm(direction)
        direction_cosines = tuple(
            float(
                action_difference[index, :14]
                @ unit_direction
                / max(norms[index], 1.0e-12)
            )
            for index in range(start, len(norms))
        )
    first_stage = {}
    outcome = {}
    for horizon in PAIRED_HORIZONS:
        stop = start + horizon
        if stop > len(action_difference):
            first_stage[horizon] = np.zeros(14, np.float64)
            outcome[horizon] = np.zeros(16, np.float64)
            continue
        first_stage[horizon] = action_difference[start:stop, :14].mean(axis=0)
        outcome[horizon] = (
            _controllable_state(plus.current_proprioception[stop - 1])
            - _controllable_state(minus.current_proprioception[stop - 1])
        )
    safety_count = int(
        plus.safety_intervention.sum() + minus.safety_intervention.sum()
    )
    severe = int(
        max(
            plus.severe_collision_count.max(initial=0),
            minus.severe_collision_count.max(initial=0),
        )
    )
    terminated = bool(plus.terminated.any() or minus.terminated.any())
    rms = (
        float(np.sqrt(np.mean(np.square(action_difference[start:, :14]))))
        if start < len(action_difference)
        else 0.0
    )
    plus_normalized = plus.actual_action / scales
    minus_normalized = minus.actual_action / scales
    plus_rms = float(
        np.sqrt(np.mean(np.square(plus_normalized[start:, :14])))
    )
    minus_rms = float(
        np.sqrt(np.mean(np.square(minus_normalized[start:, :14])))
    )
    asymmetry = abs(plus_rms - minus_rms) / max(
        (plus_rms + minus_rms) / 2.0, 1.0e-12
    )
    effect = PairedEpisodeEffect(
        task_id,
        seed,
        episode_index,
        first_stage,
        outcome,
        direction_cosines,
        rms,
        sham_equal,
        safety_count,
        severe,
        terminated,
    )
    return effect, {
        "actual_action_start": start,
        "actual_action_difference_rms": rms,
        "plus_actual_action_rms": plus_rms,
        "minus_actual_action_rms": minus_rms,
        "first_stage_relative_asymmetry": asymmetry,
        "minimum_direction_cosine": (
            min(direction_cosines) if direction_cosines else 0.0
        ),
        "safety_interventions": safety_count,
        "severe_collisions": severe,
        "terminated_early": terminated,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_branch(
    backend: PairedBranchBackend,
    *,
    task_id: str,
    seed: int,
    snapshot: PhysicalStateSnapshot,
    direction: np.ndarray,
    sign: float,
    scaling: LatentActionScaling,
    branch_name: str,
) -> BranchTrace:
    observation = backend.reset(seed=seed, task_id=task_id, initial_state=snapshot)
    backend.set_camera_rendering(False)
    proposal = []
    actual = []
    current = []
    delayed = []
    rewards = []
    safety = []
    severe = []
    terminated = []
    events = []
    for _ in range(BRANCH_STEPS):
        vector = np.concatenate(
            (
                sign
                * NORMALIZED_AMPLITUDE
                * direction
                * np.asarray(
                    (
                        scaling.base_linear,
                        scaling.base_angular,
                        *(scaling.arm_velocity,) * 12,
                    )
                ),
                (
                    observation.proprioception.left_gripper_position,
                    observation.proprioception.right_gripper_position,
                ),
            )
        )
        proposed = DualArmAction.from_vector(vector)
        outcome = backend.apply(
            dual_arm_action_frame(
                observation.timestamp_ns,
                proposed,
                source=f"p17:{branch_name}",
            )
        )
        applied = outcome.info.get("applied_action")
        if not isinstance(applied, DualArmActionFrame):
            raise TypeError("paired branch omitted actual plant action")
        physical = backend.observe()
        audit = backend.task_audit()
        proposal.append(proposed.vector())
        actual.append(applied.action.vector())
        current.append(physical.proprioception.vector())
        delayed.append(outcome.observation.proprioception.vector())
        rewards.append(float(outcome.reward))
        safety.append(bool(outcome.info.get("safety_intervened", False)))
        severe.append(int(audit.get("severe_collision_count", -1)))
        terminated.append(bool(outcome.terminated or outcome.truncated))
        events.append(
            json.dumps(
                [(event.event_type, event.details) for event in outcome.events],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        observation = outcome.observation
        if outcome.terminated or outcome.truncated:
            break
    final_snapshot = backend.capture_state_snapshot()
    return BranchTrace(
        np.asarray(proposal, np.float64),
        np.asarray(actual, np.float64),
        np.asarray(current, np.float64),
        np.asarray(delayed, np.float64),
        np.asarray(rewards, np.float64),
        np.asarray(safety, np.bool_),
        np.asarray(severe, np.int64),
        np.asarray(terminated, np.bool_),
        tuple(events),
        np.asarray(final_snapshot.runtime_state, np.float64),
    )


def _traces_equal(first: BranchTrace, second: BranchTrace) -> bool:
    arrays = (
        "proposal",
        "actual_action",
        "current_proprioception",
        "delayed_proprioception",
        "rewards",
        "safety_intervention",
        "severe_collision_count",
        "terminated",
        "final_runtime_state",
    )
    return all(
        np.array_equal(getattr(first, name), getattr(second, name))
        for name in arrays
    ) and first.event_json == second.event_json


def _controllable_state(proprioception: np.ndarray) -> np.ndarray:
    indices = (*range(6, 12), *range(18, 26), *range(29, 31))
    return np.take(np.asarray(proprioception, np.float64), indices)


def _trace_arrays(
    traces: Mapping[str, BranchTrace], direction: np.ndarray
) -> dict[str, np.ndarray]:
    arrays = {"direction": np.asarray(direction, np.float64)}
    for branch, trace in traces.items():
        arrays.update(
            {
                f"{branch}_proposal": trace.proposal,
                f"{branch}_actual_action": trace.actual_action,
                f"{branch}_current_proprioception": trace.current_proprioception,
                f"{branch}_delayed_proprioception": trace.delayed_proprioception,
                f"{branch}_reward": trace.rewards,
                f"{branch}_safety_intervention": trace.safety_intervention,
                f"{branch}_severe_collision_count": trace.severe_collision_count,
                f"{branch}_terminated": trace.terminated,
                f"{branch}_event_json": np.asarray(trace.event_json),
                f"{branch}_final_runtime_state": trace.final_runtime_state,
            }
        )
    return arrays


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
