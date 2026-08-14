"""Task-independent autonomous episode collection through the runtime contract."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from hwr.core.embodied import (
    DualArmAction,
    DualArmActionFrame,
    DualArmObservation,
)
from hwr.core.runtime import RuntimeBackend
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.data.autonomous_trajectory import AutonomousEpisode
from hwr.perception.contracts import DUAL_ARM_CAMERA_IDS
from hwr.perception.high_resolution import (
    RGB_CAMERA_IDS,
    HighResolutionVisionPreprocessor,
)
from hwr.policy.foundation_runtime import FoundationWorldModelPolicy
from hwr.train.bimanual_runtime import dual_arm_action_frame


@runtime_checkable
class AutonomousActionSource(Protocol):
    action_source: str

    @property
    def action_process(self) -> Mapping[str, object]: ...

    def reset(self, *, task_id: str, seed: int) -> None: ...

    def propose(self, observation: DualArmObservation) -> DualArmAction: ...

    def record_applied_action(self, action: DualArmAction) -> None: ...


class CurrentRLActorActionSource:
    """Stochastic actions sampled directly from the current latent Actor."""

    action_source = "rl_actor"

    def __init__(self, policy: FoundationWorldModelPolicy) -> None:
        self.policy = policy

    @property
    def action_process(self) -> Mapping[str, object]:
        return {
            "schema_version": "hwr.current-stochastic-rl-actor/v1",
            "policy_id": self.policy.policy_id,
        }

    def reset(self, *, task_id: str, seed: int) -> None:
        self.policy.reset(task_id=task_id, seed=seed)

    def propose(self, observation: DualArmObservation) -> DualArmAction:
        return self.policy.infer_stochastic((observation,)).actions[0]

    def record_applied_action(self, action: DualArmAction) -> None:
        self.policy.record_applied_action(action)


class IntrinsicRLActorActionSource(CurrentRLActorActionSource):
    """Stochastic actions from the separately trained task-independent explorer."""

    action_source = "intrinsic_rl_actor"

    @property
    def action_process(self) -> Mapping[str, object]:
        return {
            "schema_version": "hwr.current-intrinsic-rl-actor/v1",
            "policy_id": self.policy.policy_id,
            "reward_sources": [
                "world_model_uncertainty",
                "latent_state_novelty",
                "predicted_safety_cost",
                "predicted_severe_collision",
                "policy_entropy",
            ],
            "environment_reward": False,
            "task_conditioned": False,
        }


@dataclass(frozen=True)
class AutonomousCollectionConfig:
    environment_version: str
    source_commit: str
    maximum_steps: int

    def __post_init__(self) -> None:
        if not self.environment_version or not self.source_commit:
            raise ValueError("autonomous collection identities are required")
        if self.maximum_steps <= 0:
            raise ValueError("autonomous collection maximum steps must be positive")


class AutonomousEpisodeCollector:
    def __init__(
        self,
        preprocessor: HighResolutionVisionPreprocessor,
        config: AutonomousCollectionConfig,
    ) -> None:
        self.preprocessor = preprocessor
        self.config = config

    def collect(
        self,
        backend: RuntimeBackend,
        action_source: AutonomousActionSource,
        *,
        task_id: str,
        seed: int,
        initial_observation: DualArmObservation | None = None,
        snapshot_sink: list[PhysicalStateSnapshot] | None = None,
    ) -> AutonomousEpisode:
        observation = initial_observation or backend.reset(seed=seed, task_id=task_id)
        action_source.reset(task_id=task_id, seed=seed)
        observations = [observation]
        proposals: list[tuple[float, ...]] = []
        executed: list[tuple[float, ...]] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        safety_interventions: list[float] = []
        interaction_trace: list[dict[str, float]] = []
        previous_interaction = _interaction_audit(backend)
        for step in range(self.config.maximum_steps):
            proposal = action_source.propose(observation)
            frame = dual_arm_action_frame(
                observation.timestamp_ns, proposal, source=action_source.action_source
            )
            outcome = backend.apply(frame)
            applied = _applied_action(outcome.info)
            action_source.record_applied_action(applied)
            limit = step + 1 >= self.config.maximum_steps
            proposals.append(proposal.vector())
            executed.append(applied.vector())
            rewards.append(float(outcome.reward))
            terminated.append(bool(outcome.terminated))
            truncated.append(bool(outcome.truncated or limit and not outcome.terminated))
            safety_interventions.append(
                _safety_intervention(frame, outcome.info, outcome.events)
            )
            current_interaction = _interaction_audit(backend)
            interaction_trace.append(
                _interaction_transition(previous_interaction, current_interaction)
            )
            previous_interaction = current_interaction
            observation = outcome.observation
            observations.append(observation)
            if snapshot_sink is not None:
                capture = getattr(backend, "capture_state_snapshot", None)
                if not callable(capture):
                    raise TypeError("snapshot collection requires a snapshot backend")
                snapshot = capture()
                if not isinstance(snapshot, PhysicalStateSnapshot):
                    raise TypeError("runtime returned an invalid physical snapshot")
                snapshot_sink.append(snapshot)
            if outcome.terminated or outcome.truncated or limit:
                break
        arrays, preprocess_fingerprint = self._episode_arrays(
            observations,
            proposals,
            executed,
            rewards,
            terminated,
            truncated,
            safety_interventions,
            action_source.action_source,
        )
        transforms = ()
        if hasattr(backend, "legal_environment_transforms"):
            transforms = tuple(
                item.transform_id for item in backend.legal_environment_transforms()
            )
        result = backend.result()
        metadata = {
            "collector": "foundation-autonomous/v2",
            "action_process": dict(action_source.action_process),
            "success": bool(result and result.success),
            "result_reason": result.reason if result else "step_limit",
            "result_metrics": dict(result.metrics) if result else {},
            "interaction_audit": _aggregate_interaction_trace(interaction_trace),
            "interaction_trace": interaction_trace,
        }
        return AutonomousEpisode(
            episode_id=f"episode-{uuid.uuid4().hex}",
            task_id=task_id,
            seed=seed,
            instruction=observations[0].instruction.text,
            locale=observations[0].instruction.locale,
            environment_version=self.config.environment_version,
            source_commit=self.config.source_commit,
            preprocess_fingerprint=preprocess_fingerprint,
            legal_transform_ids=transforms,
            arrays=arrays,
            metadata=metadata,
        )

    def _episode_arrays(
        self,
        observations: list[DualArmObservation],
        proposals: list[tuple[float, ...]],
        executed: list[tuple[float, ...]],
        rewards: list[float],
        terminated: list[bool],
        truncated: list[bool],
        safety_interventions: list[float],
        action_source: str,
    ) -> tuple[dict[str, np.ndarray], str]:
        observation_values = [self._observation_arrays(value) for value in observations]
        fingerprints = {value["preprocess_fingerprint"] for value in observation_values}
        if len(fingerprints) != 1:
            raise ValueError("visual preprocessing changed inside an Episode")
        return {
            "rgb_uint8": np.stack([value["rgb_uint8"] for value in observation_values]),
            "raw_head_depth_m": np.stack([value["raw_head_depth_m"] for value in observation_values]),
            "head_depth_valid": np.stack([value["head_depth_valid"] for value in observation_values]),
            "camera_validity": np.stack([value["camera_validity"] for value in observation_values]),
            "frame_timestamps_ns": np.stack([value["frame_timestamps_ns"] for value in observation_values]),
            "proprioception": np.asarray(
                [value.proprioception.vector() for value in observations], np.float32
            ),
            "observation_source_sha256": np.asarray(
                [value["source_sha256"] for value in observation_values]
            ),
            "actor_proposal": np.asarray(proposals, np.float32),
            "executed_action": np.asarray(executed, np.float32),
            "reward": np.asarray(rewards, np.float32),
            "terminated": np.asarray(terminated, np.bool_),
            "truncated": np.asarray(truncated, np.bool_),
            "safety_intervention": np.asarray(safety_interventions, np.float32),
            "action_source": np.asarray([action_source] * len(executed)),
            "intrinsics": np.stack([value["intrinsics"] for value in observation_values]),
            "robot_from_camera": np.stack(
                [value["robot_from_camera"] for value in observation_values]
            ),
        }, fingerprints.pop()

    def _observation_arrays(self, observation: DualArmObservation) -> dict[str, object]:
        frame = self.preprocessor.preprocess(observation)
        cameras = {value.camera_id: value for value in observation.cameras}
        if set(cameras) != set(DUAL_ARM_CAMERA_IDS):
            raise ValueError("autonomous collection requires all four cameras")
        rgb = [_decode_camera(cameras[name], np.uint8) for name in RGB_CAMERA_IDS]
        if len({value.shape for value in rgb}) != 1:
            raise ValueError("autonomous RGB cameras must share source dimensions")
        depth = _decode_camera(cameras["head_depth"], np.float32)
        if depth.shape != rgb[0].shape[:2]:
            raise ValueError("autonomous RGB and depth source dimensions differ")
        depth_valid = np.isfinite(depth)
        depth_valid &= depth >= self.preprocessor.config.minimum_depth_m
        depth_valid &= depth <= self.preprocessor.config.maximum_depth_m
        return {
            "rgb_uint8": np.stack(rgb),
            "raw_head_depth_m": np.where(depth_valid, depth, 0.0).astype(np.float32),
            "head_depth_valid": depth_valid,
            "camera_validity": frame.camera_validity,
            "frame_timestamps_ns": frame.frame_timestamps_ns,
            "source_sha256": frame.source_sha256,
            "preprocess_fingerprint": frame.preprocess_fingerprint,
            "intrinsics": self._raw_intrinsics(observation),
            "robot_from_camera": self._raw_extrinsics(observation),
        }

    def _raw_intrinsics(self, observation: DualArmObservation) -> np.ndarray:
        dynamic = {value.camera_id: value for value in observation.camera_calibrations}
        return np.asarray(
            [
                dynamic[name].intrinsics
                if name in dynamic
                else (
                    self.preprocessor.calibrations[name].intrinsics.fx,
                    self.preprocessor.calibrations[name].intrinsics.fy,
                    self.preprocessor.calibrations[name].intrinsics.cx,
                    self.preprocessor.calibrations[name].intrinsics.cy,
                )
                for name in DUAL_ARM_CAMERA_IDS
            ],
            np.float32,
        )

    def _raw_extrinsics(self, observation: DualArmObservation) -> np.ndarray:
        dynamic = {value.camera_id: value for value in observation.camera_calibrations}
        return np.asarray(
            [
                dynamic[name].robot_from_camera
                if name in dynamic
                else self.preprocessor.calibrations[name].robot_from_camera
                for name in DUAL_ARM_CAMERA_IDS
            ],
            np.float32,
        ).reshape(4, 4, 4)


def _decode_camera(frame, dtype: np.dtype) -> np.ndarray:
    if frame.payload is None:
        raise ValueError("autonomous collection requires in-memory camera payloads")
    channels = (3,) if dtype == np.uint8 else ()
    return np.frombuffer(frame.payload, dtype=dtype).reshape(
        frame.height, frame.width, *channels
    ).copy()


def _applied_action(info) -> DualArmAction:
    frame = info.get("applied_action")
    if not isinstance(frame, DualArmActionFrame):
        raise TypeError("runtime must report the actual applied dual-arm action")
    return frame.action


def _safety_intervention(frame: DualArmActionFrame, info, events) -> float:
    """Label whether the independent safety layer altered an Actor proposal."""
    applied = info.get("applied_action")
    changed = isinstance(applied, DualArmActionFrame) and applied.action != frame.action
    intervention = bool(info.get("safety_intervened", False))
    rejected = any(event.event_type == "action_rejected" for event in events)
    return float(changed or intervention or rejected)


def _interaction_audit(backend: RuntimeBackend) -> dict[str, float]:
    """Copy a task-independent physical coverage whitelist after an Episode."""
    audit = getattr(backend, "task_audit", None)
    raw = audit() if callable(audit) else {}
    if not isinstance(raw, Mapping):
        raise TypeError("runtime interaction audit must be a mapping")
    metrics = raw.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("runtime interaction metrics must be a mapping")
    values = {
        "left_contact_steps": float(raw.get("left_contact_steps", 0.0)),
        "right_contact_steps": float(raw.get("right_contact_steps", 0.0)),
        "simultaneous_contact_steps": float(
            raw.get("simultaneous_contact_steps", 0.0)
        ),
        "maximum_controlled_rigid_displacement": float(
            metrics.get("maximum_controlled_target_progress", 0.0)
        ),
        "maximum_controlled_articulation_displacement": float(
            metrics.get("maximum_controlled_articulation_progress", 0.0)
        ),
        "severe_collision_count": float(
            raw.get(
                "severe_collision_count",
                metrics.get("severe_collisions", 0.0),
            )
        ),
    }
    if not all(np.isfinite(value) and value >= 0.0 for value in values.values()):
        raise ValueError("runtime interaction audit contains invalid values")
    return values


def _interaction_transition(
    previous: Mapping[str, float], current: Mapping[str, float]
) -> dict[str, float]:
    """Turn cumulative runtime evidence into task-blind per-transition evidence."""
    return {
        name: max(0.0, float(current[name]) - float(previous[name]))
        for name in current
    }


def _aggregate_interaction_trace(
    trace: list[Mapping[str, float]],
) -> dict[str, float]:
    names = (
        "left_contact_steps",
        "right_contact_steps",
        "simultaneous_contact_steps",
        "maximum_controlled_rigid_displacement",
        "maximum_controlled_articulation_displacement",
        "severe_collision_count",
    )
    return {
        name: sum(float(item.get(name, 0.0)) for item in trace)
        for name in names
    }
