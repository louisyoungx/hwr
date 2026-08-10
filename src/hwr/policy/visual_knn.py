"""Non-parametric visual-proprioceptive policy for local low-data training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from hwr.core.runtime import PolicySpec
from hwr.core.types import ActionFrame, ObservationFrame
from hwr.data.visual import POLICY_INPUT_FIELDS, extract_formal_policy_input


@dataclass(frozen=True)
class VisualKnnConfig:
    image_width: int
    image_height: int
    action_history: int
    neighbors: int = 5
    transition_ratio: float = 1.25
    transition_steps: int = 10

    def __post_init__(self) -> None:
        if min(
            self.image_width,
            self.image_height,
            self.action_history,
            self.neighbors,
            self.transition_ratio,
            self.transition_steps,
        ) <= 0:
            raise ValueError("visual kNN configuration must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def visual_knn_features(inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    if frozenset(inputs) != POLICY_INPUT_FIELDS:
        raise ValueError("visual kNN inputs violate the deployment whitelist")
    count, height, width = inputs["head_depth"].shape
    y = np.linspace(0, height - 1, 4).round().astype(np.int64)
    x = np.linspace(0, width - 1, 6).round().astype(np.int64)
    head = inputs["head_rgb"][:, y][:, :, x].astype(np.float32) / 255.0
    wrist = inputs["wrist_rgb"][:, y][:, :, x].astype(np.float32) / 255.0
    depth = np.nan_to_num(
        inputs["head_depth"][:, y][:, :, x].astype(np.float32),
        nan=0.0,
        posinf=5.0,
    )
    depth = np.clip(depth, 0.0, 5.0)[..., None] / 5.0
    proprioception = inputs["proprioception"].astype(np.float32, copy=True)
    yaw = proprioception[:, 15:16]
    proprioception = np.concatenate(
        (
            proprioception[:, :15],
            np.sin(yaw),
            np.cos(yaw),
            proprioception[:, 16:],
        ),
        axis=1,
    )
    history = inputs["action_history"].astype(np.float32).reshape(count, -1)
    visual = np.concatenate((head, depth, wrist), axis=3).reshape(count, -1)
    return np.ascontiguousarray(np.concatenate((proprioception, history, visual), axis=1))


def visual_knn_feature_scale(action_history: int) -> np.ndarray:
    return np.asarray(
        [1.0] * 25 + [0.30] * (action_history * 9) + [0.10] * (4 * 6 * 7),
        dtype=np.float32,
    )


class VisualKnnPolicy:
    def __init__(
        self,
        config: VisualKnnConfig,
        references: np.ndarray,
        actions: np.ndarray,
        phases: np.ndarray,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        feature_scale: np.ndarray,
        phase_names: Sequence[str],
        phase_action_mask: np.ndarray,
        *,
        policy_version: str,
        task_instructions: Mapping[str, tuple[int, str]],
        control_hz: float,
    ) -> None:
        self.config = config
        self.references = references.astype(np.float32, copy=False)
        self.actions = actions.astype(np.float32, copy=False)
        self.phases = phases.astype(np.int64, copy=False)
        self.feature_mean = feature_mean.astype(np.float32, copy=False)
        self.feature_std = feature_std.astype(np.float32, copy=False)
        self.feature_scale = feature_scale.astype(np.float32, copy=False)
        self.phase_names = tuple(phase_names)
        self.phase_action_mask = phase_action_mask.astype(bool, copy=False)
        self.policy_version = policy_version
        self.task_instructions = dict(task_instructions)
        self.control_hz = control_hz
        self._validate()
        self._phase_indices = tuple(
            np.flatnonzero(self.phases == index) for index in range(len(self.phase_names))
        )
        self._history: list[np.ndarray] = []
        self._task_id: str | None = None
        self._phase_index = 0
        self._transition_count = 0

    def _validate(self) -> None:
        count = self.references.shape[0]
        if self.actions.shape != (count, 9) or self.phases.shape != (count,):
            raise ValueError("visual kNN reference tensors are inconsistent")
        if self.phase_action_mask.shape != (len(self.phase_names), 8):
            raise ValueError("visual kNN phase action mask is invalid")
        if not self.phase_names or set(self.phases) != set(range(len(self.phase_names))):
            raise ValueError("visual kNN phase vocabulary is incomplete")

    def spec(self) -> PolicySpec:
        return PolicySpec(self.policy_version, 1, 1, self.control_hz, 6)

    def reset(self, *, task_id: str, seed: int) -> None:
        del seed
        if task_id not in self.task_instructions:
            raise ValueError(f"checkpoint has no instruction for {task_id}")
        self._task_id = task_id
        self._history = [
            np.zeros(9, dtype=np.float32) for _ in range(self.config.action_history)
        ]
        self._phase_index = 0
        self._transition_count = 0

    def infer(self, observations: Sequence[ObservationFrame]) -> tuple[ActionFrame, ...]:
        if not observations or self._task_id is None:
            raise ValueError("reset kNN policy and provide an observation")
        observation = observations[-1]
        instruction_id, _ = self.task_instructions[self._task_id]
        value = extract_formal_policy_input(
            observation,
            instruction_id=instruction_id,
            action_history=self._history,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        raw = visual_knn_features(
            {name: array[None] for name, array in value.named_arrays().items()}
        )[0]
        feature = (raw - self.feature_mean) / self.feature_std * self.feature_scale
        self._update_phase(feature)
        prediction = self._neighbor_action(feature, self._phase_index)
        action = self._action(prediction, observation)
        vector = np.asarray(
            (action.base_linear, action.base_angular, *action.arm_command, action.gripper_target),
            dtype=np.float32,
        )
        self._history = [*self._history[1:], vector]
        return (action,)

    def _distances(self, feature: np.ndarray, phase: int) -> tuple[np.ndarray, np.ndarray]:
        indices = self._phase_indices[phase]
        delta = self.references[indices] - feature
        return indices, np.mean(delta * delta, axis=1)

    def _update_phase(self, feature: np.ndarray) -> None:
        next_phase = self._phase_index + 1
        if next_phase >= len(self.phase_names):
            return
        _, current = self._distances(feature, self._phase_index)
        _, following = self._distances(feature, next_phase)
        threshold = max(float(current.min()) * self.config.transition_ratio, 0.02)
        self._transition_count = (
            self._transition_count + 1 if float(following.min()) <= threshold else 0
        )
        if self._transition_count >= self.config.transition_steps:
            self._phase_index = next_phase
            self._transition_count = 0

    def _neighbor_action(self, feature: np.ndarray, phase: int) -> np.ndarray:
        indices, distances = self._distances(feature, phase)
        count = min(self.config.neighbors, len(indices))
        selected = np.argpartition(distances, count - 1)[:count]
        weights = 1.0 / np.maximum(distances[selected], 1e-6)
        weights /= weights.sum()
        prediction = np.sum(self.actions[indices[selected]] * weights[:, None], axis=0)
        prediction[:8] = np.where(self.phase_action_mask[phase], prediction[:8], 0.0)
        prediction[8] = float(prediction[8] >= 0.5)
        return prediction

    def _action(self, prediction: np.ndarray, observation: ObservationFrame) -> ActionFrame:
        lower = (-0.5, -1.0, *([-1.0] * 6))
        upper = (0.5, 1.0, *([1.0] * 6))
        continuous = np.clip(prediction[:8], lower, upper)
        period = round(1_000_000_000 / self.control_hz)
        return ActionFrame(
            observation.timestamp_ns,
            observation.timestamp_ns,
            observation.timestamp_ns + period,
            f"learned:{self.policy_version}",
            base_linear=float(continuous[0]),
            base_angular=float(continuous[1]),
            arm_command=tuple(float(value) for value in continuous[2:]),
            gripper_target=float(prediction[8]),
            policy_version=self.policy_version,
        )

    def close(self) -> None:
        pass
