"""Runtime Policy implementation for a trained behavior model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from hwr.core.runtime import PolicySpec
from hwr.core.types import ActionFrame, ObservationFrame
from hwr.data.vectorization import observation_to_vector, vector_to_action
from hwr.policy.model import BehaviorMLP


@dataclass(frozen=True)
class Normalization:
    observation_mean: tuple[float, ...]
    observation_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Normalization":
        return cls(
            observation_mean=tuple(value["observation_mean"]),
            observation_std=tuple(value["observation_std"]),
            action_mean=tuple(value["action_mean"]),
            action_std=tuple(value["action_std"]),
        )


class NeuralPolicy:
    def __init__(
        self,
        model: BehaviorMLP,
        normalization: Normalization,
        *,
        policy_version: str,
        control_hz: float,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.normalization = normalization
        self.policy_version = policy_version
        self.control_hz = control_hz
        self.device = torch.device(device)
        self._observation_mean = np.asarray(normalization.observation_mean, dtype=np.float32)
        self._observation_std = np.asarray(normalization.observation_std, dtype=np.float32)
        self._action_mean = np.asarray(normalization.action_mean, dtype=np.float32)
        self._action_std = np.asarray(normalization.action_std, dtype=np.float32)

    def spec(self) -> PolicySpec:
        return PolicySpec(
            policy_id=self.policy_version,
            observation_history=1,
            action_horizon=1,
            control_hz=self.control_hz,
            arm_dof=2,
        )

    def reset(self, *, task_id: str, seed: int) -> None:
        del task_id, seed

    def infer(self, observations: Sequence[ObservationFrame]) -> tuple[ActionFrame, ...]:
        if not observations:
            raise ValueError("at least one observation is required")
        observation = observations[-1]
        vector = observation_to_vector(observation)
        normalized = np.clip(
            (vector - self._observation_mean) / self._observation_std,
            -5.0,
            5.0,
        )
        tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            prediction = self.model(tensor).squeeze(0).cpu().numpy()
        continuous = prediction[:4] * self._action_std + self._action_mean
        gripper = _guarded_gripper_target(observation)
        action_vector = np.concatenate(
            (continuous, np.asarray([gripper], dtype=np.float32))
        )
        return (
            vector_to_action(
                action_vector,
                observation,
                source="policy",
                policy_version=self.policy_version,
                control_hz=self.control_hz,
            ),
        )

    def close(self) -> None:
        pass


def _guarded_gripper_target(observation: ObservationFrame) -> float:
    """Keep discrete contact events behind a geometric skill guard."""
    carrying = observation.features["carrying"][0] > 0.5
    target = (
        observation.features["target_zone_relative"]
        if carrying
        else observation.features["target_object_relative"]
    )
    arm_x, arm_y = observation.joint_position
    endpoint_error = np.hypot(target[0] - arm_x, target[1] - arm_y)
    if carrying:
        return 0.0 if endpoint_error <= 0.12 else 1.0
    return 1.0 if endpoint_error <= 0.06 else 0.0
