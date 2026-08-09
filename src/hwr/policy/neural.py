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
        normalized = (vector - self._observation_mean) / self._observation_std
        tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            prediction = self.model(tensor).squeeze(0).cpu().numpy()
        continuous = prediction[:4] * self._action_std + self._action_mean
        gripper = 1.0 / (1.0 + np.exp(-float(prediction[4])))
        action_vector = np.concatenate((continuous, np.asarray([gripper], dtype=np.float32)))
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

