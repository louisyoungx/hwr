"""Task-blind temporally coherent action source for initial random RL data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch

from hwr.core.embodied import DualArmAction, DualArmObservation
from hwr.policy.latent_actions import LatentActionScaling, scale_latent_action


@dataclass(frozen=True)
class RandomRLExplorationConfig:
    """Global persistence parameters with no observation or task semantics."""

    motion_correlation: float = 0.96
    gripper_flip_probability: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.motion_correlation < 1.0:
            raise ValueError("random RL motion correlation must be in [0, 1)")
        if not 0.0 < self.gripper_flip_probability <= 1.0:
            raise ValueError("random RL gripper flip probability must be in (0, 1]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class RandomRLActionSource:
    """Seeded correlated random process without observations or action answers."""

    action_source = "random_rl_exploration"

    def __init__(
        self,
        scaling: LatentActionScaling,
        config: RandomRLExplorationConfig | None = None,
    ) -> None:
        self.scaling = scaling
        self.config = config or RandomRLExplorationConfig()
        self._rng: np.random.Generator | None = None
        self._motion: np.ndarray | None = None
        self._grippers: np.ndarray | None = None

    @property
    def action_process(self) -> Mapping[str, object]:
        return {
            "schema_version": "hwr.correlated-random-rl/v1",
            **self.config.to_dict(),
            "observation_conditioned": False,
            "task_conditioned": False,
        }

    def reset(self, *, task_id: str, seed: int) -> None:
        del task_id
        self._rng = np.random.default_rng(seed)
        self._motion = self._rng.standard_normal(14).astype(np.float32)
        self._grippers = self._rng.integers(0, 2, size=2).astype(np.float32)

    def propose(self, observation: DualArmObservation) -> DualArmAction:
        del observation
        if self._rng is None or self._motion is None or self._grippers is None:
            raise RuntimeError("random RL source must be reset")
        correlation = self.config.motion_correlation
        innovation_scale = np.sqrt(1.0 - correlation * correlation)
        innovation = self._rng.standard_normal(14).astype(np.float32)
        self._motion = correlation * self._motion + innovation_scale * innovation
        flips = self._rng.random(2) < self.config.gripper_flip_probability
        self._grippers[flips] = 1.0 - self._grippers[flips]
        normalized = np.concatenate(
            (np.tanh(self._motion), self._grippers)
        ).astype(np.float32)
        scaled = scale_latent_action(
            torch.from_numpy(normalized)[None], self.scaling
        )[0]
        return DualArmAction.from_vector(scaled.tolist())

    def record_applied_action(self, action: DualArmAction) -> None:
        del action
