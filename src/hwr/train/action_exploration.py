"""Task-agnostic temporal exploration for a dual-arm continuous action space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hwr.core.embodied import DUAL_ARM_ACTION_DIM


@dataclass(frozen=True)
class TemporalExplorationConfig:
    noise_standard_deviation: float = 0.18
    noise_correlation: float = 0.85
    action_smoothing: float = 0.65
    gripper_epsilon: float = 0.35
    gripper_hold_steps: int = 16
    base_linear_scale: float = 0.18
    base_angular_scale: float = 0.50
    arm_twist_scale: float = 0.35

    def __post_init__(self) -> None:
        fractions = (
            self.noise_standard_deviation,
            self.noise_correlation,
            self.action_smoothing,
            self.gripper_epsilon,
        )
        if not all(0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("temporal exploration fractions must be in [0, 1]")
        if self.gripper_hold_steps <= 0:
            raise ValueError("gripper exploration hold must be positive")
        if min(
            self.base_linear_scale,
            self.base_angular_scale,
            self.arm_twist_scale,
        ) <= 0.0:
            raise ValueError("temporal exploration action scales must be positive")


class TemporalActionExplorer:
    """Perturb actions without observations, task truth, goals, or action labels."""

    def __init__(
        self,
        config: TemporalExplorationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.rng = rng
        self.motion_scales = np.asarray(
            (
                config.base_linear_scale,
                config.base_angular_scale,
                *(config.arm_twist_scale,) * 12,
            )
        )
        self._noise = np.zeros(14, dtype=np.float64)
        self._previous: np.ndarray | None = None
        self._gripper_mask = np.zeros(2, dtype=bool)
        self._gripper_values = np.zeros(2, dtype=np.float64)
        self._gripper_remaining = 0

    def reset(self) -> None:
        self._noise.fill(0.0)
        self._previous = None
        self._gripper_mask.fill(False)
        self._gripper_values.fill(0.0)
        self._gripper_remaining = 0

    def perturb(self, policy_action: np.ndarray) -> np.ndarray:
        value = np.asarray(policy_action, dtype=np.float64).copy()
        if value.shape != (DUAL_ARM_ACTION_DIM,):
            raise ValueError("temporal explorer requires one 16D action")
        correlation = self.config.noise_correlation
        innovation = np.sqrt(max(0.0, 1.0 - correlation**2))
        self._noise = correlation * self._noise + innovation * self.rng.normal(
            0.0, self.config.noise_standard_deviation, 14
        )
        value[:14] = np.clip(
            value[:14] + self._noise * self.motion_scales,
            -self.motion_scales,
            self.motion_scales,
        )
        if self._previous is not None:
            smoothing = self.config.action_smoothing
            value[:14] = smoothing * self._previous[:14] + (1.0 - smoothing) * value[:14]
        self._refresh_grippers()
        value[14:] = np.clip(value[14:], 0.0, 1.0)
        value[14:] = np.where(
            self._gripper_mask, self._gripper_values, value[14:]
        )
        self._gripper_remaining -= 1
        self._previous = value.copy()
        return value

    def _refresh_grippers(self) -> None:
        if self._gripper_remaining > 0:
            return
        self._gripper_mask = self.rng.random(2) < self.config.gripper_epsilon
        self._gripper_values = self.rng.integers(0, 2, size=2).astype(np.float64)
        self._gripper_remaining = self.config.gripper_hold_steps

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "hwr.task-agnostic-action-exploration/v1",
            "observation_fields": [],
            "privileged_fields": [],
            "action_labels": False,
            "noise_process": "first-order-correlated-gaussian",
            "gripper_process": "persistent-independent-epsilon",
        }
