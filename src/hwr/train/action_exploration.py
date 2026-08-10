"""Task-agnostic temporal exploration for a dual-arm continuous action space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hwr.core.embodied import (
    DUAL_ARM_ACTION_DIM,
    DUAL_ARM_TOOL_TWIST_REFLECTION_SIGNS,
)


@dataclass(frozen=True)
class TemporalExplorationConfig:
    noise_standard_deviation: float = 0.18
    noise_correlation: float = 0.85
    action_smoothing: float = 0.65
    gripper_epsilon: float = 0.35
    gripper_hold_steps: int = 16
    policy_gripper_hold_steps: int = 12
    reflection_coupled_probability: float = 0.60
    paired_gripper_probability: float = 0.60
    global_random_burst_probability: float = 0.0
    global_random_burst_steps: int = 8
    actuator_dwell_probability: float = 0.0
    actuator_dwell_steps: int = 240
    actuator_initial_dwell_probability: float = 0.0
    actuator_dwell_closed_probability: float = 0.50
    base_linear_scale: float = 0.18
    base_angular_scale: float = 0.50
    arm_twist_scale: float = 0.35

    def __post_init__(self) -> None:
        fractions = (
            self.noise_standard_deviation,
            self.noise_correlation,
            self.action_smoothing,
            self.gripper_epsilon,
            self.reflection_coupled_probability,
            self.paired_gripper_probability,
            self.global_random_burst_probability,
            self.actuator_dwell_probability,
            self.actuator_initial_dwell_probability,
            self.actuator_dwell_closed_probability,
        )
        if not all(0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("temporal exploration fractions must be in [0, 1]")
        if min(
            self.gripper_hold_steps,
            self.policy_gripper_hold_steps,
            self.global_random_burst_steps,
            self.actuator_dwell_steps,
        ) <= 0:
            raise ValueError("exploration hold durations must be positive")
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
        self._policy_gripper_values = np.zeros(2, dtype=np.float64)
        self._policy_gripper_remaining = 0
        self._burst_action: np.ndarray | None = None
        self._burst_remaining = 0
        self._dwell_action: np.ndarray | None = None
        self._dwell_remaining = 0
        self._first_perturb = True
        self._reflection_signs = np.asarray(
            DUAL_ARM_TOOL_TWIST_REFLECTION_SIGNS, dtype=np.float64
        )

    def reset(self) -> None:
        self._noise.fill(0.0)
        self._previous = None
        self._gripper_mask.fill(False)
        self._gripper_values.fill(0.0)
        self._gripper_remaining = 0
        self._policy_gripper_values.fill(0.0)
        self._policy_gripper_remaining = 0
        self._burst_action = None
        self._burst_remaining = 0
        self._dwell_action = None
        self._dwell_remaining = 0
        self._first_perturb = True

    def perturb(self, policy_action: np.ndarray) -> np.ndarray:
        value = np.asarray(policy_action, dtype=np.float64).copy()
        if value.shape != (DUAL_ARM_ACTION_DIM,):
            raise ValueError("temporal explorer requires one 16D action")
        initial = self._first_perturb
        self._first_perturb = False
        burst = self._global_random_burst()
        if burst is not None:
            self._previous = burst.copy()
            return burst
        dwell = self._actuator_dwell(initial=initial)
        if dwell is not None:
            self._previous = dwell.copy()
            return dwell
        correlation = self.config.noise_correlation
        innovation = np.sqrt(max(0.0, 1.0 - correlation**2))
        sampled = self._sample_motion_noise()
        self._noise = correlation * self._noise + innovation * sampled
        value[:14] = np.clip(
            value[:14] + self._noise * self.motion_scales,
            -self.motion_scales,
            self.motion_scales,
        )
        if self._previous is not None:
            smoothing = self.config.action_smoothing
            value[:14] = smoothing * self._previous[:14] + (1.0 - smoothing) * value[:14]
        value[14:] = np.clip(value[14:], 0.0, 1.0)
        if self._policy_gripper_remaining <= 0:
            self._policy_gripper_values = value[14:].copy()
            self._policy_gripper_remaining = self.config.policy_gripper_hold_steps
        value[14:] = self._policy_gripper_values
        self._policy_gripper_remaining -= 1
        self._refresh_grippers()
        value[14:] = np.where(
            self._gripper_mask, self._gripper_values, value[14:]
        )
        self._gripper_remaining -= 1
        self._previous = value.copy()
        return value

    def sample_random(self) -> np.ndarray:
        """Sample without observations, goals, stages, or labeled actions."""
        value = np.empty(DUAL_ARM_ACTION_DIM, dtype=np.float64)
        value[:2] = self.rng.uniform(-self.motion_scales[:2], self.motion_scales[:2])
        right = self.rng.uniform(
            -self.config.arm_twist_scale,
            self.config.arm_twist_scale,
            6,
        )
        if self.rng.random() < self.config.reflection_coupled_probability:
            left = right * self._reflection_signs
        else:
            left = self.rng.uniform(
                -self.config.arm_twist_scale,
                self.config.arm_twist_scale,
                6,
            )
        value[2:8] = left
        value[8:14] = right
        if self.rng.random() < self.config.paired_gripper_probability:
            value[14:] = self.rng.integers(0, 2)
        else:
            value[14:] = self.rng.integers(0, 2, size=2)
        return value

    def _sample_motion_noise(self) -> np.ndarray:
        sampled = self.rng.normal(
            0.0, self.config.noise_standard_deviation, 14
        )
        if self.rng.random() < self.config.reflection_coupled_probability:
            right = self.rng.normal(
                0.0, self.config.noise_standard_deviation, 6
            )
            sampled[2:8] = right * self._reflection_signs
            sampled[8:14] = right
        return sampled

    def _refresh_grippers(self) -> None:
        if self._gripper_remaining > 0:
            return
        if self.rng.random() < self.config.paired_gripper_probability:
            self._gripper_mask.fill(
                self.rng.random() < self.config.gripper_epsilon
            )
            self._gripper_values.fill(self.rng.integers(0, 2))
        else:
            self._gripper_mask = self.rng.random(2) < self.config.gripper_epsilon
            self._gripper_values = self.rng.integers(0, 2, size=2).astype(
                np.float64
            )
        self._gripper_remaining = self.config.gripper_hold_steps

    def _global_random_burst(self) -> np.ndarray | None:
        probability = self.config.global_random_burst_probability
        if self._burst_remaining <= 0:
            if probability <= 0.0 or self.rng.random() >= probability:
                return None
            self._burst_action = self.sample_random()
            self._burst_remaining = self.config.global_random_burst_steps
        if self._burst_action is None:
            raise RuntimeError("global exploration burst has no sampled action")
        self._burst_remaining -= 1
        return self._burst_action.copy()

    def _actuator_dwell(self, *, initial: bool) -> np.ndarray | None:
        initial_probability = self.config.actuator_initial_dwell_probability
        probability = (
            initial_probability
            if initial and initial_probability > 0.0
            else self.config.actuator_dwell_probability
        )
        if self._dwell_remaining <= 0:
            if probability <= 0.0 or self.rng.random() >= probability:
                return None
            self._dwell_action = np.zeros(DUAL_ARM_ACTION_DIM, dtype=np.float64)
            if self.rng.random() < self.config.paired_gripper_probability:
                self._dwell_action[14:] = float(
                    self.rng.random()
                    < self.config.actuator_dwell_closed_probability
                )
            else:
                self._dwell_action[14:] = (
                    self.rng.random(2)
                    < self.config.actuator_dwell_closed_probability
                )
            self._dwell_remaining = self.config.actuator_dwell_steps
        if self._dwell_action is None:
            raise RuntimeError("actuator dwell has no sampled action")
        self._dwell_remaining -= 1
        return self._dwell_action.copy()

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "hwr.task-agnostic-action-exploration/v1",
            "observation_fields": [],
            "privileged_fields": [],
            "action_labels": False,
            "noise_process": "first-order-correlated-gaussian",
            "gripper_process": "persistent-mixed-paired-independent-epsilon",
            "policy_gripper_hold_steps": self.config.policy_gripper_hold_steps,
            "embodiment_prior": "stochastic-left-right-reflection-coupling",
            "global_random_bursts": {
                "probability": self.config.global_random_burst_probability,
                "hold_steps": self.config.global_random_burst_steps,
            },
            "actuator_dwell": {
                "probability": self.config.actuator_dwell_probability,
                "initial_probability": (
                    self.config.actuator_initial_dwell_probability
                ),
                "hold_steps": self.config.actuator_dwell_steps,
                "closed_probability": (
                    self.config.actuator_dwell_closed_probability
                ),
                "motion": "zero",
                "grippers": "paired-or-independent-bernoulli-binary",
            },
            "task_conditioned": False,
        }
