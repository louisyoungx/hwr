"""Checkpoint-loadable visual Policy using deployment-whitelisted observations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from hwr.core.runtime import PolicySpec
from hwr.core.types import ActionFrame, ObservationFrame
from hwr.data.visual import extract_formal_policy_input
from hwr.policy.visual_model import HouseholdVisualPolicyModel


ROUTE_POSITION_TOLERANCE = 0.035
ROUTE_YAW_TOLERANCE = 0.01
ROUTE_LOCAL_SERVO_RADIUS = 0.20


@dataclass(frozen=True)
class VisualNormalization:
    proprioception_mean: tuple[float, ...]
    proprioception_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "VisualNormalization":
        return cls(
            proprioception_mean=tuple(value["proprioception_mean"]),
            proprioception_std=tuple(value["proprioception_std"]),
            action_mean=tuple(value["action_mean"]),
            action_std=tuple(value["action_std"]),
        )


def visual_input_tensors(
    inputs: Mapping[str, np.ndarray],
    normalization: VisualNormalization,
    indices: np.ndarray | None = None,
) -> tuple[torch.Tensor, ...]:
    select = slice(None) if indices is None else indices
    head_rgb = torch.from_numpy(
        inputs["head_rgb"][select].astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)
    wrist_rgb = torch.from_numpy(
        inputs["wrist_rgb"][select].astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)
    depth_values = np.nan_to_num(
        inputs["head_depth"][select].astype(np.float32), nan=0.0, posinf=5.0
    )
    depth = torch.from_numpy(np.clip(depth_values, 0.0, 5.0)[:, None] / 5.0)
    mean = np.asarray(normalization.proprioception_mean, dtype=np.float32)
    std = np.asarray(normalization.proprioception_std, dtype=np.float32)
    proprioception = torch.from_numpy((inputs["proprioception"][select] - mean) / std)
    history = inputs["action_history"][select].astype(np.float32, copy=True)
    action_mean = np.asarray(normalization.action_mean, dtype=np.float32)
    action_std = np.asarray(normalization.action_std, dtype=np.float32)
    history[..., :8] = (history[..., :8] - action_mean) / action_std
    return (
        head_rgb,
        depth,
        wrist_rgb,
        proprioception,
        torch.from_numpy(history),
        torch.from_numpy(inputs["instruction_id"][select].astype(np.int64)),
    )


class LearnedVisualPolicy:
    def __init__(
        self,
        model: HouseholdVisualPolicyModel,
        normalization: VisualNormalization,
        *,
        policy_version: str,
        control_hz: float,
        task_instructions: Mapping[str, tuple[int, str]],
        phase_names: Sequence[str],
        phase_action_mask: Sequence[Sequence[bool]],
        phase_step_limits: Sequence[Sequence[int]] | None = None,
        navigation_routes: Mapping[
            str, Sequence[Sequence[float]]
        ] | None = None,
        arm_control_mode: str = "joint_velocity",
        base_control_mode: str = "base_velocity",
        navigation_goal_bounds: Mapping[
            str, Sequence[Sequence[float]]
        ] | None = None,
        phase_gripper_targets: Sequence[float] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.normalization = normalization
        self.policy_version = policy_version
        self.control_hz = control_hz
        self.task_instructions = dict(task_instructions)
        self.phase_names = tuple(phase_names)
        if len(self.phase_names) != model.config.phase_count:
            raise ValueError("phase vocabulary does not match the visual model")
        self.phase_action_mask = np.asarray(phase_action_mask, dtype=bool)
        if self.phase_action_mask.shape != (model.config.phase_count, 8):
            raise ValueError("phase action mask does not match the visual model")
        limits = phase_step_limits or ((0, 2**31 - 1),) * len(self.phase_names)
        self.phase_step_limits = tuple(
            (int(value[0]), int(value[1])) for value in limits
        )
        if len(self.phase_step_limits) != len(self.phase_names) or any(
            minimum < 0 or maximum <= minimum
            for minimum, maximum in self.phase_step_limits
        ):
            raise ValueError("phase step limits do not match the visual model")
        self.navigation_routes = {
            name: tuple(tuple(float(item) for item in pose) for pose in route)
            for name, route in (navigation_routes or {}).items()
        }
        if not set(self.navigation_routes).issubset(self.phase_names) or any(
            not route or any(len(pose) != 3 for pose in route)
            for route in self.navigation_routes.values()
        ):
            raise ValueError("navigation routes do not match the visual model")
        if arm_control_mode not in {
            "joint_velocity",
            "phase_terminal_joint_position",
        }:
            raise ValueError("unknown arm control mode")
        self.arm_control_mode = arm_control_mode
        if base_control_mode not in {
            "base_velocity",
            "phase_terminal_base_position",
        }:
            raise ValueError("unknown base control mode")
        self.base_control_mode = base_control_mode
        self.navigation_goal_bounds = {
            name: (
                tuple(float(value) for value in bounds[0]),
                tuple(float(value) for value in bounds[1]),
            )
            for name, bounds in (navigation_goal_bounds or {}).items()
        }
        if not set(self.navigation_goal_bounds).issubset(self.navigation_routes) or any(
            len(lower) != 2
            or len(upper) != 2
            or any(low > high for low, high in zip(lower, upper, strict=True))
            for lower, upper in self.navigation_goal_bounds.values()
        ):
            raise ValueError("navigation goal bounds do not match learned routes")
        self.phase_gripper_targets = (
            tuple(float(value) for value in phase_gripper_targets)
            if phase_gripper_targets is not None
            else None
        )
        if self.phase_gripper_targets is not None and len(
            self.phase_gripper_targets
        ) != len(self.phase_names):
            raise ValueError("phase gripper targets do not match the visual model")
        self.device = torch.device(device)
        self._history = [np.zeros(9, dtype=np.float32) for _ in range(model.config.action_history)]
        self._task_id: str | None = None
        self._phase_index = 0
        self._phase_candidate_steps = 0
        self._phase_steps = 0
        self._route_index = 0
        self._route_position_latched = False

    def spec(self) -> PolicySpec:
        return PolicySpec(self.policy_version, 1, 1, self.control_hz, 6)

    def reset(self, *, task_id: str, seed: int) -> None:
        del seed
        if task_id not in self.task_instructions:
            raise ValueError(f"checkpoint has no instruction for {task_id}")
        self._task_id = task_id
        self._history = [
            np.zeros(9, dtype=np.float32) for _ in range(self.model.config.action_history)
        ]
        self._phase_index = 0
        self._phase_candidate_steps = 0
        self._phase_steps = 0
        self._route_index = 0
        self._route_position_latched = False

    def infer(self, observations: Sequence[ObservationFrame]) -> tuple[ActionFrame, ...]:
        if not observations or self._task_id is None:
            raise ValueError("reset policy and provide at least one observation")
        observation = observations[-1]
        if observation.task_id != self._task_id:
            raise ValueError("observation task differs from reset instruction")
        instruction_id, _ = self.task_instructions[self._task_id]
        value = extract_formal_policy_input(
            observation,
            instruction_id=instruction_id,
            action_history=self._history,
            image_width=self.model.config.image_width,
            image_height=self.model.config.image_height,
        )
        batch = {name: array[None] for name, array in value.named_arrays().items()}
        tensors = tuple(
            item.to(self.device) for item in visual_input_tensors(batch, self.normalization)
        )
        with torch.inference_mode():
            output = self.model(*tensors)
        current_prediction = output.actions[0, self._phase_index].cpu().numpy()
        mean = np.asarray(self.normalization.action_mean, dtype=np.float32)
        std = np.asarray(self.normalization.action_std, dtype=np.float32)
        learned_base = current_prediction[:2] * std[:2] + mean[:2]
        learned_base = self._bounded_navigation_goal(learned_base)
        phase_index = self._select_phase(
            output.phase_logits.squeeze(0), observation.base_pose, learned_base
        )
        prediction = output.actions[0, phase_index].cpu().numpy()
        action = self._action(prediction, observation)
        vector = np.asarray(
            (action.base_linear, action.base_angular, *action.arm_command, action.gripper_target),
            dtype=np.float32,
        )
        self._history = [*self._history[1:], vector]
        return (action,)

    def _select_phase(
        self,
        logits: torch.Tensor,
        base_pose: Sequence[float] | None = None,
        learned_base: Sequence[float] | None = None,
    ) -> int:
        next_index = self._phase_index + 1
        if next_index >= len(self.phase_names):
            self._phase_steps += 1
            return self._phase_index
        minimum, maximum = self.phase_step_limits[self._phase_index]
        phase_name = self.phase_names[self._phase_index]
        if phase_name in self.navigation_routes and base_pose is not None:
            if (
                learned_base is None
                or self.base_control_mode != "phase_terminal_base_position"
            ):
                stopped = self._route_complete(base_pose)
            else:
                stopped = (
                    self._route_near_endpoint(base_pose)
                    and self._learned_route_complete(base_pose, learned_base)
                )
            candidate = self._phase_steps >= minimum and stopped
            force = False
        else:
            candidate = (
                self._phase_steps >= minimum
                and float(logits[next_index].cpu())
                > float(logits[self._phase_index].cpu())
            )
            force = self._phase_steps >= maximum
        if candidate:
            self._phase_candidate_steps += 1
        else:
            self._phase_candidate_steps = 0
        if self._phase_candidate_steps >= 10 or force:
            self._phase_index = next_index
            self._phase_candidate_steps = 0
            self._phase_steps = 0
            self._route_index = 0
            self._route_position_latched = False
        else:
            self._phase_steps += 1
        return self._phase_index

    def _action(self, prediction: np.ndarray, observation: ObservationFrame) -> ActionFrame:
        mean = np.asarray(self.normalization.action_mean, dtype=np.float32)
        std = np.asarray(self.normalization.action_std, dtype=np.float32)
        continuous = prediction[:8] * std + mean
        continuous = np.where(
            self.phase_action_mask[self._phase_index], continuous, 0.0
        )
        phase_name = self.phase_names[self._phase_index]
        if phase_name in self.navigation_routes:
            learned_goal = (
                self._bounded_navigation_goal(continuous[:2])
                if self.base_control_mode == "phase_terminal_base_position"
                else None
            )
            continuous[:2] = self._navigation_action(
                observation.base_pose, learned_goal
            )
            continuous[2:] = 0.0
        elif self.arm_control_mode == "phase_terminal_joint_position":
            target = continuous[2:].copy()
            current = np.asarray(observation.joint_position, dtype=np.float32)
            continuous[2:] = np.clip(2.0 * (target - current), -1.0, 1.0)
        continuous[:2] = np.clip(continuous[:2], (-0.5, -1.0), (0.5, 1.0))
        continuous[2:] = np.clip(continuous[2:], -1.0, 1.0)
        gripper = (
            self.phase_gripper_targets[self._phase_index]
            if self.phase_gripper_targets is not None
            else float(prediction[8] >= 0.0)
        )
        period = round(1_000_000_000 / self.control_hz)
        return ActionFrame(
            observation.timestamp_ns,
            observation.timestamp_ns,
            observation.timestamp_ns + period,
            f"learned:{self.policy_version}",
            base_linear=float(continuous[0]),
            base_angular=float(continuous[1]),
            arm_command=tuple(float(value) for value in continuous[2:]),
            gripper_target=gripper,
            policy_version=self.policy_version,
        )

    def _route_complete(self, base_pose: Sequence[float]) -> bool:
        target = self.navigation_routes[self.phase_names[self._phase_index]][-1]
        distance = math.hypot(target[0] - base_pose[0], target[1] - base_pose[1])
        self._route_position_latched |= distance <= ROUTE_POSITION_TOLERANCE
        return self._route_position_latched and abs(
            _wrap(target[2] - base_pose[2])
        ) <= ROUTE_YAW_TOLERANCE

    def _route_near_endpoint(self, base_pose: Sequence[float]) -> bool:
        target = self.navigation_routes[self.phase_names[self._phase_index]][-1]
        return math.hypot(
            target[0] - base_pose[0], target[1] - base_pose[1]
        ) <= ROUTE_LOCAL_SERVO_RADIUS

    def _learned_route_complete(
        self, base_pose: Sequence[float], learned_goal: Sequence[float]
    ) -> bool:
        route = self.navigation_routes[self.phase_names[self._phase_index]]
        distance = math.hypot(
            learned_goal[0] - base_pose[0], learned_goal[1] - base_pose[1]
        )
        self._route_position_latched |= distance <= ROUTE_POSITION_TOLERANCE
        return self._route_position_latched and abs(
            _wrap(route[-1][2] - base_pose[2])
        ) <= ROUTE_YAW_TOLERANCE

    def _bounded_navigation_goal(
        self, learned_goal: Sequence[float]
    ) -> np.ndarray:
        phase_name = self.phase_names[self._phase_index]
        bounds = self.navigation_goal_bounds.get(phase_name)
        if bounds is None:
            return np.asarray(learned_goal, dtype=np.float32)
        return np.clip(
            np.asarray(learned_goal, dtype=np.float32),
            np.asarray(bounds[0], dtype=np.float32),
            np.asarray(bounds[1], dtype=np.float32),
        )

    def _navigation_action(
        self,
        base_pose: Sequence[float],
        learned_goal: Sequence[float] | None = None,
    ) -> tuple[float, float]:
        route = self.navigation_routes[self.phase_names[self._phase_index]]
        while self._route_index < len(route) - 1:
            target = route[self._route_index]
            if math.hypot(target[0] - base_pose[0], target[1] - base_pose[1]) > 0.24:
                break
            self._route_index += 1
        target = route[self._route_index]
        distance = math.hypot(target[0] - base_pose[0], target[1] - base_pose[1])
        final = self._route_index == len(route) - 1
        if final and distance <= ROUTE_LOCAL_SERVO_RADIUS and learned_goal is not None:
            target = (float(learned_goal[0]), float(learned_goal[1]), route[-1][2])
            distance = math.hypot(target[0] - base_pose[0], target[1] - base_pose[1])
        if final and distance <= ROUTE_POSITION_TOLERANCE:
            self._route_position_latched = True
        if final and self._route_position_latched:
            yaw_error = _wrap(target[2] - base_pose[2])
            if abs(yaw_error) <= ROUTE_YAW_TOLERANCE:
                return (0.0, 0.0)
            return (0.0, float(np.clip(1.8 * yaw_error, -0.75, 0.75)))
        heading = math.atan2(target[1] - base_pose[1], target[0] - base_pose[0])
        heading_error = _wrap(heading - base_pose[2])
        direction = 1.0
        if abs(heading_error) > math.pi / 2:
            direction = -1.0
            heading_error = _wrap(heading + math.pi - base_pose[2])
        alignment = max(0.0, 1.0 - abs(heading_error) / 0.75)
        linear = direction * min(0.42, 1.1 * distance) * alignment
        angular = float(np.clip(1.8 * heading_error, -0.85, 0.85))
        return (linear, angular)

    def close(self) -> None:
        pass


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
