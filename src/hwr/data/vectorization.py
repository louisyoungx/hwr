"""Stable vector projections for the first low-dimensional policy baseline."""

from __future__ import annotations

import numpy as np

from hwr.core.types import ActionFrame, ObservationFrame


OBSERVATION_VECTOR_VERSION = "hwr.observation-vector/v1"
ACTION_VECTOR_VERSION = "hwr.action-vector/v1"


def observation_to_vector(observation: ObservationFrame) -> np.ndarray:
    heading = observation.base_pose[2]
    values = (
        *observation.joint_position,
        observation.gripper_position,
        observation.base_twist[0],
        observation.base_twist[1],
        np.sin(heading),
        np.cos(heading),
        *observation.features["target_object_relative"],
        *observation.features["target_zone_relative"],
        *observation.features["lidar"],
        *observation.features["carrying"],
        *observation.features["placed_fraction"],
    )
    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("observation vector contains non-finite values")
    return vector


def action_to_vector(action: ActionFrame) -> np.ndarray:
    values = (
        action.base_linear,
        action.base_angular,
        *action.arm_command,
        action.gripper_target,
    )
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (5,):
        raise ValueError(f"reference action vector must have 5 values, got {vector.shape}")
    return vector


def vector_to_action(
    vector: np.ndarray,
    observation: ObservationFrame,
    *,
    source: str,
    policy_version: str,
    control_hz: float,
) -> ActionFrame:
    values = np.asarray(vector, dtype=np.float32)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("policy action vector must contain five finite values")
    period_ns = round(1_000_000_000 / control_hz)
    return ActionFrame(
        created_at_ns=observation.timestamp_ns,
        valid_from_ns=observation.timestamp_ns,
        valid_until_ns=observation.timestamp_ns + period_ns,
        source=source,
        base_linear=float(values[0]),
        base_angular=float(values[1]),
        arm_command=(float(values[2]), float(values[3])),
        gripper_target=float(values[4]),
        policy_version=policy_version,
    )

