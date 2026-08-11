"""Generic tensor augmentation for simulator-declared legal transforms."""

from __future__ import annotations

from typing import Mapping

import torch

from hwr.core.embodied import DUAL_ARM_TOOL_TWIST_REFLECTION_SIGNS
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS
from hwr.tasks import BIMANUAL_GOAL_DIM


LATERAL_REFLECTION = "lateral_reflection"
ENVIRONMENT_TRANSFORM_INDICES = {LATERAL_REFLECTION: 1}


def environment_transform_index(transform_id: str) -> int:
    try:
        return ENVIRONMENT_TRANSFORM_INDICES[transform_id]
    except KeyError as exc:
        raise ValueError(f"unsupported legal environment transform: {transform_id}") from exc


def environment_transform_name(index: int) -> str:
    for name, candidate in ENVIRONMENT_TRANSFORM_INDICES.items():
        if candidate == index:
            return name
    raise ValueError(f"unsupported legal environment transform index: {index}")


def transform_goal(goal: torch.Tensor, transform_id: str) -> torch.Tensor:
    _require_lateral_reflection(transform_id)
    value = goal.clone()
    value[:, 1] *= -1
    value[:, [7, 8]] = value[:, [8, 7]]
    return value


def transform_action(value: torch.Tensor, transform_id: str) -> torch.Tensor:
    _require_lateral_reflection(transform_id)
    transformed = value.clone()
    signs = torch.tensor(
        DUAL_ARM_TOOL_TWIST_REFLECTION_SIGNS,
        dtype=value.dtype,
        device=value.device,
    )
    transformed[..., 1] *= -1
    transformed[..., 2:8] = value[..., 8:14] * signs
    transformed[..., 8:14] = value[..., 2:8] * signs
    transformed[..., 14] = value[..., 15]
    transformed[..., 15] = value[..., 14]
    return transformed


def transform_proprioception(
    value: torch.Tensor, transform_id: str
) -> torch.Tensor:
    _require_lateral_reflection(transform_id)
    transformed = value.clone()
    signs = torch.tensor(
        (-1, 1, 1, -1, 1, -1),
        dtype=value.dtype,
        device=value.device,
    )
    transformed[:, 0:6] = value[:, 12:18] * signs
    transformed[:, 6:12] = value[:, 18:24] * signs
    transformed[:, 12:18] = value[:, 0:6] * signs
    transformed[:, 18:24] = value[:, 6:12] * signs
    transformed[:, 24] = value[:, 25]
    transformed[:, 25] = value[:, 24]
    if value.shape[1] >= 31:
        transformed[:, 27] *= -1
        transformed[:, 28] *= -1
        transformed[:, 30] *= -1
    return transformed


def transform_actor_inputs(
    inputs: Mapping[str, torch.Tensor], transform_id: str
) -> dict[str, torch.Tensor]:
    _require_lateral_reflection(transform_id)
    if frozenset(inputs) != VLA_POLICY_INPUT_FIELDS:
        raise ValueError("environment augmentation received non-deployable Actor fields")
    value = {name: tensor.clone() for name, tensor in inputs.items()}
    value["head_rgb"] = torch.flip(inputs["head_rgb"], dims=(-2,))
    value["head_depth"] = torch.flip(inputs["head_depth"], dims=(-1,))
    value["head_depth_valid"] = torch.flip(
        inputs["head_depth_valid"], dims=(-1,)
    )
    value["left_wrist_rgb"] = torch.flip(
        inputs["right_wrist_rgb"], dims=(-2,)
    )
    value["right_wrist_rgb"] = torch.flip(
        inputs["left_wrist_rgb"], dims=(-2,)
    )
    value["head_points"][..., 1] *= -1
    value["camera_validity"][..., [2, 3]] = inputs["camera_validity"][
        ..., [3, 2]
    ]
    value["proprioception"] = transform_proprioception(
        inputs["proprioception"], transform_id
    )
    value["action_history"] = transform_action(
        inputs["action_history"], transform_id
    )
    return value


def transform_privileged(value: torch.Tensor, transform_id: str) -> torch.Tensor:
    _require_lateral_reflection(transform_id)
    transformed = value.clone()
    transformed[:, :BIMANUAL_GOAL_DIM] = transform_goal(
        value[:, :BIMANUAL_GOAL_DIM], transform_id
    )
    goal = slice(BIMANUAL_GOAL_DIM, 2 * BIMANUAL_GOAL_DIM)
    transformed[:, goal] = transform_goal(value[:, goal], transform_id)
    transformed[:, [24, 25]] = value[:, [25, 24]]
    transformed[:, 27] *= -1
    signs = torch.tensor(
        (-1, 1, 1, -1, 1, -1),
        dtype=value.dtype,
        device=value.device,
    )
    for left, right in ((29, 35), (41, 47)):
        transformed[:, left : left + 6] = value[:, right : right + 6] * signs
        transformed[:, right : right + 6] = value[:, left : left + 6] * signs
    transformed[:, [53, 54]] = value[:, [54, 53]]
    transformed[:, 56] *= -1
    transformed[:, 57] *= -1
    transformed[:, 59] *= -1
    return transformed


def _require_lateral_reflection(transform_id: str) -> None:
    if transform_id != LATERAL_REFLECTION:
        raise ValueError(f"unsupported legal environment transform: {transform_id}")
