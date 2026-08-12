"""Generic batch transforms selected only from environment declarations."""

from __future__ import annotations

from typing import Sequence

import torch

from hwr.perception.student_objectives import VisualTeacherTargets
from hwr.train.environment_augmentation import (
    LATERAL_REFLECTION,
    transform_action,
    transform_proprioception,
)
from hwr.train.foundation_batch import FoundationTrainingBatch


def transform_foundation_batch(
    batch: FoundationTrainingBatch,
    transforms: Sequence[str | None],
) -> FoundationTrainingBatch:
    """Apply one legal transform per sequence without task or object semantics."""
    if len(transforms) != batch.sequence_batch_size:
        raise ValueError("foundation transform count must match sequence batch")
    unsupported = sorted(
        {value for value in transforms if value not in {None, LATERAL_REFLECTION}}
    )
    if unsupported:
        raise ValueError(f"unsupported legal environment transforms: {unsupported}")
    selected = torch.tensor(
        [value == LATERAL_REFLECTION for value in transforms],
        dtype=torch.bool,
        device=batch.executed_actions.device,
    )
    if not bool(selected.any()):
        return batch
    observations = batch.observation_count
    flat_selected = selected[:, None].expand(-1, observations).reshape(-1)
    inputs = {name: value.clone() for name, value in batch.student_inputs.items()}
    _transform_visual_inputs(inputs, flat_selected)
    proprioception = batch.proprioception.clone()
    proprioception[selected] = transform_proprioception(
        proprioception[selected].flatten(0, 1), LATERAL_REFLECTION
    ).reshape_as(proprioception[selected])
    actions = batch.executed_actions.clone()
    actions[selected] = transform_action(actions[selected], LATERAL_REFLECTION)
    targets = _transform_targets(batch.visual_targets, flat_selected, inputs["rgb"])
    return FoundationTrainingBatch(
        inputs,
        targets,
        batch.sequence_batch_size,
        batch.observation_count,
        batch.language_features.clone(),
        proprioception,
        actions,
        batch.rewards.clone(),
        batch.continues.clone(),
        batch.safety.clone(),
    )


def _transform_visual_inputs(
    inputs: dict[str, torch.Tensor], selected: torch.Tensor
) -> None:
    rgb = torch.flip(inputs["rgb"][selected], dims=(-1,))
    inputs["rgb"][selected] = _swap_camera(rgb, 1, 2, camera_axis=2)
    inputs["head_depth_m"][selected] = torch.flip(
        inputs["head_depth_m"][selected], dims=(-1,)
    )
    inputs["head_depth_valid"][selected] = torch.flip(
        inputs["head_depth_valid"][selected], dims=(-1,)
    )
    validity = inputs["camera_validity"][selected]
    inputs["camera_validity"][selected] = _swap_camera(
        validity, 2, 3, camera_axis=2
    )
    width = inputs["rgb"].shape[-1]
    intrinsics = inputs["intrinsics"][selected]
    intrinsics[..., 2] = width - 1 - intrinsics[..., 2]
    inputs["intrinsics"][selected] = _swap_camera(
        intrinsics, 2, 3, camera_axis=2
    )
    extrinsics = inputs["robot_from_camera"][selected]
    robot_reflection = torch.diag(extrinsics.new_tensor((1.0, -1.0, 1.0, 1.0)))
    image_reflection = torch.diag(extrinsics.new_tensor((-1.0, 1.0, 1.0, 1.0)))
    extrinsics = robot_reflection @ extrinsics @ image_reflection
    inputs["robot_from_camera"][selected] = _swap_camera(
        extrinsics, 2, 3, camera_axis=2
    )


def _transform_targets(
    targets: VisualTeacherTargets,
    selected: torch.Tensor,
    transformed_rgb: torch.Tensor,
) -> VisualTeacherTargets:
    siglip = targets.siglip.clone()
    siglip_valid = targets.siglip_valid.clone()
    dinov2 = targets.dinov2.clone()
    dinov2_valid = targets.dinov2_valid.clone()
    for values in (siglip, dinov2):
        reflected = torch.flip(values[selected], dims=(-2,))
        values[selected] = _swap_camera(reflected, 1, 2, camera_axis=2)
    for values in (siglip_valid, dinov2_valid):
        reflected = torch.flip(values[selected], dims=(-1,))
        values[selected] = _swap_camera(reflected, 1, 2, camera_axis=2)
    reconstruction = targets.reconstruction_mask.clone()
    reflected_mask = torch.flip(reconstruction[selected], dims=(-1,))
    reconstruction[selected] = _swap_camera(
        reflected_mask, 1, 2, camera_axis=2
    )
    depth = targets.head_depth_m.clone()
    depth_valid = targets.head_depth_valid.clone()
    depth[selected] = torch.flip(depth[selected], dims=(-1,))
    depth_valid[selected] = torch.flip(depth_valid[selected], dims=(-1,))
    correspondences = _transform_correspondences(
        targets.correspondences, selected, transformed_rgb.shape[-1] // 16
    )
    return VisualTeacherTargets(
        siglip,
        siglip_valid,
        dinov2,
        dinov2_valid,
        transformed_rgb,
        reconstruction,
        depth,
        depth_valid,
        correspondences,
    )


def _transform_correspondences(
    value: torch.Tensor, selected: torch.Tensor, grid_size: int
) -> torch.Tensor:
    result = value.clone()
    if not result.numel():
        return result
    selected_rows = selected[result[:, 0].long()]
    rows = result[selected_rows]
    for camera_column, image_column in ((2, 4), (7, 9)):
        camera = rows[:, camera_column].clone()
        rows[:, camera_column] = torch.where(
            camera == 1, 2, torch.where(camera == 2, 1, camera)
        )
        rows[:, image_column] = grid_size - 1 - rows[:, image_column]
    result[selected_rows] = rows
    return result


def _swap_camera(
    value: torch.Tensor, first: int, second: int, *, camera_axis: int
) -> torch.Tensor:
    result = value.clone()
    first_index = [slice(None)] * value.ndim
    second_index = [slice(None)] * value.ndim
    first_index[camera_axis] = first
    second_index[camera_axis] = second
    result[tuple(first_index)] = value[tuple(second_index)]
    result[tuple(second_index)] = value[tuple(first_index)]
    return result
