from __future__ import annotations

from dataclasses import replace

import torch

from hwr.perception.student_objectives import VisualTeacherTargets
from hwr.train.foundation_augmentation import transform_foundation_batch
from hwr.train.foundation_batch import FoundationTrainingBatch


def _batch() -> FoundationTrainingBatch:
    batch, observations, history, size = 2, 2, 1, 32
    flattened = batch * observations
    rgb = torch.arange(flattened * history * 3 * 3 * size * size).reshape(
        flattened, history, 3, 3, size, size
    ).float()
    inputs = {
        "rgb": rgb,
        "head_depth_m": torch.arange(flattened * history * size * size).reshape(
            flattened, history, 1, size, size
        ).float(),
        "head_depth_valid": torch.ones(flattened, history, 1, size, size, dtype=torch.bool),
        "camera_validity": torch.ones(flattened, history, 4, dtype=torch.bool),
        "intrinsics": torch.ones(flattened, history, 4, 4),
        "robot_from_camera": torch.eye(4).reshape(1, 1, 1, 4, 4).expand(
            flattened, history, 4, 4, 4
        ).clone(),
        "repeated_frame": torch.zeros(flattened, history, dtype=torch.bool),
    }
    vision_language = torch.arange(flattened * history * 3 * 2 * 2).reshape(
        flattened, history, 3, 2, 2, 1
    ).float()
    targets = VisualTeacherTargets(
        vision_language,
        torch.ones(vision_language.shape[:-1], dtype=torch.bool),
        vision_language.clone(),
        torch.ones(vision_language.shape[:-1], dtype=torch.bool),
        rgb,
        torch.ones(flattened, history, 3, 1, size, size, dtype=torch.bool),
        inputs["head_depth_m"],
        inputs["head_depth_valid"],
        torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 1]], dtype=torch.long),
    )
    proprioception = torch.arange(batch * observations * 31).reshape(
        batch, observations, 31
    ).float()
    action = torch.zeros(batch, observations - 1, 16)
    action[:, :, 1] = 0.3
    action[:, :, 2] = 0.4
    action[:, :, 14] = 0.2
    action[:, :, 15] = 0.8
    proposal = action.clone()
    proposal[:, :, 0] = 0.1
    return FoundationTrainingBatch(
        inputs,
        targets,
        batch,
        observations,
        torch.ones(batch, 6),
        proprioception,
        proposal,
        action,
        torch.zeros(batch, observations - 1),
        torch.ones(batch, observations - 1),
        torch.zeros(batch, observations - 1),
        torch.zeros(batch, observations - 1),
    )


def test_foundation_augmentation_applies_only_declared_sequence_transform() -> None:
    original = _batch()

    transformed = transform_foundation_batch(
        original, ("lateral_reflection", None)
    )

    expected_head = torch.flip(original.student_inputs["rgb"][0, :, 0], dims=(-1,))
    torch.testing.assert_close(transformed.student_inputs["rgb"][0, :, 0], expected_head)
    expected_left = torch.flip(original.student_inputs["rgb"][0, :, 2], dims=(-1,))
    torch.testing.assert_close(transformed.student_inputs["rgb"][0, :, 1], expected_left)
    torch.testing.assert_close(
        transformed.student_inputs["rgb"][2:], original.student_inputs["rgb"][2:]
    )
    assert transformed.executed_actions[0, 0, 1] == -0.3
    assert transformed.executed_actions[0, 0, 14] == 0.8
    assert transformed.executed_actions[0, 0, 15] == 0.2
    assert transformed.actor_proposals[0, 0, 0] == 0.1
    assert transformed.actor_proposals[0, 0, 14] == 0.8
    assert transformed.visual_targets.correspondences[0, 4] == 1
    assert transformed.visual_targets.correspondences[0, 7] == 2


def test_foundation_augmentation_is_an_involution() -> None:
    original = _batch()
    transformed = transform_foundation_batch(
        transform_foundation_batch(original, ("lateral_reflection", None)),
        ("lateral_reflection", None),
    )

    torch.testing.assert_close(
        transformed.student_inputs["rgb"], original.student_inputs["rgb"]
    )
    torch.testing.assert_close(
        transformed.executed_actions, original.executed_actions
    )
    torch.testing.assert_close(
        transformed.actor_proposals, original.actor_proposals
    )
    torch.testing.assert_close(
        transformed.proprioception, original.proprioception
    )


def test_foundation_augmentation_supports_world_only_batch() -> None:
    transformed = transform_foundation_batch(
        replace(_batch(), visual_targets=None), ("lateral_reflection", None)
    )

    assert transformed.visual_targets is None
    assert transformed.executed_actions[0, 0, 1] == -0.3
