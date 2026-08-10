from __future__ import annotations

import numpy as np

from hwr.core.embodied import (
    DUAL_ARM_ACTION_DIM,
    DualArmObservation,
    DualArmProprioception,
    FrozenLanguageEmbedding,
    NaturalLanguageInstruction,
)
from hwr.core.types import CameraFrame
from hwr.perception import (
    CameraCalibration,
    DualArmVisionPreprocessor,
    PinholeIntrinsics,
    VisionPreprocessConfig,
)
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS, build_vla_actor_input


def _processed_vision():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
    depth = np.ones((2, 2), dtype=np.float32).tobytes()
    observation = DualArmObservation(
        timestamp_ns=0,
        sequence_id=0,
        task_id="housework/v1",
        instruction=NaturalLanguageInstruction("双手搬运托盘"),
        proprioception=DualArmProprioception(
            (0.0,) * 6,
            (0.0,) * 6,
            (0.0,) * 6,
            (0.0,) * 6,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0),
        ),
        cameras=(
            CameraFrame("head_rgb", 0, 0, 2, 2, "rgb8", payload=rgb),
            CameraFrame("head_depth", 0, 0, 2, 2, "depth32f", payload=depth),
            CameraFrame("left_wrist_rgb", 0, 0, 2, 2, "rgb8", payload=rgb),
            CameraFrame("right_wrist_rgb", 0, 0, 2, 2, "rgb8", payload=rgb),
        ),
    )
    intrinsics = PinholeIntrinsics(2, 2, 1.0, 1.0, 0.5, 0.5)
    identity = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    calibrations = {
        name: CameraCalibration("test", name, intrinsics, identity)
        for name in ("head_rgb", "head_depth", "left_wrist_rgb", "right_wrist_rgb")
    }
    preprocessor = DualArmVisionPreprocessor(
        VisionPreprocessConfig(2, 2, 4), calibrations
    )
    return preprocessor.preprocess(observation)


def test_actor_input_contains_only_deployable_continuous_fields() -> None:
    vision = _processed_vision()
    language = FrozenLanguageEmbedding("text-model/v1", "b" * 64, (0.1,) * 8)

    value = build_vla_actor_input(
        (vision, vision),
        language,
        proprioception=(0.0,) * 37,
        action_history=np.zeros((3, DUAL_ARM_ACTION_DIM), dtype=np.float32),
    )

    assert frozenset(value.named_arrays()) == VLA_POLICY_INPUT_FIELDS
    assert value.head_rgb.shape == (2, 2, 2, 3)
    assert value.head_points.shape == (2, 4, 6)
    assert value.left_wrist_rgb.shape == value.right_wrist_rgb.shape
    assert value.camera_validity.shape == (2, 4)
    assert value.instruction_embedding.shape == (8,)
    assert not any("stage" in name or "token" in name or "plan" in name for name in VLA_POLICY_INPUT_FIELDS)
    assert not hasattr(value, "task_id")
    assert not hasattr(value, "object_id")
