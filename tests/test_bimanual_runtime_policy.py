from __future__ import annotations

import numpy as np
import pytest
import torch

from hwr.core.embodied import (
    DualArmAction,
    DualArmObservation,
    DualArmProprioception,
    NaturalLanguageInstruction,
)
from hwr.core.types import CameraFrame
from hwr.perception import FrozenNgramLanguageConfig, FrozenNgramLanguageEncoder
from hwr.policy import BimanualVLAActorPolicy, VLAActorConfig, VLAActorModel
from hwr.policy.bimanual_input import (
    BimanualActorInputPipeline,
    BimanualInputConfig,
)
from hwr.policy.vla_actions import VLAActionScaling


def _observation(step: int = 0) -> DualArmObservation:
    width, height = 4, 4
    rgb = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
    depth = np.full((height, width), 1.2, dtype=np.float32)
    timestamp = step * 50_000_000
    cameras = (
        CameraFrame(
            "head_rgb", timestamp, step, width, height, "rgb8", payload=rgb.tobytes()
        ),
        CameraFrame(
            "head_depth",
            timestamp,
            step,
            width,
            height,
            "depth32f",
            payload=depth.tobytes(),
        ),
        CameraFrame(
            "left_wrist_rgb",
            timestamp,
            step,
            width,
            height,
            "rgb8",
            payload=rgb.tobytes(),
        ),
        CameraFrame(
            "right_wrist_rgb",
            timestamp,
            step,
            width,
            height,
            "rgb8",
            payload=rgb.tobytes(),
        ),
    )
    return DualArmObservation(
        timestamp,
        step,
        "runtime-policy/v1",
        NaturalLanguageInstruction("双手搬运托盘"),
        DualArmProprioception(
            (0.0,) * 6,
            (0.0,) * 6,
            (0.0,) * 6,
            (0.0,) * 6,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0),
            (0.0,) * 6,
        ),
        cameras,
    )


def _policy(*, fingerprint: str | None = None) -> BimanualVLAActorPolicy:
    input_config = BimanualInputConfig(
        4,
        4,
        image_width=4,
        image_height=4,
        point_count=4,
    )
    language = FrozenNgramLanguageEncoder(FrozenNgramLanguageConfig(dimension=8))
    model = VLAActorModel(
        VLAActorConfig(1, 1, 37, 8, 4, 1, hidden_dim=16, attention_heads=4)
    )
    expected = BimanualVLAActorPolicy(
        model,
        input_config,
        language,
        VLAActionScaling(0.18, 0.5, 0.35),
        policy_id="checkpoint-sha256",
        preprocess_fingerprint=(
            fingerprint
            or BimanualActorInputPipeline(
                input_config, language
            ).preprocessor.fingerprint
        ),
    )
    return expected


def test_runtime_policy_uses_applied_action_feedback_and_bounded_actor_output() -> None:
    policy = _policy()
    captured: list[dict[str, torch.Tensor]] = []

    def capture(_module, arguments) -> None:
        captured.append(arguments[0])

    policy.model.register_forward_pre_hook(capture)
    policy.reset(task_id="runtime-policy/v1", seed=7)
    first = policy.infer((_observation(),)).actions[0]
    applied = DualArmAction(0.1, 0.2, (0.3,) * 6, (-0.3,) * 6, 1.0, 0.0)
    policy.record_applied_action(applied)
    policy.infer((_observation(1),))

    assert abs(first.base_linear) <= 0.18
    assert abs(first.base_angular) <= 0.5
    assert all(abs(value) <= 0.35 for value in (*first.left_arm, *first.right_arm))
    assert torch.allclose(
        captured[1]["action_history"][0, 0],
        torch.tensor(applied.vector()),
    )
    assert policy.spec().policy_id == "checkpoint-sha256"


def test_runtime_policy_rejects_preprocessing_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="preprocessing differs"):
        _policy(fingerprint="wrong")
