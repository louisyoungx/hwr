from __future__ import annotations

import numpy as np

from hwr.core.embodied import ActionChunk, DualArmAction
from hwr.data.vla_dataset import (
    VLABehaviorDatasetBuilder,
    VLABehaviorSample,
)
from hwr.policy.vla_input import VLAActorInput


PREPROCESS_FINGERPRINT = "c" * 64
LANGUAGE_SHA256 = "d" * 64


def actor_input(step: int) -> VLAActorInput:
    history, height, width, points = 2, 8, 8, 8
    return VLAActorInput(
        head_rgb=np.full((history, height, width, 3), step / 20.0, dtype=np.float32),
        head_depth=np.full((history, height, width), 1.0 + step / 100.0, dtype=np.float32),
        head_depth_valid=np.ones((history, height, width), dtype=np.bool_),
        head_points=np.full((history, points, 6), step / 50.0, dtype=np.float32),
        head_point_valid=np.ones((history, points), dtype=np.bool_),
        wrist_rgb=np.full((history, height, width, 3), step / 30.0, dtype=np.float32),
        camera_validity=np.ones((history, 3), dtype=np.bool_),
        proprioception=np.full(37, step / 100.0, dtype=np.float32),
        instruction_embedding=np.linspace(-1.0, 1.0, 12, dtype=np.float32),
        action_history=np.full((2, 16), step / 100.0, dtype=np.float32),
        preprocess_fingerprint=PREPROCESS_FINGERPRINT,
        language_encoder_id="local-language/v1",
        language_weights_sha256=LANGUAGE_SHA256,
    )


def action_chunk(step: int, valid_steps: int = 3) -> ActionChunk:
    actions = []
    for offset in range(3):
        value = (step + offset) / 100.0
        actions.append(
            DualArmAction(
                base_linear=value,
                base_angular=-value,
                left_arm=(value,) * 6,
                right_arm=(-value,) * 6,
                left_gripper=float(step % 2),
                right_gripper=float((step + 1) % 2),
            )
        )
    return ActionChunk(tuple(actions), valid_steps)


def build_dataset(root, *, dataset_id: str = "vla-training"):
    builder = VLABehaviorDatasetBuilder(
        root,
        dataset_id,
        instruction="把桌面上的东西收好",
        action_chunk_size=3,
    )
    for episode in range(3):
        samples = [
            VLABehaviorSample(
                step,
                actor_input(step + episode),
                action_chunk(step, 2 if step == 5 else 3),
            )
            for step in range(6)
        ]
        builder.write_episode(f"episode-{episode}", 100 + episode, samples)
    return builder.seal()
