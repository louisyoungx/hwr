from __future__ import annotations

import numpy as np

from hwr.core.types import ActionFrame, CameraFrame, ObservationFrame
from hwr.data import (
    VisualBehaviorSample,
    VisualDatasetBuilder,
    extract_formal_policy_input,
    formal_action_vector,
    load_visual_dataset,
)
from hwr.train import (
    VisualTrainingConfig,
    load_visual_policy,
    save_visual_training_result,
    train_visual_policy,
)


TASK_ID = "visual-household/v1"
INSTRUCTION = "put away both objects"


def _observation(step: int) -> ObservationFrame:
    rgb = np.full((8, 8, 3), 20 + step, dtype=np.uint8).tobytes()
    depth = np.full((8, 8), 1.0 + 0.01 * step, dtype=np.float32).tobytes()
    return ObservationFrame(
        step,
        step,
        TASK_ID,
        "instruction_following",
        joint_position=(0.01 * step,) * 6,
        joint_velocity=(0.0,) * 6,
        gripper_position=float(step % 2),
        base_pose=(0.01 * step, 0.0, 0.0),
        base_twist=(0.1, 0.0),
        imu=(0.0,) * 6,
        cameras=(
            CameraFrame("head_rgb", step, step, 8, 8, "rgb8", payload=rgb),
            CameraFrame("head_depth", step, step, 8, 8, "depth32f", payload=depth),
            CameraFrame("wrist_rgb", step, step, 8, 8, "rgb8", payload=rgb),
        ),
    )


def _action(step: int) -> ActionFrame:
    return ActionFrame(
        step,
        step,
        step + 1,
        "expert",
        base_linear=0.1,
        arm_command=(0.01 * step,) * 6,
        gripper_target=float(step % 2),
    )


def _dataset(tmp_path):
    builder = VisualDatasetBuilder(
        tmp_path,
        "visual-training",
        task_id=TASK_ID,
        instruction=INSTRUCTION,
        image_size=(8, 8),
        action_history=2,
    )
    for episode in range(2):
        samples = []
        history = [np.zeros(9, dtype=np.float32) for _ in range(2)]
        for step in range(8):
            policy_input = extract_formal_policy_input(
                _observation(step + episode),
                instruction_id=0,
                action_history=history,
                image_width=8,
                image_height=8,
            )
            action = formal_action_vector(_action(step))
            samples.append(VisualBehaviorSample(step, policy_input, action))
            history = [history[-1], action]
        builder.write_episode(f"episode-{episode}", episode, samples)
    return load_visual_dataset(builder.seal())


def test_visual_policy_trains_saves_reloads_and_infers(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    result = train_visual_policy(
        dataset,
        VisualTrainingConfig(epochs=2, batch_size=4, device="cpu"),
    )
    model_path = save_visual_training_result(
        tmp_path / "models",
        "visual-policy",
        "v1",
        result,
        dataset_manifest=dataset.manifest,
        task_instructions={TASK_ID: (0, INSTRUCTION)},
        control_hz=20.0,
    )
    policy = load_visual_policy(model_path)
    policy.reset(task_id=TASK_ID, seed=100)

    action = policy.infer((_observation(9),))[0]

    assert len(dataset) == 16
    assert len(result.history) == 2
    assert np.isfinite(result.best_validation_loss)
    assert len(action.arm_command) == 6
    assert action.source == "learned:visual-policy:v1"
    assert action.policy_version == "visual-policy:v1"
