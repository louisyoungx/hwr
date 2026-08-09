from __future__ import annotations

import json

import numpy as np
import pytest

from hwr.core.types import ActionFrame, CameraFrame, ObservationFrame
from hwr.data import (
    POLICY_INPUT_FIELDS,
    VisualBehaviorSample,
    VisualDatasetBuilder,
    extract_formal_policy_input,
    formal_action_vector,
    verify_visual_dataset,
)


def _observation(*, features=None, task_stage="instruction_following") -> ObservationFrame:
    rgb = bytes(range(18))
    depth = np.arange(6, dtype=np.float32).tobytes()
    cameras = (
        CameraFrame("head_rgb", 0, 0, 3, 2, "rgb8", payload=rgb),
        CameraFrame("head_depth", 0, 0, 3, 2, "depth32f", payload=depth),
        CameraFrame("wrist_rgb", 0, 0, 3, 2, "rgb8", payload=rgb),
    )
    return ObservationFrame(
        0,
        0,
        "formal/v1",
        task_stage,
        joint_position=(0.0,) * 6,
        joint_velocity=(0.0,) * 6,
        gripper_position=0.2,
        base_pose=(1.0, 2.0, 0.3),
        base_twist=(0.1, -0.1),
        imu=(0.0,) * 6,
        cameras=cameras,
        features={} if features is None else features,
    )


def _action() -> ActionFrame:
    return ActionFrame(0, 0, 1, "expert", 0.1, -0.2, (0.0,) * 6, 1.0)


def test_formal_projection_contains_only_deployment_fields() -> None:
    value = extract_formal_policy_input(
        _observation(),
        instruction_id=2,
        action_history=[np.zeros(9), np.ones(9)],
        image_width=2,
        image_height=2,
    )

    assert frozenset(value.named_arrays()) == POLICY_INPUT_FIELDS
    assert value.head_rgb.shape == (2, 2, 3)
    assert value.head_depth.shape == (2, 2)
    assert value.proprioception.shape == (24,)
    assert value.action_history.shape == (2, 9)


@pytest.mark.parametrize(
    ("features", "task_stage"),
    [({"object_truth": (1.0, 2.0, 3.0)}, "instruction_following"), ({}, "grasp")],
)
def test_formal_projection_rejects_privileged_observation_fields(features, task_stage) -> None:
    with pytest.raises(ValueError, match="privileged"):
        extract_formal_policy_input(
            _observation(features=features, task_stage=task_stage),
            instruction_id=0,
            action_history=[np.zeros(9)],
            image_width=2,
            image_height=2,
        )


def test_visual_dataset_round_trip_and_checksum(tmp_path) -> None:
    policy_input = extract_formal_policy_input(
        _observation(),
        instruction_id=0,
        action_history=[np.zeros(9), np.zeros(9)],
        image_width=2,
        image_height=2,
    )
    builder = VisualDatasetBuilder(
        tmp_path,
        "formal-demo",
        task_id="formal/v1",
        instruction="put both objects away",
        image_size=(2, 2),
        action_history=2,
    )
    builder.write_episode(
        "episode-000",
        11,
        [VisualBehaviorSample(0, policy_input, formal_action_vector(_action()))],
    )
    path = builder.seal()

    manifest = verify_visual_dataset(path)

    assert manifest["sample_count"] == 1
    assert manifest["seeds"] == [11]
    assert manifest["policy_input_fields"] == sorted(POLICY_INPUT_FIELDS)


def test_visual_dataset_verifier_rejects_manifest_input_leak(tmp_path) -> None:
    policy_input = extract_formal_policy_input(
        _observation(),
        instruction_id=0,
        action_history=[np.zeros(9)],
        image_width=2,
        image_height=2,
    )
    builder = VisualDatasetBuilder(
        tmp_path,
        "leaky",
        task_id="formal/v1",
        instruction="test",
        image_size=(2, 2),
        action_history=1,
    )
    builder.write_episode(
        "episode-000",
        3,
        [VisualBehaviorSample(0, policy_input, formal_action_vector(_action()))],
    )
    path = builder.seal()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy_input_fields"].append("object_truth")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="whitelist"):
        verify_visual_dataset(path)
