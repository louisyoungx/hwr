from __future__ import annotations

import numpy as np

from hwr.core.types import ActionFrame, CameraFrame, ObservationFrame
from hwr.data import (
    VisualBehaviorSample,
    VisualDatasetBuilder,
    compact_household_phase,
    compact_visual_dataset,
    extract_formal_policy_input,
    formal_action_vector,
    load_visual_dataset,
)


def _input():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
    depth = np.ones((2, 2), dtype=np.float32).tobytes()
    observation = ObservationFrame(
        0,
        0,
        "formal/v1",
        "instruction_following",
        joint_position=(0.0,) * 6,
        joint_velocity=(0.0,) * 6,
        base_pose=(0.0, 0.0, 0.0),
        base_twist=(0.0, 0.0),
        imu=(0.0,) * 6,
        cameras=(
            CameraFrame("head_rgb", 0, 0, 2, 2, "rgb8", payload=rgb),
            CameraFrame("head_depth", 0, 0, 2, 2, "depth32f", payload=depth),
            CameraFrame("wrist_rgb", 0, 0, 2, 2, "rgb8", payload=rgb),
        ),
    )
    return extract_formal_policy_input(
        observation,
        instruction_id=0,
        action_history=[np.zeros(9, dtype=np.float32)],
        image_width=2,
        image_height=2,
    )


def test_macro_phase_mapping_groups_micro_stages() -> None:
    assert compact_household_phase("nav_object_cup") == "approach_object_cup"
    assert compact_household_phase("arm_object_above_cup") == "approach_object_cup"
    assert compact_household_phase("grip_object_cup") == "grasp_object_cup"
    assert compact_household_phase("contact_pull_drawer") == "open_drawer"


def test_visual_phase_compaction_preserves_inputs_and_actions(tmp_path) -> None:
    value = _input()
    action = formal_action_vector(ActionFrame(0, 0, 1, "expert", arm_command=(0.0,) * 6))
    source_root = tmp_path / "source"
    builder = VisualDatasetBuilder(
        source_root,
        "micro",
        task_id="formal/v1",
        instruction="test",
        image_size=(2, 2),
        action_history=1,
    )
    builder.declare_phase_order(("nav_object_cup", "arm_object_above_cup", "grip_object_cup"))
    builder.write_episode(
        "episode-0",
        1,
        [
            VisualBehaviorSample(0, value, action, phase="nav_object_cup"),
            VisualBehaviorSample(1, value, action, phase="arm_object_above_cup"),
            VisualBehaviorSample(2, value, action, phase="grip_object_cup"),
        ],
    )
    source = builder.seal()

    compacted = load_visual_dataset(
        compact_visual_dataset(source, tmp_path / "output", "macro")
    )

    assert compacted.phase_names == ("approach_object_cup", "grasp_object_cup")
    assert compacted.phases.tolist() == [
        "approach_object_cup",
        "approach_object_cup",
        "grasp_object_cup",
    ]
    np.testing.assert_array_equal(compacted.actions, np.stack((action, action, action)))
