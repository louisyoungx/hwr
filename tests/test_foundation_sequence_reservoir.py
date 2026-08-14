from __future__ import annotations

import numpy as np

from hwr.data.autonomous_trajectory import (
    AppendableAutonomousTrajectoryStore,
    AutonomousEpisode,
)
from hwr.train.foundation_sequence_reservoir import (
    append_episode_sequence_evidence,
    count_source_episodes,
)


def _episode() -> AutonomousEpisode:
    observations = 65
    transitions = observations - 1
    proprioception = np.zeros((observations, 37), np.float32)
    proprioception[40:, 0] = np.arange(observations - 40)
    arrays = {
        "rgb_uint8": np.zeros((observations, 3, 8, 8, 3), np.uint8),
        "raw_head_depth_m": np.ones((observations, 8, 8), np.float32),
        "head_depth_valid": np.ones((observations, 8, 8), np.bool_),
        "camera_validity": np.ones((observations, 4), np.bool_),
        "frame_timestamps_ns": np.arange(observations)[:, None].repeat(4, 1),
        "proprioception": proprioception,
        "observation_source_sha256": np.asarray(
            [f"{value:064x}" for value in range(observations)]
        ),
        "actor_proposal": np.zeros((transitions, 16), np.float32),
        "executed_action": np.zeros((transitions, 16), np.float32),
        "reward": np.zeros(transitions, np.float32),
        "terminated": np.asarray([False] * (transitions - 1) + [True]),
        "truncated": np.zeros(transitions, np.bool_),
        "safety_intervention": np.zeros(transitions, np.float32),
        "action_source": np.asarray(["random_rl_exploration"] * transitions),
        "intrinsics": np.ones((observations, 4, 4), np.float32),
        "robot_from_camera": np.repeat(
            np.eye(4, dtype=np.float32)[None, None], observations * 4, axis=0
        ).reshape(observations, 4, 4, 4),
    }
    return AutonomousEpisode(
        "source-episode",
        "fixture/v1",
        7,
        "移动",
        "zh-CN",
        "fixture/v1",
        "a" * 40,
        "b" * 64,
        (),
        arrays,
        {
            "interaction_audit": {"severe_collision_count": 0},
            "interaction_trace": [
                {
                    "left_contact_steps": float(24 <= step < 28),
                    "right_contact_steps": float(24 <= step < 28),
                    "simultaneous_contact_steps": float(24 <= step < 28),
                    "maximum_controlled_rigid_displacement": (
                        0.005 if 24 <= step < 28 else 0.0
                    ),
                    "maximum_controlled_articulation_displacement": 0.0,
                    "severe_collision_count": 0.0,
                }
                for step in range(transitions)
            ],
        },
    )


def test_sequence_reservoir_retains_bounded_continuous_source_evidence(
    tmp_path,
) -> None:
    store = AppendableAutonomousTrajectoryStore(tmp_path, "reservoir")

    excerpts = append_episode_sequence_evidence(
        store,
        _episode(),
        sequence_transitions=16,
        windows_per_episode=2,
    )

    assert len(excerpts) == 2
    assert store.manifest["transition_count"] == 32
    assert count_source_episodes(store.manifest) == 1
    ranges = [
        item.metadata["sequence_reservoir"] for item in excerpts
    ]
    assert ranges[0]["transition_stop"] <= ranges[1]["transition_start"]
    assert excerpts[-1].arrays["terminated"][-1]
    assert all(len(item.arrays["proprioception"]) == 17 for item in excerpts)
    assert any(
        item.metadata["interaction_audit"]["simultaneous_contact_steps"] > 0
        for item in excerpts
    )
    assert sum(item.metadata["visual_supervision"] for item in excerpts) == 1
    assert all("interaction_trace" not in item.metadata for item in excerpts)
