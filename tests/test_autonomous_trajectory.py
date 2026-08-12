from __future__ import annotations

import numpy as np
import pytest

from hwr.data.autonomous_trajectory import (
    AutonomousEpisode,
    AutonomousTrajectoryDatasetBuilder,
    verify_autonomous_trajectory_dataset,
)
from hwr.data.trajectory_windows import AutonomousTrajectoryWindows


def _arrays(observations: int = 5) -> dict[str, np.ndarray]:
    transitions = observations - 1
    return {
        "rgb_uint8": np.zeros((observations, 3, 8, 8, 3), dtype=np.uint8),
        "raw_head_depth_m": np.ones((observations, 8, 8), dtype=np.float32),
        "head_depth_valid": np.ones((observations, 8, 8), dtype=np.bool_),
        "camera_validity": np.ones((observations, 4), dtype=np.bool_),
        "frame_timestamps_ns": np.arange(observations * 4).reshape(observations, 4),
        "proprioception": np.zeros((observations, 37), dtype=np.float32),
        "observation_source_sha256": np.asarray([f"{value:064x}" for value in range(observations)]),
        "actor_proposal": np.zeros((transitions, 16), dtype=np.float32),
        "executed_action": np.zeros((transitions, 16), dtype=np.float32),
        "reward": np.arange(transitions, dtype=np.float32),
        "terminated": np.asarray([False] * (transitions - 1) + [True]),
        "truncated": np.zeros(transitions, dtype=np.bool_),
        "safety_cost": np.zeros(transitions, dtype=np.float32),
        "action_source": np.asarray(["rl_actor"] * transitions),
        "intrinsics": np.ones((4, 4), dtype=np.float32),
        "robot_from_camera": np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0),
    }


def _episode(**overrides) -> AutonomousEpisode:
    values = {
        "episode_id": "episode-0001",
        "task_id": "fixture/v1",
        "seed": 7,
        "instruction": "双手搬运容器",
        "locale": "zh-CN",
        "environment_version": "fixture-env/v1",
        "source_commit": "a" * 40,
        "preprocess_fingerprint": "b" * 64,
        "legal_transform_ids": ("reflect_lateral",),
        "arrays": _arrays(),
        "metadata": {"randomization_seed": 7},
    }
    values.update(overrides)
    return AutonomousEpisode(**values)


def test_autonomous_trajectory_round_trip_and_continuous_windows(tmp_path) -> None:
    builder = AutonomousTrajectoryDatasetBuilder(tmp_path, "autonomous-v1")
    builder.write_episode(_episode())
    path = builder.seal()

    manifest = verify_autonomous_trajectory_dataset(path)
    windows = AutonomousTrajectoryWindows(path, transitions=2)

    assert manifest["transition_count"] == 4
    assert len(windows) == 3
    assert windows[1]["rgb_uint8"].shape[0] == 3
    assert windows[1]["executed_action"].shape == (2, 16)
    assert windows[1]["reward"].tolist() == [1.0, 2.0]
    assert windows.shard_metadata(0)["instruction"] == "双手搬运容器"


def test_autonomous_trajectory_rejects_non_rl_action_sources() -> None:
    arrays = _arrays()
    arrays["action_source"][0] = "expert"

    with pytest.raises(ValueError, match="not autonomous RL"):
        _episode(arrays=arrays)


def test_autonomous_trajectory_rejects_action_supervision_metadata() -> None:
    with pytest.raises(ValueError, match="forbidden action supervision"):
        _episode(metadata={"teacher_action": [0.0] * 16})


def test_autonomous_trajectory_rejects_early_terminal_and_field_leaks() -> None:
    arrays = _arrays()
    arrays["terminated"][1] = True
    with pytest.raises(ValueError, match="cannot continue"):
        _episode(arrays=arrays)
    arrays = _arrays()
    arrays["task_stage"] = np.zeros(4)
    with pytest.raises(ValueError, match="field whitelist"):
        _episode(arrays=arrays)


def test_autonomous_trajectory_verification_detects_corruption(tmp_path) -> None:
    builder = AutonomousTrajectoryDatasetBuilder(tmp_path, "autonomous-v1")
    shard = builder.write_episode(_episode())
    path = builder.seal()
    shard.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_autonomous_trajectory_dataset(path)
