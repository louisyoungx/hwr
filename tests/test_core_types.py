from __future__ import annotations

import pytest

from hwr.core.clock import DeterministicClock
from hwr.core.types import ActionFrame, CameraFrame, ObservationFrame, SafetyState


def test_observation_round_trip() -> None:
    observation = ObservationFrame(
        timestamp_ns=10,
        sequence_id=2,
        task_id="tidy_table",
        task_stage="approach",
        joint_position=(0.1, 0.2),
        joint_velocity=(0.0, 0.1),
        gripper_position=0.5,
        base_pose=(1.0, 2.0, 0.3),
        base_twist=(0.1, -0.1),
        cameras=(CameraFrame("head", 10, 2, 640, 480),),
        features={"target": (2.0, 3.0)},
        safety_state=SafetyState.DEGRADED,
        quality={"sync_error_ms": 1.2},
    )

    restored = ObservationFrame.from_dict(observation.to_dict())

    assert restored == observation


def test_action_rejects_inverted_window() -> None:
    with pytest.raises(ValueError, match="inverted"):
        ActionFrame(10, 20, 19, "test")


def test_deterministic_clock_never_moves_backwards() -> None:
    clock = DeterministicClock(100)
    assert clock.advance_seconds(0.5) == 500_000_100
    with pytest.raises(ValueError, match="backwards"):
        clock.advance_ns(-1)


def test_camera_payload_is_available_online_but_omitted_from_schema() -> None:
    camera = CameraFrame(
        camera_id="head_rgb",
        timestamp_ns=10,
        frame_index=2,
        width=2,
        height=1,
        payload=b"\x01\x02\x03\x04\x05\x06",
    )
    observation = ObservationFrame(
        timestamp_ns=10,
        sequence_id=2,
        task_id="visual/v1",
        task_stage="instruction_following",
        cameras=(camera,),
    )

    assert camera.payload == b"\x01\x02\x03\x04\x05\x06"
    assert "payload" not in observation.to_dict()["cameras"][0]
    assert ObservationFrame.from_dict(observation.to_dict()).cameras[0].payload is None


def test_camera_payload_size_is_validated() -> None:
    with pytest.raises(ValueError, match="expected 12"):
        CameraFrame("head_rgb", 0, 0, 2, 2, payload=b"short")
