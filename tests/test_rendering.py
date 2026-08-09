from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hwr.render import VideoConfig, capture_rollout, write_rollout_video
from hwr.render.raster import RolloutRasterizer
from hwr.scenarios import ExpertPolicy, PickPlaceExpert, debug_pick_place_task
from hwr.sim import RobotSpec


def _successful_trace():
    robot = RobotSpec()
    task = debug_pick_place_task()
    policy = ExpertPolicy(PickPlaceExpert(robot))
    return robot, task, capture_rollout(task, robot, policy, seed=7)


def test_capture_rollout_retains_closed_loop_result_and_frames() -> None:
    _, task, trace = _successful_trace()

    assert trace.task_id == task.task_id
    assert trace.policy_id == "rule-expert/v1"
    assert trace.result.success
    assert len(trace.frames) == trace.result.metrics["steps"] + 1
    assert trace.frames[-1].snapshot.task_stage == "complete"
    assert trace.frames[-1].events[-1].event_type == "task_succeeded"


def test_rasterizer_renders_trace_without_mutating_it() -> None:
    robot, task, trace = _successful_trace()
    expected_snapshot = trace.frames[-1].snapshot

    image = RolloutRasterizer(panel_width=320, height=480).render_grid(
        (trace,),
        (task,),
        robot,
        (len(trace.frames) - 1,),
    )

    assert image.mode == "RGB"
    assert image.size == (320, 480)
    assert trace.frames[-1].snapshot == expected_snapshot


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_video_encoder_writes_browser_compatible_mp4(tmp_path: Path) -> None:
    robot, task, trace = _successful_trace()
    output_path = tmp_path / "rollout.mp4"

    result = write_rollout_video(
        output_path,
        (trace,),
        (task,),
        robot,
        config=VideoConfig(
            frames_per_second=10,
            playback_speed=100.0,
            intro_seconds=0.0,
            outro_seconds=0.0,
            panel_width=320,
            height=480,
        ),
    )

    assert result.path == output_path
    assert result.frame_count >= 2
    assert output_path.read_bytes()[4:8] == b"ftyp"
    assert output_path.stat().st_size > 1_000


@pytest.mark.parametrize("fps,speed", [(0, 1.0), (20, 0.0)])
def test_video_config_rejects_non_positive_timing(fps: int, speed: float) -> None:
    with pytest.raises(ValueError):
        VideoConfig(frames_per_second=fps, playback_speed=speed)
