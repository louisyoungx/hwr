"""Closed-loop rollout capture and presentation adapters."""

from hwr.render.bimanual_video import BimanualVideoRecorder, BimanualVideoResult
from hwr.render.rollout import RolloutFrame, RolloutTrace, capture_rollout
from hwr.render.video import VideoConfig, VideoResult, write_rollout_video

__all__ = [
    "BimanualVideoRecorder",
    "BimanualVideoResult",
    "RolloutFrame",
    "RolloutTrace",
    "VideoConfig",
    "VideoResult",
    "capture_rollout",
    "write_rollout_video",
]
