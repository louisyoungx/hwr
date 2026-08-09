"""Closed-loop rollout capture and presentation adapters."""

from hwr.render.rollout import RolloutFrame, RolloutTrace, capture_rollout
from hwr.render.video import VideoConfig, VideoResult, write_rollout_video

__all__ = [
    "RolloutFrame",
    "RolloutTrace",
    "VideoConfig",
    "VideoResult",
    "capture_rollout",
    "write_rollout_video",
]
