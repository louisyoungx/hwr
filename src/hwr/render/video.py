"""Encode synchronized rollout traces as a browser-compatible MP4 video."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from hwr.render.raster import RolloutRasterizer
from hwr.render.rollout import RolloutTrace
from hwr.sim import HouseholdTaskSpec, RobotSpec


@dataclass(frozen=True)
class VideoConfig:
    frames_per_second: int = 20
    playback_speed: float = 2.0
    intro_seconds: float = 1.0
    outro_seconds: float = 2.0
    panel_width: int = 480
    height: int = 720
    crf: int = 20

    def __post_init__(self) -> None:
        if self.frames_per_second <= 0 or self.playback_speed <= 0:
            raise ValueError("frame rate and playback speed must be positive")
        if self.intro_seconds < 0 or self.outro_seconds < 0:
            raise ValueError("hold durations must be non-negative")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")


@dataclass(frozen=True)
class VideoResult:
    path: Path
    frame_count: int
    width: int
    height: int
    duration_seconds: float


class FFmpegWriter:
    """Small raw-RGB pipe around the system FFmpeg executable."""

    def __init__(self, path: Path, *, width: int, height: int, config: VideoConfig) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is required to encode replay videos")
        if width % 2 or height % 2:
            raise ValueError("H.264 video dimensions must be even")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = width
        self.height = height
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(config.frames_per_second),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(config.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append(self, image: Image.Image) -> None:
        if image.size != (self.width, self.height):
            raise ValueError("video frame dimensions changed during encoding")
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is unavailable")
        self._process.stdin.write(image.convert("RGB").tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = b"" if self._process.stderr is None else self._process.stderr.read()
        return_code = self._process.wait()
        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {message}")

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait()


def write_rollout_video(
    output_path: Path,
    traces: tuple[RolloutTrace, ...],
    tasks: tuple[HouseholdTaskSpec, ...],
    robot_spec: RobotSpec,
    *,
    config: VideoConfig | None = None,
) -> VideoResult:
    """Render traces on one timeline and atomically replace the requested MP4."""
    if not traces or len(traces) != len(tasks):
        raise ValueError("video requires matching non-empty traces and tasks")
    settings = config or VideoConfig()
    rasterizer = RolloutRasterizer(
        panel_width=settings.panel_width,
        height=settings.height,
    )
    width = settings.panel_width * len(traces)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    writer = FFmpegWriter(temporary_path, width=width, height=settings.height, config=settings)
    frame_count = 0
    try:
        first = rasterizer.render_grid(
            traces,
            tasks,
            robot_spec,
            tuple(0 for _ in traces),
        )
        for _ in range(round(settings.intro_seconds * settings.frames_per_second)):
            writer.append(first)
            frame_count += 1
        max_duration = max(trace.duration_seconds for trace in traces)
        motion_frames = max(1, math.ceil(max_duration / settings.playback_speed * settings.frames_per_second))
        for output_index in range(motion_frames + 1):
            playback_time = output_index / settings.frames_per_second
            simulation_time = playback_time * settings.playback_speed
            indices = tuple(
                min(round(simulation_time * trace.control_hz), len(trace.frames) - 1)
                for trace in traces
            )
            writer.append(rasterizer.render_grid(traces, tasks, robot_spec, indices))
            frame_count += 1
        final_indices = tuple(len(trace.frames) - 1 for trace in traces)
        final = rasterizer.render_grid(traces, tasks, robot_spec, final_indices)
        for _ in range(round(settings.outro_seconds * settings.frames_per_second)):
            writer.append(final)
            frame_count += 1
        writer.close()
        os.replace(temporary_path, output_path)
    except BaseException:
        writer.abort()
        temporary_path.unlink(missing_ok=True)
        raise
    return VideoResult(
        path=output_path,
        frame_count=frame_count,
        width=width,
        height=settings.height,
        duration_seconds=frame_count / settings.frames_per_second,
    )
