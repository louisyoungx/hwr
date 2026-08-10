"""Synchronized, uncut MP4 recording for four bimanual evaluation views."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image

from hwr.render.video import FFmpegWriter, VideoConfig


BIMANUAL_VIDEO_VIEWS = (
    "third_person",
    "head_rgb",
    "left_wrist_rgb",
    "right_wrist_rgb",
)


@dataclass(frozen=True)
class BimanualVideoResult:
    paths: Mapping[str, Path]
    frame_count: int
    width: int
    height: int
    frames_per_second: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.frames_per_second


class BimanualVideoRecorder:
    """Write every supplied control frame to four synchronized video streams."""

    def __init__(
        self,
        output_directory: Path,
        basename: str,
        *,
        width: int,
        height: int,
        frames_per_second: int = 20,
        crf: int = 20,
    ) -> None:
        if not basename or width <= 0 or height <= 0:
            raise ValueError("bimanual video identity and dimensions are required")
        output_directory.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.frames_per_second = int(frames_per_second)
        config = VideoConfig(
            frames_per_second=self.frames_per_second,
            playback_speed=1.0,
            intro_seconds=0.0,
            outro_seconds=0.0,
            panel_width=self.width,
            height=self.height,
            crf=crf,
        )
        self.paths = {
            view: output_directory / f"{basename}.{view}.mp4"
            for view in BIMANUAL_VIDEO_VIEWS
        }
        self.temporary_paths = {
            view: path.with_name(f"{path.stem}.tmp{path.suffix}")
            for view, path in self.paths.items()
        }
        self.writers = {
            view: FFmpegWriter(
                self.temporary_paths[view],
                width=self.width,
                height=self.height,
                config=config,
            )
            for view in BIMANUAL_VIDEO_VIEWS
        }
        self.frame_count = 0
        self._closed = False

    def append(self, frames: Mapping[str, bytes]) -> None:
        if self._closed:
            raise RuntimeError("bimanual video recorder is closed")
        if set(frames) != set(BIMANUAL_VIDEO_VIEWS):
            raise ValueError("bimanual evidence views are incomplete")
        expected = self.width * self.height * 3
        if any(len(frames[view]) != expected for view in BIMANUAL_VIDEO_VIEWS):
            raise ValueError("bimanual evidence RGB payload size differs")
        for view in BIMANUAL_VIDEO_VIEWS:
            image = Image.frombytes(
                "RGB", (self.width, self.height), frames[view]
            )
            self.writers[view].append(image)
        self.frame_count += 1

    def close(self) -> BimanualVideoResult:
        if self._closed:
            raise RuntimeError("bimanual video recorder is already closed")
        try:
            for writer in self.writers.values():
                writer.close()
            for view, path in self.paths.items():
                os.replace(self.temporary_paths[view], path)
        except BaseException:
            self.abort()
            raise
        self._closed = True
        return BimanualVideoResult(
            dict(self.paths),
            self.frame_count,
            self.width,
            self.height,
            self.frames_per_second,
        )

    def abort(self) -> None:
        if self._closed:
            return
        for writer in self.writers.values():
            writer.abort()
        for path in self.temporary_paths.values():
            path.unlink(missing_ok=True)
        self._closed = True
