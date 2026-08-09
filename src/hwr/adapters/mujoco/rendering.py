"""Offscreen RGB and metric-depth capture through project camera frames."""

from __future__ import annotations

import numpy as np
import mujoco

from hwr.core.types import CameraFrame


class MujocoCameraRenderer:
    def __init__(self, model: mujoco.MjModel, *, width: int, height: int) -> None:
        self.model = model
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def rgb(
        self,
        data: mujoco.MjData,
        camera_name: str,
        *,
        timestamp_ns: int,
        frame_index: int,
    ) -> CameraFrame:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=camera_name)
        pixels = np.ascontiguousarray(self.renderer.render(), dtype=np.uint8)
        return CameraFrame(
            camera_id=camera_name,
            timestamp_ns=timestamp_ns,
            frame_index=frame_index,
            width=self.width,
            height=self.height,
            encoding="rgb8",
            payload=pixels.tobytes(),
        )

    def depth(
        self,
        data: mujoco.MjData,
        camera_name: str,
        *,
        timestamp_ns: int,
        frame_index: int,
    ) -> CameraFrame:
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=camera_name)
        pixels = np.ascontiguousarray(self.renderer.render(), dtype=np.float32)
        return CameraFrame(
            camera_id=camera_name,
            timestamp_ns=timestamp_ns,
            frame_index=frame_index,
            width=self.width,
            height=self.height,
            encoding="depth32f",
            payload=pixels.tobytes(),
        )

    def close(self) -> None:
        self.renderer.close()
