"""Read-only high-resolution evidence views from a bimanual MuJoCo runtime."""

from __future__ import annotations

from typing import Mapping

from hwr.adapters.mujoco.dual_arm_backend import MujocoDualArmBackend
from hwr.adapters.mujoco.rendering import MujocoCameraRenderer
from hwr.core.embodied import DualArmObservation


BIMANUAL_EVIDENCE_VIEWS = (
    "third_person",
    "head_rgb",
    "left_wrist_rgb",
    "right_wrist_rgb",
)
_MUJOCO_CAMERA_BY_VIEW = {
    "third_person": "third_person",
    "head_rgb": "head_rgb",
    "left_wrist_rgb": "left_wrist_rgb",
    "right_wrist_rgb": "wrist_rgb",
}


class MujocoBimanualEvidenceSource:
    """Own a separate renderer so video resolution cannot alter Actor input."""

    def __init__(
        self,
        backend: MujocoDualArmBackend,
        *,
        width: int = 640,
        height: int = 480,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("evidence dimensions must be positive")
        self.backend = backend
        self.width = int(width)
        self.height = int(height)
        self.renderer = MujocoCameraRenderer(
            backend.model, width=self.width, height=self.height
        )

    def capture(
        self, observation: DualArmObservation
    ) -> Mapping[str, bytes]:
        if observation.task_id != self.backend.config.task_id:
            raise ValueError("evidence observation and backend tasks differ")
        return {
            view: self.renderer.rgb(
                self.backend.data,
                camera_name,
                timestamp_ns=observation.timestamp_ns,
                frame_index=observation.sequence_id,
                camera_id=view,
            ).payload
            or b""
            for view, camera_name in _MUJOCO_CAMERA_BY_VIEW.items()
        }

    def close(self) -> None:
        self.renderer.close()
