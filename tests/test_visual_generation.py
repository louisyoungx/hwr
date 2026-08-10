from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hwr.core.runtime import StepOutcome
from hwr.core.types import ActionFrame, CameraFrame, EpisodeResult, ObservationFrame
from hwr.data import generate_visual_expert_dataset, verify_visual_dataset


@dataclass(frozen=True)
class _Task:
    task_id: str = "visual-test/v1"
    instruction: str = "move the objects"
    max_steps: int = 4
    control_hz: float = 20.0


def _observation(step: int) -> ObservationFrame:
    rgb = np.full((2, 2, 3), step, dtype=np.uint8).tobytes()
    depth = np.full((2, 2), step, dtype=np.float32).tobytes()
    return ObservationFrame(
        step,
        step,
        "visual-test/v1",
        "instruction_following",
        joint_position=(0.0,) * 6,
        joint_velocity=(0.0,) * 6,
        base_pose=(0.0, 0.0, 0.0),
        base_twist=(0.0, 0.0),
        imu=(0.0,) * 6,
        cameras=(
            CameraFrame("head_rgb", step, step, 2, 2, "rgb8", payload=rgb),
            CameraFrame("head_depth", step, step, 2, 2, "depth32f", payload=depth),
            CameraFrame("wrist_rgb", step, step, 2, 2, "rgb8", payload=rgb),
        ),
    )


class _Environment:
    def __init__(self) -> None:
        self.step = 0
        self._result = None

    def reset(self, *, seed: int, task_id: str) -> ObservationFrame:
        del seed, task_id
        self.step = 0
        return _observation(0)

    def observe(self) -> ObservationFrame:
        return _observation(self.step)

    def apply(self, action: ActionFrame) -> StepOutcome:
        self.step += 1
        terminated = self.step == 4
        if terminated:
            self._result = EpisodeResult(True, "complete", self.step)
        return StepOutcome(
            _observation(self.step),
            terminated=terminated,
            info={"applied_action": action},
        )

    def result(self):
        return self._result

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _Output:
    action: ActionFrame
    stage: str = "test_phase"
    privileged_label: bool = True


class _Expert:
    failed = False
    phase_names = ("test_phase",)

    def action(self, observation: ObservationFrame) -> _Output:
        return _Output(
            ActionFrame(
                observation.timestamp_ns,
                observation.timestamp_ns,
                observation.timestamp_ns + 1,
                "privileged-test",
                arm_command=(0.0,) * 6,
            )
        )


def test_visual_collection_keeps_episode_split_and_stride(tmp_path) -> None:
    path = generate_visual_expert_dataset(
        tmp_path,
        "demo",
        _Task(),
        _Environment,
        lambda environment: _Expert(),
        [10, 11],
        image_size=(2, 2),
        action_history=2,
        sample_stride=2,
    )

    manifest = verify_visual_dataset(path)

    assert manifest["episode_count"] == 2
    assert manifest["sample_count"] == 4
    assert manifest["seeds"] == [10, 11]
    assert manifest["metadata"]["expert_fields_are_labels_only"] is True
