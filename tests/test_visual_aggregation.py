from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hwr.core.runtime import PolicySpec, StepOutcome
from hwr.core.types import ActionFrame, CameraFrame, EpisodeResult, ObservationFrame
from hwr.data import (
    VisualBehaviorSample,
    VisualDatasetBuilder,
    aggregate_visual_policy_dataset,
    extract_formal_policy_input,
    formal_action_vector,
    load_visual_dataset,
)


TASK_ID = "formal/v1"


def _observation(step: int) -> ObservationFrame:
    rgb = np.full((2, 2, 3), step, dtype=np.uint8).tobytes()
    depth = np.ones((2, 2), dtype=np.float32).tobytes()
    return ObservationFrame(
        step,
        step,
        TASK_ID,
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

    def reset(self, *, seed: int, task_id: str):
        del seed, task_id
        self.step = 0
        return _observation(0)

    def apply(self, action: ActionFrame) -> StepOutcome:
        self.step += 1
        terminated = self.step == 4
        if terminated:
            self._result = EpisodeResult(False, "rollout_complete", self.step)
        return StepOutcome(
            _observation(self.step),
            terminated=terminated,
            info={"applied_action": action},
        )

    def observe(self):
        return _observation(self.step)

    def result(self):
        return self._result

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _ExpertOutput:
    action: ActionFrame
    stage: str = "nav_object_cup"
    privileged_label: bool = True


class _Expert:
    failed = False

    def action(self, observation):
        return _ExpertOutput(
            ActionFrame(
                observation.timestamp_ns,
                observation.timestamp_ns,
                observation.timestamp_ns + 1,
                "privileged",
                base_linear=0.25,
                arm_command=(0.0,) * 6,
            )
        )


class _Policy:
    def spec(self):
        return PolicySpec("learned-test", 1, 1, 20.0, 6)

    def reset(self, *, task_id: str, seed: int) -> None:
        del task_id, seed

    def infer(self, observations):
        observation = observations[-1]
        return (
            ActionFrame(
                observation.timestamp_ns,
                observation.timestamp_ns,
                observation.timestamp_ns + 1,
                "learned:test",
                base_linear=-0.1,
                arm_command=(0.0,) * 6,
                policy_version="test",
            ),
        )

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _Task:
    task_id: str = TASK_ID
    instruction: str = "test"
    max_steps: int = 4


def test_visual_dagger_labels_policy_visited_states_with_expert_actions(tmp_path) -> None:
    base_root = tmp_path / "base"
    builder = VisualDatasetBuilder(
        base_root,
        "macro-base",
        task_id=TASK_ID,
        instruction="test",
        image_size=(2, 2),
        action_history=1,
    )
    builder.declare_phase_order(("approach_object_cup",))
    policy_input = extract_formal_policy_input(
        _observation(0),
        instruction_id=0,
        action_history=[np.zeros(9)],
        image_width=2,
        image_height=2,
    )
    action = formal_action_vector(
        ActionFrame(0, 0, 1, "expert", base_linear=0.25, arm_command=(0.0,) * 6)
    )
    builder.write_episode(
        "base-0",
        1,
        [VisualBehaviorSample(0, policy_input, action, phase="approach_object_cup")],
    )
    base = builder.seal()

    path = aggregate_visual_policy_dataset(
        tmp_path / "aggregated",
        "dagger-1",
        base,
        _Task(),
        _Environment,
        lambda environment: _Expert(),
        _Policy(),
        [10],
        sample_stride=1,
    )
    dataset = load_visual_dataset(path)

    assert dataset.manifest["episode_count"] == 2
    assert dataset.manifest["metadata"]["expert_action_probability"] == 0.0
    assert len(dataset) == 5
    np.testing.assert_allclose(dataset.actions[-4:, 0], 0.25)
    np.testing.assert_allclose(dataset.inputs["action_history"][-1, -1, 0], -0.1)
