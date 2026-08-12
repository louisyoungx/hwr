from __future__ import annotations

from dataclasses import replace

import numpy as np

from hwr.core.embodied import (
    DualArmAction,
    DualArmObservation,
    DualArmProprioception,
    NaturalLanguageInstruction,
)
from hwr.core.runtime import LegalEnvironmentTransform, RuntimeStepOutcome
from hwr.core.types import CameraFrame, EpisodeEvent, EpisodeResult
from hwr.perception.contracts import (
    DUAL_ARM_CAMERA_IDS,
    CameraCalibration,
    PinholeIntrinsics,
)
from hwr.perception.high_resolution import (
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
)
from hwr.policy.latent_actions import LatentActionScaling
from hwr.train.foundation_collection import (
    AutonomousCollectionConfig,
    AutonomousEpisodeCollector,
    RandomRLActionSource,
)


def _calibrations(size: int = 160) -> dict[str, CameraCalibration]:
    return {
        name: CameraCalibration(
            f"fixture-{name}",
            name,
            PinholeIntrinsics(size, size, 100.0, 100.0, size / 2, size / 2),
            tuple(np.eye(4).reshape(-1)),
        )
        for name in DUAL_ARM_CAMERA_IDS
    }


def _observation(sequence: int, task_id: str = "fixture/v1") -> DualArmObservation:
    size = 160
    timestamp = sequence * 50_000_000
    rgb = np.full((size, size, 3), sequence, np.uint8).tobytes()
    depth = np.ones((size, size), np.float32).tobytes()
    cameras = tuple(
        CameraFrame(
            name,
            timestamp,
            sequence,
            size,
            size,
            "depth32f" if name == "head_depth" else "rgb8",
            payload=depth if name == "head_depth" else rgb,
        )
        for name in DUAL_ARM_CAMERA_IDS
    )
    proprioception = DualArmProprioception(
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        0.0,
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0),
    )
    return DualArmObservation(
        timestamp,
        sequence,
        task_id,
        NaturalLanguageInstruction("双手移动物体"),
        proprioception,
        cameras,
    )


class _Backend:
    def __init__(self) -> None:
        self.sequence = 0
        self._result = None

    def reset(self, *, seed: int, task_id: str):
        del seed
        self.sequence = 0
        self.task_id = task_id
        self._result = None
        return _observation(0, task_id)

    def observe(self):
        return _observation(self.sequence, self.task_id)

    def apply(self, frame):
        self.sequence += 1
        changed = self.sequence == 1
        action = (
            DualArmAction(0.0, 0.0, (0.0,) * 6, (0.0,) * 6, 0.0, 0.0)
            if changed
            else frame.action
        )
        applied = replace(frame, action=action, source="safety" if changed else frame.source)
        terminal = self.sequence == 2
        if terminal:
            self._result = EpisodeResult(True, "fixture_success", self.sequence)
        events = (
            EpisodeEvent(self.sequence, "action_clamped", "safety"),
        ) if changed else ()
        return RuntimeStepOutcome(
            _observation(self.sequence, self.task_id),
            reward=float(self.sequence),
            terminated=terminal,
            events=events,
            info={"applied_action": applied, "safety_intervened": changed},
        )

    def legal_environment_transforms(self):
        return (LegalEnvironmentTransform("lateral_reflection"),)

    def result(self):
        return self._result

    def close(self):
        pass


def test_collector_records_raw_observations_and_actual_safety_actions() -> None:
    preprocessor = HighResolutionVisionPreprocessor(
        HighResolutionVisionConfig(), _calibrations()
    )
    collector = AutonomousEpisodeCollector(
        preprocessor,
        AutonomousCollectionConfig("fixture-env/v1", "abc123", maximum_steps=5),
    )
    episode = collector.collect(
        _Backend(),
        RandomRLActionSource(LatentActionScaling()),
        task_id="fixture/v1",
        seed=9,
    )

    assert episode.arrays["rgb_uint8"].shape == (3, 3, 160, 160, 3)
    assert episode.arrays["raw_head_depth_m"].shape == (3, 160, 160)
    assert episode.arrays["executed_action"].shape == (2, 16)
    assert not np.array_equal(
        episode.arrays["actor_proposal"][0], episode.arrays["executed_action"][0]
    )
    assert episode.arrays["safety_cost"].tolist() == [1.0, 0.0]
    assert episode.arrays["action_source"].tolist() == [
        "random_rl_exploration", "random_rl_exploration"
    ]
    assert episode.arrays["terminated"].tolist() == [False, True]
    assert episode.legal_transform_ids == ("lateral_reflection",)


def test_random_rl_source_is_seeded_and_observation_independent() -> None:
    first = RandomRLActionSource(LatentActionScaling())
    second = RandomRLActionSource(LatentActionScaling())
    first.reset(task_id="a", seed=7)
    second.reset(task_id="b", seed=7)

    assert first.propose(_observation(0, "a")).vector() == second.propose(
        _observation(0, "b")
    ).vector()
