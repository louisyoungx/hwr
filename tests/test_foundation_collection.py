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
    IntrinsicRLActorActionSource,
)
from hwr.train.foundation_exploration import (
    RandomRLActionSource,
    RandomRLExplorationConfig,
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
        NaturalLanguageInstruction("Move the object with both hands"),
        proprioception,
        cameras,
    )


class _Backend:
    def __init__(
        self, *, intervention_sequence: int = 1, terminal_sequence: int = 2
    ) -> None:
        self.sequence = 0
        self._result = None
        self.intervention_sequence = intervention_sequence
        self.terminal_sequence = terminal_sequence

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
        changed = self.sequence == self.intervention_sequence
        action = (
            DualArmAction(0.0, 0.0, (0.0,) * 6, (0.0,) * 6, 0.0, 0.0)
            if changed
            else frame.action
        )
        applied = replace(frame, action=action, source="safety" if changed else frame.source)
        terminal = self.sequence == self.terminal_sequence
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
    assert episode.arrays["safety_intervention"].tolist() == [1.0, 0.0]
    assert episode.arrays["action_source"].tolist() == [
        "random_rl_exploration", "random_rl_exploration"
    ]
    assert episode.arrays["terminated"].tolist() == [False, True]
    assert episode.legal_transform_ids == ("lateral_reflection",)
    assert episode.metadata["action_process"] == {
        "schema_version": "hwr.correlated-random-rl/v1",
        "motion_correlation": 0.96,
        "gripper_flip_probability": 0.05,
        "observation_conditioned": False,
        "task_conditioned": False,
    }
    assert len(episode.metadata["interaction_trace"]) == 2
    assert episode.metadata["interaction_audit"] == {
        "left_contact_steps": 0.0,
        "right_contact_steps": 0.0,
        "simultaneous_contact_steps": 0.0,
        "maximum_controlled_rigid_displacement": 0.0,
        "maximum_controlled_articulation_displacement": 0.0,
        "severe_collision_count": 0.0,
    }


def test_collector_stops_after_retaining_safety_intervention_evidence() -> None:
    preprocessor = HighResolutionVisionPreprocessor(
        HighResolutionVisionConfig(), _calibrations()
    )
    collector = AutonomousEpisodeCollector(
        preprocessor,
        AutonomousCollectionConfig(
            "fixture-env/v1",
            "abc123",
            maximum_steps=5,
            stop_after_safety_intervention=True,
            minimum_stop_steps=2,
        ),
    )
    episode = collector.collect(
        _Backend(intervention_sequence=1, terminal_sequence=4),
        RandomRLActionSource(LatentActionScaling()),
        task_id="fixture/v1",
        seed=9,
    )

    assert episode.arrays["executed_action"].shape == (2, 16)
    assert episode.arrays["safety_intervention"].tolist() == [1.0, 0.0]
    assert episode.arrays["terminated"].tolist() == [False, False]
    assert episode.arrays["truncated"].tolist() == [False, True]
    assert episode.metadata["result_reason"] == "safety_intervention_evidence"
    assert (
        episode.metadata["collection_stop_reason"]
        == "safety_intervention_evidence"
    )


def test_collector_bounds_retained_episode_buffers() -> None:
    preprocessor = HighResolutionVisionPreprocessor(
        HighResolutionVisionConfig(), _calibrations()
    )
    collector = AutonomousEpisodeCollector(
        preprocessor,
        AutonomousCollectionConfig(
            "fixture-env/v1",
            "abc123",
            maximum_steps=5,
            retained_transition_capacity=2,
        ),
    )
    episode = collector.collect(
        _Backend(intervention_sequence=1, terminal_sequence=5),
        RandomRLActionSource(LatentActionScaling()),
        task_id="fixture/v1",
        seed=9,
    )

    assert episode.arrays["executed_action"].shape == (2, 16)
    assert episode.arrays["rgb_uint8"][:, 0, 0, 0, 0].tolist() == [3, 4, 5]
    assert episode.arrays["safety_intervention"].tolist() == [0.0, 0.0]
    assert episode.metadata["collection_transition_count"] == 5
    assert episode.metadata["retained_transition_count"] == 2
    assert len(episode.metadata["interaction_trace"]) == 2


def test_random_rl_source_is_seeded_and_observation_independent() -> None:
    first = RandomRLActionSource(LatentActionScaling())
    second = RandomRLActionSource(LatentActionScaling())
    first.reset(task_id="a", seed=7)
    second.reset(task_id="b", seed=7)

    assert first.propose(_observation(0, "a")).vector() == second.propose(
        _observation(0, "b")
    ).vector()


def test_random_rl_source_has_persistent_motion_and_gripper_dwell() -> None:
    source = RandomRLActionSource(
        LatentActionScaling(),
        RandomRLExplorationConfig(
            motion_correlation=0.96,
            gripper_flip_probability=0.05,
        ),
    )
    source.reset(task_id="fixture/v1", seed=19)
    actions = np.asarray(
        [source.propose(_observation(step % 256)).vector() for step in range(512)]
    )

    motion_correlation = np.corrcoef(
        actions[:-1, :14].reshape(-1),
        actions[1:, :14].reshape(-1),
    )[0, 1]
    gripper_flip_rate = np.not_equal(
        actions[:-1, 14:], actions[1:, 14:]
    ).mean()

    assert motion_correlation > 0.90
    assert 0.02 < gripper_flip_rate < 0.08
    assert np.all(np.abs(actions[:, 0]) <= 0.18)
    assert set(np.unique(actions[:, 14:])) == {0.0, 1.0}


def test_random_rl_exploration_config_rejects_iid_gripper_flicker() -> None:
    with np.testing.assert_raises(ValueError):
        RandomRLExplorationConfig(motion_correlation=1.0)
    with np.testing.assert_raises(ValueError):
        RandomRLExplorationConfig(gripper_flip_probability=0.0)


def test_intrinsic_rl_source_declares_no_environment_reward() -> None:
    policy = type("Policy", (), {"policy_id": "intrinsic-fixture"})()
    source = IntrinsicRLActorActionSource(policy)

    assert source.action_source == "intrinsic_rl_actor"
    assert source.action_process["environment_reward"] is False
    assert source.action_process["task_conditioned"] is False
