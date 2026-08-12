from __future__ import annotations

import numpy as np

from hwr.data.autonomous_trajectory import (
    AutonomousEpisode,
    AutonomousTrajectoryDatasetBuilder,
)
from hwr.data.foundation_cache import FoundationFeatureCache
from hwr.data.foundation_features import (
    materialize_language_features,
    materialize_visual_features,
)
from hwr.data.foundation_loading import (
    FoundationPreparedFeatures,
    FoundationSequenceBatchLoader,
)
from hwr.perception.contracts import (
    DUAL_ARM_CAMERA_IDS,
    CameraCalibration,
    PinholeIntrinsics,
)
from hwr.perception.foundation import (
    DenseVisualFeatures,
    FoundationModelLock,
    SemanticLanguageFeatures,
    WeightArtifact,
    language_source_sha256,
)
from hwr.perception.high_resolution import (
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
)
from hwr.perception.student import VisualStudentConfig


class _VisionProvider:
    def __init__(self, role: str, dimension: int, marker: str) -> None:
        self._lock = FoundationModelLock(
            f"fixture/{marker}",
            marker * 40,
            role,
            "Apache-2.0",
            dimension,
            (WeightArtifact(f"{marker}.bin", marker * 64, 1),),
        )

    @property
    def model_lock(self):
        return self._lock

    def encode_vision(self, rgb, camera_valid, source_sha256):
        values = np.full((3, 2, 2, self._lock.output_dimension), 0.25, np.float32)
        valid = np.broadcast_to(camera_valid[:, None, None], (3, 2, 2)).copy()
        return DenseVisualFeatures(
            values, valid, self._lock.lock_sha256, source_sha256
        )


class _LanguageProvider:
    def __init__(self) -> None:
        self._lock = FoundationModelLock(
            "fixture/language",
            "c" * 40,
            "language",
            "Apache-2.0",
            6,
            (WeightArtifact("c.bin", "c" * 64, 1),),
        )

    @property
    def model_lock(self):
        return self._lock

    def encode_language(self, text, locale):
        return SemanticLanguageFeatures(
            np.arange(1, 7, dtype=np.float32),
            self._lock.lock_sha256,
            language_source_sha256(text, locale),
        )


def _preprocessor() -> HighResolutionVisionPreprocessor:
    size = 160
    calibrations = {
        name: CameraCalibration(
            f"fixture-{name}",
            name,
            PinholeIntrinsics(size, size, 100.0, 100.0, 80.0, 80.0),
            tuple(np.eye(4).reshape(-1)),
        )
        for name in DUAL_ARM_CAMERA_IDS
    }
    return HighResolutionVisionPreprocessor(
        HighResolutionVisionConfig(), calibrations
    )


def _dataset(tmp_path, fingerprint: str):
    observations = 3
    transitions = observations - 1
    arrays = {
        "rgb_uint8": np.zeros((observations, 3, 160, 160, 3), np.uint8),
        "raw_head_depth_m": np.ones((observations, 160, 160), np.float32),
        "head_depth_valid": np.ones((observations, 160, 160), np.bool_),
        "camera_validity": np.ones((observations, 4), np.bool_),
        "frame_timestamps_ns": np.arange(observations)[:, None].repeat(4, 1),
        "proprioception": np.zeros((observations, 31), np.float32),
        "observation_source_sha256": np.asarray(
            [f"{index + 1:064x}" for index in range(observations)]
        ),
        "actor_proposal": np.zeros((transitions, 16), np.float32),
        "executed_action": np.ones((transitions, 16), np.float32) * 0.1,
        "reward": np.asarray([0.0, 1.0], np.float32),
        "terminated": np.asarray([False, True]),
        "truncated": np.zeros(transitions, np.bool_),
        "safety_cost": np.asarray([0.0, 1.0], np.float32),
        "action_source": np.asarray(["rl_actor"] * transitions),
        "intrinsics": np.ones((4, 4), np.float32),
        "robot_from_camera": np.repeat(np.eye(4, dtype=np.float32)[None], 4, 0),
    }
    episode = AutonomousEpisode(
        "episode-1",
        "fixture/v1",
        7,
        "双手移动容器",
        "zh-CN",
        "fixture-env/v1",
        "abc123",
        fingerprint,
        ("lateral_reflection",),
        arrays,
    )
    builder = AutonomousTrajectoryDatasetBuilder(tmp_path, "dataset")
    builder.write_episode(episode)
    return builder.seal()


def test_materialized_foundation_features_build_continuous_training_batch(tmp_path) -> None:
    preprocessor = _preprocessor()
    dataset = _dataset(tmp_path, preprocessor.fingerprint)
    cache = FoundationFeatureCache(tmp_path / "cache")
    siglip = materialize_visual_features(
        dataset,
        cache,
        preprocessor,
        _VisionProvider("vision_language", 7, "a"),
        tmp_path / "siglip.json",
    )
    dinov2 = materialize_visual_features(
        dataset,
        cache,
        preprocessor,
        _VisionProvider("dense_vision", 5, "b"),
        tmp_path / "dinov2.json",
    )
    language = materialize_language_features(
        dataset, cache, _LanguageProvider(), tmp_path / "language.json"
    )
    student = VisualStudentConfig(
        image_size=160,
        visual_history=2,
        backbone_dimensions=(8, 12, 16, 24),
        backbone_depths=(1, 1, 1, 1),
        feature_dimension=8,
        state_queries=2,
        attention_heads=2,
        fusion_layers=1,
        temporal_layers=1,
        formal=False,
    )
    loader = FoundationSequenceBatchLoader(
        dataset,
        cache,
        preprocessor,
        student,
        FoundationPreparedFeatures(siglip, dinov2, language),
        transitions=2,
    )

    batch = loader.build([0])

    assert batch.sequence_batch_size == 1
    assert batch.observation_count == 3
    assert batch.student_inputs["rgb"].shape == (3, 2, 3, 3, 160, 160)
    assert batch.visual_targets.siglip.shape == (3, 2, 3, 2, 2, 7)
    assert batch.visual_targets.dinov2.shape == (3, 2, 3, 2, 2, 5)
    assert batch.language_features.shape == (1, 6)
    assert batch.executed_actions.shape == (1, 2, 16)
    assert batch.continues.tolist() == [[1.0, 0.0]]
    assert batch.visual_targets.correspondences.shape[1] == 10
