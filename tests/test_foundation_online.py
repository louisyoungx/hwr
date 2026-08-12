from __future__ import annotations

from dataclasses import replace

import numpy as np

from hwr.core.embodied import (
    DualArmObservation,
    DualArmProprioception,
    NaturalLanguageInstruction,
)
from hwr.core.runtime import LegalEnvironmentTransform, RuntimeStepOutcome
from hwr.core.types import CameraFrame, EpisodeResult
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
from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
)
from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.policy.latent_value import LatentValueModel
from hwr.train.foundation_online import (
    FoundationOnlineTrainingConfig,
    FoundationOnlineTrainingRunner,
    FoundationProviderFactories,
    FoundationTaskInterface,
)
from hwr.train.foundation_setup import FoundationLearningStack
from hwr.train.foundation_trainer import (
    FoundationTrainerConfig,
    FoundationWorldModelTrainer,
)
from hwr.train.imagination_rl import ImaginationRLConfig
from hwr.world_model import (
    ActionConditionedWorldModel,
    WorldModelConfig,
    WorldModelLoss,
    WorldModelLossConfig,
)


TASK_IDS = ("fixture-a/v1", "fixture-b/v1", "fixture-c/v1")


class _VisionProvider:
    def __init__(self, role: str, dimension: int, marker: str) -> None:
        self._lock = FoundationModelLock(
            f"fixture/{marker}", marker * 40, role, "Apache-2.0", dimension,
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
            "fixture/language", "c" * 40, "language", "Apache-2.0", 6,
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


def _observation(task_id: str, sequence: int) -> DualArmObservation:
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
        (0.0,) * 6, (0.0,) * 6, (0.0,) * 6, (0.0,) * 6,
        0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0),
    )
    return DualArmObservation(
        timestamp,
        sequence,
        task_id,
        NaturalLanguageInstruction(f"执行 {task_id} 的双臂任务"),
        proprioception,
        cameras,
    )


class _Backend:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.sequence = 0
        self._result = None

    def reset(self, *, seed: int, task_id: str):
        del seed
        if task_id != self.task_id:
            raise ValueError("fixture task differs")
        self.sequence = 0
        self._result = None
        return _observation(task_id, 0)

    def observe(self):
        return _observation(self.task_id, self.sequence)

    def apply(self, frame):
        self.sequence += 1
        terminal = self.sequence == 2
        if terminal:
            self._result = EpisodeResult(True, "fixture_success", self.sequence)
        return RuntimeStepOutcome(
            _observation(self.task_id, self.sequence),
            reward=float(self.sequence),
            terminated=terminal,
            info={"applied_action": replace(frame), "safety_intervened": False},
        )

    def result(self):
        return self._result

    def legal_environment_transforms(self):
        return (LegalEnvironmentTransform("lateral_reflection"),)

    def close(self):
        pass


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
    return HighResolutionVisionPreprocessor(HighResolutionVisionConfig(), calibrations)


def _stack() -> FoundationLearningStack:
    visual_config = VisualStudentConfig(
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
    world_config = WorldModelConfig(
        visual_dimension=8,
        language_dimension=6,
        proprioception_dimension=31,
        action_dimension=16,
        observation_embedding_dimension=12,
        deterministic_dimension=10,
        stochastic_variables=3,
        stochastic_classes=4,
        hidden_dimension=16,
        prior_ensemble=2,
        reward_bins=11,
        formal=False,
    )
    student = VisualStudentModel(visual_config)
    visual_objective = VisualFoundationObjectives(
        VisualObjectiveConfig(
            student_dimension=8, siglip_dimension=7, dinov2_dimension=5
        )
    )
    world = ActionConditionedWorldModel(world_config)
    actor = LatentActor(
        LatentActorConfig(
            world_config.feature_dimension,
            hidden_dimension=16,
            hidden_layers=2,
            formal=False,
        )
    )
    value = LatentValueModel(
        world_config.feature_dimension, bins=11, hidden_dimension=16, hidden_layers=2
    )
    trainer = FoundationWorldModelTrainer(
        student,
        visual_objective,
        world,
        WorldModelLoss(world_config, WorldModelLossConfig()),
        actor,
        value,
        ImaginationRLConfig(horizon=2, value_bins=11, value_symlog_limit=5.0),
        FoundationTrainerConfig(),
    )
    return FoundationLearningStack(trainer, LatentActionScaling())


def test_online_runner_uses_one_loop_for_random_then_current_rl_actions(tmp_path) -> None:
    tasks = {name: FoundationTaskInterface(name, 2) for name in TASK_IDS}
    providers = FoundationProviderFactories(
        lambda: _VisionProvider("vision_language", 7, "a"),
        lambda: _VisionProvider("dense_vision", 5, "b"),
        _LanguageProvider,
    )
    runner = FoundationOnlineTrainingRunner(
        tasks,
        lambda task_id, width, height: _Backend(task_id),
        _preprocessor(),
        providers,
        _stack(),
        FoundationOnlineTrainingConfig(
            episodes=6,
            initial_random_episodes=3,
            collection_episodes_per_cycle=3,
            updates_per_cycle=1,
            batch_size=1,
            sequence_transitions=2,
            camera_width=160,
            camera_height=160,
            replay_transition_capacity=6,
            published_checkpoint_retention=1,
            seed=7,
        ),
        tmp_path / "run",
        source_commit="abc123",
    )

    result = runner.train()

    assert result.update_count == 2
    assert {record.task_id for record in result.records} == set(TASK_IDS)
    assert [record.action_source for record in result.records[:3]] == [
        "random_rl_exploration"
    ] * 3
    assert [record.action_source for record in result.records[3:]] == [
        "rl_actor"
    ] * 3
    assert result.latest_checkpoint.is_dir()
    assert result.latest_deployment.is_dir()
    assert runner.store.manifest["transition_count"] <= 6
    assert len(list((tmp_path / "run/checkpoints").glob("update-*"))) == 1
    assert len(list((tmp_path / "run/deployments").glob("update-*"))) == 1
    assert runner.task_sampler.audit()["distance_thresholds"] is False
