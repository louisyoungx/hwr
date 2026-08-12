from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hwr.core.embodied import (
    DualArmObservation,
    DualArmProprioception,
    NaturalLanguageInstruction,
)
from hwr.perception.high_resolution import HighResolutionVision
from hwr.perception.language_cache import StaticLanguageFeatureResolver
from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.policy.foundation_runtime import FoundationWorldModelPolicy
from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.world_model import (
    ActionConditionedWorldModel,
    DeployableWorldModelStateFilter,
    WorldModelConfig,
)


class _FixturePreprocessor:
    config = SimpleNamespace(student_image_size=32)

    def preprocess(self, observation):
        del observation
        size = 32
        return HighResolutionVision(
            teacher_rgb=np.zeros((3, 224, 224, 3), np.float32),
            student_rgb=np.zeros((3, size, size, 3), np.float32),
            student_head_depth_m=np.ones((size, size), np.float32),
            student_head_depth_valid=np.ones((size, size), np.bool_),
            camera_validity=np.ones(4, np.bool_),
            frame_timestamps_ns=np.arange(4, dtype=np.int64),
            student_intrinsics=np.ones((4, 4), np.float32),
            robot_from_camera=np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0),
            preprocess_fingerprint="a" * 64,
            source_sha256="b" * 64,
        )


def _observation() -> DualArmObservation:
    proprioception = DualArmProprioception(
        (0.0,) * 6, (0.0,) * 6, (0.0,) * 6, (0.0,) * 6,
        0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0),
    )
    return DualArmObservation(
        100, 0, "fixture/v1", NaturalLanguageInstruction("双手搬运容器"),
        proprioception, (),
    )


def _policy() -> FoundationWorldModelPolicy:
    visual_config = VisualStudentConfig(
        image_size=32,
        visual_history=2,
        backbone_dimensions=(16, 24, 32, 48),
        backbone_depths=(1, 1, 1, 1),
        feature_dimension=16,
        state_queries=2,
        attention_heads=4,
        fusion_layers=1,
        temporal_layers=1,
        formal=False,
    )
    world_config = WorldModelConfig(
        visual_dimension=16,
        language_dimension=6,
        proprioception_dimension=31,
        observation_embedding_dimension=16,
        deterministic_dimension=16,
        stochastic_variables=4,
        stochastic_classes=4,
        hidden_dimension=32,
        prior_ensemble=3,
        reward_bins=21,
    )
    training_world = ActionConditionedWorldModel(world_config)
    world = DeployableWorldModelStateFilter.from_world_model(training_world)
    actor = LatentActor(
        LatentActorConfig(
            world_config.feature_dimension, hidden_dimension=32, hidden_layers=2
        )
    )
    resolver = StaticLanguageFeatureResolver(
        {("zh-CN", "双手搬运容器"): np.ones(6, np.float32)},
        encoder_lock_sha256="c" * 64,
        output_dimension=6,
    )
    return FoundationWorldModelPolicy(
        VisualStudentModel(visual_config),
        world,
        actor,
        _FixturePreprocessor(),
        resolver,
        LatentActionScaling(),
        policy_id="fixture-policy",
    )


def test_foundation_runtime_requires_actual_action_feedback_between_inferences() -> None:
    policy = _policy()
    policy.reset(task_id="fixture/v1", seed=3)

    first = policy.infer((_observation(),))
    with pytest.raises(RuntimeError, match="feedback"):
        policy.infer((_observation(),))
    policy.record_applied_action(first.actions[0])
    second = policy.infer((_observation(),))

    assert len(first.actions[0].vector()) == 16
    assert len(second.actions[0].vector()) == 16
    assert abs(first.actions[0].base_linear) <= 0.18
    assert 0.0 <= first.actions[0].left_gripper <= 1.0


def test_foundation_runtime_has_no_teacher_critic_or_reward_component() -> None:
    policy = _policy()

    forbidden = {"teacher", "critic", "reward", "planner", "expert"}
    assert forbidden.isdisjoint(policy.__dict__)
    assert not any("reward" in name for name, _ in policy.world_model.named_modules())
    assert not any("head" in name for name, _ in policy.world_model.named_modules())
    assert policy.spec().action_horizon == 1
    assert policy.spec().observation_history == 2


def test_static_language_resolver_rejects_unprepared_instruction() -> None:
    policy = _policy()
    policy.reset(task_id="fixture/v1", seed=3)
    observation = _observation()
    observation = DualArmObservation(
        **{**observation.__dict__, "instruction": NaturalLanguageInstruction("未缓存指令")}
    )

    with pytest.raises(KeyError, match="not prepared"):
        policy.infer((observation,))
