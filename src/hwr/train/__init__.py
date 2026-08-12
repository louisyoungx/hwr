"""Training APIs with dependency-light lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "TrainingConfig": ("hwr.train.trainer", "TrainingConfig"),
    "TrainingResult": ("hwr.train.trainer", "TrainingResult"),
    "train_behavior_policy": ("hwr.train.trainer", "train_behavior_policy"),
    "load_policy": ("hwr.train.registry", "load_policy"),
    "save_training_result": ("hwr.train.registry", "save_training_result"),
    "VisualTrainingConfig": ("hwr.train.visual_trainer", "VisualTrainingConfig"),
    "VisualTrainingResult": ("hwr.train.visual_trainer", "VisualTrainingResult"),
    "train_visual_policy": ("hwr.train.visual_trainer", "train_visual_policy"),
    "load_visual_policy": ("hwr.train.visual_registry", "load_visual_policy"),
    "save_visual_training_result": ("hwr.train.visual_registry", "save_visual_training_result"),
    "load_visual_knn_policy": ("hwr.train.visual_knn", "load_visual_knn_policy"),
    "save_visual_knn_policy": ("hwr.train.visual_knn", "save_visual_knn_policy"),
    "VLABehaviorTrainingConfig": ("hwr.train.vla_trainer", "VLABehaviorTrainingConfig"),
    "VLABehaviorTrainingResult": ("hwr.train.vla_trainer", "VLABehaviorTrainingResult"),
    "train_vla_behavior_cloning": ("hwr.train.vla_trainer", "train_vla_behavior_cloning"),
    "load_deployable_vla_actor": ("hwr.train.vla_registry", "load_deployable_vla_actor"),
    "save_vla_actor_checkpoint": ("hwr.train.vla_registry", "save_vla_actor_checkpoint"),
    "save_vla_behavior_result": ("hwr.train.vla_registry", "save_vla_behavior_result"),
    "AsymmetricActorCriticTrainer": ("hwr.train.asymmetric_rl", "AsymmetricActorCriticTrainer"),
    "AsymmetricRLBatch": ("hwr.train.asymmetric_rl", "AsymmetricRLBatch"),
    "AsymmetricRLConfig": ("hwr.train.asymmetric_rl", "AsymmetricRLConfig"),
    "AsymmetricReplayBuffer": ("hwr.train.asymmetric_replay", "AsymmetricReplayBuffer"),
    "TemporalActionExplorer": ("hwr.train.action_exploration", "TemporalActionExplorer"),
    "TemporalExplorationConfig": ("hwr.train.action_exploration", "TemporalExplorationConfig"),
    "load_asymmetric_training_checkpoint": (
        "hwr.train.asymmetric_registry",
        "load_asymmetric_training_checkpoint",
    ),
    "save_asymmetric_training_checkpoint": (
        "hwr.train.asymmetric_registry",
        "save_asymmetric_training_checkpoint",
    ),
    "AutomaticCurriculum": ("hwr.train.curriculum", "AutomaticCurriculum"),
    "CurriculumConfig": ("hwr.train.curriculum", "CurriculumConfig"),
    "CurriculumUpdate": ("hwr.train.curriculum", "CurriculumUpdate"),
    "LearningFrontierCandidate": ("hwr.train.learning_frontier", "LearningFrontierCandidate"),
    "LearningFrontierConfig": ("hwr.train.learning_frontier", "LearningFrontierConfig"),
    "LearningFrontierEntry": ("hwr.train.learning_frontier", "LearningFrontierEntry"),
    "LearningSignal": ("hwr.train.learning_frontier", "LearningSignal"),
    "TaskAgnosticLearningFrontier": (
        "hwr.train.learning_frontier",
        "TaskAgnosticLearningFrontier",
    ),
    "AutonomousEpisode": ("hwr.train.autonomous_replay", "AutonomousEpisode"),
    "AutonomousReplayAddResult": ("hwr.train.autonomous_replay", "AutonomousReplayAddResult"),
    "AutonomousReplayBuffer": ("hwr.train.autonomous_replay", "AutonomousReplayBuffer"),
    "transform_batch": ("hwr.train.autonomous_replay", "transform_batch"),
    "TaskPartitionedAutonomousReplayBuffer": (
        "hwr.train.task_replay",
        "TaskPartitionedAutonomousReplayBuffer",
    ),
    "OutcomeAdaptiveTaskSampler": ("hwr.train.task_sampling", "OutcomeAdaptiveTaskSampler"),
    "OutcomeAdaptiveTaskSamplingConfig": (
        "hwr.train.task_sampling",
        "OutcomeAdaptiveTaskSamplingConfig",
    ),
    "TaskOutcome": ("hwr.train.task_sampling", "TaskOutcome"),
    "NStepTargets": ("hwr.train.n_step", "NStepTargets"),
    "build_n_step_targets": ("hwr.train.n_step", "build_n_step_targets"),
    "TrainingEpisodeRecord": ("hwr.train.bimanual_records", "TrainingEpisodeRecord"),
    "BimanualRLTrainingConfig": ("hwr.train.bimanual_config", "BimanualRLTrainingConfig"),
    "BimanualTrainingResult": ("hwr.train.bimanual_training", "BimanualTrainingResult"),
    "BimanualTrainingRunner": ("hwr.train.bimanual_training", "BimanualTrainingRunner"),
    "fork_bimanual_training_run": ("hwr.train.bimanual_registry", "fork_bimanual_training_run"),
    "load_bimanual_actor": ("hwr.train.bimanual_registry", "load_bimanual_actor"),
    "resume_bimanual_training_run": ("hwr.train.bimanual_registry", "resume_bimanual_training_run"),
    "save_bimanual_live_progress": ("hwr.train.bimanual_registry", "save_bimanual_live_progress"),
    "save_bimanual_training_run": ("hwr.train.bimanual_registry", "save_bimanual_training_run"),
    "verify_bimanual_training_run": ("hwr.train.bimanual_registry", "verify_bimanual_training_run"),
    "ImaginationActorCritic": ("hwr.train.imagination_rl", "ImaginationActorCritic"),
    "ImaginationRLConfig": ("hwr.train.imagination_rl", "ImaginationRLConfig"),
    "lambda_returns": ("hwr.train.imagination_rl", "lambda_returns"),
    "optimize_imagination_step": ("hwr.train.imagination_rl", "optimize_imagination_step"),
    "FoundationTrainingBatch": (
        "hwr.train.foundation_batch",
        "FoundationTrainingBatch",
    ),
    "FoundationTrainerConfig": (
        "hwr.train.foundation_trainer",
        "FoundationTrainerConfig",
    ),
    "FoundationWorldModelTrainer": (
        "hwr.train.foundation_trainer",
        "FoundationWorldModelTrainer",
    ),
    "export_foundation_deployment": (
        "hwr.train.foundation_registry",
        "export_foundation_deployment",
    ),
    "load_foundation_deployment": (
        "hwr.train.foundation_registry",
        "load_foundation_deployment",
    ),
    "load_foundation_training_checkpoint": (
        "hwr.train.foundation_registry",
        "load_foundation_training_checkpoint",
    ),
    "save_foundation_training_checkpoint": (
        "hwr.train.foundation_registry",
        "save_foundation_training_checkpoint",
    ),
    "AutonomousCollectionConfig": (
        "hwr.train.foundation_collection",
        "AutonomousCollectionConfig",
    ),
    "AutonomousEpisodeCollector": (
        "hwr.train.foundation_collection",
        "AutonomousEpisodeCollector",
    ),
    "CurrentRLActorActionSource": (
        "hwr.train.foundation_collection",
        "CurrentRLActorActionSource",
    ),
    "RandomRLActionSource": (
        "hwr.train.foundation_collection",
        "RandomRLActionSource",
    ),
    "transform_foundation_batch": (
        "hwr.train.foundation_augmentation",
        "transform_foundation_batch",
    ),
    "build_foundation_learning_stack": (
        "hwr.train.foundation_setup",
        "build_foundation_learning_stack",
    ),
    "FoundationEnvironmentFactory": (
        "hwr.train.foundation_online",
        "FoundationEnvironmentFactory",
    ),
    "FoundationOnlineTrainingConfig": (
        "hwr.train.foundation_online",
        "FoundationOnlineTrainingConfig",
    ),
    "FoundationOnlineTrainingRunner": (
        "hwr.train.foundation_online",
        "FoundationOnlineTrainingRunner",
    ),
    "FoundationProviderFactories": (
        "hwr.train.foundation_online",
        "FoundationProviderFactories",
    ),
    "FoundationTaskInterface": (
        "hwr.train.foundation_online",
        "FoundationTaskInterface",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
