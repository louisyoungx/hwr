"""Local behavior policy training and model registry."""

from hwr.train.registry import load_policy, save_training_result
from hwr.train.trainer import TrainingConfig, TrainingResult, train_behavior_policy
from hwr.train.visual_registry import load_visual_policy, save_visual_training_result
from hwr.train.visual_knn import load_visual_knn_policy, save_visual_knn_policy
from hwr.train.visual_trainer import (
    VisualTrainingConfig,
    VisualTrainingResult,
    train_visual_policy,
)
from hwr.train.vla_registry import (
    load_deployable_vla_actor,
    save_vla_actor_checkpoint,
    save_vla_behavior_result,
)
from hwr.train.vla_trainer import (
    VLABehaviorTrainingConfig,
    VLABehaviorTrainingResult,
    train_vla_behavior_cloning,
)
from hwr.train.asymmetric_rl import (
    AsymmetricActorCriticTrainer,
    AsymmetricRLBatch,
    AsymmetricRLConfig,
)
from hwr.train.asymmetric_replay import AsymmetricReplayBuffer
from hwr.train.action_exploration import (
    TemporalActionExplorer,
    TemporalExplorationConfig,
)
from hwr.train.asymmetric_registry import (
    load_asymmetric_training_checkpoint,
    save_asymmetric_training_checkpoint,
)
from hwr.train.curriculum import AutomaticCurriculum, CurriculumConfig, CurriculumUpdate
from hwr.train.goal_replay import (
    GoalConditionedReplayBuffer,
    GoalEpisode,
    GoalReplayAddResult,
    hindsight_relabel,
    mirror_batch,
)
from hwr.train.bimanual_training import (
    BimanualRLTrainingConfig,
    BimanualTrainingResult,
    BimanualTrainingRunner,
    TrainingEpisodeRecord,
    load_default_bimanual_training_catalogs,
)
from hwr.train.bimanual_registry import (
    load_bimanual_actor,
    resume_bimanual_training_run,
    save_bimanual_training_run,
    verify_bimanual_training_run,
)

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "VisualTrainingConfig",
    "VisualTrainingResult",
    "load_policy",
    "load_visual_policy",
    "load_visual_knn_policy",
    "save_training_result",
    "save_visual_training_result",
    "save_visual_knn_policy",
    "train_behavior_policy",
    "train_visual_policy",
    "VLABehaviorTrainingConfig",
    "VLABehaviorTrainingResult",
    "load_deployable_vla_actor",
    "save_vla_behavior_result",
    "train_vla_behavior_cloning",
    "AsymmetricActorCriticTrainer",
    "AsymmetricRLBatch",
    "AsymmetricRLConfig",
    "AsymmetricReplayBuffer",
    "TemporalActionExplorer",
    "TemporalExplorationConfig",
    "load_asymmetric_training_checkpoint",
    "save_asymmetric_training_checkpoint",
    "save_vla_actor_checkpoint",
    "AutomaticCurriculum",
    "CurriculumConfig",
    "CurriculumUpdate",
    "GoalConditionedReplayBuffer",
    "GoalEpisode",
    "GoalReplayAddResult",
    "hindsight_relabel",
    "mirror_batch",
    "BimanualRLTrainingConfig",
    "BimanualTrainingResult",
    "BimanualTrainingRunner",
    "TrainingEpisodeRecord",
    "load_default_bimanual_training_catalogs",
    "load_bimanual_actor",
    "resume_bimanual_training_run",
    "save_bimanual_training_run",
    "verify_bimanual_training_run",
]
