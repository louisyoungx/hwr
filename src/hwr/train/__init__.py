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
from hwr.train.vla_registry import load_deployable_vla_actor, save_vla_behavior_result
from hwr.train.vla_trainer import (
    VLABehaviorTrainingConfig,
    VLABehaviorTrainingResult,
    train_vla_behavior_cloning,
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
]
