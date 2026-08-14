"""Independent Episode-level validation for the learned collision head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from hwr.data.foundation_loading import FoundationSequenceBatchLoader
from hwr.train.foundation_trainer import FoundationWorldModelTrainer


COLLISION_VALIDATION_SCHEMA = "hwr.foundation-collision-validation/v1"


@dataclass(frozen=True)
class CollisionValidationCriteria:
    minimum_positive_episodes_per_task: int = 8
    minimum_negative_episodes_per_task: int = 8
    minimum_recall: float = 0.80
    minimum_pr_auc: float = 0.50
    maximum_brier_score: float = 0.10

    def __post_init__(self) -> None:
        if min(
            self.minimum_positive_episodes_per_task,
            self.minimum_negative_episodes_per_task,
        ) < 0 or (
            self.minimum_positive_episodes_per_task
            + self.minimum_negative_episodes_per_task
            <= 0
        ):
            raise ValueError("collision validation Episode counts are invalid")
        if not 0.0 <= self.minimum_recall <= 1.0:
            raise ValueError("collision validation recall is invalid")
        if not 0.0 <= self.minimum_pr_auc <= 1.0:
            raise ValueError("collision validation PR-AUC is invalid")
        if not 0.0 <= self.maximum_brier_score <= 1.0:
            raise ValueError("collision validation Brier score is invalid")


def evaluate_foundation_collision_validation(
    trainer: FoundationWorldModelTrainer,
    loader: FoundationSequenceBatchLoader,
    task_ids: tuple[str, ...],
    criteria: CollisionValidationCriteria,
    *,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate one terminal sequence from every eligible holdout Episode."""
    if batch_size <= 0 or not task_ids:
        raise ValueError("collision validation batch configuration is invalid")
    selected = _terminal_windows_by_task(loader, task_ids)
    partitions = {
        task_id: _evaluate_partition(
            trainer, loader, selected[task_id], criteria, batch_size=batch_size
        )
        for task_id in task_ids
    }
    return {
        "schema_version": COLLISION_VALIDATION_SCHEMA,
        "passed": all(value["passed"] is True for value in partitions.values()),
        "criteria": asdict(criteria),
        "partitions": partitions,
        "window_selection": [
            _window_identity(loader.window_metadata(index))
            for task_id in task_ids
            for index in selected[task_id]
        ],
        "task_semantic_fields": [],
    }


def _terminal_windows_by_task(
    loader: FoundationSequenceBatchLoader, task_ids: tuple[str, ...]
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, dict[str, int]] = {task_id: {} for task_id in task_ids}
    for index in range(len(loader)):
        metadata = loader.window_metadata(index)
        task_id = str(metadata["task_id"])
        if task_id not in grouped:
            continue
        if int(metadata["transition_stop"]) != int(metadata["transition_count"]):
            continue
        episode_id = str(metadata["episode_id"])
        if episode_id in grouped[task_id]:
            raise ValueError("collision validation has duplicate terminal windows")
        grouped[task_id][episode_id] = index
    return {
        task_id: tuple(value for _, value in sorted(episodes.items()))
        for task_id, episodes in grouped.items()
    }


def _evaluate_partition(
    trainer: FoundationWorldModelTrainer,
    loader: FoundationSequenceBatchLoader,
    indices: tuple[int, ...],
    criteria: CollisionValidationCriteria,
    *,
    batch_size: int,
) -> dict[str, object]:
    probabilities: list[float] = []
    labels: list[float] = []
    for start in range(0, len(indices), batch_size):
        batch = loader.build(
            indices[start : start + batch_size], include_visual_targets=False
        )
        predicted = trainer.severe_collision_probabilities(batch).numpy()
        targets = batch.severe_collisions.detach().cpu().numpy()
        if predicted.shape != targets.shape:
            raise ValueError("collision validation prediction shape differs")
        probabilities.extend(np.max(predicted, axis=1).tolist())
        labels.extend(np.max(targets, axis=1).tolist())
    return _binary_episode_report(probabilities, labels, criteria)


def _binary_episode_report(
    probabilities: list[float],
    labels: list[float],
    criteria: CollisionValidationCriteria,
) -> dict[str, object]:
    probability = np.asarray(probabilities, np.float64)
    target = np.asarray(labels, np.float64)
    if probability.shape != target.shape or not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("collision validation labels are invalid")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("collision validation probabilities are invalid")
    positives = int(target.sum())
    negatives = int(target.size - positives)
    recall = float(((probability >= 0.5) & (target == 1.0)).sum() / positives) if positives else 0.0
    pr_auc = _average_precision(probability, target)
    brier = float(np.mean((probability - target) ** 2)) if target.size else 1.0
    checks = {
        "minimum_positive_episodes": positives
        >= criteria.minimum_positive_episodes_per_task,
        "minimum_negative_episodes": negatives
        >= criteria.minimum_negative_episodes_per_task,
        "minimum_recall": recall >= criteria.minimum_recall,
        "minimum_pr_auc": pr_auc >= criteria.minimum_pr_auc,
        "maximum_brier_score": brier <= criteria.maximum_brier_score,
    }
    return {
        "passed": all(checks.values()),
        "episode_count": int(target.size),
        "positive_episode_count": positives,
        "negative_episode_count": negatives,
        "recall": recall,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "positive_rate": float(target.mean()) if target.size else 0.0,
        "checks": checks,
    }


def _average_precision(probability: np.ndarray, target: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-probability, kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float((precision * ranked).sum() / positives)


def _window_identity(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "task_id": str(metadata["task_id"]),
        "episode_id": str(metadata["episode_id"]),
        "seed": int(metadata["seed"]),
        "transition_start": int(metadata["transition_start"]),
        "transition_stop": int(metadata["transition_stop"]),
    }
