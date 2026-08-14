"""Independent holdout validation for the learned safety action rewrite model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from hwr.data.foundation_loading import FoundationSequenceBatchLoader
from hwr.train.foundation_holdout import ACTION_EXECUTION_VALIDATION_PHASE
from hwr.train.foundation_trainer import FoundationWorldModelTrainer


ACTION_EXECUTION_VALIDATION_SCHEMA = "hwr.foundation-action-execution-validation/v1"


@dataclass(frozen=True)
class ActionExecutionValidationCriteria:
    minimum_positive_episodes_per_task: int = 8
    minimum_negative_episodes_per_task: int = 8
    minimum_recall: float = 0.80
    minimum_pr_auc: float = 0.50
    maximum_brier_score: float = 0.10
    maximum_intervention_normalized_rmse: float = 0.15
    maximum_identity_normalized_rmse: float = 0.05
    maximum_out_of_bounds_rate: float = 0.0

    def __post_init__(self) -> None:
        counts = (
            self.minimum_positive_episodes_per_task,
            self.minimum_negative_episodes_per_task,
        )
        probabilities = (
            self.minimum_recall,
            self.minimum_pr_auc,
            self.maximum_brier_score,
            self.maximum_out_of_bounds_rate,
        )
        if min(counts) < 0 or sum(counts) <= 0:
            raise ValueError("action execution validation Episode counts are invalid")
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("action execution validation probabilities are invalid")
        if min(
            self.maximum_intervention_normalized_rmse,
            self.maximum_identity_normalized_rmse,
        ) <= 0.0:
            raise ValueError("action execution validation error limits are invalid")


def evaluate_foundation_action_execution_validation(
    trainer: FoundationWorldModelTrainer,
    loader: FoundationSequenceBatchLoader,
    task_ids: tuple[str, ...],
    criteria: ActionExecutionValidationCriteria,
    *,
    batch_size: int,
) -> dict[str, object]:
    if batch_size <= 0 or not task_ids:
        raise ValueError("action execution validation batch configuration is invalid")
    selected = _terminal_windows_by_task(loader, task_ids)
    partitions = {
        task_id: _evaluate_partition(
            trainer, loader, selected[task_id], criteria, batch_size=batch_size
        )
        for task_id in task_ids
    }
    return {
        "schema_version": ACTION_EXECUTION_VALIDATION_SCHEMA,
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
        episode = metadata.get("metadata", {})
        task_id = str(metadata["task_id"])
        if (
            task_id not in grouped
            or not isinstance(episode, Mapping)
            or episode.get("holdout_phase") != ACTION_EXECUTION_VALIDATION_PHASE
            or int(metadata["transition_stop"]) != int(metadata["transition_count"])
        ):
            continue
        episode_id = str(metadata["episode_id"])
        if episode_id in grouped[task_id]:
            raise ValueError("action execution validation has duplicate terminal windows")
        grouped[task_id][episode_id] = index
    return {
        task_id: tuple(value for _, value in sorted(episodes.items()))
        for task_id, episodes in grouped.items()
    }


def _evaluate_partition(
    trainer: FoundationWorldModelTrainer,
    loader: FoundationSequenceBatchLoader,
    indices: tuple[int, ...],
    criteria: ActionExecutionValidationCriteria,
    *,
    batch_size: int,
) -> dict[str, object]:
    probabilities: list[np.ndarray] = []
    interventions: list[np.ndarray] = []
    predicted_actions: list[np.ndarray] = []
    proposals: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    episode_labels: list[bool] = []
    for start in range(0, len(indices), batch_size):
        batch = loader.build(
            indices[start : start + batch_size], include_visual_targets=False
        )
        probability, predicted = trainer.action_execution_validation_predictions(batch)
        target = batch.safety_interventions.detach().cpu().numpy()
        probabilities.append(probability.numpy())
        predicted_actions.append(predicted.numpy())
        interventions.append(target)
        proposals.append(batch.actor_proposals.detach().cpu().numpy())
        executed_actions.append(batch.executed_actions.detach().cpu().numpy())
        episode_labels.extend(bool(row.any()) for row in target > 0.5)
    return _action_execution_report(
        np.concatenate(probabilities) if probabilities else np.empty((0, 0)),
        np.concatenate(interventions) if interventions else np.empty((0, 0)),
        np.concatenate(predicted_actions) if predicted_actions else np.empty((0, 0, 0)),
        np.concatenate(proposals) if proposals else np.empty((0, 0, 0)),
        np.concatenate(executed_actions) if executed_actions else np.empty((0, 0, 0)),
        np.asarray(episode_labels, np.bool_),
        trainer.world_model.config.action_minimum,
        trainer.world_model.config.action_maximum,
        criteria,
    )


def _action_execution_report(
    probability: np.ndarray,
    intervention: np.ndarray,
    predicted: np.ndarray,
    proposal: np.ndarray,
    executed: np.ndarray,
    episode_labels: np.ndarray,
    lower_values: tuple[float, ...],
    upper_values: tuple[float, ...],
    criteria: ActionExecutionValidationCriteria,
) -> dict[str, object]:
    target = intervention > 0.5
    lower = np.asarray(lower_values, np.float64)
    upper = np.asarray(upper_values, np.float64)
    scale = upper - lower
    normalized_error = (predicted - executed) / scale
    intervention_rmse = _masked_rmse(normalized_error, target)
    identity_error = (predicted - proposal) / scale
    identity_rmse = _masked_rmse(identity_error, ~target)
    out_of_bounds = (predicted < lower) | (predicted > upper)
    out_of_bounds_rate = float(out_of_bounds.mean()) if predicted.size else 1.0
    flat_probability = probability.reshape(-1)
    flat_target = target.reshape(-1).astype(np.float64)
    predicted_positive = flat_probability >= 0.5
    positives = int(flat_target.sum())
    recall = (
        float(flat_target[predicted_positive].sum() / positives) if positives else 0.0
    )
    pr_auc = _average_precision(flat_probability, flat_target)
    brier = (
        float(np.mean((flat_probability - flat_target) ** 2))
        if flat_target.size
        else 1.0
    )
    positive_episodes = int(episode_labels.sum())
    negative_episodes = int((~episode_labels).sum())
    checks = {
        "minimum_positive_episodes": (
            positive_episodes >= criteria.minimum_positive_episodes_per_task
        ),
        "minimum_negative_episodes": (
            negative_episodes >= criteria.minimum_negative_episodes_per_task
        ),
        "minimum_recall": recall >= criteria.minimum_recall,
        "minimum_pr_auc": pr_auc >= criteria.minimum_pr_auc,
        "maximum_brier_score": brier <= criteria.maximum_brier_score,
        "maximum_intervention_normalized_rmse": (
            intervention_rmse <= criteria.maximum_intervention_normalized_rmse
        ),
        "maximum_identity_normalized_rmse": (
            identity_rmse <= criteria.maximum_identity_normalized_rmse
        ),
        "maximum_out_of_bounds_rate": (
            out_of_bounds_rate <= criteria.maximum_out_of_bounds_rate
        ),
    }
    return {
        "passed": all(checks.values()),
        "episode_count": int(episode_labels.size),
        "positive_episode_count": positive_episodes,
        "negative_episode_count": negative_episodes,
        "intervention_transition_count": positives,
        "non_intervention_transition_count": int(flat_target.size - positives),
        "recall": recall,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "intervention_action_normalized_rmse": intervention_rmse,
        "identity_action_normalized_rmse": identity_rmse,
        "out_of_bounds_rate": out_of_bounds_rate,
        "checks": checks,
    }


def _masked_rmse(error: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(np.sqrt(np.mean(error[mask] ** 2)))


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
