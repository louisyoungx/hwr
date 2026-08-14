from __future__ import annotations

from types import SimpleNamespace

import torch

from hwr.train.foundation_collision_validation import (
    CollisionValidationCriteria,
    evaluate_foundation_collision_validation,
)
from hwr.train.foundation_holdout import COLLISION_VALIDATION_PHASE


class _Loader:
    def __init__(self) -> None:
        self.metadata = []
        self.values = []
        for task_id in ("task-a/v1", "task-b/v1"):
            for episode_index, collided in enumerate((True, False)):
                self.metadata.append({
                    "task_id": task_id,
                    "episode_id": f"{task_id}-{episode_index}",
                    "seed": episode_index + 10,
                    "transition_count": 4,
                    "transition_start": 2,
                    "transition_stop": 4,
                    "metadata": {
                        "holdout_phase": COLLISION_VALIDATION_PHASE
                    },
                })
                self.values.append((collided, 0.9 if collided else 0.1))

    def __len__(self):
        return len(self.metadata)

    def window_metadata(self, index):
        return self.metadata[index]

    def build(self, indices, *, include_visual_targets=True):
        assert include_visual_targets is False
        targets = []
        predicted = []
        for index in indices:
            collided, probability = self.values[index]
            targets.append((0.0, float(collided)))
            predicted.append((0.05, probability))
        return SimpleNamespace(
            severe_collisions=torch.tensor(targets),
            predicted=torch.tensor(predicted),
        )


class _Trainer:
    def severe_collision_counterfactual_probabilities(self, batch, *, shuffle_seed):
        del shuffle_seed
        return batch.predicted, 1.0 - batch.predicted


def test_collision_validation_uses_independent_terminal_episodes() -> None:
    report = evaluate_foundation_collision_validation(
        _Trainer(),
        _Loader(),
        ("task-a/v1", "task-b/v1"),
        CollisionValidationCriteria(
            minimum_positive_episodes_per_task=1,
            minimum_negative_episodes_per_task=1,
            minimum_recall=0.8,
            minimum_pr_auc=0.8,
            maximum_brier_score=0.02,
        ),
        batch_size=2,
    )

    assert report["passed"] is True
    assert len(report["window_selection"]) == 4
    for task in report["partitions"].values():
        assert task["positive_episode_count"] == 1
        assert task["negative_episode_count"] == 1
        assert task["recall"] == 1.0
        assert task["pr_auc"] == 1.0
        assert task["brier_score"] < 0.02
        assert task["terminal_alignment_rate"] == 1.0
        assert task["false_positive_rate"] == 0.0
        assert task["shuffled_to_true_brier_ratio"] > 1.0


def test_collision_validation_fails_without_both_episode_classes() -> None:
    loader = _Loader()
    loader.metadata = loader.metadata[:1]
    loader.values = loader.values[:1]

    report = evaluate_foundation_collision_validation(
        _Trainer(),
        loader,
        ("task-a/v1",),
        CollisionValidationCriteria(1, 1, 0.8, 0.8, 0.02),
        batch_size=1,
    )

    task = report["partitions"]["task-a/v1"]
    assert report["passed"] is False
    assert task["checks"]["minimum_negative_episodes"] is False
