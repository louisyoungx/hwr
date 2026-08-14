from __future__ import annotations

from types import SimpleNamespace

import torch

from hwr.train.foundation_action_execution_validation import (
    ActionExecutionValidationCriteria,
    evaluate_foundation_action_execution_validation,
)
from hwr.train.foundation_holdout import ACTION_EXECUTION_VALIDATION_PHASE


class _Loader:
    def __init__(self) -> None:
        self.metadata = []
        self.values = []
        for task_id in ("task-a/v1", "task-b/v1"):
            for episode_index, intervened in enumerate((True, False)):
                self.metadata.append({
                    "task_id": task_id,
                    "episode_id": f"{task_id}-{episode_index}",
                    "seed": episode_index + 20,
                    "transition_count": 2,
                    "transition_start": 0,
                    "transition_stop": 2,
                    "metadata": {
                        "holdout_phase": ACTION_EXECUTION_VALIDATION_PHASE
                    },
                })
                self.values.append(intervened)

    def __len__(self):
        return len(self.metadata)

    def window_metadata(self, index):
        return self.metadata[index]

    def build(self, indices, *, include_visual_targets=True):
        assert include_visual_targets is False
        labels = []
        proposals = []
        executed = []
        for index in indices:
            intervened = self.values[index]
            label = torch.tensor([float(intervened), 0.0])
            proposal = torch.tensor([[0.8, -0.8], [0.2, -0.2]])
            actual = proposal.clone()
            if intervened:
                actual[0] = torch.tensor([0.5, -0.5])
            labels.append(label)
            proposals.append(proposal)
            executed.append(actual)
        return SimpleNamespace(
            safety_interventions=torch.stack(labels),
            actor_proposals=torch.stack(proposals),
            executed_actions=torch.stack(executed),
        )


class _Trainer:
    def __init__(self) -> None:
        self.world_model = SimpleNamespace(
            config=SimpleNamespace(
                action_minimum=(-1.0, -1.0),
                action_maximum=(1.0, 1.0),
            )
        )

    def action_execution_validation_predictions(self, batch):
        labels = batch.safety_interventions
        probabilities = torch.where(labels > 0.5, 0.95, 0.05)
        return probabilities, batch.executed_actions.clone()


def test_action_execution_validation_requires_bounded_rewrite_and_identity() -> None:
    report = evaluate_foundation_action_execution_validation(
        _Trainer(),
        _Loader(),
        ("task-a/v1", "task-b/v1"),
        ActionExecutionValidationCriteria(
            minimum_positive_episodes_per_task=1,
            minimum_negative_episodes_per_task=1,
            minimum_pr_auc=0.9,
            maximum_brier_score=0.01,
        ),
        batch_size=2,
    )

    assert report["passed"] is True
    assert len(report["window_selection"]) == 4
    for task in report["partitions"].values():
        assert task["positive_episode_count"] == 1
        assert task["negative_episode_count"] == 1
        assert task["intervention_action_normalized_rmse"] == 0.0
        assert task["identity_action_normalized_rmse"] == 0.0
        assert task["out_of_bounds_rate"] == 0.0


def test_action_execution_validation_rejects_out_of_contract_predictions() -> None:
    trainer = _Trainer()

    def invalid(batch):
        return torch.full_like(batch.safety_interventions, 0.5), torch.full_like(
            batch.executed_actions, 2.0
        )

    trainer.action_execution_validation_predictions = invalid
    report = evaluate_foundation_action_execution_validation(
        trainer,
        _Loader(),
        ("task-a/v1", "task-b/v1"),
        ActionExecutionValidationCriteria(1, 1),
        batch_size=2,
    )

    assert report["passed"] is False
    assert all(
        task["checks"]["maximum_out_of_bounds_rate"] is False
        for task in report["partitions"].values()
    )
