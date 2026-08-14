"""One unified batch for visual, dynamics, and imagined RL updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from hwr.perception.student import VISUAL_STUDENT_INPUT_FIELDS
from hwr.perception.student_objectives import VisualTeacherTargets


@dataclass(frozen=True)
class FoundationTrainingBatch:
    student_inputs: Mapping[str, torch.Tensor]
    visual_targets: VisualTeacherTargets | None
    sequence_batch_size: int
    observation_count: int
    language_features: torch.Tensor
    proprioception: torch.Tensor
    actor_proposals: torch.Tensor
    executed_actions: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    safety_interventions: torch.Tensor
    severe_collisions: torch.Tensor

    def __post_init__(self) -> None:
        if frozenset(self.student_inputs) != VISUAL_STUDENT_INPUT_FIELDS:
            raise ValueError("foundation batch student inputs violate the field whitelist")
        if min(self.sequence_batch_size, self.observation_count) <= 0:
            raise ValueError("foundation batch sequence dimensions must be positive")
        flattened = self.sequence_batch_size * self.observation_count
        if any(value.shape[0] != flattened for value in self.student_inputs.values()):
            raise ValueError("foundation batch student inputs do not cover every observation")
        expected_prefixes = {
            "language_features": (self.sequence_batch_size,),
            "proprioception": (self.sequence_batch_size, self.observation_count),
            "actor_proposals": (
                self.sequence_batch_size,
                self.observation_count - 1,
            ),
            "executed_actions": (
                self.sequence_batch_size,
                self.observation_count - 1,
            ),
            "rewards": (self.sequence_batch_size, self.observation_count - 1),
            "continues": (self.sequence_batch_size, self.observation_count - 1),
            "safety_interventions": (
                self.sequence_batch_size,
                self.observation_count - 1,
            ),
            "severe_collisions": (
                self.sequence_batch_size,
                self.observation_count - 1,
            ),
        }
        values = {
            "language_features": self.language_features,
            "proprioception": self.proprioception,
            "actor_proposals": self.actor_proposals,
            "executed_actions": self.executed_actions,
            "rewards": self.rewards,
            "continues": self.continues,
            "safety_interventions": self.safety_interventions,
            "severe_collisions": self.severe_collisions,
        }
        mismatches = {
            name: (tuple(values[name].shape), prefix)
            for name, prefix in expected_prefixes.items()
            if tuple(values[name].shape[: len(prefix)]) != prefix
        }
        if mismatches:
            raise ValueError(f"foundation batch sequence shapes are invalid: {mismatches}")
        floating = (
            *self.student_inputs.values(),
            self.language_features,
            self.proprioception,
            self.actor_proposals,
            self.executed_actions,
            self.rewards,
            self.safety_interventions,
            self.severe_collisions,
        )
        if not all(torch.isfinite(value).all() for value in floating):
            raise ValueError("foundation batch contains non-finite values")


def detach_student_inputs(
    inputs: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in inputs.items()}
