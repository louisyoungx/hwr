"""Closed-loop policy evaluation."""

from hwr.eval.evaluator import EvaluationReport, evaluate_policy
from hwr.eval.stability import (
    MultiObjectStabilityCriterion,
    PlacementSample,
    StabilityConfig,
    StablePlacementCriterion,
    TargetVolume,
)

__all__ = [
    "EvaluationReport",
    "MultiObjectStabilityCriterion",
    "PlacementSample",
    "StabilityConfig",
    "StablePlacementCriterion",
    "TargetVolume",
    "evaluate_policy",
]
