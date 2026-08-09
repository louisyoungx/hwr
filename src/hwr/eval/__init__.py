"""Closed-loop policy evaluation."""

from hwr.eval.evaluator import EvaluationReport, evaluate_policy
from hwr.eval.stability import StabilityConfig, StablePlacementCriterion, TargetVolume

__all__ = [
    "EvaluationReport",
    "StabilityConfig",
    "StablePlacementCriterion",
    "TargetVolume",
    "evaluate_policy",
]
