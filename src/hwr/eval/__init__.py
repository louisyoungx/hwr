"""Closed-loop policy evaluation."""

from hwr.eval.evaluator import EvaluationReport, evaluate_policy
from hwr.eval.formal_visual import FormalEvaluationReport, evaluate_formal_visual_policy
from hwr.eval.stability import (
    MultiObjectStabilityCriterion,
    PlacementSample,
    StabilityConfig,
    StablePlacementCriterion,
    TargetVolume,
)

__all__ = [
    "EvaluationReport",
    "FormalEvaluationReport",
    "MultiObjectStabilityCriterion",
    "PlacementSample",
    "StabilityConfig",
    "StablePlacementCriterion",
    "TargetVolume",
    "evaluate_policy",
    "evaluate_formal_visual_policy",
]
