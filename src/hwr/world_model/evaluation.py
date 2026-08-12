"""Counterfactual action causality and open-loop world model diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from hwr.world_model.distributions import two_hot_symlog
from hwr.world_model.model import ActionConditionedWorldModel, WorldModelPriorRollout


@dataclass(frozen=True)
class CounterfactualCausalityReport:
    true_action_error: float
    shuffled_action_error: float
    shuffled_to_true_ratio: float
    true_horizon_errors: tuple[float, ...]
    shuffled_horizon_errors: tuple[float, ...]
    uncertainty_by_horizon: tuple[float, ...]
    error_components: tuple[str, ...] = ("visual_latent", "proprioception")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionCausalityCriteria:
    minimum_shuffled_to_true_ratio: float = 1.05
    minimum_worse_horizon_fraction: float = 0.60

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_shuffled_to_true_ratio)
            or self.minimum_shuffled_to_true_ratio <= 1.0
        ):
            raise ValueError("action causality ratio must be greater than one")
        if (
            not math.isfinite(self.minimum_worse_horizon_fraction)
            or not 0.0 < self.minimum_worse_horizon_fraction <= 1.0
        ):
            raise ValueError("action causality horizon fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def assess_action_causality(
    report: CounterfactualCausalityReport,
    criteria: ActionCausalityCriteria | None = None,
) -> dict[str, object]:
    """Require shuffled executed actions to degrade open-loop predictions."""
    settings = criteria or ActionCausalityCriteria()
    horizons = len(report.true_horizon_errors)
    if horizons == 0 or len(report.shuffled_horizon_errors) != horizons:
        raise ValueError("action causality report has invalid horizon evidence")
    worse = sum(
        shuffled > true
        for true, shuffled in zip(
            report.true_horizon_errors, report.shuffled_horizon_errors, strict=True
        )
    )
    fraction = worse / horizons
    ratio_passed = (
        report.shuffled_to_true_ratio
        >= settings.minimum_shuffled_to_true_ratio
    )
    horizon_passed = fraction >= settings.minimum_worse_horizon_fraction
    return {
        "passed": ratio_passed and horizon_passed,
        "criteria": settings.to_dict(),
        "shuffled_to_true_ratio": report.shuffled_to_true_ratio,
        "worse_horizon_count": worse,
        "horizon_count": horizons,
        "worse_horizon_fraction": fraction,
        "ratio_passed": ratio_passed,
        "horizon_passed": horizon_passed,
    }


def evaluate_action_causality(
    model: ActionConditionedWorldModel,
    visual: torch.Tensor,
    language: torch.Tensor,
    proprioception: torch.Tensor,
    executed_actions: torch.Tensor,
    rewards: torch.Tensor | None = None,
    continues: torch.Tensor | None = None,
    safety: torch.Tensor | None = None,
) -> CounterfactualCausalityReport:
    if executed_actions.shape[1] < 2:
        raise ValueError("action causality evaluation requires at least two transitions")
    was_training = model.training
    outcomes = (rewards, continues, safety)
    if any(value is None for value in outcomes) and not all(
        value is None for value in outcomes
    ):
        raise ValueError("action causality outcomes must be supplied together")
    model.eval()
    try:
        with torch.inference_mode():
            initial = model.initial_posterior(
                visual[:, 0], language, proprioception[:, 0]
            )
            true_rollout = model.rollout_prior(initial, executed_actions, sample=False)
            shuffled = torch.roll(executed_actions, shifts=1, dims=1)
            shuffled_rollout = model.rollout_prior(initial, shuffled, sample=False)
            targets = (rewards, continues, safety)
            true_errors = _horizon_errors(
                model, true_rollout, visual[:, 1:], proprioception[:, 1:], targets
            )
            shuffled_errors = _horizon_errors(
                model,
                shuffled_rollout,
                visual[:, 1:],
                proprioception[:, 1:],
                targets,
            )
    finally:
        model.train(was_training)
    true_mean = float(true_errors.mean().cpu())
    shuffled_mean = float(shuffled_errors.mean().cpu())
    return CounterfactualCausalityReport(
        true_action_error=true_mean,
        shuffled_action_error=shuffled_mean,
        shuffled_to_true_ratio=shuffled_mean / max(true_mean, 1.0e-8),
        true_horizon_errors=tuple(float(value) for value in true_errors.cpu()),
        shuffled_horizon_errors=tuple(float(value) for value in shuffled_errors.cpu()),
        uncertainty_by_horizon=tuple(
            float(value) for value in true_rollout.uncertainty.mean(dim=0).cpu()
        ),
        error_components=(
            "visual_latent",
            "proprioception",
            *(("reward", "continue", "safety") if rewards is not None else ()),
        ),
    )


def _horizon_errors(
    model: ActionConditionedWorldModel,
    rollout: WorldModelPriorRollout,
    visual: torch.Tensor,
    proprioception: torch.Tensor,
    outcomes: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
) -> torch.Tensor:
    visual_error = (rollout.visual_prediction - visual).square().mean(dim=-1)
    proprioception_error = (
        rollout.proprioception_prediction - proprioception
    ).square().mean(dim=-1)
    total = visual_error + proprioception_error
    rewards, continues, safety = outcomes
    if rewards is not None and continues is not None and safety is not None:
        reward_target = two_hot_symlog(
            rewards,
            bins=model.config.reward_bins,
            limit=model.config.reward_symlog_limit,
        )
        reward_error = -(
            reward_target * rollout.reward_logits.log_softmax(dim=-1)
        ).sum(dim=-1)
        continue_error = nn.functional.binary_cross_entropy_with_logits(
            rollout.continue_logits, continues.float(), reduction="none"
        )
        safety_error = nn.functional.binary_cross_entropy_with_logits(
            rollout.safety_logits, safety.float(), reduction="none"
        )
        total = total + reward_error + continue_error + safety_error
    return total.mean(dim=0)
