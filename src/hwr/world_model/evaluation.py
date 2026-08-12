"""Counterfactual action causality and open-loop world model diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import nn

from hwr.world_model.distributions import two_hot_symlog
from hwr.world_model.model import ActionConditionedWorldModel, WorldModelPriorRollout


ACTION_CAUSALITY_COMPONENTS = (
    "visual_latent",
    "proprioception",
    "reward",
    "continue",
    "safety",
)


@dataclass(frozen=True)
class CounterfactualComponentReport:
    true_error: float
    shuffled_error: float
    shuffled_to_true_ratio: float
    true_horizon_errors: tuple[float, ...]
    shuffled_horizon_errors: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.true_horizon_errors
            or len(self.true_horizon_errors) != len(self.shuffled_horizon_errors)
        ):
            raise ValueError("component causality horizon evidence is invalid")
        _validate_error_summary(
            self.true_error,
            self.shuffled_error,
            self.shuffled_to_true_ratio,
            self.true_horizon_errors,
            self.shuffled_horizon_errors,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualCausalityReport:
    true_action_error: float
    shuffled_action_error: float
    shuffled_to_true_ratio: float
    true_horizon_errors: tuple[float, ...]
    shuffled_horizon_errors: tuple[float, ...]
    uncertainty_by_horizon: tuple[float, ...]
    component_reports: Mapping[str, CounterfactualComponentReport]
    error_components: tuple[str, ...] = ("visual_latent", "proprioception")
    sample_count: int = 1

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("action causality sample count must be positive")
        if tuple(self.component_reports) != self.error_components:
            raise ValueError("action causality component evidence differs")
        horizons = len(self.true_horizon_errors)
        if (
            horizons == 0
            or len(self.shuffled_horizon_errors) != horizons
            or len(self.uncertainty_by_horizon) != horizons
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.uncertainty_by_horizon
            )
            or any(
                len(value.true_horizon_errors) != horizons
                for value in self.component_reports.values()
            )
        ):
            raise ValueError("action causality report has invalid horizon evidence")
        true_components = tuple(
            sum(value.true_horizon_errors[index] for value in self.component_reports.values())
            for index in range(horizons)
        )
        shuffled_components = tuple(
            sum(
                value.shuffled_horizon_errors[index]
                for value in self.component_reports.values()
            )
            for index in range(horizons)
        )
        if not _series_close(self.true_horizon_errors, true_components) or not _series_close(
            self.shuffled_horizon_errors, shuffled_components
        ):
            raise ValueError("aggregate causality errors differ from components")
        _validate_error_summary(
            self.true_action_error,
            self.shuffled_action_error,
            self.shuffled_to_true_ratio,
            self.true_horizon_errors,
            self.shuffled_horizon_errors,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_error_summary(
    true_error: float,
    shuffled_error: float,
    ratio: float,
    true_horizons: tuple[float, ...],
    shuffled_horizons: tuple[float, ...],
) -> None:
    values = (true_error, shuffled_error, ratio, *true_horizons, *shuffled_horizons)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("action causality errors must be finite and non-negative")
    expected_true = sum(true_horizons) / len(true_horizons)
    expected_shuffled = sum(shuffled_horizons) / len(shuffled_horizons)
    expected_ratio = expected_shuffled / max(expected_true, 1.0e-8)
    if not all(
        math.isclose(left, right, rel_tol=1.0e-5, abs_tol=1.0e-7)
        for left, right in (
            (true_error, expected_true),
            (shuffled_error, expected_shuffled),
            (ratio, expected_ratio),
        )
    ):
        raise ValueError("action causality summary differs from horizon evidence")


def _series_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(first, second, rel_tol=1.0e-5, abs_tol=1.0e-7)
        for first, second in zip(left, right, strict=True)
    )


def counterfactual_report_from_dict(
    value: Mapping[str, object],
) -> CounterfactualCausalityReport:
    """Rebuild and validate one serialized counterfactual evidence record."""
    raw_components = value.get("component_reports")
    if not isinstance(raw_components, Mapping):
        raise ValueError("serialized action causality components are missing")
    components = {
        str(name): _component_report_from_dict(component)
        for name, component in raw_components.items()
    }
    return CounterfactualCausalityReport(
        true_action_error=float(value["true_action_error"]),
        shuffled_action_error=float(value["shuffled_action_error"]),
        shuffled_to_true_ratio=float(value["shuffled_to_true_ratio"]),
        true_horizon_errors=tuple(float(item) for item in value["true_horizon_errors"]),
        shuffled_horizon_errors=tuple(
            float(item) for item in value["shuffled_horizon_errors"]
        ),
        uncertainty_by_horizon=tuple(
            float(item) for item in value["uncertainty_by_horizon"]
        ),
        component_reports=components,
        error_components=tuple(str(item) for item in value["error_components"]),
        sample_count=int(value["sample_count"]),
    )


def _component_report_from_dict(value: object) -> CounterfactualComponentReport:
    if not isinstance(value, Mapping):
        raise ValueError("serialized action causality component is invalid")
    return CounterfactualComponentReport(
        true_error=float(value["true_error"]),
        shuffled_error=float(value["shuffled_error"]),
        shuffled_to_true_ratio=float(value["shuffled_to_true_ratio"]),
        true_horizon_errors=tuple(float(item) for item in value["true_horizon_errors"]),
        shuffled_horizon_errors=tuple(
            float(item) for item in value["shuffled_horizon_errors"]
        ),
    )


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
    """Require shuffled actions to degrade every predicted result separately."""
    settings = criteria or ActionCausalityCriteria()
    aggregate = _assess_error_series(
        report.shuffled_to_true_ratio,
        report.true_horizon_errors,
        report.shuffled_horizon_errors,
        settings,
    )
    components = {
        name: _assess_error_series(
            value.shuffled_to_true_ratio,
            value.true_horizon_errors,
            value.shuffled_horizon_errors,
            settings,
        )
        for name, value in report.component_reports.items()
    }
    all_components_passed = all(
        value["passed"] is True for value in components.values()
    )
    return {
        **aggregate,
        "passed": aggregate["passed"] and all_components_passed,
        "aggregate_passed": aggregate["passed"],
        "all_components_passed": all_components_passed,
        "components": components,
    }


def _assess_error_series(
    ratio: float,
    true_errors: tuple[float, ...],
    shuffled_errors: tuple[float, ...],
    criteria: ActionCausalityCriteria,
) -> dict[str, object]:
    horizons = len(true_errors)
    if horizons == 0 or len(shuffled_errors) != horizons:
        raise ValueError("action causality error series is invalid")
    worse = sum(
        shuffled > true
        for true, shuffled in zip(
            true_errors, shuffled_errors, strict=True
        )
    )
    fraction = worse / horizons
    ratio_passed = ratio >= criteria.minimum_shuffled_to_true_ratio
    horizon_passed = fraction >= criteria.minimum_worse_horizon_fraction
    return {
        "passed": ratio_passed and horizon_passed,
        "criteria": criteria.to_dict(),
        "shuffled_to_true_ratio": ratio,
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
    actor_proposals: torch.Tensor,
    executed_actions: torch.Tensor,
    rewards: torch.Tensor | None = None,
    continues: torch.Tensor | None = None,
    safety: torch.Tensor | None = None,
    *,
    shuffle_seed: int = 0,
) -> CounterfactualCausalityReport:
    if executed_actions.shape[1] < 2 or actor_proposals.shape != executed_actions.shape:
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
            true_rollout = model.rollout_prior(
                initial, executed_actions, actor_proposals, sample=False
            )
            paired = torch.cat((actor_proposals, executed_actions), dim=-1)
            shuffled_pair = deterministic_action_derangement(
                paired, seed=shuffle_seed
            )
            proposal_dimension = actor_proposals.shape[-1]
            shuffled_proposals = shuffled_pair[..., :proposal_dimension]
            shuffled_executed = shuffled_pair[..., proposal_dimension:]
            shuffled_rollout = model.rollout_prior(
                initial, shuffled_executed, shuffled_proposals, sample=False
            )
            targets = (rewards, continues, safety)
            true_components = _horizon_component_errors(
                model, true_rollout, visual[:, 1:], proprioception[:, 1:], targets
            )
            shuffled_components = _horizon_component_errors(
                model,
                shuffled_rollout,
                visual[:, 1:],
                proprioception[:, 1:],
                targets,
            )
    finally:
        model.train(was_training)
    component_reports = {
        name: _component_report(values, shuffled_components[name])
        for name, values in true_components.items()
    }
    true_errors = torch.stack(tuple(true_components.values())).sum(dim=0)
    shuffled_errors = torch.stack(tuple(shuffled_components.values())).sum(dim=0)
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
        component_reports=component_reports,
        error_components=tuple(component_reports),
        sample_count=int(executed_actions.shape[0]),
    )


def _component_report(
    true_errors: torch.Tensor, shuffled_errors: torch.Tensor
) -> CounterfactualComponentReport:
    true_values = tuple(float(value) for value in true_errors.cpu())
    shuffled_values = tuple(float(value) for value in shuffled_errors.cpu())
    true_mean = sum(true_values) / len(true_values)
    shuffled_mean = sum(shuffled_values) / len(shuffled_values)
    return CounterfactualComponentReport(
        true_mean,
        shuffled_mean,
        shuffled_mean / max(true_mean, 1.0e-8),
        true_values,
        shuffled_values,
    )


def deterministic_action_derangement(
    executed_actions: torch.Tensor, *, seed: int
) -> torch.Tensor:
    """Permute all batch/time actions with no fixed points and no new values."""
    if seed < 0 or executed_actions.ndim != 3:
        raise ValueError("action derangement seed or tensor shape is invalid")
    flattened = executed_actions.flatten(0, 1)
    count = flattened.shape[0]
    if count < 2:
        raise ValueError("action derangement requires at least two actions")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cycle = torch.randperm(count, generator=generator)
    sources = torch.empty_like(cycle)
    sources[cycle] = torch.roll(cycle, shifts=1)
    sources = sources.to(executed_actions.device)
    return flattened[sources].reshape_as(executed_actions)


def aggregate_action_causality_reports(
    reports: tuple[CounterfactualCausalityReport, ...],
) -> CounterfactualCausalityReport:
    """Combine equal-horizon reports using their evaluated sequence counts."""
    if not reports:
        raise ValueError("action causality aggregation requires reports")
    horizon_count = len(reports[0].true_horizon_errors)
    components = reports[0].error_components
    if horizon_count == 0 or any(
        len(value.true_horizon_errors) != horizon_count
        or len(value.shuffled_horizon_errors) != horizon_count
        or len(value.uncertainty_by_horizon) != horizon_count
        or value.error_components != components
        for value in reports
    ):
        raise ValueError("action causality reports cannot be aggregated")
    weights = torch.tensor(
        [value.sample_count for value in reports], dtype=torch.float64
    )
    weights /= weights.sum()

    def average(name: str) -> tuple[float, ...]:
        values = torch.tensor(
            [getattr(value, name) for value in reports], dtype=torch.float64
        )
        return tuple(float(item) for item in (values * weights[:, None]).sum(dim=0))

    component_reports: dict[str, CounterfactualComponentReport] = {}
    for component in components:
        true_component = _average_component_horizons(
            reports, weights, component, "true_horizon_errors"
        )
        shuffled_component = _average_component_horizons(
            reports, weights, component, "shuffled_horizon_errors"
        )
        true_component_mean = sum(true_component) / horizon_count
        shuffled_component_mean = sum(shuffled_component) / horizon_count
        component_reports[component] = CounterfactualComponentReport(
            true_component_mean,
            shuffled_component_mean,
            shuffled_component_mean / max(true_component_mean, 1.0e-8),
            true_component,
            shuffled_component,
        )
    true_horizons = tuple(
        sum(value.true_horizon_errors[index] for value in component_reports.values())
        for index in range(horizon_count)
    )
    shuffled_horizons = tuple(
        sum(
            value.shuffled_horizon_errors[index]
            for value in component_reports.values()
        )
        for index in range(horizon_count)
    )
    true_mean = sum(true_horizons) / horizon_count
    shuffled_mean = sum(shuffled_horizons) / horizon_count
    return CounterfactualCausalityReport(
        true_mean,
        shuffled_mean,
        shuffled_mean / max(true_mean, 1.0e-8),
        true_horizons,
        shuffled_horizons,
        average("uncertainty_by_horizon"),
        component_reports,
        components,
        sum(value.sample_count for value in reports),
    )


def _average_component_horizons(
    reports: tuple[CounterfactualCausalityReport, ...],
    weights: torch.Tensor,
    component: str,
    field: str,
) -> tuple[float, ...]:
    values = torch.tensor(
        [
            getattr(report.component_reports[component], field)
            for report in reports
        ],
        dtype=torch.float64,
    )
    return tuple(float(item) for item in (values * weights[:, None]).sum(dim=0))


def _horizon_component_errors(
    model: ActionConditionedWorldModel,
    rollout: WorldModelPriorRollout,
    visual: torch.Tensor,
    proprioception: torch.Tensor,
    outcomes: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    visual_error = (rollout.visual_prediction - visual).square().mean(dim=-1)
    proprioception_error = (
        rollout.proprioception_prediction - proprioception
    ).square().mean(dim=-1)
    result = {
        "visual_latent": visual_error.mean(dim=0),
        "proprioception": proprioception_error.mean(dim=0),
    }
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
        result.update(
            {
                "reward": reward_error.mean(dim=0),
                "continue": continue_error.mean(dim=0),
                "safety": safety_error.mean(dim=0),
            }
        )
    return result
