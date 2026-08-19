"""Leakage-resistant posterior overshooting preflight for R0001-P06."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMSequence, RSSMState


OVERSHOOTING_HORIZONS = (1, 2, 4, 8)


@dataclass(frozen=True)
class OvershootingPairs:
    starts: tuple[int, ...]
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.starts
            or len(self.starts) != len(self.targets)
            or any(target <= start for start, target in zip(self.starts, self.targets))
        ):
            raise ValueError("overshooting target pairs are invalid")


def build_overshooting_pairs(
    transition_count: int, horizon: int
) -> OvershootingPairs:
    if transition_count <= 0 or not 1 <= horizon <= transition_count:
        raise ValueError("overshooting dimensions are invalid")
    starts = tuple(range(transition_count - horizon + 1))
    return OvershootingPairs(starts, tuple(start + horizon for start in starts))


def evaluate_posterior_overshooting(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    executed_actions: torch.Tensor,
    *,
    horizons: Sequence[int] = OVERSHOOTING_HORIZONS,
) -> dict[str, object]:
    horizon_values = tuple(int(value) for value in horizons)
    if (
        executed_actions.ndim != 3
        or executed_actions.shape[0] != sequence.deterministic.shape[0]
        or executed_actions.shape[1] + 1 != sequence.deterministic.shape[1]
        or executed_actions.shape[-1] != model.config.action_dimension
        or not horizon_values
        or len(set(horizon_values)) != len(horizon_values)
    ):
        raise ValueError("posterior overshooting input shapes are invalid")
    actions = executed_actions.detach().clone().requires_grad_(True)
    true_losses = _condition_losses(model, sequence, actions, horizon_values)
    true_total = torch.stack(
        tuple(value["total"] for value in true_losses.values())
    ).mean()
    action_gradient = torch.autograd.grad(true_total, actions)[0]
    with torch.no_grad():
        zero_losses = _condition_losses(
            model, sequence, torch.zeros_like(actions), horizon_values
        )
        shifted_losses = _condition_losses(
            model, sequence, torch.roll(actions, shifts=1, dims=1), horizon_values
        )
    target_requires_gradient = any(
        tensor.requires_grad
        for horizon in horizon_values
        for tensor in _posterior_targets(
            sequence,
            build_overshooting_pairs(executed_actions.shape[1], horizon),
        )
    )
    report = {
        "schema_version": "hwr.posterior-overshooting-preflight/v1",
        "horizons": list(horizon_values),
        "conditions": {
            "true_action": _loss_report(true_losses),
            "zero_action": _loss_report(zero_losses),
            "shifted_action": _loss_report(shifted_losses),
        },
        "action_gradient_norm": float(action_gradient.norm().detach().cpu()),
        "action_gradient_finite": bool(torch.isfinite(action_gradient).all()),
        "target_requires_gradient": target_requires_gradient,
        "target_pairs": {
            str(horizon): {
                "starts": list(
                    build_overshooting_pairs(executed_actions.shape[1], horizon).starts
                ),
                "targets": list(
                    build_overshooting_pairs(executed_actions.shape[1], horizon).targets
                ),
            }
            for horizon in horizon_values
        },
    }
    report["assessment"] = assess_posterior_overshooting(report)
    return report


def assess_posterior_overshooting(
    report: Mapping[str, object],
) -> dict[str, object]:
    conditions = report["conditions"]
    true_values = conditions["true_action"]["horizon_total_losses"]
    zero_values = conditions["zero_action"]["horizon_total_losses"]
    shifted_values = conditions["shifted_action"]["horizon_total_losses"]
    names = tuple(true_values)
    finite = all(
        math.isfinite(float(values[name]))
        for values in (true_values, zero_values, shifted_values)
        for name in names
    )
    true_better_count = sum(
        float(true_values[name]) < float(zero_values[name])
        and float(true_values[name]) < float(shifted_values[name])
        for name in names
    )
    true_mean = float(conditions["true_action"]["mean_total_loss"])
    zero_mean = float(conditions["zero_action"]["mean_total_loss"])
    shifted_mean = float(conditions["shifted_action"]["mean_total_loss"])
    checks = {
        "all_losses_finite": finite,
        "true_better_on_three_of_four_horizons": true_better_count >= 3,
        "true_at_least_five_percent_better_than_zero": (
            true_mean <= 0.95 * zero_mean
        ),
        "true_at_least_five_percent_better_than_shifted": (
            true_mean <= 0.95 * shifted_mean
        ),
        "finite_action_gradient": report.get("action_gradient_finite") is True,
        "action_gradient_norm_above_1e_6": (
            float(report.get("action_gradient_norm", 0.0)) > 1.0e-6
        ),
        "posterior_targets_stopped": (
            report.get("target_requires_gradient") is False
        ),
    }
    return {
        "decision": "preflight_passed" if all(checks.values()) else "preflight_failed",
        "passed": all(checks.values()),
        "true_better_horizon_count": true_better_count,
        "true_to_zero_ratio": true_mean / max(zero_mean, 1.0e-12),
        "true_to_shifted_ratio": true_mean / max(shifted_mean, 1.0e-12),
        "checks": checks,
    }


def aggregate_posterior_overshooting(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(reports) != 24:
        raise ValueError("overshooting preflight requires 24 source Episode reports")
    horizons = tuple(str(value) for value in OVERSHOOTING_HORIZONS)
    conditions = {}
    for condition in ("true_action", "zero_action", "shifted_action"):
        losses = {
            horizon: sum(
                float(report["conditions"][condition]["horizon_total_losses"][horizon])
                for report in reports
            )
            / len(reports)
            for horizon in horizons
        }
        conditions[condition] = {
            "mean_total_loss": sum(losses.values()) / len(losses),
            "horizon_total_losses": losses,
        }
    aggregate = {
        "schema_version": "hwr.posterior-overshooting-preflight-aggregate/v1",
        "episode_count": len(reports),
        "horizons": list(OVERSHOOTING_HORIZONS),
        "conditions": conditions,
        "action_gradient_norm": sum(
            float(report["action_gradient_norm"]) for report in reports
        )
        / len(reports),
        "action_gradient_finite": all(
            report["action_gradient_finite"] is True for report in reports
        ),
        "target_requires_gradient": any(
            report["target_requires_gradient"] is True for report in reports
        ),
    }
    aggregate["assessment"] = assess_posterior_overshooting(aggregate)
    return aggregate


def _condition_losses(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    actions: torch.Tensor,
    horizons: tuple[int, ...],
) -> dict[int, dict[str, torch.Tensor]]:
    return {
        horizon: _horizon_loss(model, sequence, actions, horizon)
        for horizon in horizons
    }


def _horizon_loss(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    actions: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    pairs = build_overshooting_pairs(actions.shape[1], horizon)
    initial = RSSMState(
        torch.cat(
            [sequence.deterministic[:, start] for start in pairs.starts], dim=0
        ).detach(),
        torch.cat(
            [sequence.stochastic[:, start] for start in pairs.starts], dim=0
        ).detach(),
    )
    action_chunks = torch.cat(
        [actions[:, start : start + horizon] for start in pairs.starts], dim=0
    )
    proposals = torch.zeros_like(action_chunks)
    rollout = model.rollout_prior(initial, action_chunks, proposals, sample=False)
    predicted_deterministic = rollout.states.deterministic[:, -1]
    predicted_logits = rollout.states.prior_logits[:, -1]
    target_deterministic, target_logits = _posterior_targets(sequence, pairs)
    deterministic = torch.nn.functional.mse_loss(
        predicted_deterministic, target_deterministic
    )
    stochastic = _categorical_kl(target_logits, predicted_logits).mean()
    return {
        "deterministic": deterministic,
        "stochastic": stochastic,
        "total": deterministic + stochastic,
    }


def _posterior_targets(
    sequence: RSSMSequence, pairs: OvershootingPairs
) -> tuple[torch.Tensor, torch.Tensor]:
    deterministic = torch.cat(
        [sequence.deterministic[:, target] for target in pairs.targets], dim=0
    ).detach()
    logits = torch.cat(
        [sequence.posterior_logits[:, target] for target in pairs.targets], dim=0
    ).detach()
    return deterministic, logits


def _categorical_kl(
    target_logits: torch.Tensor, predicted_logits: torch.Tensor
) -> torch.Tensor:
    target = target_logits.softmax(dim=-1)
    log_target = target_logits.log_softmax(dim=-1)
    log_predicted = predicted_logits.log_softmax(dim=-1)
    return (target * (log_target - log_predicted)).sum(dim=-1).sum(dim=-1)


def _loss_report(
    losses: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, object]:
    totals = {
        str(horizon): float(value["total"].detach().cpu())
        for horizon, value in losses.items()
    }
    return {
        "mean_total_loss": sum(totals.values()) / len(totals),
        "horizon_total_losses": totals,
        "horizons": {
            str(horizon): {
                name: float(item.detach().cpu())
                for name, item in value.items()
            }
            for horizon, value in losses.items()
        },
    }
