"""Gradient dead-zone diagnostic for the R0001-P19 free-nats hypothesis."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch

from hwr.world_model.model import ActionConditionedWorldModel
from hwr.world_model.rssm import RSSMSequence


FREE_NATS_CONDITIONS = {
    "current": 1.0,
    "candidate": 0.1,
    "raw": None,
}


def evaluate_free_nats_deadzone(
    model: ActionConditionedWorldModel,
    sequence: RSSMSequence,
    executed_actions: torch.Tensor,
) -> dict[str, object]:
    if (
        executed_actions.ndim != 3
        or executed_actions.shape[0] != sequence.prior_logits.shape[0]
        or executed_actions.shape[1] + 1 != sequence.prior_logits.shape[1]
        or executed_actions.shape[-1] != model.config.action_dimension
        or not executed_actions.requires_grad
    ):
        raise ValueError("free-nats diagnostic input shapes are invalid")
    raw_transition = _categorical_kl(
        sequence.posterior_logits[:, 1:].detach(),
        sequence.prior_logits[:, 1:],
    )
    parameters = tuple(
        parameter
        for module in (
            model.rssm.transition_input,
            model.rssm.recurrent,
            model.rssm.prior,
        )
        for parameter in module.parameters()
    )
    conditions = {
        name: _gradient_report(
            raw_transition,
            floor,
            parameters,
            executed_actions,
            retain_graph=name != "raw",
        )
        for name, floor in FREE_NATS_CONDITIONS.items()
    }
    raw_gradient = conditions["raw"]["parameter_gradient"]
    candidate_gradient = conditions["candidate"]["parameter_gradient"]
    report = {
        "schema_version": "hwr.free-nats-deadzone-diagnostic/v1",
        "raw_kl_values": [
            float(value) for value in raw_transition.detach().cpu().reshape(-1)
        ],
        "raw_kl": _distribution_report(raw_transition),
        "conditions": {
            name: {
                key: value
                for key, value in condition.items()
                if key != "parameter_gradient"
            }
            for name, condition in conditions.items()
        },
        "candidate_raw_parameter_gradient_cosine": _cosine(
            candidate_gradient, raw_gradient
        ),
    }
    report["assessment"] = assess_free_nats_deadzone(report)
    return report


def aggregate_free_nats_deadzone(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(reports) != 24:
        raise ValueError("free-nats diagnostic requires 24 source Episode reports")
    raw_values = [
        float(value)
        for report in reports
        for value in report["raw_kl_values"]
    ]
    conditions = {
        name: {
            "loss": _mean(reports, name, "loss"),
            "parameter_gradient_norm": _mean(
                reports, name, "parameter_gradient_norm"
            ),
            "action_gradient_norm": _mean(
                reports, name, "action_gradient_norm"
            ),
            "parameter_gradient_finite": all(
                report["conditions"][name]["parameter_gradient_finite"] is True
                for report in reports
            ),
            "action_gradient_finite": all(
                report["conditions"][name]["action_gradient_finite"] is True
                for report in reports
            ),
        }
        for name in FREE_NATS_CONDITIONS
    }
    aggregate = {
        "schema_version": "hwr.free-nats-deadzone-aggregate/v1",
        "episode_count": len(reports),
        "transition_count": len(raw_values),
        "raw_kl_values": raw_values,
        "raw_kl": _distribution_report(
            torch.tensor(raw_values, dtype=torch.float64)
        ),
        "conditions": conditions,
        "candidate_raw_parameter_gradient_cosine": sum(
            float(report["candidate_raw_parameter_gradient_cosine"])
            for report in reports
        )
        / len(reports),
    }
    aggregate["assessment"] = assess_free_nats_deadzone(aggregate)
    return aggregate


def assess_free_nats_deadzone(
    report: Mapping[str, object],
) -> dict[str, object]:
    raw = report["raw_kl"]
    conditions = report["conditions"]
    current = conditions["current"]
    candidate = conditions["candidate"]
    raw_condition = conditions["raw"]
    checks = {
        "raw_kl_below_1_fraction_at_least_0_80": (
            float(raw["below_1_fraction"]) >= 0.80
        ),
        "current_parameter_gradient_at_most_1e_8": (
            float(current["parameter_gradient_norm"]) <= 1.0e-8
        ),
        "raw_parameter_gradient_above_1e_6": (
            float(raw_condition["parameter_gradient_norm"]) > 1.0e-6
        ),
        "candidate_parameter_gradient_above_1e_6": (
            float(candidate["parameter_gradient_norm"]) > 1.0e-6
        ),
        "candidate_action_gradient_above_1e_6": (
            float(candidate["action_gradient_norm"]) > 1.0e-6
        ),
        "candidate_gradients_finite": (
            candidate["parameter_gradient_finite"] is True
            and candidate["action_gradient_finite"] is True
        ),
        "candidate_raw_gradient_cosine_at_least_0_90": (
            float(report["candidate_raw_parameter_gradient_cosine"]) >= 0.90
        ),
    }
    return {
        "decision": "diagnostic_passed" if all(checks.values()) else "diagnostic_failed",
        "passed": all(checks.values()),
        "checks": checks,
    }


def _gradient_report(
    raw_transition: torch.Tensor,
    floor: float | None,
    parameters: tuple[torch.nn.Parameter, ...],
    actions: torch.Tensor,
    *,
    retain_graph: bool,
) -> dict[str, object]:
    loss = (
        raw_transition.mean()
        if floor is None
        else raw_transition.clamp_min(floor).mean()
    )
    gradients = torch.autograd.grad(
        loss,
        (*parameters, actions),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    parameter_gradient = _flatten_gradients(gradients[:-1], parameters)
    action_gradient = (
        gradients[-1]
        if gradients[-1] is not None
        else torch.zeros_like(actions)
    )
    return {
        "loss": float(loss.detach().cpu()),
        "parameter_gradient_norm": float(
            parameter_gradient.norm().detach().cpu()
        ),
        "action_gradient_norm": float(action_gradient.norm().detach().cpu()),
        "parameter_gradient_finite": bool(
            torch.isfinite(parameter_gradient).all()
        ),
        "action_gradient_finite": bool(torch.isfinite(action_gradient).all()),
        "parameter_gradient": parameter_gradient.detach().cpu(),
    }


def _flatten_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: tuple[torch.nn.Parameter, ...],
) -> torch.Tensor:
    values = [
        gradient.reshape(-1)
        if gradient is not None
        else torch.zeros_like(parameter).reshape(-1)
        for gradient, parameter in zip(gradients, parameters, strict=True)
    ]
    return torch.cat(values)


def _categorical_kl(
    target_logits: torch.Tensor, prior_logits: torch.Tensor
) -> torch.Tensor:
    target = target_logits.softmax(dim=-1)
    log_target = target_logits.log_softmax(dim=-1)
    log_prior = prior_logits.log_softmax(dim=-1)
    return (target * (log_target - log_prior)).sum(dim=-1).sum(dim=-1)


def _distribution_report(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().cpu().double().numpy().reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("raw free-nats KL values are invalid")
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
        "below_0_1_fraction": float(np.mean(array < 0.1)),
        "below_1_fraction": float(np.mean(array < 1.0)),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float((left.norm() * right.norm()).detach().cpu())
    if denominator <= 1.0e-20:
        return 0.0
    value = float((left @ right).detach().cpu()) / denominator
    return max(-1.0, min(1.0, value))


def _mean(
    reports: Sequence[Mapping[str, object]], condition: str, metric: str
) -> float:
    values = [
        float(report["conditions"][condition][metric]) for report in reports
    ]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("free-nats aggregate contains non-finite metrics")
    return sum(values) / len(values)
