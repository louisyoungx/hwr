"""Counterfactual action causality and open-loop world model diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from hwr.world_model.model import ActionConditionedWorldModel, WorldModelPriorRollout


@dataclass(frozen=True)
class CounterfactualCausalityReport:
    true_action_error: float
    shuffled_action_error: float
    shuffled_to_true_ratio: float
    true_horizon_errors: tuple[float, ...]
    shuffled_horizon_errors: tuple[float, ...]
    uncertainty_by_horizon: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_action_causality(
    model: ActionConditionedWorldModel,
    visual: torch.Tensor,
    language: torch.Tensor,
    proprioception: torch.Tensor,
    executed_actions: torch.Tensor,
) -> CounterfactualCausalityReport:
    if executed_actions.shape[1] < 2:
        raise ValueError("action causality evaluation requires at least two transitions")
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        initial = model.initial_posterior(
            visual[:, 0], language, proprioception[:, 0]
        )
        true_rollout = model.rollout_prior(initial, executed_actions, sample=False)
        shuffled = torch.roll(executed_actions, shifts=1, dims=1)
        shuffled_rollout = model.rollout_prior(initial, shuffled, sample=False)
        true_errors = _horizon_errors(
            true_rollout, visual[:, 1:], proprioception[:, 1:]
        )
        shuffled_errors = _horizon_errors(
            shuffled_rollout, visual[:, 1:], proprioception[:, 1:]
        )
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
    )


def _horizon_errors(
    rollout: WorldModelPriorRollout,
    visual: torch.Tensor,
    proprioception: torch.Tensor,
) -> torch.Tensor:
    visual_error = (rollout.visual_prediction - visual).square().mean(dim=-1)
    proprioception_error = (
        rollout.proprioception_prediction - proprioception
    ).square().mean(dim=-1)
    return (visual_error + proprioception_error).mean(dim=0)
