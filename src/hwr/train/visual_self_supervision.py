"""Label-free temporal objectives for deployable visual Actor inputs."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from hwr.policy.vla_model import VLAActorModel


def temporal_visual_contrastive_loss(
    actor: VLAActorModel,
    target_actor: VLAActorModel,
    current_inputs: Mapping[str, torch.Tensor],
    next_inputs: Mapping[str, torch.Tensor],
    actor_weights: torch.Tensor | None,
    *,
    temperature: float,
) -> torch.Tensor:
    """Match autonomous temporal successors without goals or action labels."""
    if temperature <= 0.0:
        raise ValueError("visual contrastive temperature must be positive")
    reference = next(iter(current_inputs.values()))
    eligible = (
        actor_weights > 0
        if actor_weights is not None
        else torch.ones(reference.shape[0], dtype=torch.bool, device=reference.device)
    )
    indices = torch.nonzero(eligible).flatten()
    if indices.numel() < 2:
        return reference.sum() * 0.0
    current = actor.visual_representation(_select(current_inputs, indices))
    with torch.no_grad():
        successor = target_actor.visual_representation(_select(next_inputs, indices))
    logits = current @ successor.transpose(0, 1) / temperature
    labels = torch.arange(indices.numel(), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


def _select(
    values: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {name: value[indices] for name, value in values.items()}
