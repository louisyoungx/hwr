"""Distributional reward utilities used by world model and imagined value heads."""

from __future__ import annotations

import torch


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(value.abs())


def symexp(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.expm1(value.abs())


def two_hot_symlog(
    value: torch.Tensor, *, bins: int, limit: float
) -> torch.Tensor:
    transformed = symlog(value).clamp(-limit, limit)
    position = (transformed + limit) * (bins - 1) / (2.0 * limit)
    lower = position.floor().long().clamp(0, bins - 1)
    upper = (lower + 1).clamp(0, bins - 1)
    upper_weight = position - lower.to(position.dtype)
    lower_weight = 1.0 - upper_weight
    target = torch.zeros(*value.shape, bins, device=value.device, dtype=value.dtype)
    target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target


def reward_expectation(logits: torch.Tensor, *, limit: float) -> torch.Tensor:
    bins = logits.shape[-1]
    support = torch.linspace(-limit, limit, bins, device=logits.device, dtype=logits.dtype)
    return symexp((logits.softmax(dim=-1) * support).sum(dim=-1))
