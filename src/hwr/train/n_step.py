"""Multi-step return construction for delayed physical action effects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class NStepTargets:
    rewards: tuple[float, ...]
    next_indices: tuple[int, ...]
    bootstrap_discounts: tuple[float, ...]
    done: tuple[float, ...]


def build_n_step_targets(
    rewards: Sequence[float],
    done: Sequence[float],
    *,
    horizon: int,
    discount: float,
) -> NStepTargets:
    """Accumulate returns without crossing an Episode terminal transition."""
    if not rewards or len(rewards) != len(done):
        raise ValueError("n-step rewards and terminals must have equal non-zero length")
    if horizon <= 0 or not 0.0 <= discount <= 1.0:
        raise ValueError("n-step horizon or discount is invalid")
    if not all(math.isfinite(float(value)) for value in (*rewards, *done)):
        raise ValueError("n-step inputs must be finite")
    totals: list[float] = []
    next_indices: list[int] = []
    discounts: list[float] = []
    terminals: list[float] = []
    for start in range(len(rewards)):
        total = 0.0
        terminal = False
        end = start
        for offset in range(horizon):
            index = start + offset
            if index >= len(rewards):
                break
            total += discount**offset * float(rewards[index])
            end = index
            terminal = float(done[index]) > 0.5
            if terminal:
                break
        steps = end - start + 1
        totals.append(total)
        next_indices.append(end)
        discounts.append(0.0 if terminal else discount**steps)
        terminals.append(float(terminal))
    return NStepTargets(
        tuple(totals),
        tuple(next_indices),
        tuple(discounts),
        tuple(terminals),
    )
