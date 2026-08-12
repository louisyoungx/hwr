"""Episode-local, task-agnostic signals for autonomous RL curricula."""

from __future__ import annotations

from typing import Sequence


def reward_improvement_speeds(
    rewards: Sequence[float], *, smoothing: float = 0.90
) -> tuple[float, ...]:
    """Measure local reward improvement without copying an Episode score to states."""
    if not rewards:
        return ()
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("reward improvement smoothing must be in [0, 1)")
    baseline = float(rewards[0])
    values = [0.0]
    for reward in rewards[1:]:
        current = float(reward)
        values.append(current - baseline)
        baseline = smoothing * baseline + (1.0 - smoothing) * current
    return tuple(values)


def failure_boundary_step(
    safety_interventions: Sequence[float], *, terminated_failure: bool
) -> int:
    """Return the last safe state before an environment-declared failure only."""
    if not terminated_failure or len(safety_interventions) < 2:
        return -1
    safe = [
        index
        for index, intervention in enumerate(safety_interventions[:-1])
        if intervention <= 0.0
    ]
    return safe[-1] if safe else -1
