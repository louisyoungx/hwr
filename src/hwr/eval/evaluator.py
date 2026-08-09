"""Evaluate any Policy through the common RuntimeBackend contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from hwr.core.runtime import Policy, RuntimeBackend
from hwr.sim.specs import HouseholdTaskSpec


@dataclass(frozen=True)
class EvaluationReport:
    task_id: str
    episode_count: int
    success_count: int
    success_rate: float
    average_steps: float
    average_collisions: float
    reasons: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_policy(
    task_spec: HouseholdTaskSpec,
    environment_factory: Callable[[], RuntimeBackend],
    policy: Policy,
    seeds: Sequence[int],
) -> EvaluationReport:
    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    successes = 0
    total_steps = 0.0
    total_collisions = 0.0
    reasons: dict[str, int] = {}
    for seed in seeds:
        environment = environment_factory()
        observation = environment.reset(seed=seed, task_id=task_spec.task_id)
        policy.reset(task_id=task_spec.task_id, seed=seed)
        for _ in range(task_spec.max_steps):
            action_chunk = policy.infer((observation,))
            if not action_chunk:
                raise RuntimeError("policy returned an empty action chunk")
            outcome = environment.apply(action_chunk[0])
            observation = outcome.observation
            if outcome.terminated or outcome.truncated:
                break
        result = environment.result()
        environment.close()
        if result is None:
            raise RuntimeError("evaluation episode has no result")
        successes += int(result.success)
        total_steps += result.metrics.get("steps", 0.0)
        total_collisions += result.metrics.get("collisions", 0.0)
        reasons[result.reason] = reasons.get(result.reason, 0) + 1
    episode_count = len(seeds)
    return EvaluationReport(
        task_id=task_spec.task_id,
        episode_count=episode_count,
        success_count=successes,
        success_rate=successes / episode_count,
        average_steps=total_steps / episode_count,
        average_collisions=total_collisions / episode_count,
        reasons=reasons,
    )

