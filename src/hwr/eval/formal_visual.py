"""Closed-loop evaluation that accepts only reloaded learned visual policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from hwr.core.runtime import Policy, RuntimeBackend


class AuditedRuntimeBackend(RuntimeBackend, Protocol):
    def audit_snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FormalEpisodeEvaluation:
    seed: int
    success: bool
    reason: str
    steps: int
    stable_steps: int
    severe_collisions: int
    maximum_forbidden_force: float
    action_sources: tuple[str, ...]
    audit: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FormalEvaluationReport:
    task_id: str
    policy_id: str
    episodes: tuple[FormalEpisodeEvaluation, ...]

    @property
    def success_rate(self) -> float:
        return sum(episode.success for episode in self.episodes) / len(self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "hwr.formal-visual-evaluation/v1",
            "task_id": self.task_id,
            "policy_id": self.policy_id,
            "episode_count": len(self.episodes),
            "success_count": sum(episode.success for episode in self.episodes),
            "success_rate": self.success_rate,
            "severe_collision_count": sum(
                episode.severe_collisions for episode in self.episodes
            ),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def evaluate_formal_visual_policy(
    task_id: str,
    max_steps: int,
    environment_factory: Callable[[], AuditedRuntimeBackend],
    policy: Policy,
    seeds: Sequence[int],
) -> FormalEvaluationReport:
    if len(seeds) < 1:
        raise ValueError("formal evaluation needs at least one seed")
    episodes = tuple(
        _evaluate_episode(task_id, max_steps, environment_factory, policy, seed)
        for seed in seeds
    )
    return FormalEvaluationReport(task_id, policy.spec().policy_id, episodes)


def _evaluate_episode(
    task_id: str,
    max_steps: int,
    environment_factory: Callable[[], AuditedRuntimeBackend],
    policy: Policy,
    seed: int,
) -> FormalEpisodeEvaluation:
    environment = environment_factory()
    sources: set[str] = set()
    steps = 0
    try:
        observation = environment.reset(seed=seed, task_id=task_id)
        policy.reset(task_id=task_id, seed=seed)
        for steps in range(1, max_steps + 1):
            actions = policy.infer((observation,))
            if len(actions) != 1:
                raise ValueError("formal baseline requires one action per control step")
            action = actions[0]
            if not action.source.startswith("learned:") or action.policy_version is None:
                raise ValueError("formal evaluation rejected a non-learned action source")
            sources.add(action.source)
            outcome = environment.apply(action)
            observation = outcome.observation
            if outcome.terminated or outcome.truncated:
                break
        result = environment.result()
        audit = environment.audit_snapshot()
        if result is None:
            raise RuntimeError("formal runtime produced no terminal result")
        metrics = result.metrics
        return FormalEpisodeEvaluation(
            seed=seed,
            success=result.success,
            reason=result.reason,
            steps=steps,
            stable_steps=int(metrics.get("stable_steps", audit.get("stable_steps", 0))),
            severe_collisions=int(
                metrics.get("severe_collisions", audit.get("severe_collision_count", 0))
            ),
            maximum_forbidden_force=float(
                metrics.get(
                    "maximum_forbidden_force", audit.get("maximum_forbidden_force", 0.0)
                )
            ),
            action_sources=tuple(sorted(sources)),
            audit=audit,
        )
    finally:
        environment.close()
