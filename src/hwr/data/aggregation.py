"""Dataset aggregation from states visited by a learned policy."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Protocol, Sequence

from hwr.core.runtime import LegacyPolicy, LegacyRuntimeBackend
from hwr.data.dataset import BehaviorDataset, BehaviorSample, write_behavior_dataset
from hwr.data.vectorization import action_to_vector, observation_to_vector
from hwr.sim.specs import HouseholdTaskSpec


class ObservationExpert(Protocol):
    def action(self, observation): ...


def _base_samples(dataset: BehaviorDataset) -> list[BehaviorSample]:
    return [dataset.sample(index) for index in range(len(dataset))]


def aggregate_policy_dataset(
    root: Path,
    dataset_id: str,
    base_dataset: BehaviorDataset,
    task_spec: HouseholdTaskSpec,
    environment_factory: Callable[[], LegacyRuntimeBackend],
    expert: ObservationExpert,
    policy: LegacyPolicy,
    seeds: Sequence[int],
    *,
    expert_action_probability: float = 0.2,
) -> Path:
    if not seeds:
        raise ValueError("aggregation requires at least one seed")
    if not 0.0 <= expert_action_probability <= 1.0:
        raise ValueError("expert action probability must be between zero and one")
    samples = _base_samples(base_dataset)
    rng = random.Random(sum(seeds) + len(samples))
    for episode_index, seed in enumerate(seeds):
        episode_id = f"{dataset_id}-visited-{episode_index:05d}"
        environment = environment_factory()
        observation = environment.reset(seed=seed, task_id=task_spec.task_id)
        policy.reset(task_id=task_spec.task_id, seed=seed)
        for step_index in range(task_spec.max_steps):
            expert_action = expert.action(observation)
            learner_action = policy.infer((observation,))[0]
            samples.append(
                BehaviorSample(
                    episode_id=episode_id,
                    step_index=step_index,
                    task_id=task_spec.task_id,
                    observation=observation_to_vector(observation),
                    action=action_to_vector(expert_action),
                )
            )
            applied_action = (
                expert_action if rng.random() < expert_action_probability else learner_action
            )
            outcome = environment.apply(applied_action)
            observation = outcome.observation
            if outcome.terminated or outcome.truncated:
                break
        environment.close()
    return write_behavior_dataset(
        root,
        dataset_id,
        samples,
        metadata={
            "source": "dataset_aggregation",
            "parent_dataset": base_dataset.manifest["dataset_id"],
            "parent_checksum": base_dataset.manifest["checksum"],
            "seeds": list(seeds),
            "expert_action_probability": expert_action_probability,
        },
    )
