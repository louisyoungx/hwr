"""Generate canonical Episode records and policy samples from an expert."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, Sequence

from hwr.core.runtime import RuntimeBackend
from hwr.core.types import EpisodeMetadata, StepRecord
from hwr.data.dataset import BehaviorSample, write_behavior_dataset
from hwr.data.episode import EpisodeRecorder
from hwr.data.vectorization import action_to_vector, observation_to_vector
from hwr.sim.specs import HouseholdTaskSpec


class ObservationExpert(Protocol):
    def action(self, observation): ...


def generate_expert_dataset(
    root: Path,
    dataset_id: str,
    task_spec: HouseholdTaskSpec,
    environment_factory: Callable[[], RuntimeBackend],
    expert: ObservationExpert,
    seeds: Sequence[int],
) -> Path:
    if not seeds:
        raise ValueError("at least one seed is required")
    episodes_root = root / f"{dataset_id}-episodes"
    episodes_root.mkdir(parents=True, exist_ok=False)
    samples: list[BehaviorSample] = []
    for episode_index, seed in enumerate(seeds):
        episode_id = f"{dataset_id}-{episode_index:05d}"
        environment = environment_factory()
        observation = environment.reset(seed=seed, task_id=task_spec.task_id)
        recorder = EpisodeRecorder(
            episodes_root,
            EpisodeMetadata(
                episode_id=episode_id,
                task_id=task_spec.task_id,
                robot_spec_version="mobile_manipulator_2d/v1",
                task_spec_version=task_spec.spec_version,
                source_type="simulation_expert",
                seed=seed,
                started_at_ns=observation.timestamp_ns,
            ),
        )
        for step_index in range(task_spec.max_steps):
            proposed = expert.action(observation)
            outcome = environment.apply(proposed)
            applied = outcome.info["applied_action"]
            recorder.append_step(StepRecord(observation, proposed, applied))
            for event in outcome.events:
                recorder.append_event(event)
            samples.append(
                BehaviorSample(
                    episode_id=episode_id,
                    step_index=step_index,
                    task_id=task_spec.task_id,
                    observation=observation_to_vector(observation),
                    action=action_to_vector(applied),
                )
            )
            observation = outcome.observation
            if outcome.terminated or outcome.truncated:
                break
        result = environment.result()
        environment.close()
        if result is None:
            raise RuntimeError("environment did not produce an episode result")
        recorder.close(result)
        if not result.success:
            raise RuntimeError(f"expert failed seed {seed}: {result.reason}")
    return write_behavior_dataset(
        root,
        dataset_id,
        samples,
        metadata={"source": "simulation_expert", "seeds": list(seeds)},
    )

