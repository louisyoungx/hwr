"""DAgger collection on visual-policy visited states with expert-only labels."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from hwr.core.runtime import Policy, RuntimeBackend
from hwr.core.types import ActionFrame, ObservationFrame
from hwr.data.visual import (
    FormalPolicyInput,
    VisualBehaviorSample,
    VisualDatasetBuilder,
    extract_formal_policy_input,
    formal_action_vector,
)
from hwr.data.visual_loading import LoadedVisualDataset, load_visual_dataset
from hwr.data.visual_phases import compact_household_phase


class AggregationTask(Protocol):
    task_id: str
    instruction: str
    max_steps: int


class AggregationExpertOutput(Protocol):
    action: ActionFrame
    stage: str
    privileged_label: bool


class AggregationExpert(Protocol):
    failed: bool

    def action(self, observation: ObservationFrame) -> AggregationExpertOutput: ...


def _sample(dataset: LoadedVisualDataset, index: int) -> VisualBehaviorSample:
    return VisualBehaviorSample(
        int(dataset.step_indices[index]),
        FormalPolicyInput(**{name: dataset.inputs[name][index] for name in dataset.inputs}),
        dataset.actions[index],
        phase=str(dataset.phases[index]),
    )


def _copy_base(builder: VisualDatasetBuilder, dataset: LoadedVisualDataset) -> None:
    offset = 0
    for shard in dataset.manifest["shards"]:
        count = int(shard["sample_count"])
        samples = [_sample(dataset, index) for index in range(offset, offset + count)]
        builder.write_episode(str(shard["episode_id"]), int(shard["seed"]), samples)
        offset += count


def aggregate_visual_policy_dataset(
    root: Path,
    dataset_id: str,
    base_path: Path,
    task: AggregationTask,
    environment_factory: Callable[[], RuntimeBackend],
    expert_factory: Callable[[RuntimeBackend], AggregationExpert],
    policy: Policy,
    seeds: Sequence[int],
    *,
    max_steps: int | None = None,
    sample_stride: int = 4,
) -> Path:
    if not seeds or sample_stride <= 0:
        raise ValueError("aggregation seeds and sample stride are required")
    base = load_visual_dataset(base_path)
    image_size = tuple(int(value) for value in base.manifest["image_size"])
    action_history = int(base.manifest["action_history"])
    metadata = dict(base.manifest["metadata"])
    metadata.update(
        {
            "source": "visual_dagger_expert_labels",
            "parent_dataset_id": base.manifest["dataset_id"],
            "aggregation_seeds": list(seeds),
            "behavior_policy_id": policy.spec().policy_id,
            "expert_action_probability": 0.0,
        }
    )
    builder = VisualDatasetBuilder(
        root,
        dataset_id,
        task_id=task.task_id,
        instruction=task.instruction,
        image_size=image_size,
        action_history=action_history,
        metadata=metadata,
    )
    builder.declare_phase_order(base.phase_names)
    _copy_base(builder, base)
    rollout_steps = task.max_steps if max_steps is None else min(max_steps, task.max_steps)
    for episode_index, seed in enumerate(seeds):
        _collect_visited_episode(
            builder,
            f"{dataset_id}-visited-{episode_index:05d}",
            seed,
            task,
            environment_factory,
            expert_factory,
            policy,
            rollout_steps,
            image_size,
            action_history,
            sample_stride,
        )
    return builder.seal()


def _collect_visited_episode(
    builder: VisualDatasetBuilder,
    episode_id: str,
    seed: int,
    task: AggregationTask,
    environment_factory: Callable[[], RuntimeBackend],
    expert_factory: Callable[[RuntimeBackend], AggregationExpert],
    policy: Policy,
    max_steps: int,
    image_size: tuple[int, int],
    action_history: int,
    sample_stride: int,
) -> None:
    environment = environment_factory()
    samples: list[VisualBehaviorSample] = []
    history = [np.zeros(9, dtype=np.float32) for _ in range(action_history)]
    try:
        observation = environment.reset(seed=seed, task_id=task.task_id)
        expert = expert_factory(environment)
        policy.reset(task_id=task.task_id, seed=seed)
        for step_index in range(max_steps):
            label = expert.action(observation)
            actions = policy.infer((observation,))
            if not label.privileged_label or len(actions) != 1:
                raise ValueError("aggregation requires one learned action and one expert label")
            behavior = actions[0]
            if not behavior.source.startswith("learned:"):
                raise ValueError("aggregation behavior must come from a learned policy")
            if step_index % sample_stride == 0:
                samples.append(
                    VisualBehaviorSample(
                        step_index,
                        extract_formal_policy_input(
                            observation,
                            instruction_id=0,
                            action_history=history,
                            image_width=image_size[0],
                            image_height=image_size[1],
                        ),
                        formal_action_vector(label.action),
                        phase=compact_household_phase(label.stage),
                    )
                )
            outcome = environment.apply(behavior)
            applied = outcome.info.get("applied_action")
            if not isinstance(applied, ActionFrame):
                raise ValueError("runtime did not report its applied aggregation action")
            history = [*history[1:], formal_action_vector(applied)]
            observation = outcome.observation
            if outcome.terminated or outcome.truncated or expert.failed:
                break
        if not samples:
            raise RuntimeError(f"aggregation seed {seed} produced no samples")
        builder.write_episode(episode_id, seed, samples)
    finally:
        environment.close()
