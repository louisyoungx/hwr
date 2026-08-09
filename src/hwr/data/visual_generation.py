"""Engine-independent collection of whitelisted visual expert demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from hwr.core.runtime import RuntimeBackend
from hwr.core.types import ActionFrame, ObservationFrame
from hwr.data.visual import (
    VisualBehaviorSample,
    VisualDatasetBuilder,
    extract_formal_policy_input,
    formal_action_vector,
)


class VisualTaskSpec(Protocol):
    task_id: str
    instruction: str
    max_steps: int
    control_hz: float


class PrivilegedExpertOutput(Protocol):
    action: ActionFrame
    privileged_label: bool


class VisualExpert(Protocol):
    failed: bool

    def action(self, observation: ObservationFrame) -> PrivilegedExpertOutput: ...


def generate_visual_expert_dataset(
    root: Path,
    dataset_id: str,
    task: VisualTaskSpec,
    environment_factory: Callable[[], RuntimeBackend],
    expert_factory: Callable[[RuntimeBackend], VisualExpert],
    seeds: Sequence[int],
    *,
    image_size: tuple[int, int] = (48, 36),
    action_history: int = 8,
    sample_stride: int = 4,
) -> Path:
    """Run successful physical episodes and store deployment-visible inputs only."""
    if not seeds or sample_stride <= 0:
        raise ValueError("seeds and a positive sample stride are required")
    builder = VisualDatasetBuilder(
        root,
        dataset_id,
        task_id=task.task_id,
        instruction=task.instruction,
        image_size=image_size,
        action_history=action_history,
        metadata={
            "source": "privileged_simulation_expert",
            "expert_fields_are_labels_only": True,
            "sample_stride": sample_stride,
            "control_hz": task.control_hz,
        },
    )
    for episode_index, seed in enumerate(seeds):
        _collect_episode(
            builder,
            episode_id=f"{dataset_id}-{episode_index:05d}",
            seed=seed,
            task=task,
            environment_factory=environment_factory,
            expert_factory=expert_factory,
            image_size=image_size,
            action_history=action_history,
            sample_stride=sample_stride,
        )
    return builder.seal()


def _collect_episode(
    builder: VisualDatasetBuilder,
    *,
    episode_id: str,
    seed: int,
    task: VisualTaskSpec,
    environment_factory: Callable[[], RuntimeBackend],
    expert_factory: Callable[[RuntimeBackend], VisualExpert],
    image_size: tuple[int, int],
    action_history: int,
    sample_stride: int,
) -> None:
    environment = environment_factory()
    samples: list[VisualBehaviorSample] = []
    history = [np.zeros(9, dtype=np.float32) for _ in range(action_history)]
    expert: VisualExpert | None = None
    try:
        observation = environment.reset(seed=seed, task_id=task.task_id)
        expert = expert_factory(environment)
        for step_index in range(task.max_steps):
            output = expert.action(observation)
            if not output.privileged_label:
                raise ValueError("formal demonstration lacks privileged-label provenance")
            policy_input = None
            if step_index % sample_stride == 0:
                policy_input = extract_formal_policy_input(
                    observation,
                    instruction_id=0,
                    action_history=history,
                    image_width=image_size[0],
                    image_height=image_size[1],
                )
            outcome = environment.apply(output.action)
            applied = outcome.info.get("applied_action")
            if not isinstance(applied, ActionFrame):
                raise ValueError("runtime did not report its applied action")
            applied_vector = formal_action_vector(applied)
            if policy_input is not None:
                samples.append(
                    VisualBehaviorSample(step_index, policy_input, applied_vector)
                )
            history = [*history[1:], applied_vector]
            observation = outcome.observation
            if outcome.terminated or outcome.truncated or expert.failed:
                break
        result = environment.result()
        if expert.failed:
            raise RuntimeError(f"expert controller failed seed {seed}")
        if result is None or not result.success:
            reason = "missing result" if result is None else result.reason
            raise RuntimeError(f"expert failed seed {seed}: {reason}")
        builder.write_episode(episode_id, seed, samples)
    finally:
        environment.close()
