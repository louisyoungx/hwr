"""Task-partitioned replay that prevents one scene from owning all capacity."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from hwr.train.asymmetric_rl import AsymmetricRLBatch
from hwr.train.goal_replay import (
    GoalConditionedReplayBuffer,
    GoalEpisode,
    GoalReplayAddResult,
)


def _concatenate(
    batches: Sequence[AsymmetricRLBatch],
) -> AsymmetricRLBatch:
    if not batches:
        raise ValueError("cannot concatenate empty task replay samples")
    combine = lambda field: {
        name: torch.cat([getattr(batch, field)[name] for batch in batches])
        for name in getattr(batches[0], field)
    }

    def optional(field: str):
        values = [getattr(batch, field) for batch in batches]
        return torch.cat(values) if all(value is not None for value in values) else None

    return AsymmetricRLBatch(
        actor_inputs=combine("actor_inputs"),
        next_actor_inputs=combine("next_actor_inputs"),
        privileged_state=torch.cat([batch.privileged_state for batch in batches]),
        next_privileged_state=torch.cat(
            [batch.next_privileged_state for batch in batches]
        ),
        action_chunks=torch.cat([batch.action_chunks for batch in batches]),
        stop_decisions=torch.cat([batch.stop_decisions for batch in batches]),
        rewards=torch.cat([batch.rewards for batch in batches]),
        done=torch.cat([batch.done for batch in batches]),
        actor_weights=optional("actor_weights"),
        proposed_action_chunks=optional("proposed_action_chunks"),
        safety_costs=optional("safety_costs"),
    )


class TaskPartitionedGoalReplayBuffer:
    """Keep bounded per-task stores and draw equal-sized task sub-batches."""

    def __init__(
        self, capacity: int, task_ids: Sequence[str], *, seed: int = 0
    ) -> None:
        identities = tuple(sorted(set(task_ids)))
        if capacity < len(identities) or not identities:
            raise ValueError("partitioned replay capacity or task identities are invalid")
        self.capacity = int(capacity)
        self.task_ids = identities
        per_task = max(1, capacity // len(identities))
        self.partitions = {
            task_id: GoalConditionedReplayBuffer(
                per_task, seed=seed ^ _stable_seed(task_id)
            )
            for task_id in identities
        }

    @property
    def size(self) -> int:
        return sum(partition.size for partition in self.partitions.values())

    @property
    def failure_size(self) -> int:
        return sum(
            partition.failure_size for partition in self.partitions.values()
        )

    @property
    def discovery_size(self) -> int:
        return sum(
            partition.discovery_size for partition in self.partitions.values()
        )

    @property
    def episode_count(self) -> int:
        return sum(
            partition.episode_count for partition in self.partitions.values()
        )

    @property
    def hindsight_count(self) -> int:
        return sum(
            partition.hindsight_count for partition in self.partitions.values()
        )

    @property
    def mirror_count(self) -> int:
        return sum(
            partition.mirror_count for partition in self.partitions.values()
        )

    def task_sizes(self) -> dict[str, int]:
        return {
            task_id: partition.size
            for task_id, partition in self.partitions.items()
        }

    def add_episode(
        self, task_id: str, episode: GoalEpisode
    ) -> GoalReplayAddResult:
        try:
            partition = self.partitions[task_id]
        except KeyError as exc:
            raise ValueError(f"partitioned replay does not know {task_id}") from exc
        return partition.add_episode(episode)

    def sample(
        self,
        batch_size: int,
        *,
        failure_fraction: float = 0.35,
        discovery_fraction: float = 0.35,
    ) -> AsymmetricRLBatch:
        active = [
            partition
            for partition in self.partitions.values()
            if partition.size > 0
        ]
        if batch_size <= 0 or sum(partition.size for partition in active) < batch_size:
            raise ValueError("partitioned replay cannot fill the requested batch")
        counts = _balanced_counts(batch_size, [item.size for item in active])
        batches = [
            partition.sample(
                count,
                failure_fraction=failure_fraction,
                discovery_fraction=discovery_fraction,
            )
            for partition, count in zip(active, counts, strict=True)
            if count
        ]
        return _concatenate(batches)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "task_ids": self.task_ids,
            "partitions": {
                task_id: partition.state_dict()
                for task_id, partition in self.partitions.items()
            },
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if int(value["capacity"]) != self.capacity:
            raise ValueError("partitioned replay checkpoint capacity differs")
        if tuple(value["task_ids"]) != self.task_ids:
            raise ValueError("partitioned replay checkpoint tasks differ")
        states = value["partitions"]
        for task_id, partition in self.partitions.items():
            partition.load_state_dict(states[task_id])


def _balanced_counts(batch_size: int, sizes: Sequence[int]) -> list[int]:
    counts = [0] * len(sizes)
    remaining = batch_size
    while remaining:
        progressed = False
        for index, size in enumerate(sizes):
            if counts[index] < size and remaining:
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("partitioned replay stores cannot fill batch")
    return counts


def _stable_seed(task_id: str) -> int:
    value = 2166136261
    for byte in task_id.encode("utf-8"):
        value = (value ^ byte) * 16777619 & 0xFFFFFFFF
    return value
