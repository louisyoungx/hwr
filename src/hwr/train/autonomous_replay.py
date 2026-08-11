"""Failure-aware autonomous replay with sample-time environment transforms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from hwr.train.asymmetric_replay import AsymmetricReplayBuffer
from hwr.train.asymmetric_rl import AsymmetricRLBatch
from hwr.train.environment_augmentation import (
    environment_transform_index,
    environment_transform_name,
    transform_action,
    transform_actor_inputs,
    transform_privileged,
)
from hwr.train.learning_signals import reward_improvement_speeds


PROGRESS_REPLAY_SCHEMA = "hwr.task-agnostic-reward-improvement-speed/v4"
AUTONOMOUS_REPLAY_STORAGE_SCHEMA = "hwr.autonomous-replay-storage/v1"
SAMPLE_AUGMENTATION_PROBABILITY = 0.50


@dataclass(frozen=True)
class AutonomousEpisode:
    batch: AsymmetricRLBatch
    success: bool
    legal_transforms: tuple[str, ...]
    reward_improvements: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.batch.rewards.numel() == 0:
            raise ValueError("autonomous replay episode cannot be empty")
        if self.reward_improvements is not None:
            if self.reward_improvements.shape != self.batch.rewards.shape:
                raise ValueError("reward improvement scores must align with replay")
            if not torch.isfinite(self.reward_improvements).all():
                raise ValueError("reward improvement scores must be finite")
        for name in self.legal_transforms:
            environment_transform_index(name)


@dataclass(frozen=True)
class AutonomousReplayAddResult:
    original_count: int
    augmentation_count: int
    failure_return_count: int


def _slice_batch(batch: AsymmetricRLBatch, indices: torch.Tensor) -> AsymmetricRLBatch:
    select = lambda values: {name: value[indices] for name, value in values.items()}
    return AsymmetricRLBatch(
        actor_inputs=select(batch.actor_inputs),
        next_actor_inputs=select(batch.next_actor_inputs),
        privileged_state=batch.privileged_state[indices],
        next_privileged_state=batch.next_privileged_state[indices],
        action_chunks=batch.action_chunks[indices],
        stop_decisions=batch.stop_decisions[indices],
        rewards=batch.rewards[indices],
        done=batch.done[indices],
        actor_weights=(
            batch.actor_weights[indices] if batch.actor_weights is not None else None
        ),
        augmentation_transform_indices=(
            batch.augmentation_transform_indices[indices]
            if batch.augmentation_transform_indices is not None
            else None
        ),
        proposed_action_chunks=(
            batch.proposed_action_chunks[indices]
            if batch.proposed_action_chunks is not None
            else None
        ),
        safety_costs=(
            batch.safety_costs[indices] if batch.safety_costs is not None else None
        ),
        bootstrap_discounts=(
            batch.bootstrap_discounts[indices]
            if batch.bootstrap_discounts is not None
            else None
        ),
    )


def _with_actor_weights(batch: AsymmetricRLBatch, value: float) -> AsymmetricRLBatch:
    return AsymmetricRLBatch(
        actor_inputs=batch.actor_inputs,
        next_actor_inputs=batch.next_actor_inputs,
        privileged_state=batch.privileged_state,
        next_privileged_state=batch.next_privileged_state,
        action_chunks=batch.action_chunks,
        stop_decisions=batch.stop_decisions,
        rewards=batch.rewards,
        done=batch.done,
        actor_weights=torch.full_like(batch.rewards, value),
        augmentation_transform_indices=batch.augmentation_transform_indices,
        proposed_action_chunks=batch.proposed_action_chunks,
        safety_costs=batch.safety_costs,
        bootstrap_discounts=batch.bootstrap_discounts,
    )


def _with_transform_index(
    batch: AsymmetricRLBatch, index: int
) -> AsymmetricRLBatch:
    return AsymmetricRLBatch(
        actor_inputs=batch.actor_inputs,
        next_actor_inputs=batch.next_actor_inputs,
        privileged_state=batch.privileged_state,
        next_privileged_state=batch.next_privileged_state,
        action_chunks=batch.action_chunks,
        stop_decisions=batch.stop_decisions,
        rewards=batch.rewards,
        done=batch.done,
        actor_weights=batch.actor_weights,
        augmentation_transform_indices=torch.full_like(
            batch.rewards, index, dtype=torch.int64
        ),
        proposed_action_chunks=batch.proposed_action_chunks,
        safety_costs=batch.safety_costs,
        bootstrap_discounts=batch.bootstrap_discounts,
    )


def transform_batch(
    batch: AsymmetricRLBatch, transform_id: str
) -> AsymmetricRLBatch:
    if batch.privileged_state.shape[1] != 62:
        raise ValueError("environment augmentation requires the bimanual critic layout")
    return AsymmetricRLBatch(
        actor_inputs=transform_actor_inputs(batch.actor_inputs, transform_id),
        next_actor_inputs=transform_actor_inputs(
            batch.next_actor_inputs, transform_id
        ),
        privileged_state=transform_privileged(
            batch.privileged_state, transform_id
        ),
        next_privileged_state=transform_privileged(
            batch.next_privileged_state, transform_id
        ),
        action_chunks=transform_action(batch.action_chunks, transform_id),
        stop_decisions=batch.stop_decisions.clone(),
        rewards=batch.rewards.clone(),
        done=batch.done.clone(),
        actor_weights=(
            batch.actor_weights.clone() if batch.actor_weights is not None else None
        ),
        augmentation_transform_indices=(
            batch.augmentation_transform_indices.clone()
            if batch.augmentation_transform_indices is not None
            else None
        ),
        proposed_action_chunks=(
            transform_action(batch.proposed_action_chunks, transform_id)
            if batch.proposed_action_chunks is not None
            else None
        ),
        safety_costs=(
            batch.safety_costs.clone() if batch.safety_costs is not None else None
        ),
        bootstrap_discounts=(
            batch.bootstrap_discounts.clone()
            if batch.bootstrap_discounts is not None
            else None
        ),
    )


def _concat_batches(first: AsymmetricRLBatch, second: AsymmetricRLBatch) -> AsymmetricRLBatch:
    combine = lambda left, right: {
        name: torch.cat((left[name], right[name])) for name in left
    }
    weights = (
        torch.cat((first.actor_weights, second.actor_weights))
        if first.actor_weights is not None and second.actor_weights is not None
        else None
    )
    transform_indices = (
        torch.cat(
            (
                first.augmentation_transform_indices,
                second.augmentation_transform_indices,
            )
        )
        if first.augmentation_transform_indices is not None
        and second.augmentation_transform_indices is not None
        else None
    )
    proposals = (
        torch.cat((first.proposed_action_chunks, second.proposed_action_chunks))
        if first.proposed_action_chunks is not None
        and second.proposed_action_chunks is not None
        else None
    )
    safety = (
        torch.cat((first.safety_costs, second.safety_costs))
        if first.safety_costs is not None and second.safety_costs is not None
        else None
    )
    bootstrap = (
        torch.cat((first.bootstrap_discounts, second.bootstrap_discounts))
        if first.bootstrap_discounts is not None
        and second.bootstrap_discounts is not None
        else None
    )
    return AsymmetricRLBatch(
        actor_inputs=combine(first.actor_inputs, second.actor_inputs),
        next_actor_inputs=combine(first.next_actor_inputs, second.next_actor_inputs),
        privileged_state=torch.cat((first.privileged_state, second.privileged_state)),
        next_privileged_state=torch.cat(
            (first.next_privileged_state, second.next_privileged_state)
        ),
        action_chunks=torch.cat((first.action_chunks, second.action_chunks)),
        stop_decisions=torch.cat((first.stop_decisions, second.stop_decisions)),
        rewards=torch.cat((first.rewards, second.rewards)),
        done=torch.cat((first.done, second.done)),
        actor_weights=weights,
        augmentation_transform_indices=transform_indices,
        proposed_action_chunks=proposals,
        safety_costs=safety,
        bootstrap_discounts=bootstrap,
    )


class AutonomousReplayBuffer:
    """Keep autonomous transitions without synthetic action or goal labels."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        self.regular = AsymmetricReplayBuffer(capacity, seed=seed)
        self.failures = AsymmetricReplayBuffer(capacity, seed=seed ^ 0xFA17)
        self.discoveries = AsymmetricReplayBuffer(
            max(1, capacity // 8), seed=seed ^ 0xD15C
        )
        self.progress_events = AsymmetricReplayBuffer(
            max(1, capacity // 8), seed=seed ^ 0xA091
        )
        self.safety_events = AsymmetricReplayBuffer(
            max(1, capacity // 8), seed=seed ^ 0x5AFE
        )
        self._generator = torch.Generator().manual_seed(seed ^ 0x4E2)
        self.episode_count = 0
        self.legacy_discarded_hindsight_count = 0
        self.legacy_discarded_reward_priority_count = 0
        self.augmentation_count = 0
        self._load_migration_discarded_reward_priority_count = 0
        self._load_migration_rebuilt_reward_priority_count = 0

    @property
    def size(self) -> int:
        return self.regular.size

    @property
    def failure_size(self) -> int:
        return self.failures.size

    @property
    def discovery_size(self) -> int:
        return self.discoveries.size

    @property
    def safety_size(self) -> int:
        return self.safety_events.size

    @property
    def progress_size(self) -> int:
        return self.progress_events.size

    def add_episode(self, episode: AutonomousEpisode) -> AutonomousReplayAddResult:
        original = _with_actor_weights(episode.batch, 1.0)
        augmentation_count = 0
        transforms = episode.legal_transforms
        if transforms:
            index = environment_transform_index(transforms[0])
            original = _with_transform_index(original, index)
            augmentation_count = original.rewards.shape[0]
        self.regular.add(original)
        if not episode.success:
            self.failures.add(original)
        self._add_discoveries(original)
        self._add_progress_events(original, episode.reward_improvements)
        self._add_safety_events(original)
        count = original.rewards.shape[0]
        self.episode_count += 1
        self.augmentation_count += augmentation_count
        return AutonomousReplayAddResult(
            original_count=count,
            augmentation_count=augmentation_count,
            failure_return_count=count if not episode.success else 0,
        )

    def sample(
        self,
        batch_size: int,
        *,
        failure_fraction: float = 0.35,
        discovery_fraction: float = 0.35,
        progress_fraction: float = 0.0,
        safety_fraction: float = 0.15,
    ) -> AsymmetricRLBatch:
        fractions = (
            failure_fraction,
            discovery_fraction,
            progress_fraction,
            safety_fraction,
        )
        if not all(0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("replay fractions must be in [0, 1]")
        if sum(fractions) > 1.0 + 1e-9:
            raise ValueError("replay fractions cannot exceed one batch")
        safety_count = min(round(batch_size * safety_fraction), self.safety_size)
        safety_count = min(safety_count, max(0, batch_size - 1))
        progress_count = min(
            round(batch_size * progress_fraction), self.progress_size
        )
        progress_count = min(
            progress_count, max(0, batch_size - safety_count - 1)
        )
        discovery_count = min(
            round(batch_size * discovery_fraction), self.discovery_size
        )
        discovery_count = min(
            discovery_count,
            max(0, batch_size - safety_count - progress_count - 1),
        )
        failure_count = min(round(batch_size * failure_fraction), self.failure_size)
        failure_count = min(
            failure_count,
            max(
                0,
                batch_size
                - safety_count
                - progress_count
                - discovery_count
                - 1,
            ),
        )
        regular_count = (
            batch_size
            - safety_count
            - progress_count
            - discovery_count
            - failure_count
        )
        result = self.regular.sample(regular_count)
        if failure_count:
            result = _concat_batches(result, self.failures.sample(failure_count))
        if discovery_count:
            result = _concat_batches(result, self.discoveries.sample(discovery_count))
        if progress_count:
            result = _concat_batches(
                result, self.progress_events.sample(progress_count)
            )
        if safety_count:
            result = _concat_batches(result, self.safety_events.sample(safety_count))
        return _sample_time_augmentation(result, self._generator)

    def _add_discoveries(self, batch: AsymmetricRLBatch) -> None:
        actor_eligible = (
            batch.actor_weights > 0
            if batch.actor_weights is not None
            else torch.ones_like(batch.rewards, dtype=torch.bool)
        )
        if batch.safety_costs is not None:
            actor_eligible &= batch.safety_costs <= 0.0
        state = torch.nn.functional.normalize(batch.privileged_state, dim=1)
        next_state = torch.nn.functional.normalize(
            batch.next_privileged_state, dim=1
        )
        novelty = 1.0 - (state * next_state).sum(dim=1)
        indices = _top_ranked_indices(novelty, actor_eligible)
        if indices.numel():
            self.discoveries.add(_slice_batch(batch, indices))

    def _add_safety_events(self, batch: AsymmetricRLBatch) -> None:
        if batch.safety_costs is None:
            return
        actor_eligible = (
            batch.actor_weights > 0
            if batch.actor_weights is not None
            else torch.ones_like(batch.safety_costs, dtype=torch.bool)
        )
        indices = torch.nonzero(
            (batch.safety_costs > 0.5) & actor_eligible
        ).flatten()
        if indices.numel():
            self.safety_events.add(_slice_batch(batch, indices))

    def _add_progress_events(
        self,
        batch: AsymmetricRLBatch,
        scores: torch.Tensor | None = None,
        *,
        keep: int | None = None,
    ) -> int:
        actor_eligible = (
            batch.actor_weights > 0
            if batch.actor_weights is not None
            else torch.ones_like(batch.rewards, dtype=torch.bool)
        )
        if batch.safety_costs is not None:
            actor_eligible &= batch.safety_costs <= 0.0
        values = scores if scores is not None else _sequence_reward_improvements(batch)
        values = values.to(batch.rewards.device, dtype=batch.rewards.dtype)
        actor_eligible &= values > 0.0
        indices = _top_ranked_indices(values, actor_eligible, keep=keep)
        if indices.numel():
            self.progress_events.add(_slice_batch(batch, indices))
        return int(indices.numel())

    def _rebuild_progress_events(self) -> int:
        if not self.regular.size:
            return 0
        return self._add_progress_events(
            self.regular.chronological(),
            keep=self.progress_events.capacity,
        )

    def priority_migration_audit(self) -> dict[str, object] | None:
        discarded = self._load_migration_discarded_reward_priority_count
        rebuilt = self._load_migration_rebuilt_reward_priority_count
        if not discarded and not rebuilt:
            return None
        return {
            "schema_version": "hwr.reward-priority-migration/v1",
            "reason": "derived_priority_changed_to_local_reward_improvement_speed",
            "discarded_legacy_priority_rows": discarded,
            "rebuilt_priority_rows": rebuilt,
            "primary_autonomous_rows_retained": self.regular.size,
            "action_labels": False,
            "task_semantic_fields": [],
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "autonomous_replay_storage_schema": AUTONOMOUS_REPLAY_STORAGE_SCHEMA,
            "regular": self.regular.state_dict(),
            "failures": self.failures.state_dict(),
            "discoveries": self.discoveries.state_dict(),
            "progress_events": self.progress_events.state_dict(),
            "progress_event_schema": PROGRESS_REPLAY_SCHEMA,
            "safety_events": self.safety_events.state_dict(),
            "generator_state": self._generator.get_state(),
            "episode_count": self.episode_count,
            "legacy_discarded_hindsight_count": (
                self.legacy_discarded_hindsight_count
            ),
            "legacy_discarded_reward_priority_count": (
                self.legacy_discarded_reward_priority_count
            ),
            "augmentation_count": self.augmentation_count,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        self._load_migration_discarded_reward_priority_count = 0
        self._load_migration_rebuilt_reward_priority_count = 0
        current_schema = (
            value.get("autonomous_replay_storage_schema")
            == AUTONOMOUS_REPLAY_STORAGE_SCHEMA
        )
        if current_schema:
            self.regular.load_state_dict(value["regular"])
            self.failures.load_state_dict(value["failures"])
            self.legacy_discarded_hindsight_count = int(
                value.get("legacy_discarded_hindsight_count", 0)
            )
        else:
            self._load_expanded_legacy_stores(value)
        if current_schema and "discoveries" in value:
            self.discoveries.load_state_dict(value["discoveries"])
        elif self.regular.size:
            self._add_discoveries(self.regular.all())
        if current_schema and "safety_events" in value:
            self.safety_events.load_state_dict(value["safety_events"])
        elif self.regular.size:
            self._add_safety_events(self.regular.all())
        if (
            current_schema
            and "progress_events" in value
            and value.get("progress_event_schema") == PROGRESS_REPLAY_SCHEMA
        ):
            self.progress_events.load_state_dict(value["progress_events"])
        elif self.regular.size:
            discarded = int(value.get("progress_events", {}).get("size", 0))
            rebuilt = self._rebuild_progress_events()
            self._load_migration_discarded_reward_priority_count = discarded
            self._load_migration_rebuilt_reward_priority_count = rebuilt
        self._generator.set_state(value["generator_state"])
        self.episode_count = int(value["episode_count"])
        self.legacy_discarded_reward_priority_count = (
            int(value.get("legacy_discarded_reward_priority_count", 0))
            + self._load_migration_discarded_reward_priority_count
        )
        self.augmentation_count = int(
            value.get("augmentation_count", value.get("mirror_count", 0))
        )

    def _load_expanded_legacy_stores(self, value: Mapping[str, object]) -> None:
        regular = _load_legacy_buffer(value["regular"])
        if regular.size:
            batch = regular.all()
            actor = torch.nonzero(batch.actor_weights > 0).flatten()
            critic = torch.nonzero(batch.actor_weights <= 0).flatten()
            if actor.numel():
                self.regular.add(_slice_batch(batch, actor))
            self.legacy_discarded_hindsight_count += int(critic.numel())
        self.legacy_discarded_hindsight_count = max(
            self.legacy_discarded_hindsight_count,
            int(value.get("hindsight_count", 0)),
        )
        failures = _load_legacy_buffer(value["failures"])
        if failures.size:
            batch = failures.all()
            actor = torch.nonzero(batch.actor_weights > 0).flatten()
            if actor.numel():
                self.failures.add(_slice_batch(batch, actor))


def _load_legacy_buffer(value: Mapping[str, object]) -> AsymmetricReplayBuffer:
    buffer = AsymmetricReplayBuffer(int(value["capacity"]))
    buffer.load_state_dict(value)
    return buffer


def _sample_time_augmentation(
    batch: AsymmetricRLBatch, generator: torch.Generator
) -> AsymmetricRLBatch:
    indices = batch.augmentation_transform_indices
    if indices is None or not torch.count_nonzero(indices):
        return batch
    result = batch
    for index in torch.unique(indices).tolist():
        if int(index) <= 0:
            continue
        selected = (indices == index) & (
            torch.rand(indices.shape, generator=generator) < SAMPLE_AUGMENTATION_PROBABILITY
        )
        if selected.any():
            transformed = transform_batch(
                result, environment_transform_name(int(index))
            )
            result = _choose_batch_rows(result, transformed, selected)
    return result


def _choose_batch_rows(
    original: AsymmetricRLBatch,
    transformed: AsymmetricRLBatch,
    selected: torch.Tensor,
) -> AsymmetricRLBatch:
    choose = lambda left, right: torch.where(
        selected.reshape((-1,) + (1,) * (left.ndim - 1)), right, left
    )
    mapping = lambda field: {
        name: choose(getattr(original, field)[name], getattr(transformed, field)[name])
        for name in getattr(original, field)
    }
    optional = lambda field: (
        choose(getattr(original, field), getattr(transformed, field))
        if getattr(original, field) is not None
        else None
    )
    return AsymmetricRLBatch(
        actor_inputs=mapping("actor_inputs"),
        next_actor_inputs=mapping("next_actor_inputs"),
        privileged_state=choose(original.privileged_state, transformed.privileged_state),
        next_privileged_state=choose(
            original.next_privileged_state, transformed.next_privileged_state
        ),
        action_chunks=choose(original.action_chunks, transformed.action_chunks),
        stop_decisions=choose(original.stop_decisions, transformed.stop_decisions),
        rewards=choose(original.rewards, transformed.rewards),
        done=choose(original.done, transformed.done),
        actor_weights=optional("actor_weights"),
        augmentation_transform_indices=optional("augmentation_transform_indices"),
        proposed_action_chunks=optional("proposed_action_chunks"),
        safety_costs=optional("safety_costs"),
        bootstrap_discounts=optional("bootstrap_discounts"),
    )


def _top_ranked_indices(
    values: torch.Tensor,
    eligible: torch.Tensor,
    *,
    keep: int | None = None,
) -> torch.Tensor:
    indices = torch.nonzero(eligible).flatten()
    if not indices.numel():
        return indices
    count = (
        max(1, math.ceil(math.sqrt(indices.numel())))
        if keep is None
        else max(0, min(int(keep), indices.numel()))
    )
    order = torch.argsort(values[indices], descending=True, stable=True)
    return indices[order[:count]]


def _sequence_reward_improvements(batch: AsymmetricRLBatch) -> torch.Tensor:
    rewards = batch.rewards.detach().cpu().tolist()
    done = batch.done.detach().cpu().tolist()
    values: list[float] = []
    start = 0
    for index in range(1, len(rewards)):
        if done[index - 1] > 0.5 and done[index] <= 0.5:
            values.extend(reward_improvement_speeds(rewards[start:index]))
            start = index
    values.extend(reward_improvement_speeds(rewards[start:]))
    return torch.tensor(values, dtype=batch.rewards.dtype)
