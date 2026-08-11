"""Failure-aware replay with critic-only HER and legal environment transforms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from hwr.tasks import BIMANUAL_GOAL_DIM
from hwr.train.asymmetric_replay import AsymmetricReplayBuffer
from hwr.train.asymmetric_rl import AsymmetricRLBatch
from hwr.train.environment_augmentation import (
    environment_transform_index,
    transform_action,
    transform_actor_inputs,
    transform_privileged,
)


PROGRESS_REPLAY_SCHEMA = "hwr.task-agnostic-reward-improvement/v3"


@dataclass(frozen=True)
class GoalEpisode:
    batch: AsymmetricRLBatch
    achieved_goals: torch.Tensor
    next_achieved_goals: torch.Tensor
    desired_goals: torch.Tensor
    success: bool
    legal_transforms: tuple[str, ...]

    def __post_init__(self) -> None:
        count = self.batch.rewards.shape[0]
        expected = (count, BIMANUAL_GOAL_DIM)
        for value in (
            self.achieved_goals,
            self.next_achieved_goals,
            self.desired_goals,
        ):
            if tuple(value.shape) != expected:
                raise ValueError("goal episode tensors have invalid shapes")
        state = self.batch.privileged_state
        next_state = self.batch.next_privileged_state
        if state.shape[1] < 2 * BIMANUAL_GOAL_DIM or next_state.shape != state.shape:
            raise ValueError("goal episode privileged state layout is invalid")
        if not torch.allclose(state[:, :BIMANUAL_GOAL_DIM], self.achieved_goals):
            raise ValueError("privileged state achieved-goal prefix differs")
        desired_slice = state[:, BIMANUAL_GOAL_DIM : 2 * BIMANUAL_GOAL_DIM]
        if not torch.allclose(desired_slice, self.desired_goals):
            raise ValueError("privileged state desired-goal prefix differs")
        for name in self.legal_transforms:
            environment_transform_index(name)


@dataclass(frozen=True)
class GoalReplayAddResult:
    original_count: int
    hindsight_count: int
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


def _hindsight_success(achieved: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    position = torch.linalg.vector_norm(achieved[:, :3] - desired[:, :3], dim=1)
    articulation = (achieved[:, 6] - desired[:, 6]).abs()
    relations = (achieved[:, 7:11] - desired[:, 7:11]).abs().amax(dim=1)
    safe = achieved[:, 11] == 0
    return (position <= 0.05) & (articulation <= 0.05) & (relations <= 0.5) & safe


def _hindsight_reward(achieved: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    position = torch.linalg.vector_norm(achieved[:, :3] - desired[:, :3], dim=1)
    articulation = (achieved[:, 6] - desired[:, 6]).abs()
    relations = (achieved[:, 7:11] - desired[:, 7:11]).abs().mean(dim=1)
    return -position - 0.25 * articulation - 0.25 * relations + _hindsight_success(
        achieved, desired
    ).to(achieved.dtype)


def hindsight_relabel(
    episode: GoalEpisode, generator: torch.Generator
) -> AsymmetricRLBatch:
    count = episode.batch.rewards.shape[0]
    future = torch.tensor(
        [
            int(torch.randint(index, count, (1,), generator=generator))
            for index in range(count)
        ],
        dtype=torch.long,
    )
    desired = episode.next_achieved_goals[future].clone()
    desired[:, 11] = 0.0
    state = episode.batch.privileged_state.clone()
    next_state = episode.batch.next_privileged_state.clone()
    goal_slice = slice(BIMANUAL_GOAL_DIM, 2 * BIMANUAL_GOAL_DIM)
    state[:, goal_slice] = desired
    next_state[:, goal_slice] = desired
    success = _hindsight_success(episode.next_achieved_goals, desired)
    bootstrap = episode.batch.bootstrap_discounts
    if bootstrap is not None:
        bootstrap = torch.where(success, torch.zeros_like(bootstrap), bootstrap)
    return AsymmetricRLBatch(
        actor_inputs=episode.batch.actor_inputs,
        next_actor_inputs=episode.batch.next_actor_inputs,
        privileged_state=state,
        next_privileged_state=next_state,
        action_chunks=episode.batch.action_chunks,
        stop_decisions=episode.batch.stop_decisions,
        rewards=_hindsight_reward(episode.next_achieved_goals, desired),
        done=success.to(episode.batch.done.dtype),
        actor_weights=torch.zeros_like(episode.batch.rewards),
        augmentation_transform_indices=(
            episode.batch.augmentation_transform_indices
        ),
        proposed_action_chunks=episode.batch.proposed_action_chunks,
        safety_costs=episode.batch.safety_costs,
        bootstrap_discounts=bootstrap,
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


class GoalConditionedReplayBuffer:
    """Store no labels; failed episodes receive dedicated return sampling."""

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
        self.hindsight_count = 0
        self.augmentation_count = 0

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

    def add_episode(self, episode: GoalEpisode) -> GoalReplayAddResult:
        original = _with_actor_weights(episode.batch, 1.0)
        hindsight = hindsight_relabel(episode, self._generator)
        batches = [original, hindsight]
        augmentation_count = 0
        transforms = episode.legal_transforms
        if transforms:
            index = environment_transform_index(transforms[0])
            original = _with_transform_index(original, index)
            hindsight = _with_transform_index(hindsight, index)
            batches = [original, hindsight]
        for transform_id in transforms:
            index = environment_transform_index(transform_id)
            transformed_original = transform_batch(
                _with_transform_index(original, index), transform_id
            )
            transformed_hindsight = transform_batch(
                _with_transform_index(hindsight, index), transform_id
            )
            batches.extend((transformed_original, transformed_hindsight))
            augmentation_count += 2 * original.rewards.shape[0]
        for batch in batches:
            self.regular.add(batch)
            if not episode.success:
                self.failures.add(batch)
        self._add_discoveries(original)
        self._add_progress_events(original)
        self._add_safety_events(original)
        for transform_id in transforms:
            transformed = transform_batch(original, transform_id)
            self._add_discoveries(transformed)
            self._add_progress_events(transformed)
            self._add_safety_events(transformed)
        count = original.rewards.shape[0]
        self.episode_count += 1
        self.hindsight_count += count
        self.augmentation_count += augmentation_count
        return GoalReplayAddResult(
            original_count=count,
            hindsight_count=count,
            augmentation_count=augmentation_count,
            failure_return_count=sum(batch.rewards.shape[0] for batch in batches)
            if not episode.success
            else 0,
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
        regular = self.regular.sample(regular_count)
        result = regular
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
        return result

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

    def _add_progress_events(self, batch: AsymmetricRLBatch) -> None:
        actor_eligible = (
            batch.actor_weights > 0
            if batch.actor_weights is not None
            else torch.ones_like(batch.rewards, dtype=torch.bool)
        )
        if batch.safety_costs is not None:
            actor_eligible &= batch.safety_costs <= 0.0
        indices = _top_ranked_indices(batch.rewards, actor_eligible)
        if indices.numel():
            self.progress_events.add(_slice_batch(batch, indices))
    def state_dict(self) -> dict[str, object]:
        return {
            "regular": self.regular.state_dict(),
            "failures": self.failures.state_dict(),
            "discoveries": self.discoveries.state_dict(),
            "progress_events": self.progress_events.state_dict(),
            "progress_event_schema": PROGRESS_REPLAY_SCHEMA,
            "safety_events": self.safety_events.state_dict(),
            "generator_state": self._generator.get_state(),
            "episode_count": self.episode_count,
            "hindsight_count": self.hindsight_count,
            "augmentation_count": self.augmentation_count,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        self.regular.load_state_dict(value["regular"])
        self.failures.load_state_dict(value["failures"])
        if "discoveries" in value:
            self.discoveries.load_state_dict(value["discoveries"])
        elif self.regular.size:
            self._add_discoveries(self.regular.all())
        if "safety_events" in value:
            self.safety_events.load_state_dict(value["safety_events"])
        elif self.regular.size:
            self._add_safety_events(self.regular.all())
        if (
            "progress_events" in value
            and value.get("progress_event_schema") == PROGRESS_REPLAY_SCHEMA
        ):
            self.progress_events.load_state_dict(value["progress_events"])
        elif self.regular.size:
            self._add_progress_events(self.regular.all())
        self._generator.set_state(value["generator_state"])
        self.episode_count = int(value["episode_count"])
        self.hindsight_count = int(value["hindsight_count"])
        self.augmentation_count = int(
            value.get("augmentation_count", value.get("mirror_count", 0))
        )


def _top_ranked_indices(
    values: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    indices = torch.nonzero(eligible).flatten()
    if not indices.numel():
        return indices
    keep = max(1, math.ceil(math.sqrt(indices.numel())))
    order = torch.argsort(values[indices], descending=True, stable=True)
    return indices[order[:keep]]
