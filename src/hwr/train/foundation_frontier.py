"""Foundation-runner integration for the task-agnostic physical-state frontier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from hwr.core.embodied import DualArmObservation
from hwr.core.runtime import RuntimeBackend
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.data.autonomous_trajectory import AutonomousEpisode
from hwr.train.foundation_learning_signals import EpisodeLearningEvidence
from hwr.train.learning_frontier import (
    LearningFrontierBackend,
    LearningFrontierCandidate,
    LearningFrontierConfig,
    LearningFrontierEntry,
    LearningSignal,
    PreparedLearningFrontierReset,
    TaskAgnosticLearningFrontier,
    prepare_learning_frontier_reset,
)
from hwr.train.learning_signals import (
    failure_boundary_step,
    reward_improvement_speeds,
)


@dataclass
class PreparedFoundationFrontierCollection:
    initial_observation: DualArmObservation | None
    snapshots: list[PhysicalStateSnapshot]
    entry: LearningFrontierEntry | None
    reset: PreparedLearningFrontierReset | None


@dataclass(frozen=True)
class FoundationFrontierEpisodeResult:
    entries_added: int
    reset_applied: bool
    reset_validated: bool
    reset_reproduced: bool
    source_episode: int
    source_step: int


@dataclass(frozen=True)
class _PendingFrontierEpisode:
    episode_index: int
    snapshots: tuple[PhysicalStateSnapshot, ...]
    entry: LearningFrontierEntry | None
    reset: PreparedLearningFrontierReset | None


class FoundationLearningFrontierController:
    """Own reset selection and snapshot evidence outside policy inputs."""

    def __init__(
        self,
        task_ids: tuple[str, ...],
        config: LearningFrontierConfig,
        *,
        seed: int,
        episode_seed_base: int,
    ) -> None:
        self.frontier = TaskAgnosticLearningFrontier(task_ids, config)
        self.rng = np.random.default_rng(seed)
        self.episode_seed_base = episode_seed_base
        self._pending: dict[str, _PendingFrontierEpisode] = {}

    def prepare_collection(
        self,
        backend: RuntimeBackend,
        *,
        task_id: str,
        episode_index: int,
        episode_seed: int,
        resets_enabled: bool,
    ) -> PreparedFoundationFrontierCollection:
        if not isinstance(backend, LearningFrontierBackend):
            return PreparedFoundationFrontierCollection(None, [], None, None)
        entry = self.frontier.select(task_id, self.rng) if resets_enabled else None
        source_index = entry.source_episode if entry is not None else episode_index
        prepared = prepare_learning_frontier_reset(
            backend,
            entry,
            task_id=task_id,
            episode_seed=episode_seed,
            source_seed=self.episode_seed_base + source_index * 104729,
        )
        return PreparedFoundationFrontierCollection(
            prepared.observation, [], entry, prepared
        )

    def remember(
        self,
        episode_id: str,
        episode_index: int,
        prepared: PreparedFoundationFrontierCollection,
    ) -> None:
        if episode_id in self._pending:
            raise ValueError("frontier Episode evidence was already registered")
        if prepared.reset is None:
            return
        self._pending[episode_id] = _PendingFrontierEpisode(
            episode_index,
            tuple(prepared.snapshots),
            prepared.entry,
            prepared.reset,
        )

    def consider(
        self,
        episode: AutonomousEpisode,
        evidence: EpisodeLearningEvidence,
    ) -> FoundationFrontierEpisodeResult:
        pending = self._pending.pop(episode.episode_id, None)
        if pending is None:
            return FoundationFrontierEpisodeResult(0, False, False, False, -1, -1)
        candidates = self._candidates(episode, evidence, pending)
        added = self.frontier.consider_episode(episode.task_id, candidates)
        entry = pending.entry
        reset = pending.reset
        return FoundationFrontierEpisodeResult(
            added,
            bool(reset and reset.applied),
            bool(reset and reset.validated),
            bool(reset and reset.reproduced),
            entry.source_episode if entry is not None else -1,
            entry.source_step if entry is not None else -1,
        )

    def state_dict(self) -> dict[str, object]:
        if self._pending:
            raise RuntimeError("cannot checkpoint unconsumed frontier evidence")
        return {
            "frontier": self.frontier.state_dict(),
            "rng_state": self.rng.bit_generator.state,
            "episode_seed_base": self.episode_seed_base,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if int(value["episode_seed_base"]) != self.episode_seed_base:
            raise ValueError("frontier Episode seed base differs")
        self.frontier.load_state_dict(value["frontier"])
        self.rng.bit_generator.state = value["rng_state"]

    def audit(self) -> dict[str, object]:
        value = self.frontier.audit()
        value["reset_enable_gate"] = "physical_action_identifiability"
        value["policy_inputs"] = False
        value["environment_reward_used_for_action"] = False
        return value

    def _candidates(
        self,
        episode: AutonomousEpisode,
        evidence: EpisodeLearningEvidence,
        pending: _PendingFrontierEpisode,
    ) -> tuple[LearningFrontierCandidate, ...]:
        arrays = episode.arrays
        rewards = tuple(float(value) for value in arrays["reward"])
        interventions = tuple(
            float(value) for value in arrays["safety_intervention"]
        )
        terminated_failure = bool(arrays["terminated"][-1]) and not bool(
            episode.metadata["success"]
        )
        boundary = failure_boundary_step(
            interventions, terminated_failure=terminated_failure
        )
        improvement = reward_improvement_speeds(rewards)
        candidates = []
        for window in evidence.windows:
            step = window.source_step
            if (
                step < 0
                or step >= len(pending.snapshots)
                or bool(arrays["terminated"][step])
                or bool(arrays["truncated"][step])
            ):
                continue
            candidates.append(
                LearningFrontierCandidate(
                    pending.snapshots[step],
                    window.state_embedding,
                    LearningSignal(
                        self.frontier.state_novelty(
                            episode.task_id, window.state_embedding
                        ),
                        window.td_error,
                        improvement[step],
                        float(step == boundary),
                        safe=interventions[step] <= 0.0,
                    ),
                    pending.episode_index,
                    step,
                )
            )
        return tuple(candidates)
