"""Task-independent Episode evidence updates for the foundation runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hwr.data.autonomous_trajectory import AutonomousEpisode
from hwr.train.foundation_frontier import FoundationLearningFrontierController
from hwr.train.foundation_learning_signals import EpisodeLearningEvidence
from hwr.train.foundation_online_types import FoundationEpisodeRecord
from hwr.train.learning_signals import failure_boundary_step
from hwr.train.task_sampling import OutcomeAdaptiveTaskSampler, TaskOutcome


def record_foundation_learning_outcomes(
    episodes: Sequence[AutonomousEpisode],
    learning_evidence: Mapping[str, EpisodeLearningEvidence],
    task_sampler: OutcomeAdaptiveTaskSampler,
    frontier: FoundationLearningFrontierController,
    records: list[FoundationEpisodeRecord],
    *,
    update_count: int,
) -> None:
    for episode in episodes:
        arrays = episode.arrays
        evidence = learning_evidence[episode.episode_id]
        signal = evidence.summary
        episode_return = float(arrays["reward"].sum())
        safety_rate = float(arrays["safety_intervention"].mean())
        success = bool(episode.metadata["success"])
        terminated_failure = bool(arrays["terminated"][-1]) and not success
        boundary = failure_boundary_step(
            arrays["safety_intervention"],
            terminated_failure=terminated_failure,
        )
        boundary_signal = (
            float(boundary + 1) / len(arrays["safety_intervention"])
            if boundary >= 0
            else 0.0
        )
        improvement = task_sampler.reward_improvement(
            episode.task_id, episode_return
        )
        task_sampler.record(
            episode.task_id,
            TaskOutcome(
                episode_return,
                signal.state_novelty,
                signal.td_error,
                improvement,
                boundary_signal,
                success,
                safety_rate,
            ),
        )
        frontier_result = frontier.consider(episode, evidence)
        records.append(
            FoundationEpisodeRecord(
                len(records),
                episode.task_id,
                episode.seed,
                str(arrays["action_source"][0]),
                episode_return,
                success,
                safety_rate,
                len(arrays["executed_action"]),
                update_count,
                signal.state_novelty,
                signal.td_error,
                improvement,
                boundary_signal,
                frontier_result.entries_added,
                frontier_result.reset_applied,
                frontier_result.reset_validated,
                frontier_result.reset_reproduced,
                frontier_result.source_episode,
                frontier_result.source_step,
            )
        )
