"""Build the independent diagnostics used to admit learned Actors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hwr.data.autonomous_trajectory import AppendableAutonomousTrajectoryStore
from hwr.data.foundation_cache import FoundationFeatureCache
from hwr.data.foundation_loading import (
    FoundationPreparedFeatures,
    FoundationSequenceBatchLoader,
)
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.policy.latent_actions import LatentActionScaling
from hwr.train.foundation_action_probe import evaluate_foundation_data_action_probe
from hwr.train.foundation_actor_readiness import FoundationActorReadinessTracker
from hwr.train.foundation_collision_validation import (
    CollisionValidationCriteria,
    evaluate_foundation_collision_validation,
)
from hwr.train.foundation_diagnostics import (
    evaluate_foundation_action_causality_audit,
)
from hwr.train.foundation_holdout import (
    HOLDOUT_COLLECTOR,
    causality_batches_by_task,
    causality_window_manifest,
    select_causality_windows,
)
from hwr.train.foundation_interaction_coverage import summarize_interaction_coverage
from hwr.train.foundation_metrics import summarize_replay_action_coverage
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_sequence_reservoir import count_source_episodes
from hwr.train.foundation_trainer import FoundationWorldModelTrainer
from hwr.world_model.evaluation import ActionCausalityCriteria


@dataclass(frozen=True)
class FoundationAdmissionResult:
    diagnostic: dict[str, object]
    readiness: dict[str, object]
    warmup_actor: str | None


def evaluate_foundation_actor_admission(
    trainer: FoundationWorldModelTrainer,
    replay_store: AppendableAutonomousTrajectoryStore,
    holdout_store: AppendableAutonomousTrajectoryStore,
    cache: FoundationFeatureCache,
    preprocessor: HighResolutionVisionPreprocessor,
    prepared: FoundationPreparedFeatures,
    action_scaling: LatentActionScaling,
    readiness_tracker: FoundationActorReadinessTracker,
    config: FoundationOnlineTrainingConfig,
    task_ids: tuple[str, ...],
) -> FoundationAdmissionResult:
    loader = FoundationSequenceBatchLoader(
        holdout_store.path,
        cache,
        preprocessor,
        trainer.visual_student.config,
        prepared,
        transitions=config.sequence_transitions,
        device=str(next(trainer.actor.parameters()).device),
    )
    selected = select_causality_windows(
        loader,
        task_ids,
        windows_per_task=config.causality_audit_windows_per_task,
        selection_seed=config.seed,
    )
    diagnostic = evaluate_foundation_action_causality_audit(
        trainer,
        causality_batches_by_task(
            loader, selected, batch_size=config.causality_audit_batch_size
        ),
        ActionCausalityCriteria(
            config.minimum_action_causality_ratio,
            config.minimum_action_causality_horizon_fraction,
        ),
        shuffle_seed=config.seed,
        shuffle_repeats=config.causality_shuffle_repeats,
    )
    diagnostic["window_selection"] = causality_window_manifest(loader, selected)
    diagnostic["holdout_collector"] = HOLDOUT_COLLECTOR
    action_coverage = summarize_replay_action_coverage(
        replay_store.path, replay_store.manifest, action_scaling
    )
    probe = evaluate_foundation_data_action_probe(
        replay_store.path,
        replay_store.manifest,
        holdout_store.path,
        holdout_store.manifest,
        bootstrap_seed=config.seed,
    )
    interaction = summarize_interaction_coverage(
        replay_store.path,
        replay_store.manifest,
        minimum_displacement=config.minimum_interaction_displacement,
        minimum_transitions=config.sequence_transitions,
    )
    collision_validation = evaluate_foundation_collision_validation(
        trainer,
        loader,
        task_ids,
        CollisionValidationCriteria(
            config.minimum_collision_validation_positive_episodes_per_task,
            config.minimum_collision_validation_negative_episodes_per_task,
            config.minimum_collision_validation_recall,
            config.minimum_collision_validation_pr_auc,
            config.maximum_collision_validation_brier_score,
            config.maximum_collision_validation_false_positive_rate,
            config.minimum_collision_validation_terminal_alignment,
            config.minimum_collision_validation_action_sensitivity_ratio,
        ),
        batch_size=config.causality_audit_batch_size,
    )
    diagnostic["collision_validation"] = collision_validation
    readiness = readiness_tracker.assess(
        diagnostic,
        probe,
        action_coverage,
        interaction,
        collision_validation,
        replay_episodes=count_source_episodes(replay_store.manifest),
    )
    warmup = _required_warmup(readiness_tracker)
    return FoundationAdmissionResult(diagnostic, readiness, warmup)


def _required_warmup(tracker: FoundationActorReadinessTracker) -> str | None:
    if tracker.task_actor_unlocked and tracker.task_actor_update_count == 0:
        return "task"
    if (
        tracker.exploration_unlocked
        and tracker.exploration_actor_update_count == 0
        and not tracker.task_actor_unlocked
    ):
        return "exploration"
    return None
