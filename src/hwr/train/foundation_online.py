"""One task-blind collect-materialize-update loop for all household tasks."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np

from hwr.core.runtime import RuntimeBackend
from hwr.data.autonomous_trajectory import AppendableAutonomousTrajectoryStore
from hwr.data.foundation_cache import FoundationFeatureCache
from hwr.data.foundation_features import LANGUAGE_PREPROCESS_SHA256, file_sha256
from hwr.data.foundation_loading import FoundationPreparedFeatures
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.policy.foundation_runtime import FoundationWorldModelPolicy
from hwr.train.accelerator_memory import release_unused_accelerator_memory
from hwr.train.foundation_actor_readiness import (
    FoundationActorReadinessTracker,
    actor_readiness_criteria_from_config,
    failed_exploration_calibration_checks,
)
from hwr.train.foundation_admission import evaluate_foundation_actor_admission
from hwr.train.foundation_collection import (
    AutonomousCollectionConfig,
    AutonomousEpisodeCollector,
    CurrentRLActorActionSource,
    IntrinsicRLActorActionSource,
)
from hwr.train.foundation_diagnostics import (
    foundation_action_causality_qualified,
    publish_action_causality_report,
)
from hwr.train.foundation_holdout import collect_causality_holdout
from hwr.train.foundation_learning_signals import (
    EpisodeLearningEvidence,
    evaluate_replay_episode_learning_evidence,
)
from hwr.train.foundation_frontier import FoundationLearningFrontierController
from hwr.train.foundation_materialization import materialize_foundation_replay_features
from hwr.train.foundation_metrics import (
    FoundationMetricsStore,
    build_foundation_cycle_metrics,
    publish_foundation_progress,
)
from hwr.train.foundation_exploration import RandomRLActionSource, RandomRLExplorationConfig
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_online_types import (
    FoundationEnvironmentFactory,
    FoundationEpisodeRecord,
    FoundationOnlineTrainingResult,
    FoundationProviderFactories,
    FoundationTaskInterface,
)
from hwr.train.foundation_outcomes import record_foundation_learning_outcomes
from hwr.train.foundation_registry import (
    ACTION_CAUSALITY_SCHEMA,
    export_foundation_deployment,
    foundation_deployment_qualified,
    file_sha256 as registry_file_sha256,
    load_foundation_training_checkpoint,
    prune_versioned_artifacts,
    require_foundation_lineage,
    save_foundation_training_checkpoint,
)
from hwr.train.foundation_run_manifest import write_or_verify_foundation_run_manifest
from hwr.train.foundation_recovery import (
    capture_torch_rng_state,
    clear_replay_archive,
    publish_runner_progress,
    restore_torch_rng_state,
    restore_runner_progress,
)
from hwr.train.foundation_replay_features import (
    discard_visual_feature_sources,
    language_resolver_from_replay,
)
from hwr.train.foundation_resource_budget import require_foundation_resource_budget
from hwr.train.foundation_setup import FoundationLearningStack
from hwr.train.foundation_sequence_reservoir import append_episode_sequence_evidence
from hwr.train.foundation_cycle_updates import run_replay_updates, warm_start_actor
from hwr.train.learning_frontier import LearningFrontierConfig
from hwr.train.task_sampling import OutcomeAdaptiveTaskSampler
from hwr.world_model.deploy import DeployableWorldModelStateFilter
class FoundationOnlineTrainingRunner:
    """All tasks share the same models, optimizer, replay, and update code."""

    def __init__(
        self,
        tasks: Mapping[str, FoundationTaskInterface],
        environment_factory: FoundationEnvironmentFactory,
        preprocessor: HighResolutionVisionPreprocessor,
        providers: FoundationProviderFactories,
        learning_stack: FoundationLearningStack,
        config: FoundationOnlineTrainingConfig,
        run_path: Path,
        *,
        source_commit: str,
        development_ready_sha256: str,
        execution: Mapping[str, object],
    ) -> None:
        if len(tasks) != 3 or set(tasks) != {item.task_id for item in tasks.values()}:
            raise ValueError("foundation training requires exactly three task interfaces")
        if not source_commit:
            raise ValueError("foundation training source commit is required")
        if (
            len(development_ready_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in development_ready_sha256
            )
        ):
            raise ValueError("foundation development readiness hash is invalid")
        self.tasks = dict(tasks)
        self.task_ids = tuple(sorted(tasks))
        self.environment_factory = environment_factory
        self.preprocessor = preprocessor
        self.providers = providers
        self.stack = learning_stack
        self.config = config
        self.run_path = run_path
        self.source_commit = source_commit
        self.development_ready_sha256 = development_ready_sha256
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.store = AppendableAutonomousTrajectoryStore(
            self.run_path / "replay", "autonomous"
        )
        self.causality_store = AppendableAutonomousTrajectoryStore(
            self.run_path / "causality-holdout", "autonomous"
        )
        self.cache = FoundationFeatureCache(self.run_path / "feature-cache")
        self.task_sampler = OutcomeAdaptiveTaskSampler(self.task_ids)
        self.rng = np.random.default_rng(config.seed)
        self.records: list[FoundationEpisodeRecord] = []
        self.random_exploration = RandomRLExplorationConfig(
            config.random_exploration_motion_correlation,
            config.random_exploration_gripper_flip_probability,
        )
        self.latest_checkpoint: Path | None = None
        self.latest_deployment: Path | None = None
        self.latest_action_causality: dict[str, object] | None = None
        self.latest_action_causality_report: Path | None = None
        self.completed_cycles = 0
        self.actor_readiness = FoundationActorReadinessTracker(
            actor_readiness_criteria_from_config(config)
        )
        self.frontier = FoundationLearningFrontierController(
            self.task_ids,
            LearningFrontierConfig(
                capacity_per_task=config.learning_frontier_capacity_per_task,
                reset_probability=config.learning_frontier_reset_probability,
                candidates_per_episode=(
                    config.learning_frontier_candidates_per_episode
                ),
                signature_uniform_fraction=(
                    config.learning_frontier_signature_uniform_fraction
                ),
                maximum_entries_per_source_signature=(
                    config.learning_frontier_maximum_entries_per_source_signature
                ),
            ),
            seed=config.seed + 1_000_003,
            episode_seed_base=config.seed,
        )
        self.metrics_store = FoundationMetricsStore(
            self.run_path,
            source_commit=self.source_commit,
            target_episodes=self.config.episodes,
        )
        write_or_verify_foundation_run_manifest(
            self.run_path,
            source_commit=self.source_commit,
            development_ready_sha256=self.development_ready_sha256,
            training_config=self.config.to_dict(),
            tasks=[asdict(self.tasks[name]) for name in self.task_ids],
            preprocessing={
                "fingerprint": self.preprocessor.fingerprint,
                "config": asdict(self.preprocessor.config),
            },
            execution=execution,
        )

    def train(self) -> FoundationOnlineTrainingResult:
        require_foundation_resource_budget(
            self.run_path, self.config, task_count=len(self.task_ids)
        )
        environments = {
            task_id: self.environment_factory(
                task_id, self.config.camera_width, self.config.camera_height
            )
            for task_id in self.task_ids
        }
        cycle = self.completed_cycles
        try:
            self._publish_progress("preparing_causality_holdout", cycle)
            self._prepare_causality_holdout(environments)
            while len(self.records) < self.config.episodes:
                next_cycle = cycle + 1
                timings: dict[str, float] = {}
                started = time.perf_counter()
                self._publish_progress("collecting", next_cycle)
                collected = self._collect_cycle(environments)
                self._bound_replay_storage()
                timings["collection_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                self._publish_progress("materializing_features", next_cycle)
                prepared, causality_prepared = self._materialize_features()
                timings["materialization_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                metrics = self._update_cycle(prepared, next_cycle)
                timings["update_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                self._publish_progress("evaluating", next_cycle, metrics)
                evidence = self._episode_learning_signals(prepared, collected)
                newly_unlocked = self._evaluate_action_causality(
                    causality_prepared, metrics
                )
                if newly_unlocked is not None:
                    warmup = self._warm_start_actor(
                        prepared, next_cycle, newly_unlocked
                    )
                    metrics.update(warmup)
                self._record_learning_outcomes(collected, evidence)
                timings["evaluation_seconds"] = time.perf_counter() - started
                cycle = next_cycle
                self.completed_cycles = cycle
                if cycle % self.config.checkpoint_interval_cycles == 0:
                    started = time.perf_counter()
                    self._publish_progress("checkpointing", cycle, metrics)
                    self._checkpoint(cycle, prepared)
                    timings["checkpoint_seconds"] = time.perf_counter() - started
                self.metrics_store.publish_cycle(
                    cycle,
                    build_foundation_cycle_metrics(
                        collected,
                        metrics,
                        timings,
                        self.stack.action_scaling,
                        update_count=self.stack.trainer.update_count,
                        episode_count=len(self.records),
                        action_causality=(
                            self.latest_action_causality["assessment"]
                            if self.latest_action_causality is not None
                            else None
                        ),
                        actor_readiness=self.actor_readiness.last_assessment,
                        learning_frontier=self.frontier.audit(),
                    ),
                )
                if cycle % self.config.checkpoint_interval_cycles == 0:
                    self._raise_if_calibration_failed()
        finally:
            for environment in environments.values():
                environment.close()
        if self.latest_checkpoint is None:
            prepared, causality_prepared = self._materialize_features()
            self._evaluate_action_causality(causality_prepared, {})
            self._checkpoint(cycle, prepared)
        return self.result()

    def result(self) -> FoundationOnlineTrainingResult:
        if (
            self.latest_checkpoint is None
            or self.latest_action_causality_report is None
        ):
            raise RuntimeError("foundation training has no published checkpoint")
        if (
            self.latest_deployment is None
            or self.latest_action_causality is None
            or not foundation_action_causality_qualified(
                self.latest_action_causality
            )
        ):
            raise RuntimeError(
                "foundation training has no causality-qualified deployment"
            )
        return FoundationOnlineTrainingResult(
            tuple(self.records),
            self.stack.trainer.update_count,
            self.store.path,
            self.latest_checkpoint,
            self.latest_deployment,
            self.latest_action_causality_report,
        )

    def resume_latest(self) -> None:
        latest_path = self.run_path / "latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(latest_path)
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if latest.get("schema_version") != "hwr.foundation-online-latest/v1":
            raise ValueError("resumed latest schema differs")
        checkpoint = self.run_path / latest["training_checkpoint"]
        report = self.run_path / latest["action_causality_report"]
        if registry_file_sha256(report) != latest["action_causality_sha256"]:
            raise ValueError("resumed action causality report hash differs")
        diagnostic = json.loads(report.read_text(encoding="utf-8"))
        if (
            diagnostic.get("schema_version") != ACTION_CAUSALITY_SCHEMA
            or diagnostic.get("source_commit") != self.source_commit
            or int(diagnostic.get("update_count", -1))
            != int(latest.get("update_count", -2))
        ):
            raise ValueError("resumed action causality lineage differs")
        checkpoint_manifest = json.loads(
            (checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
        require_foundation_lineage(
            checkpoint_manifest.get("lineage"), source_commit=self.source_commit
        )
        expected = checkpoint_manifest.get("training_diagnostics")
        if not isinstance(expected, Mapping):
            raise ValueError("resumed checkpoint diagnostics are missing")
        if (
            expected.get("action_causality_report_sha256")
            != latest["action_causality_sha256"]
            or expected.get("action_causality_passed")
            != foundation_action_causality_qualified(diagnostic)
        ):
            raise ValueError("resumed checkpoint diagnostic provenance differs")
        deployment = latest.get("deployment")
        deployment_path = (
            self.run_path / str(deployment) if deployment is not None else None
        )
        if deployment_path is not None:
            deployment_manifest = json.loads(
                (deployment_path / "manifest.json").read_text(encoding="utf-8")
            )
            if not foundation_deployment_qualified(expected):
                raise ValueError("unqualified checkpoint exposed a deployment")
            if deployment_manifest.get("training_diagnostics") != expected:
                raise ValueError("resumed deployment diagnostic provenance differs")
            artifact = checkpoint / str(checkpoint_manifest["artifact_file"])
            if deployment_manifest.get(
                "training_checkpoint_sha256"
            ) != registry_file_sha256(artifact):
                raise ValueError("resumed deployment checkpoint hash differs")
        elif foundation_deployment_qualified(expected):
            raise ValueError("qualified checkpoint is missing its deployment")
        load_foundation_training_checkpoint(checkpoint, self.stack.trainer)
        restored = restore_runner_progress(
            self.run_path,
            checkpoint,
            latest,
            self.store,
            self.causality_store,
            replay_archive=self.run_path / "recovery/replay-prune-archive",
        )
        replay_sha = registry_file_sha256(self.store.path / "manifest.json")
        audit_sha = registry_file_sha256(
            self.causality_store.path / "manifest.json"
        )
        if (
            checkpoint_manifest.get("data_manifest_sha256") != replay_sha
            or diagnostic.get("training_data_manifest_sha256") != replay_sha
            or diagnostic.get("audit_data_manifest_sha256") != audit_sha
        ):
            raise ValueError("resumed checkpoint data provenance differs")
        self.task_sampler.load_state_dict(restored.task_sampler)
        self.actor_readiness.load_state_dict(restored.actor_readiness)
        self.frontier.load_state_dict(restored.learning_frontier)
        self.rng.bit_generator.state = restored.rng_state
        restore_torch_rng_state(
            restored.torch_rng_state,
            next(self.stack.trainer.actor.parameters()).device,
        )
        self.records = [FoundationEpisodeRecord(**item) for item in restored.records]
        self.completed_cycles = restored.cycle
        self.metrics_store.rollback_after(restored.cycle)
        self._discard_cached_visual_sources(
            restored.discarded_observation_sources
        )
        self.latest_checkpoint = checkpoint
        self.latest_deployment = deployment_path
        self.latest_action_causality_report = report
        self.latest_action_causality = diagnostic

    def _collect_cycle(
        self, environments: Mapping[str, RuntimeBackend]
    ) -> list[object]:
        limit = min(
            self.config.collection_episodes_per_cycle,
            self.config.episodes - len(self.records),
        )
        collected = []
        for _ in range(limit):
            episode_index = len(self.records) + len(collected)
            task_id, _ = self.task_sampler.sample(self.rng)
            task = self.tasks[task_id]
            seed = self.config.seed + episode_index * 104729
            if self.actor_readiness.task_actor_ready_for_collection:
                source = CurrentRLActorActionSource(self._collection_policy())
            elif self.actor_readiness.exploration_ready_for_collection:
                source = IntrinsicRLActorActionSource(
                    self._collection_policy(exploration=True)
                )
            else:
                source = self._random_action_source()
            collector = AutonomousEpisodeCollector(
                self.preprocessor,
                AutonomousCollectionConfig(
                    "mujoco-bimanual-runtime/v2",
                    self.source_commit,
                    task.maximum_steps,
                ),
            )
            prepared = self.frontier.prepare_collection(
                environments[task_id],
                task_id=task_id,
                episode_index=episode_index,
                episode_seed=seed,
                resets_enabled=self.actor_readiness.exploration_ready_for_collection,
            )
            episode = collector.collect(
                environments[task_id],
                source,
                task_id=task_id,
                seed=seed,
                initial_observation=prepared.initial_observation,
                snapshot_sink=(
                    prepared.snapshots if prepared.reset is not None else None
                ),
            )
            self.frontier.remember(episode.episode_id, episode_index, prepared)
            append_episode_sequence_evidence(
                self.store,
                episode,
                sequence_transitions=self.config.sequence_transitions,
                windows_per_episode=self.config.replay_windows_per_episode,
            )
            collected.append(episode)
        return collected

    def _collection_policy(
        self, *, exploration: bool = False
    ) -> FoundationWorldModelPolicy:
        resolver = language_resolver_from_replay(
            self.store.manifest["shards"],
            self.cache,
            self.run_path / "features/language.json",
        )
        trainer = self.stack.trainer
        device = str(next(trainer.actor.parameters()).device)
        return FoundationWorldModelPolicy(
            copy.deepcopy(trainer.visual_student),
            DeployableWorldModelStateFilter.from_world_model(trainer.world_model),
            copy.deepcopy(
                trainer.exploration_actor if exploration else trainer.actor
            ),
            self.preprocessor,
            resolver,
            self.stack.action_scaling,
            policy_id=(
                f"foundation-{'exploration-' if exploration else ''}actor-"
                f"update-{trainer.update_count}"
            ),
            device=device,
        )

    def _prepare_causality_holdout(
        self, environments: Mapping[str, RuntimeBackend]
    ) -> None:
        collect_causality_holdout(
            self.causality_store,
            environments,
            {task_id: task.maximum_steps for task_id, task in self.tasks.items()},
            self.preprocessor,
            self.stack.action_scaling,
            exploration_config=self.random_exploration,
            episodes_per_task=self.config.causality_holdout_episodes_per_task,
            windows_per_episode=(
                self.config.causality_audit_windows_per_task
                // self.config.causality_holdout_episodes_per_task
            ),
            sequence_transitions=self.config.sequence_transitions,
            retained_transitions_per_episode=(
                self.config.causality_holdout_transitions_per_episode
            ),
            maximum_attempts_per_episode=(
                self.config.causality_holdout_maximum_attempts_per_episode
            ),
            base_seed=self.config.seed,
            source_commit=self.source_commit,
        )

    def _random_action_source(self) -> RandomRLActionSource:
        return RandomRLActionSource(
            self.stack.action_scaling,
            self.random_exploration,
        )

    def _materialize_features(
        self,
    ) -> tuple[FoundationPreparedFeatures, FoundationPreparedFeatures]:
        return materialize_foundation_replay_features(
            self.store.path,
            self.causality_store.path,
            self.cache,
            self.preprocessor,
            self.run_path / "features",
            self.run_path / "causality-holdout/features",
            vision_language_factory=self.providers.vision_language,
            dense_vision_factory=self.providers.dense_vision,
            language_factory=self.providers.language,
        )

    def _bound_replay_storage(self) -> None:
        base, remainder = divmod(
            self.config.replay_transition_capacity, len(self.task_ids)
        )
        capacities = {
            task_id: base + int(index < remainder)
            for index, task_id in enumerate(self.task_ids)
        }
        evicted_sources = self.store.prune_to_task_capacities(
            capacities,
            recovery_archive=self.run_path / "recovery/replay-prune-archive",
        )
        self._discard_cached_visual_sources(evicted_sources)

    def _discard_cached_visual_sources(self, sources: tuple[str, ...]) -> None:
        discard_visual_feature_sources(
            tuple(sources),
            self.cache,
            self.preprocessor,
            (
                self.run_path / "features/vision-language.json",
                self.run_path / "features/dense-vision.json",
            ),
        )

    def _update_cycle(
        self, prepared: FoundationPreparedFeatures, cycle: int
    ) -> dict[str, float]:
        result = run_replay_updates(
            self.stack.trainer,
            self.store.path,
            self.cache,
            self.preprocessor,
            prepared,
            self.rng,
            self.config,
            updates=self.config.updates_per_cycle,
            train_task_actor=self.actor_readiness.task_actor_ready_for_collection,
            train_exploration_actor=(
                self.actor_readiness.exploration_ready_for_collection
                and not self.actor_readiness.task_actor_ready_for_collection
            ),
            progress=lambda values, completed: self._publish_progress(
                "updating",
                cycle,
                values,
                completed_updates=completed,
            ),
        )
        if self.actor_readiness.task_actor_ready_for_collection:
            self.actor_readiness.record_task_actor_updates(
                self.config.updates_per_cycle
            )
        elif self.actor_readiness.exploration_ready_for_collection:
            self.actor_readiness.record_exploration_actor_updates(
                self.config.updates_per_cycle
            )
        return result

    def _publish_progress(
        self,
        stage: str,
        cycle: int,
        metrics: Mapping[str, float] | None = None,
        *,
        completed_updates: int = 0,
    ) -> None:
        publish_foundation_progress(
            self.metrics_store,
            stage,
            cycle,
            self.stack.trainer.update_count,
            len(self.records),
            self.config.updates_per_cycle,
            metrics=metrics,
            completed_updates=completed_updates,
        )

    def _evaluate_action_causality(
        self,
        prepared: FoundationPreparedFeatures,
        metrics: dict[str, float],
    ) -> str | None:
        result = evaluate_foundation_actor_admission(
            self.stack.trainer,
            self.store,
            self.causality_store,
            self.cache,
            self.preprocessor,
            prepared,
            self.stack.action_scaling,
            self.actor_readiness,
            self.config,
            self.task_ids,
        )
        self.latest_action_causality = result.diagnostic
        assessment = result.diagnostic["assessment"]
        metrics["world/action_causality_ratio"] = float(
            assessment["shuffled_to_true_ratio"]
        )
        metrics["world/action_causality_passed"] = float(
            foundation_action_causality_qualified(result.diagnostic)
        )
        metrics["actor_readiness/unlocked"] = float(result.readiness["unlocked"])
        metrics["actor_readiness/consecutive_passes"] = float(
            result.readiness["consecutive_passes"]
        )
        return result.warmup_actor

    def _warm_start_actor(
        self,
        prepared: FoundationPreparedFeatures,
        cycle: int,
        actor_kind: str,
    ) -> dict[str, float]:
        if actor_kind not in {"exploration", "task"}:
            raise ValueError("unknown foundation Actor warmup kind")
        self._publish_progress("warming_actor", cycle)
        metrics = warm_start_actor(
            self.stack.trainer,
            self.store.path,
            self.cache,
            self.preprocessor,
            prepared,
            self.rng,
            self.config,
            train_task_actor=actor_kind == "task",
        )
        self.actor_readiness.record_actor_warmup(
            actor_kind, metrics.assessment, metrics.update_count
        )
        if metrics.assessment["passed"] is not True:
            failed = [
                name
                for name, passed in metrics.assessment["checks"].items()
                if passed is not True
            ]
            raise RuntimeError(
                "foundation Actor warmup failed stability checks: "
                + ", ".join(failed)
            )
        return {
            **{f"warmup/{name}": value for name, value in metrics.metrics.items()},
            "warmup/update_count": float(metrics.update_count),
            "warmup/stability_passed": 1.0,
        }

    def _raise_if_calibration_failed(self) -> None:
        if len(self.records) < self.config.calibration_early_stop_episodes:
            return
        failed = failed_exploration_calibration_checks(
            self.actor_readiness.last_assessment
        )
        if failed:
            raise RuntimeError(
                "foundation calibration stopped early after missing evidence: "
                + ", ".join(failed)
            )

    def _episode_learning_signals(
        self,
        prepared: FoundationPreparedFeatures,
        episodes: list[object],
    ) -> dict[str, EpisodeLearningEvidence]:
        return evaluate_replay_episode_learning_evidence(
            self.stack.trainer,
            self.store.path,
            self.cache,
            self.preprocessor,
            prepared,
            [episode.episode_id for episode in episodes],
            transitions=self.config.sequence_transitions,
            maximum_windows=self.config.learning_signal_windows_per_episode,
        )

    def _record_learning_outcomes(
        self,
        episodes: list[object],
        learning_evidence: Mapping[str, EpisodeLearningEvidence],
    ) -> None:
        record_foundation_learning_outcomes(
            episodes,
            learning_evidence,
            self.task_sampler,
            self.frontier,
            self.records,
            update_count=self.stack.trainer.update_count,
        )

    def _checkpoint(
        self, cycle: int, prepared: FoundationPreparedFeatures
    ) -> None:
        if self.latest_action_causality is None:
            raise RuntimeError("checkpoint requires action causality evidence")
        dataset_sha = file_sha256(self.store.path / "manifest.json")
        audit_dataset_sha = file_sha256(
            self.causality_store.path / "manifest.json"
        )
        version = f"update-{self.stack.trainer.update_count:09d}"
        causality = publish_action_causality_report(
            self.run_path / "diagnostics/action-causality" / version,
            self.latest_action_causality,
            source_commit=self.source_commit,
            update_count=self.stack.trainer.update_count,
            training_data_manifest_sha256=dataset_sha,
            audit_data_manifest_sha256=audit_dataset_sha,
        )
        causality_sha = registry_file_sha256(causality)
        training_diagnostics = {
            "action_causality_report_sha256": causality_sha,
            "action_causality_passed": bool(
                foundation_action_causality_qualified(
                    self.latest_action_causality
                )
            ),
            "actor_readiness_unlocked": self.actor_readiness.task_actor_unlocked,
            "task_actor_update_count": self.actor_readiness.task_actor_update_count,
        }
        checkpoint = self.run_path / "checkpoints" / version
        save_foundation_training_checkpoint(
            checkpoint,
            self.stack.trainer,
            source_commit=self.source_commit,
            data_manifest_sha256=dataset_sha,
            training_diagnostics=training_diagnostics,
        )
        checkpoint_sha = registry_file_sha256(checkpoint / "training-state.pt")
        self.latest_checkpoint = checkpoint
        self.latest_action_causality_report = causality
        self.latest_deployment = None
        if foundation_deployment_qualified(training_diagnostics):
            deployment = self.run_path / "deployments" / version
            export_foundation_deployment(
                deployment,
                self.stack.trainer.visual_student,
                self.stack.trainer.world_model,
                self.stack.trainer.actor,
                self.stack.action_scaling,
                source_commit=self.source_commit,
                training_checkpoint_sha256=checkpoint_sha,
                training_diagnostics=training_diagnostics,
                preprocessing={
                    "fingerprint": self.preprocessor.fingerprint,
                    "config": asdict(self.preprocessor.config),
                },
                language_cache={
                    "encoder_lock_sha256": prepared.language.encoder_lock_sha256,
                    "preprocess_sha256": LANGUAGE_PREPROCESS_SHA256,
                    "dimension": prepared.language.output_dimension,
                },
            )
            self.latest_deployment = deployment
        self._save_runner_state(cycle)
        prune_versioned_artifacts(
            self.run_path / "checkpoints",
            self.config.published_checkpoint_retention,
        )
        prune_versioned_artifacts(
            self.run_path / "deployments",
            self.config.published_checkpoint_retention,
        )
        prune_versioned_artifacts(
            self.run_path / "diagnostics/action-causality",
            self.config.published_checkpoint_retention,
        )

    def _save_runner_state(self, cycle: int) -> None:
        if self.latest_checkpoint is None or self.latest_action_causality_report is None:
            raise RuntimeError("runner progress requires checkpoint artifacts")
        publish_runner_progress(
            self.run_path,
            self.latest_checkpoint,
            self.latest_deployment,
            self.latest_action_causality_report,
            cycle=cycle,
            update_count=self.stack.trainer.update_count,
            rng_state=self.rng.bit_generator.state,
            torch_rng_state=capture_torch_rng_state(
                next(self.stack.trainer.actor.parameters()).device
            ),
            task_sampler=self.task_sampler.state_dict(),
            actor_readiness=self.actor_readiness.state_dict(),
            learning_frontier=self.frontier.state_dict(),
            records=[asdict(item) for item in self.records],
            replay_manifest=self.store.manifest,
            causality_manifest=self.causality_store.manifest,
        )
        clear_replay_archive(self.run_path / "recovery/replay-prune-archive")
