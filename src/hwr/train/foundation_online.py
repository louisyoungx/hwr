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
from hwr.data.foundation_features import (
    LANGUAGE_PREPROCESS_SHA256,
    file_sha256,
)
from hwr.data.foundation_loading import (
    FoundationPreparedFeatures,
    FoundationSequenceBatchLoader,
)
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.perception.language_cache import StaticLanguageFeatureResolver
from hwr.policy.foundation_runtime import FoundationWorldModelPolicy
from hwr.train.accelerator_memory import (
    release_accelerator_memory_after_step,
    release_unused_accelerator_memory,
)
from hwr.train.foundation_augmentation import transform_foundation_batch
from hwr.train.foundation_action_probe import evaluate_foundation_data_action_probe
from hwr.train.foundation_actor_readiness import (
    FoundationActorReadinessCriteria,
    FoundationActorReadinessTracker,
)
from hwr.train.foundation_collection import (
    AutonomousCollectionConfig,
    AutonomousEpisodeCollector,
    CurrentRLActorActionSource,
)
from hwr.train.foundation_diagnostics import (
    evaluate_foundation_action_causality_audit,
    foundation_action_causality_qualified,
    publish_action_causality_report,
)
from hwr.train.foundation_holdout import (
    causality_batches_by_task,
    causality_window_manifest,
    collect_causality_holdout,
    select_causality_windows,
)
from hwr.train.foundation_learning_signals import (
    EpisodeLearningSignals,
    evaluate_episode_learning_signals,
)
from hwr.train.foundation_materialization import materialize_foundation_replay_features
from hwr.train.foundation_metrics import (
    FoundationMetricsProgress,
    FoundationMetricsStore,
    build_foundation_cycle_metrics,
    mean_metrics,
    summarize_replay_action_coverage,
)
from hwr.train.foundation_exploration import (
    RandomRLActionSource,
    RandomRLExplorationConfig,
)
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_online_types import (
    FoundationEnvironmentFactory,
    FoundationEpisodeRecord,
    FoundationOnlineTrainingResult,
    FoundationProviderFactories,
    FoundationTaskInterface,
)
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
from hwr.train.foundation_setup import FoundationLearningStack
from hwr.train.learning_signals import failure_boundary_step
from hwr.train.task_sampling import OutcomeAdaptiveTaskSampler, TaskOutcome
from hwr.world_model.deploy import DeployableWorldModelStateFilter
from hwr.world_model.evaluation import ActionCausalityCriteria

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
        self.latest_checkpoint: Path | None = None
        self.latest_deployment: Path | None = None
        self.latest_action_causality: dict[str, object] | None = None
        self.latest_action_causality_report: Path | None = None
        self.completed_cycles = 0
        self.actor_readiness = FoundationActorReadinessTracker(
            FoundationActorReadinessCriteria(
                config.minimum_actor_readiness_episodes,
                config.actor_readiness_consecutive_passes,
                config.minimum_active_action_dimension_fraction,
                config.minimum_action_effective_rank,
                config.minimum_data_action_probe_ratio,
                config.minimum_data_action_probe_ratio_p05,
            )
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
        )

    def train(self) -> FoundationOnlineTrainingResult:
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
                signals = self._episode_learning_signals(prepared, collected)
                self._evaluate_action_causality(causality_prepared, metrics)
                self._record_learning_outcomes(collected, signals)
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
                    ),
                )
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
            source = (
                CurrentRLActorActionSource(self._collection_policy())
                if self.actor_readiness.unlocked
                else self._random_action_source()
            )
            collector = AutonomousEpisodeCollector(
                self.preprocessor,
                AutonomousCollectionConfig(
                    "mujoco-bimanual-runtime/v2",
                    self.source_commit,
                    task.maximum_steps,
                ),
            )
            episode = collector.collect(
                environments[task_id], source, task_id=task_id, seed=seed
            )
            self.store.append(episode)
            collected.append(episode)
        return collected

    def _collection_policy(self) -> FoundationWorldModelPolicy:
        resolver = self._language_resolver()
        trainer = self.stack.trainer
        device = str(next(trainer.actor.parameters()).device)
        return FoundationWorldModelPolicy(
            copy.deepcopy(trainer.visual_student),
            DeployableWorldModelStateFilter.from_world_model(trainer.world_model),
            copy.deepcopy(trainer.actor),
            self.preprocessor,
            resolver,
            self.stack.action_scaling,
            policy_id=f"foundation-actor-update-{trainer.update_count}",
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
            exploration_config=self._random_exploration_config(),
            episodes_per_task=self.config.causality_holdout_episodes_per_task,
            base_seed=self.config.seed,
            source_commit=self.source_commit,
        )

    def _random_exploration_config(self) -> RandomRLExplorationConfig:
        return RandomRLExplorationConfig(
            self.config.random_exploration_motion_correlation,
            self.config.random_exploration_gripper_flip_probability,
        )

    def _random_action_source(self) -> RandomRLActionSource:
        return RandomRLActionSource(
            self.stack.action_scaling,
            self._random_exploration_config(),
        )

    def _language_resolver(self) -> StaticLanguageFeatureResolver:
        return language_resolver_from_replay(
            self.store.manifest["shards"],
            self.cache,
            self.run_path / "features/language.json",
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
        loader = FoundationSequenceBatchLoader(
            self.store.path,
            self.cache,
            self.preprocessor,
            self.stack.trainer.visual_student.config,
            prepared,
            transitions=self.config.sequence_transitions,
            device=str(next(self.stack.trainer.actor.parameters()).device),
        )
        metrics: list[dict[str, float]] = []
        for _ in range(self.config.updates_per_cycle):
            indices = self.rng.integers(0, len(loader), size=self.config.batch_size)
            batch = loader.build([int(value) for value in indices])
            transforms = [self._sample_transform(loader, int(value)) for value in indices]
            batch = transform_foundation_batch(batch, transforms)
            metrics.append(
                self.stack.trainer.train_step(
                    batch, train_task_actor=self.actor_readiness.unlocked
                )
            )
            release_accelerator_memory_after_step(len(metrics))
            if len(metrics) % self.config.metrics_publish_interval_updates == 0:
                self._publish_progress(
                    "updating",
                    cycle,
                    mean_metrics(metrics),
                    completed_updates=len(metrics),
                )
        names = metrics[0]
        result = {
            name: float(sum(item[name] for item in metrics) / len(metrics))
            for name in names
        }
        if self.actor_readiness.unlocked:
            self.actor_readiness.record_task_actor_updates(
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
        self.metrics_store.publish_progress(
            FoundationMetricsProgress(
                stage,
                cycle,
                self.stack.trainer.update_count,
                len(self.records),
                self.config.updates_per_cycle if stage == "updating" else 0,
                completed_updates,
            ),
            metrics=metrics,
        )

    def _evaluate_action_causality(
        self,
        prepared: FoundationPreparedFeatures,
        metrics: dict[str, float],
    ) -> None:
        loader = FoundationSequenceBatchLoader(
            self.causality_store.path,
            self.cache,
            self.preprocessor,
            self.stack.trainer.visual_student.config,
            prepared,
            transitions=self.config.sequence_transitions,
            device=str(next(self.stack.trainer.actor.parameters()).device),
        )
        selected = select_causality_windows(
            loader,
            self.task_ids,
            windows_per_task=self.config.causality_audit_windows_per_task,
            selection_seed=self.config.seed,
        )
        diagnostic = evaluate_foundation_action_causality_audit(
            self.stack.trainer,
            causality_batches_by_task(
                loader,
                selected,
                batch_size=self.config.causality_audit_batch_size,
            ),
            ActionCausalityCriteria(
                self.config.minimum_action_causality_ratio,
                self.config.minimum_action_causality_horizon_fraction,
            ),
            shuffle_seed=self.config.seed,
            shuffle_repeats=self.config.causality_shuffle_repeats,
        )
        diagnostic["window_selection"] = causality_window_manifest(loader, selected)
        diagnostic["holdout_collector"] = "foundation-causality-holdout/v1"
        self.latest_action_causality = diagnostic
        assessment = diagnostic["assessment"]
        metrics["world/action_causality_ratio"] = float(
            assessment["shuffled_to_true_ratio"]
        )
        metrics["world/action_causality_passed"] = float(
            foundation_action_causality_qualified(diagnostic)
        )
        replay_coverage = summarize_replay_action_coverage(
            self.store.path, self.store.manifest, self.stack.action_scaling
        )
        probe = evaluate_foundation_data_action_probe(
            self.store.path,
            self.store.manifest,
            self.causality_store.path,
            self.causality_store.manifest,
            bootstrap_seed=self.config.seed,
        )
        readiness = self.actor_readiness.assess(
            diagnostic,
            probe,
            replay_coverage,
            replay_episodes=int(self.store.manifest["episode_count"]),
        )
        metrics["actor_readiness/unlocked"] = float(readiness["unlocked"])
        metrics["actor_readiness/consecutive_passes"] = float(
            readiness["consecutive_passes"]
        )

    def _sample_transform(
        self, loader: FoundationSequenceBatchLoader, index: int
    ) -> str | None:
        legal = loader.legal_transform_ids(index)
        if not legal or self.rng.random() >= self.config.augmentation_probability:
            return None
        return str(self.rng.choice(legal))

    def _episode_learning_signals(
        self,
        prepared: FoundationPreparedFeatures,
        episodes: list[object],
    ) -> dict[str, EpisodeLearningSignals]:
        loader = FoundationSequenceBatchLoader(
            self.store.path,
            self.cache,
            self.preprocessor,
            self.stack.trainer.visual_student.config,
            prepared,
            transitions=self.config.sequence_transitions,
            device=str(next(self.stack.trainer.actor.parameters()).device),
        )
        return evaluate_episode_learning_signals(
            self.stack.trainer,
            loader,
            [episode.episode_id for episode in episodes],
            maximum_windows=self.config.learning_signal_windows_per_episode,
        )

    def _record_learning_outcomes(
        self,
        episodes: list[object],
        learning_signals: Mapping[str, EpisodeLearningSignals],
    ) -> None:
        for episode in episodes:
            arrays = episode.arrays
            signal = learning_signals[episode.episode_id]
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
            improvement = self.task_sampler.reward_improvement(
                episode.task_id, episode_return
            )
            self.task_sampler.record(
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
            self.records.append(
                FoundationEpisodeRecord(
                    len(self.records),
                    episode.task_id,
                    episode.seed,
                    str(arrays["action_source"][0]),
                    episode_return,
                    success,
                    safety_rate,
                    len(arrays["executed_action"]),
                    self.stack.trainer.update_count,
                    signal.state_novelty,
                    signal.td_error,
                    improvement,
                    boundary_signal,
                )
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
            "actor_readiness_unlocked": self.actor_readiness.unlocked,
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
            records=[asdict(item) for item in self.records],
            replay_manifest=self.store.manifest,
            causality_manifest=self.causality_store.manifest,
        )
        clear_replay_archive(self.run_path / "recovery/replay-prune-archive")
