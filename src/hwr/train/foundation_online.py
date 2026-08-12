"""One task-blind collect-materialize-update loop for all household tasks."""

from __future__ import annotations

import copy
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

from hwr.core.runtime import RuntimeBackend
from hwr.data.autonomous_trajectory import AppendableAutonomousTrajectoryStore
from hwr.data.foundation_cache import FoundationCacheKey, FoundationFeatureCache
from hwr.data.foundation_features import (
    LANGUAGE_PREPROCESS_SHA256,
    file_sha256,
    load_feature_index,
    materialize_language_features,
    materialize_visual_features,
)
from hwr.data.foundation_loading import (
    FoundationPreparedFeatures,
    FoundationSequenceBatchLoader,
)
from hwr.perception.foundation import (
    FrozenLanguageFeatureProvider,
    FrozenVisionFeatureProvider,
    language_source_sha256,
)
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.perception.language_cache import StaticLanguageFeatureResolver
from hwr.policy.foundation_runtime import FoundationWorldModelPolicy
from hwr.train.foundation_augmentation import transform_foundation_batch
from hwr.train.foundation_collection import (
    AutonomousCollectionConfig,
    AutonomousEpisodeCollector,
    CurrentRLActorActionSource,
    RandomRLActionSource,
)
from hwr.train.foundation_diagnostics import (
    evaluate_foundation_action_causality_audit,
    publish_action_causality_report,
)
from hwr.train.foundation_holdout import (
    causality_batches_by_task,
    causality_window_manifest,
    collect_causality_holdout,
    select_causality_windows,
)
from hwr.train.foundation_registry import (
    export_foundation_deployment,
    file_sha256 as registry_file_sha256,
    load_foundation_training_checkpoint,
    prune_versioned_artifacts,
    save_foundation_training_checkpoint,
)
from hwr.train.foundation_recovery import (
    clear_replay_archive,
    publish_runner_progress,
    restore_runner_progress,
)
from hwr.train.foundation_setup import FoundationLearningStack
from hwr.train.learning_signals import failure_boundary_step
from hwr.train.task_sampling import OutcomeAdaptiveTaskSampler, TaskOutcome
from hwr.world_model.deploy import DeployableWorldModelStateFilter
from hwr.world_model.evaluation import ActionCausalityCriteria


@dataclass(frozen=True)
class FoundationTaskInterface:
    task_id: str
    maximum_steps: int

    def __post_init__(self) -> None:
        if not self.task_id or self.maximum_steps <= 0:
            raise ValueError("foundation task interface is invalid")


@dataclass(frozen=True)
class FoundationOnlineTrainingConfig:
    episodes: int = 120
    initial_random_episodes: int = 6
    collection_episodes_per_cycle: int = 3
    updates_per_cycle: int = 200
    batch_size: int = 2
    sequence_transitions: int = 16
    camera_width: int = 256
    camera_height: int = 192
    augmentation_probability: float = 0.5
    checkpoint_interval_cycles: int = 1
    replay_transition_capacity: int = 18000
    published_checkpoint_retention: int = 3
    minimum_action_causality_ratio: float = 1.05
    minimum_action_causality_horizon_fraction: float = 0.60
    causality_holdout_episodes_per_task: int = 2
    causality_audit_windows_per_task: int = 8
    causality_audit_batch_size: int = 2
    seed: int = 20260812

    def __post_init__(self) -> None:
        positive = (
            self.episodes,
            self.initial_random_episodes,
            self.collection_episodes_per_cycle,
            self.updates_per_cycle,
            self.batch_size,
            self.sequence_transitions,
            self.camera_width,
            self.camera_height,
            self.checkpoint_interval_cycles,
            self.replay_transition_capacity,
            self.published_checkpoint_retention,
            self.causality_holdout_episodes_per_task,
            self.causality_audit_windows_per_task,
            self.causality_audit_batch_size,
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("foundation online training dimensions are invalid")
        if self.initial_random_episodes > self.episodes:
            raise ValueError("initial random Episodes exceed total Episodes")
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError("foundation augmentation probability is invalid")
        if min(self.camera_width, self.camera_height) < 160:
            raise ValueError("foundation online training requires high-resolution cameras")
        if self.replay_transition_capacity < self.sequence_transitions * 3:
            raise ValueError("foundation replay capacity cannot retain one window per task")
        if self.causality_audit_windows_per_task % self.causality_audit_batch_size:
            raise ValueError("causality audit batch size must divide task window count")
        ActionCausalityCriteria(
            self.minimum_action_causality_ratio,
            self.minimum_action_causality_horizon_fraction,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FoundationEnvironmentFactory(Protocol):
    def __call__(
        self, task_id: str, camera_width: int, camera_height: int
    ) -> RuntimeBackend: ...


@dataclass(frozen=True)
class FoundationProviderFactories:
    vision_language: Callable[[], FrozenVisionFeatureProvider]
    dense_vision: Callable[[], FrozenVisionFeatureProvider]
    language: Callable[[], FrozenLanguageFeatureProvider]


@dataclass(frozen=True)
class FoundationEpisodeRecord:
    episode_index: int
    task_id: str
    seed: int
    action_source: str
    episode_return: float
    success: bool
    safety_cost_rate: float
    environment_steps: int
    update_count: int


@dataclass(frozen=True)
class FoundationOnlineTrainingResult:
    records: tuple[FoundationEpisodeRecord, ...]
    update_count: int
    replay_path: Path
    latest_checkpoint: Path
    latest_deployment: Path
    latest_action_causality_report: Path


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
    ) -> None:
        if len(tasks) != 3 or set(tasks) != {item.task_id for item in tasks.values()}:
            raise ValueError("foundation training requires exactly three task interfaces")
        if not source_commit:
            raise ValueError("foundation training source commit is required")
        self.tasks = dict(tasks)
        self.task_ids = tuple(sorted(tasks))
        self.environment_factory = environment_factory
        self.preprocessor = preprocessor
        self.providers = providers
        self.stack = learning_stack
        self.config = config
        self.run_path = run_path
        self.source_commit = source_commit
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
        self._write_or_verify_run_manifest()

    def train(self) -> FoundationOnlineTrainingResult:
        environments = {
            task_id: self.environment_factory(
                task_id, self.config.camera_width, self.config.camera_height
            )
            for task_id in self.task_ids
        }
        cycle = self.completed_cycles
        try:
            self._prepare_causality_holdout(environments)
            while len(self.records) < self.config.episodes:
                collected = self._collect_cycle(environments)
                self._bound_replay_storage()
                prepared, causality_prepared = self._materialize_features()
                metrics = self._update_cycle(prepared)
                self._evaluate_action_causality(causality_prepared, metrics)
                self._record_learning_outcomes(collected, metrics)
                cycle += 1
                self.completed_cycles = cycle
                if cycle % self.config.checkpoint_interval_cycles == 0:
                    self._checkpoint(cycle, prepared)
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
            or self.latest_action_causality["assessment"]["passed"] is not True
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
            diagnostic.get("schema_version")
            != "hwr.foundation-action-causality/v2"
            or diagnostic.get("source_commit") != self.source_commit
            or int(diagnostic.get("update_count", -1))
            != int(latest.get("update_count", -2))
        ):
            raise ValueError("resumed action causality lineage differs")
        expected = {
            "action_causality_report_sha256": latest["action_causality_sha256"],
            "action_causality_passed": diagnostic["assessment"]["passed"],
        }
        checkpoint_manifest = json.loads(
            (checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
        if checkpoint_manifest.get("training_diagnostics") != expected:
            raise ValueError("resumed checkpoint diagnostic provenance differs")
        deployment = latest.get("deployment")
        deployment_path = (
            self.run_path / str(deployment) if deployment is not None else None
        )
        if deployment_path is not None:
            deployment_manifest = json.loads(
                (deployment_path / "manifest.json").read_text(encoding="utf-8")
            )
            if expected["action_causality_passed"] is not True:
                raise ValueError("failed causality checkpoint exposed a deployment")
            if deployment_manifest.get("training_diagnostics") != expected:
                raise ValueError("resumed deployment diagnostic provenance differs")
            artifact = checkpoint / str(checkpoint_manifest["artifact_file"])
            if deployment_manifest.get(
                "training_checkpoint_sha256"
            ) != registry_file_sha256(artifact):
                raise ValueError("resumed deployment checkpoint hash differs")
        elif expected["action_causality_passed"] is True:
            raise ValueError("passed causality checkpoint is missing its deployment")
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
        self.rng.bit_generator.state = restored.rng_state
        self.records = [FoundationEpisodeRecord(**item) for item in restored.records]
        self.completed_cycles = restored.cycle
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
            random_phase = episode_index < self.config.initial_random_episodes
            source = (
                RandomRLActionSource(self.stack.action_scaling)
                if random_phase
                else CurrentRLActorActionSource(self._collection_policy())
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
            episodes_per_task=self.config.causality_holdout_episodes_per_task,
            base_seed=self.config.seed,
            source_commit=self.source_commit,
        )

    def _language_resolver(self) -> StaticLanguageFeatureResolver:
        features = {}
        language_index_path = self.run_path / "features/language.json"
        if not language_index_path.is_file():
            raise RuntimeError("language features must be materialized before Actor collection")
        from hwr.data.foundation_features import load_feature_index

        index = load_feature_index(language_index_path)
        for shard in self.store.manifest["shards"]:
            text, locale = shard["instruction"], shard["locale"]
            source = language_source_sha256(text, locale)
            key = FoundationCacheKey(
                "language", source, index.encoder_lock_sha256, index.preprocess_sha256
            )
            features[(locale, text)] = self.cache.load_language(key).values.copy()
        return StaticLanguageFeatureResolver(
            features,
            encoder_lock_sha256=index.encoder_lock_sha256,
            output_dimension=index.output_dimension,
        )

    def _materialize_features(
        self,
    ) -> tuple[FoundationPreparedFeatures, FoundationPreparedFeatures]:
        output = self.run_path / "features"
        causality_output = self.run_path / "causality-holdout/features"
        output.mkdir(parents=True, exist_ok=True)
        causality_output.mkdir(parents=True, exist_ok=True)
        vision_language_provider = self.providers.vision_language()
        vision_language = materialize_visual_features(
            self.store.path,
            self.cache,
            self.preprocessor,
            vision_language_provider,
            output / "vision-language.json",
        )
        causality_vision_language = materialize_visual_features(
            self.causality_store.path,
            self.cache,
            self.preprocessor,
            vision_language_provider,
            causality_output / "vision-language.json",
        )
        del vision_language_provider
        gc.collect()
        dense = self.providers.dense_vision()
        dense_vision = materialize_visual_features(
            self.store.path,
            self.cache,
            self.preprocessor,
            dense,
            output / "dense-vision.json",
        )
        causality_dense_vision = materialize_visual_features(
            self.causality_store.path,
            self.cache,
            self.preprocessor,
            dense,
            causality_output / "dense-vision.json",
        )
        del dense
        gc.collect()
        language_provider = self.providers.language()
        language = materialize_language_features(
            self.store.path,
            self.cache,
            language_provider,
            output / "language.json",
        )
        causality_language = materialize_language_features(
            self.causality_store.path,
            self.cache,
            language_provider,
            causality_output / "language.json",
        )
        del language_provider
        gc.collect()
        return (
            FoundationPreparedFeatures(vision_language, dense_vision, language),
            FoundationPreparedFeatures(
                causality_vision_language,
                causality_dense_vision,
                causality_language,
            ),
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
        evicted_sources = tuple(sources)
        if not evicted_sources:
            return
        for filename in ("vision-language.json", "dense-vision.json"):
            path = self.run_path / "features" / filename
            if not path.is_file():
                continue
            index = load_feature_index(path)
            for source in evicted_sources:
                self.cache.discard(
                    FoundationCacheKey(
                        "visual",
                        source,
                        index.encoder_lock_sha256,
                        self.preprocessor.fingerprint,
                    )
                )

    def _update_cycle(
        self, prepared: FoundationPreparedFeatures
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
            metrics.append(self.stack.trainer.train_step(batch))
        names = metrics[0]
        result = {
            name: float(sum(item[name] for item in metrics) / len(metrics))
            for name in names
        }
        return result

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
        )
        diagnostic["window_selection"] = causality_window_manifest(loader, selected)
        diagnostic["holdout_collector"] = "foundation-causality-holdout/v1"
        self.latest_action_causality = diagnostic
        assessment = diagnostic["assessment"]
        metrics["world/action_causality_ratio"] = float(
            assessment["shuffled_to_true_ratio"]
        )
        metrics["world/action_causality_passed"] = float(assessment["passed"])

    def _sample_transform(
        self, loader: FoundationSequenceBatchLoader, index: int
    ) -> str | None:
        legal = loader.legal_transform_ids(index)
        if not legal or self.rng.random() >= self.config.augmentation_probability:
            return None
        return str(self.rng.choice(legal))

    def _record_learning_outcomes(
        self, episodes: list[object], metrics: Mapping[str, float]
    ) -> None:
        novelty = max(0.0, metrics["imagination/imagined_uncertainty"])
        td_error = max(0.0, metrics["imagination/td_error"])
        for episode in episodes:
            arrays = episode.arrays
            episode_return = float(arrays["reward"].sum())
            safety_rate = float(arrays["safety_cost"].mean())
            success = bool(episode.metadata["success"])
            terminated_failure = bool(arrays["terminated"][-1]) and not success
            boundary = failure_boundary_step(
                arrays["safety_cost"], terminated_failure=terminated_failure
            )
            boundary_signal = (
                float(boundary + 1) / len(arrays["safety_cost"])
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
                    novelty,
                    td_error,
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
                self.latest_action_causality["assessment"]["passed"]
            ),
        }
        checkpoint = self.run_path / "checkpoints" / (
            version
        )
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
        if training_diagnostics["action_causality_passed"] is True:
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
            task_sampler=self.task_sampler.state_dict(),
            records=[asdict(item) for item in self.records],
            replay_manifest=self.store.manifest,
            causality_manifest=self.causality_store.manifest,
        )
        clear_replay_archive(
            self.run_path / "recovery/replay-prune-archive"
        )

    def _write_or_verify_run_manifest(self) -> None:
        manifest = {
            "schema_version": "hwr.foundation-online-run/v1",
            "source_commit": self.source_commit,
            "training_config": self.config.to_dict(),
            "tasks": [asdict(self.tasks[name]) for name in self.task_ids],
            "preprocessing": {
                "fingerprint": self.preprocessor.fingerprint,
                "config": asdict(self.preprocessor.config),
            },
            "lineage": {
                "action_sources": ["random_rl_exploration", "rl_actor"],
                "expert_policies": [],
                "demonstration_datasets": [],
                "behavior_cloning": False,
                "legacy_p_series_parent": None,
            },
        }
        path = self.run_path / "run-manifest.json"
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("foundation run manifest differs on resume")
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
