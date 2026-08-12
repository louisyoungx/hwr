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
import torch

from hwr.core.runtime import RuntimeBackend
from hwr.data.autonomous_trajectory import AppendableAutonomousTrajectoryStore
from hwr.data.foundation_cache import FoundationCacheKey, FoundationFeatureCache
from hwr.data.foundation_features import (
    LANGUAGE_PREPROCESS_SHA256,
    file_sha256,
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
from hwr.train.foundation_registry import (
    export_foundation_deployment,
    file_sha256 as registry_file_sha256,
    load_foundation_training_checkpoint,
    save_foundation_training_checkpoint,
)
from hwr.train.foundation_setup import FoundationLearningStack
from hwr.train.learning_signals import failure_boundary_step
from hwr.train.task_sampling import OutcomeAdaptiveTaskSampler, TaskOutcome
from hwr.world_model.deploy import DeployableWorldModelStateFilter


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
    batch_size: int = 4
    sequence_transitions: int = 16
    camera_width: int = 256
    camera_height: int = 192
    augmentation_probability: float = 0.5
    checkpoint_interval_cycles: int = 1
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
        )
        if min(positive) <= 0 or self.seed < 0:
            raise ValueError("foundation online training dimensions are invalid")
        if self.initial_random_episodes > self.episodes:
            raise ValueError("initial random Episodes exceed total Episodes")
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError("foundation augmentation probability is invalid")
        if min(self.camera_width, self.camera_height) < 160:
            raise ValueError("foundation online training requires high-resolution cameras")

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
        self.cache = FoundationFeatureCache(self.run_path / "feature-cache")
        self.task_sampler = OutcomeAdaptiveTaskSampler(self.task_ids)
        self.rng = np.random.default_rng(config.seed)
        self.records: list[FoundationEpisodeRecord] = []
        self.latest_checkpoint: Path | None = None
        self.latest_deployment: Path | None = None
        self._write_or_verify_run_manifest()

    def train(self) -> FoundationOnlineTrainingResult:
        environments = {
            task_id: self.environment_factory(
                task_id, self.config.camera_width, self.config.camera_height
            )
            for task_id in self.task_ids
        }
        cycle = 0
        try:
            while len(self.records) < self.config.episodes:
                collected = self._collect_cycle(environments)
                prepared = self._materialize_features()
                metrics = self._update_cycle(prepared)
                self._record_learning_outcomes(collected, metrics)
                cycle += 1
                if cycle % self.config.checkpoint_interval_cycles == 0:
                    self._checkpoint(cycle, prepared)
        finally:
            for environment in environments.values():
                environment.close()
        if self.latest_checkpoint is None:
            prepared = self._materialize_features()
            self._checkpoint(cycle, prepared)
        return self.result()

    def result(self) -> FoundationOnlineTrainingResult:
        if self.latest_checkpoint is None or self.latest_deployment is None:
            raise RuntimeError("foundation training has no published checkpoint")
        return FoundationOnlineTrainingResult(
            tuple(self.records),
            self.stack.trainer.update_count,
            self.store.path,
            self.latest_checkpoint,
            self.latest_deployment,
        )

    def resume_latest(self) -> None:
        latest_path = self.run_path / "latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(latest_path)
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        checkpoint = self.run_path / latest["training_checkpoint"]
        load_foundation_training_checkpoint(checkpoint, self.stack.trainer)
        state = torch.load(
            self.run_path / "runner-state.pt", map_location="cpu", weights_only=True
        )
        self.task_sampler.load_state_dict(state["task_sampler"])
        self.rng.bit_generator.state = state["rng_state"]
        self.records = [FoundationEpisodeRecord(**item) for item in state["records"]]
        self.latest_checkpoint = checkpoint
        self.latest_deployment = self.run_path / latest["deployment"]

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

    def _materialize_features(self) -> FoundationPreparedFeatures:
        output = self.run_path / "features"
        output.mkdir(parents=True, exist_ok=True)
        vision_language = self.providers.vision_language()
        siglip = materialize_visual_features(
            self.store.path,
            self.cache,
            self.preprocessor,
            vision_language,
            output / "siglip.json",
        )
        del vision_language
        gc.collect()
        dense = self.providers.dense_vision()
        dinov2 = materialize_visual_features(
            self.store.path,
            self.cache,
            self.preprocessor,
            dense,
            output / "dinov2.json",
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
        del language_provider
        gc.collect()
        return FoundationPreparedFeatures(siglip, dinov2, language)

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
        return {
            name: float(sum(item[name] for item in metrics) / len(metrics))
            for name in names
        }

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
        dataset_sha = file_sha256(self.store.path / "manifest.json")
        checkpoint = self.run_path / "checkpoints" / (
            f"update-{self.stack.trainer.update_count:09d}"
        )
        save_foundation_training_checkpoint(
            checkpoint,
            self.stack.trainer,
            source_commit=self.source_commit,
            data_manifest_sha256=dataset_sha,
        )
        checkpoint_sha = registry_file_sha256(checkpoint / "training-state.pt")
        deployment = self.run_path / "deployments" / (
            f"update-{self.stack.trainer.update_count:09d}"
        )
        export_foundation_deployment(
            deployment,
            self.stack.trainer.visual_student,
            self.stack.trainer.world_model,
            self.stack.trainer.actor,
            self.stack.action_scaling,
            source_commit=self.source_commit,
            training_checkpoint_sha256=checkpoint_sha,
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
        self.latest_checkpoint = checkpoint
        self.latest_deployment = deployment
        self._save_runner_state(cycle)

    def _save_runner_state(self, cycle: int) -> None:
        temporary = self.run_path / "runner-state.pt.tmp"
        torch.save(
            {
                "cycle": cycle,
                "rng_state": self.rng.bit_generator.state,
                "task_sampler": self.task_sampler.state_dict(),
                "records": [asdict(item) for item in self.records],
            },
            temporary,
        )
        os.replace(temporary, self.run_path / "runner-state.pt")
        records_path = self.run_path / "episodes.jsonl"
        records_temporary = records_path.with_suffix(".jsonl.tmp")
        records_temporary.write_text(
            "".join(
                json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n"
                for item in self.records
            ),
            encoding="utf-8",
        )
        os.replace(records_temporary, records_path)
        latest = {
            "schema_version": "hwr.foundation-online-latest/v1",
            "training_checkpoint": str(self.latest_checkpoint.relative_to(self.run_path)),
            "deployment": str(self.latest_deployment.relative_to(self.run_path)),
            "episode_count": len(self.records),
            "update_count": self.stack.trainer.update_count,
        }
        temporary_json = self.run_path / "latest.json.tmp"
        temporary_json.write_text(
            json.dumps(latest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_json, self.run_path / "latest.json")

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
