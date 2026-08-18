"""Frozen replay execution helpers for the R0001-P05 batch-arm experiment."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from hwr.data.autonomous_trajectory import AppendableAutonomousTrajectoryStore
from hwr.data.foundation_cache import FoundationFeatureCache
from hwr.data.foundation_features import load_feature_index
from hwr.data.foundation_loading import (
    FoundationPreparedFeatures,
    FoundationSequenceBatchLoader,
)
from hwr.perception.high_resolution import (
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
)
from hwr.perception.student import VisualStudentConfig
from hwr.policy.bimanual_input import (
    BimanualInputConfig,
    default_four_camera_calibrations,
)
from hwr.train.accelerator_memory import release_accelerator_memory_after_step
from hwr.train.foundation_actor_readiness import (
    FoundationActorReadinessTracker,
    actor_readiness_criteria_from_config,
)
from hwr.train.foundation_admission import evaluate_foundation_actor_admission
from hwr.train.foundation_batch_arms import BatchArmSchedule
from hwr.train.foundation_metrics import mean_metrics
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_recovery import (
    capture_torch_rng_state,
    restore_torch_rng_state,
)
from hwr.train.foundation_trainer import FoundationWorldModelTrainer


@dataclass(frozen=True)
class FrozenBatchReplayInputs:
    run_path: Path
    replay_store: AppendableAutonomousTrajectoryStore
    audit_store: AppendableAutonomousTrajectoryStore
    cache: FoundationFeatureCache
    preprocessor: HighResolutionVisionPreprocessor
    training_features: FoundationPreparedFeatures
    audit_features: FoundationPreparedFeatures
    training_loader: FoundationSequenceBatchLoader
    config: FoundationOnlineTrainingConfig
    task_ids: tuple[str, ...]


def load_frozen_batch_replay_inputs(
    root: Path,
    input_run: Path,
    *,
    device: str,
) -> FrozenBatchReplayInputs:
    run_path = input_run if input_run.is_absolute() else root / input_run
    config = FoundationOnlineTrainingConfig(
        **_read_config(root / "configs/foundation/online-training-v1.json")
    )
    preprocessor = _preprocessor(config)
    student = _visual_student_config(root / "configs/foundation/visual-student-v1.json")
    training_features = FoundationPreparedFeatures(
        load_feature_index(run_path / "features/vision-language.json"),
        load_feature_index(run_path / "features/dense-vision.json"),
        load_feature_index(run_path / "features/language.json"),
    )
    audit_features = FoundationPreparedFeatures(
        None,
        None,
        load_feature_index(run_path / "causality-holdout/features/language.json"),
    )
    cache = FoundationFeatureCache(run_path / "feature-cache")
    replay_store = AppendableAutonomousTrajectoryStore(
        run_path / "replay", "autonomous"
    )
    audit_store = AppendableAutonomousTrajectoryStore(
        run_path / "causality-holdout", "autonomous"
    )
    loader = FoundationSequenceBatchLoader(
        replay_store.path,
        cache,
        preprocessor,
        student,
        training_features,
        transitions=config.sequence_transitions,
        device=device,
    )
    task_ids = tuple(
        sorted({str(value["task_id"]) for value in replay_store.manifest["shards"]})
    )
    if task_ids != tuple(
        sorted(
            (
                "clear_dining_table_3d/v1",
                "store_kitchen_items_3d/v1",
                "tidy_living_room_3d/v1",
            )
        )
    ):
        raise ValueError("frozen batch replay task identities differ")
    if any(loader.legal_transform_ids(index) for index in range(len(loader))):
        raise ValueError("frozen batch replay unexpectedly has legal augmentation")
    return FrozenBatchReplayInputs(
        run_path,
        replay_store,
        audit_store,
        cache,
        preprocessor,
        training_features,
        audit_features,
        loader,
        config,
        task_ids,
    )


def train_frozen_batch_arm(
    trainer: FoundationWorldModelTrainer,
    inputs: FrozenBatchReplayInputs,
    schedule: BatchArmSchedule,
    arm: str,
    *,
    start_update: int,
    stop_update: int,
    progress_interval: int,
    progress: Callable[[int, Mapping[str, float], float], None] | None = None,
) -> dict[str, float]:
    if (
        start_update < 0
        or stop_update > len(schedule.steps)
        or start_update >= stop_update
        or progress_interval <= 0
        or trainer.update_count != start_update
    ):
        raise ValueError("frozen batch replay update range is invalid")
    metrics: list[dict[str, float]] = []
    started = time.perf_counter()
    for update in range(start_update, stop_update):
        indices = schedule.steps[update].indices(arm)
        include_visual = trainer.visual_update_due
        expected_visual = update % trainer.config.visual_update_interval == 0
        if include_visual != expected_visual:
            raise RuntimeError("batch replay visual schedule drifted")
        batch = inputs.training_loader.build(
            indices, include_visual_targets=include_visual
        )
        step = trainer.train_step(
            batch,
            train_task_actor=False,
            train_exploration_actor=False,
        )
        step["trainer/source_episodes_per_batch"] = float(
            len(
                {
                    _source_episode(inputs.training_loader, index)
                    for index in indices
                }
            )
        )
        step["trainer/unique_windows_per_batch"] = float(len(set(indices)))
        metrics.append(step)
        release_accelerator_memory_after_step(len(metrics))
        completed = update + 1
        if progress is not None and completed % progress_interval == 0:
            progress(
                completed,
                mean_metrics(metrics[-progress_interval:]),
                time.perf_counter() - started,
            )
    return mean_metrics(metrics)


def evaluate_frozen_batch_arm(
    trainer: FoundationWorldModelTrainer,
    inputs: FrozenBatchReplayInputs,
) -> dict[str, object]:
    tracker = FoundationActorReadinessTracker(
        actor_readiness_criteria_from_config(inputs.config)
    )
    result = evaluate_foundation_actor_admission(
        trainer,
        inputs.replay_store,
        inputs.audit_store,
        inputs.cache,
        inputs.preprocessor,
        inputs.audit_features,
        trainer.imagination.action_scaling,
        tracker,
        inputs.config,
        inputs.task_ids,
    )
    return {
        "diagnostic": result.diagnostic,
        "readiness": result.readiness,
    }


def module_state_sha256(*modules: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(f"{module_index}:{name}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_input_identity(inputs: FrozenBatchReplayInputs) -> dict[str, object]:
    return {
        "run_path": str(inputs.run_path),
        "replay_manifest_sha256": _file_sha256(
            inputs.replay_store.path / "manifest.json"
        ),
        "audit_manifest_sha256": _file_sha256(
            inputs.audit_store.path / "manifest.json"
        ),
        "training_feature_indices": {
            "vision_language": inputs.training_features.vision_language.to_dict(),
            "dense_vision": inputs.training_features.dense_vision.to_dict(),
            "language": inputs.training_features.language.to_dict(),
        },
        "audit_language_index": inputs.audit_features.language.to_dict(),
        "window_count": len(inputs.training_loader),
        "source_episode_count": len(
            {
                _source_episode(inputs.training_loader, index)
                for index in range(len(inputs.training_loader))
            }
        ),
        "task_ids": list(inputs.task_ids),
        "legal_transform_ids": [],
    }


def save_batch_replay_checkpoint(
    path: Path,
    trainer: FoundationWorldModelTrainer,
    *,
    arm: str,
    seed: int,
    schedule_sha256: str,
    input_identity: Mapping[str, object],
) -> Path:
    if (
        arm not in {"duplicate", "same_source", "cross_source"}
        or seed < 0
        or len(schedule_sha256) != 64
        or trainer.update_count <= 0
    ):
        raise ValueError("batch replay checkpoint identity is invalid")
    state = {
        "arm": arm,
        "seed": seed,
        "schedule_sha256": schedule_sha256,
        "input_identity": dict(input_identity),
        "visual_student": trainer.visual_student.state_dict(),
        "visual_objective": trainer.visual_objective.state_dict(),
        "world_model": trainer.world_model.state_dict(),
        "optimizers": trainer.optimizer_state_dict(),
        "torch_rng_state": capture_torch_rng_state(
            next(trainer.world_model.parameters()).device
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return path


def load_batch_replay_checkpoint(
    path: Path,
    trainer: FoundationWorldModelTrainer,
    *,
    arm: str,
    seed: int,
    schedule_sha256: str,
    input_identity: Mapping[str, object],
) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "arm": arm,
        "seed": seed,
        "schedule_sha256": schedule_sha256,
        "input_identity": dict(input_identity),
    }
    if any(state.get(name) != value for name, value in expected.items()):
        raise ValueError("batch replay checkpoint identity differs")
    trainer.visual_student.load_state_dict(state["visual_student"])
    trainer.visual_objective.load_state_dict(state["visual_objective"])
    trainer.world_model.load_state_dict(state["world_model"])
    trainer.load_optimizer_state_dict(state["optimizers"])
    restore_torch_rng_state(
        state["torch_rng_state"],
        next(trainer.world_model.parameters()).device,
    )


def _source_episode(loader: FoundationSequenceBatchLoader, index: int) -> str:
    metadata = loader.window_metadata(index)
    episode = metadata.get("metadata", {})
    reservoir = (
        episode.get("sequence_reservoir", {})
        if isinstance(episode, Mapping)
        else {}
    )
    source = str(reservoir.get("source_episode_id", ""))
    if not source:
        raise ValueError("frozen batch replay source Episode is missing")
    return source


def _preprocessor(
    config: FoundationOnlineTrainingConfig,
) -> HighResolutionVisionPreprocessor:
    raw = BimanualInputConfig(
        config.camera_width,
        config.camera_height,
        image_width=160,
        image_height=160,
    )
    return HighResolutionVisionPreprocessor(
        HighResolutionVisionConfig(), default_four_camera_calibrations(raw)
    )


def _visual_student_config(path: Path) -> VisualStudentConfig:
    value = _read_config(path)
    value["backbone_dimensions"] = tuple(value["backbone_dimensions"])
    value["backbone_depths"] = tuple(value["backbone_depths"])
    return VisualStudentConfig(**value)


def _read_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not str(value.pop("schema_version", "")).startswith("hwr."):
        raise ValueError(f"foundation config has no schema identity: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
