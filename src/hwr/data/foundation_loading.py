"""Build unified training batches from raw autonomous sequence windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from hwr.data.foundation_cache import FoundationCacheKey, FoundationFeatureCache
from hwr.data.foundation_features import (
    FoundationFeatureIndex,
    file_sha256,
    trajectory_vision_frame,
)
from hwr.data.trajectory_windows import AutonomousTrajectoryWindows
from hwr.perception.foundation import language_source_sha256
from hwr.perception.geometric_correspondence import (
    batch_correspondence_indices,
    build_cross_camera_patch_correspondences,
)
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.perception.student import VisualStudentConfig
from hwr.perception.student_input import VisualStudentInputAssembler
from hwr.perception.student_objectives import VisualTeacherTargets
from hwr.train.foundation_batch import FoundationTrainingBatch


@dataclass(frozen=True)
class FoundationPreparedFeatures:
    vision_language: FoundationFeatureIndex | None
    dense_vision: FoundationFeatureIndex | None
    language: FoundationFeatureIndex

    def __post_init__(self) -> None:
        if (
            self.vision_language is not None
            and self.vision_language.role != "vision_language"
        ):
            raise ValueError("prepared vision-language feature role is invalid")
        if self.dense_vision is not None and self.dense_vision.role != "dense_vision":
            raise ValueError("prepared dense-vision feature role is invalid")
        if self.language.role != "language":
            raise ValueError("prepared language feature role is invalid")
        datasets = {
            self.language.dataset_sha256,
            *(
                [self.vision_language.dataset_sha256]
                if self.vision_language is not None
                else []
            ),
            *(
                [self.dense_vision.dataset_sha256]
                if self.dense_vision is not None
                else []
            ),
        }
        if len(datasets) != 1:
            raise ValueError("prepared features refer to different trajectory datasets")


class FoundationSequenceBatchLoader:
    def __init__(
        self,
        dataset_path: Path,
        cache: FoundationFeatureCache,
        preprocessor: HighResolutionVisionPreprocessor,
        student_config: VisualStudentConfig,
        features: FoundationPreparedFeatures,
        *,
        transitions: int,
        device: str = "cpu",
    ) -> None:
        self.windows = AutonomousTrajectoryWindows(
            dataset_path, transitions=transitions
        )
        self.cache = cache
        self.preprocessor = preprocessor
        self.student_config = student_config
        self.features = features
        self.device = device
        dataset_digest = file_sha256(dataset_path / "manifest.json")
        if features.language.dataset_sha256 != dataset_digest:
            raise ValueError("prepared features refer to a different trajectory manifest")
        if (
            features.vision_language is not None
            and features.vision_language.preprocess_sha256 != preprocessor.fingerprint
        ):
            raise ValueError("vision-language feature preprocessing differs")
        if (
            features.dense_vision is not None
            and features.dense_vision.preprocess_sha256 != preprocessor.fingerprint
        ):
            raise ValueError("dense-vision feature preprocessing differs")

    def __len__(self) -> int:
        return len(self.windows)

    def legal_transform_ids(self, index: int) -> tuple[str, ...]:
        return tuple(self.windows.shard_metadata(index)["legal_transform_ids"])

    def window_metadata(self, index: int) -> dict[str, object]:
        return self.windows.window_metadata(index)

    def window_shard_index(self, index: int) -> int:
        return self.windows.indices[index].shard_index

    def build(
        self,
        indices: Sequence[int],
        *,
        include_visual_targets: bool = True,
    ) -> FoundationTrainingBatch:
        if not indices:
            raise ValueError("foundation batch requires at least one sequence window")
        sequences = [
            self._sequence(index, include_visual_targets=include_visual_targets)
            for index in indices
        ]
        observation_count = self.windows.transitions + 1
        student_inputs = {
            name: torch.from_numpy(
                np.stack(
                    [item for sequence in sequences for item in sequence["inputs"][name]]
                )
            ).to(self.device)
            for name in sequences[0]["inputs"]
        }
        targets = (
            self._visual_targets(sequences, student_inputs)
            if include_visual_targets
            else None
        )
        return FoundationTrainingBatch(
            student_inputs,
            targets,
            len(sequences),
            observation_count,
            torch.from_numpy(np.stack([value["language"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(np.stack([value["proprioception"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(np.stack([value["actor_proposals"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(np.stack([value["executed_action"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(np.stack([value["reward"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(np.stack([value["continue"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(
                np.stack([value["safety_interventions"] for value in sequences])
            ).to(
                self.device
            ),
            torch.from_numpy(
                np.stack([value["severe_collisions"] for value in sequences])
            ).to(self.device),
        )

    def _sequence(
        self, index: int, *, include_visual_targets: bool
    ) -> dict[str, object]:
        arrays = self.windows[index]
        metadata = self.windows.shard_metadata(index)
        frames = [
            trajectory_vision_frame(arrays, item, metadata, self.preprocessor)
            for item in range(self.windows.transitions + 1)
        ]
        assembler = VisualStudentInputAssembler(
            visual_history=self.student_config.visual_history,
            image_size=self.student_config.image_size,
        )
        named: dict[str, list[np.ndarray]] = {}
        sources: list[tuple[str, ...]] = []
        correspondences: list[list[np.ndarray]] = []
        frame_history: list[object] = []
        grid_size = self.student_config.image_size // 16
        for frame in frames:
            assembled = assembler.build(frame)
            for name, value in assembled.named_arrays().items():
                named.setdefault(name, []).append(value)
            if include_visual_targets:
                sources.append(assembled.source_sha256)
                frame_history.append(frame)
                padded = [frame_history[0]] * (
                    self.student_config.visual_history - len(frame_history)
                ) + frame_history[-self.student_config.visual_history :]
                correspondences.append(
                    [
                        build_cross_camera_patch_correspondences(
                            value, feature_grid_size=grid_size
                        )
                        for value in padded
                    ]
                )
        language_source = language_source_sha256(
            str(metadata["instruction"]), str(metadata["locale"])
        )
        language_key = FoundationCacheKey(
            "language",
            language_source,
            self.features.language.encoder_lock_sha256,
            self.features.language.preprocess_sha256,
        )
        terminal = arrays["terminated"].astype(bool) | arrays["truncated"].astype(bool)
        return {
            "inputs": named,
            "sources": sources,
            "correspondences": correspondences,
            "language": self.cache.load_language(language_key).values.copy(),
            "proprioception": arrays["proprioception"].astype(np.float32),
            "actor_proposals": arrays["actor_proposal"].astype(np.float32),
            "executed_action": arrays["executed_action"].astype(np.float32),
            "reward": arrays["reward"].astype(np.float32),
            "continue": (~terminal).astype(np.float32),
            "safety_interventions": arrays["safety_intervention"].astype(np.float32),
            "severe_collisions": (
                arrays["terminated"].astype(np.float32)
                if metadata.get("metadata", {}).get("result_reason")
                == "severe_collision"
                else np.zeros_like(arrays["reward"], dtype=np.float32)
            ),
        }

    def _teacher_arrays(
        self,
        sequences: list[dict[str, object]],
        index: FoundationFeatureIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        sources = tuple(
            dict.fromkeys(
                source
                for sequence in sequences
                for history in sequence["sources"]
                for source in history
            )
        )
        loaded = {}
        for source in sources:
            key = FoundationCacheKey(
                "visual", source, index.encoder_lock_sha256, index.preprocess_sha256
            )
            loaded[source] = self.cache.load_visual(key)
        values: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for sequence in sequences:
            for history in sequence["sources"]:
                history_values: list[np.ndarray] = []
                history_valid: list[np.ndarray] = []
                for source in history:
                    feature = loaded[source]
                    history_values.append(feature.values)
                    history_valid.append(feature.valid)
                values.append(np.stack(history_values))
                valid.append(np.stack(history_valid))
        return np.stack(values), np.stack(valid)

    def _visual_targets(
        self,
        sequences: list[dict[str, object]],
        student_inputs: dict[str, torch.Tensor],
    ) -> VisualTeacherTargets:
        if self.features.vision_language is None or self.features.dense_vision is None:
            raise ValueError("visual targets require materialized teacher features")
        vision_language, vision_language_valid = self._teacher_arrays(
            sequences, self.features.vision_language
        )
        dense_vision, dense_vision_valid = self._teacher_arrays(
            sequences, self.features.dense_vision
        )
        return VisualTeacherTargets(
            torch.from_numpy(vision_language).to(self.device),
            torch.from_numpy(vision_language_valid).to(self.device),
            torch.from_numpy(dense_vision).to(self.device),
            torch.from_numpy(dense_vision_valid).to(self.device),
            student_inputs["rgb"],
            self._reconstruction_mask(student_inputs["rgb"]),
            student_inputs["head_depth_m"],
            student_inputs["head_depth_valid"],
            batch_correspondence_indices(
                [
                    item
                    for sequence in sequences
                    for item in sequence["correspondences"]
                ],
                device=self.device,
            ),
        )

    def _reconstruction_mask(self, rgb: torch.Tensor) -> torch.Tensor:
        batch, history, cameras, _, height, width = rgb.shape
        rows = torch.arange(height, device=rgb.device)[:, None]
        columns = torch.arange(width, device=rgb.device)[None, :]
        checker = ((rows // 16 + columns // 16) % 4 == 0)
        return checker.reshape(1, 1, 1, 1, height, width).expand(
            batch, history, cameras, 1, height, width
        )
