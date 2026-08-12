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
    siglip: FoundationFeatureIndex
    dinov2: FoundationFeatureIndex
    language: FoundationFeatureIndex

    def __post_init__(self) -> None:
        if self.siglip.role != "vision_language":
            raise ValueError("prepared SigLIP feature role is invalid")
        if self.dinov2.role != "dense_vision":
            raise ValueError("prepared DINO feature role is invalid")
        if self.language.role != "language":
            raise ValueError("prepared language feature role is invalid")
        datasets = {
            self.siglip.dataset_sha256,
            self.dinov2.dataset_sha256,
            self.language.dataset_sha256,
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
        if features.siglip.dataset_sha256 != dataset_digest:
            raise ValueError("prepared features refer to a different trajectory manifest")
        if features.siglip.preprocess_sha256 != preprocessor.fingerprint:
            raise ValueError("SigLIP feature preprocessing differs")
        if features.dinov2.preprocess_sha256 != preprocessor.fingerprint:
            raise ValueError("DINO feature preprocessing differs")

    def __len__(self) -> int:
        return len(self.windows)

    def legal_transform_ids(self, index: int) -> tuple[str, ...]:
        return tuple(self.windows.shard_metadata(index)["legal_transform_ids"])

    def build(self, indices: Sequence[int]) -> FoundationTrainingBatch:
        if not indices:
            raise ValueError("foundation batch requires at least one sequence window")
        sequences = [self._sequence(index) for index in indices]
        observation_count = self.windows.transitions + 1
        student_inputs = {
            name: torch.from_numpy(
                np.stack(
                    [item for sequence in sequences for item in sequence["inputs"][name]]
                )
            ).to(self.device)
            for name in sequences[0]["inputs"]
        }
        siglip, siglip_valid = self._teacher_arrays(sequences, self.features.siglip)
        dinov2, dinov2_valid = self._teacher_arrays(sequences, self.features.dinov2)
        targets = VisualTeacherTargets(
            torch.from_numpy(siglip).to(self.device),
            torch.from_numpy(siglip_valid).to(self.device),
            torch.from_numpy(dinov2).to(self.device),
            torch.from_numpy(dinov2_valid).to(self.device),
            student_inputs["rgb"],
            self._reconstruction_mask(student_inputs["rgb"]),
            student_inputs["head_depth_m"],
            student_inputs["head_depth_valid"],
            batch_correspondence_indices(
                [item for sequence in sequences for item in sequence["correspondences"]],
                device=self.device,
            ),
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
            torch.from_numpy(np.stack([value["executed_action"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(np.stack([value["reward"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(np.stack([value["continue"] for value in sequences])).to(
                self.device
            ),
            torch.from_numpy(np.stack([value["safety"] for value in sequences])).to(
                self.device
            ),
        )

    def _sequence(self, index: int) -> dict[str, object]:
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
            "executed_action": arrays["executed_action"].astype(np.float32),
            "reward": arrays["reward"].astype(np.float32),
            "continue": (~terminal).astype(np.float32),
            "safety": arrays["safety_cost"].astype(np.float32),
        }

    def _teacher_arrays(
        self,
        sequences: list[dict[str, object]],
        index: FoundationFeatureIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        values: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for sequence in sequences:
            for history in sequence["sources"]:
                history_values: list[np.ndarray] = []
                history_valid: list[np.ndarray] = []
                for source in history:
                    key = FoundationCacheKey(
                        "visual", source, index.encoder_lock_sha256, index.preprocess_sha256
                    )
                    feature = self.cache.load_visual(key)
                    history_values.append(feature.values.copy())
                    history_valid.append(feature.valid.copy())
                values.append(np.stack(history_values))
                valid.append(np.stack(history_valid))
        return np.stack(values), np.stack(valid)

    def _reconstruction_mask(self, rgb: torch.Tensor) -> torch.Tensor:
        batch, history, cameras, _, height, width = rgb.shape
        rows = torch.arange(height, device=rgb.device)[:, None]
        columns = torch.arange(width, device=rgb.device)[None, :]
        checker = ((rows // 16 + columns // 16) % 4 == 0)
        return checker.reshape(1, 1, 1, 1, height, width).expand(
            batch, history, cameras, 1, height, width
        )
