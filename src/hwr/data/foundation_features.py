"""Offline materialization of frozen continuous foundation features."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hwr.core.embodied import (
    DualArmObservation,
    DualArmProprioception,
    FrameCameraCalibration,
    NaturalLanguageInstruction,
)
from hwr.core.types import CameraFrame
from hwr.data.autonomous_trajectory import verify_autonomous_trajectory_dataset
from hwr.data.foundation_cache import (
    FoundationCacheKey,
    FoundationFeatureCache,
)
from hwr.perception.contracts import DUAL_ARM_CAMERA_IDS
from hwr.perception.foundation import (
    FrozenLanguageFeatureProvider,
    FrozenVisionFeatureProvider,
    language_source_sha256,
)
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor


FOUNDATION_FEATURE_INDEX_SCHEMA = "hwr.foundation-feature-index/v1"
LANGUAGE_PREPROCESS_SHA256 = hashlib.sha256(
    b"hwr.normalized-language-content/v1"
).hexdigest()


@dataclass(frozen=True)
class FoundationFeatureIndex:
    kind: str
    role: str
    dataset_sha256: str
    encoder_lock_sha256: str
    preprocess_sha256: str
    output_dimension: int
    entry_count: int
    schema_version: str = FOUNDATION_FEATURE_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.kind not in {"visual", "language"}:
            raise ValueError("foundation feature index kind is invalid")
        if not self.role or min(self.output_dimension, self.entry_count) <= 0:
            raise ValueError("foundation feature index dimensions are invalid")
        for value in (
            self.dataset_sha256,
            self.encoder_lock_sha256,
            self.preprocess_sha256,
        ):
            if len(value) != 64:
                raise ValueError("foundation feature index requires SHA-256 identities")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_vision_frame(
    arrays: Mapping[str, np.ndarray],
    observation_index: int,
    metadata: Mapping[str, Any],
    preprocessor: HighResolutionVisionPreprocessor,
):
    """Rebuild deterministic teacher/student views from one raw trajectory record."""
    rgb = arrays["rgb_uint8"][observation_index]
    depth = arrays["raw_head_depth_m"][observation_index]
    if rgb.ndim != 4 or depth.shape != rgb.shape[1:3]:
        raise ValueError("trajectory source camera shapes are invalid")
    height, width = depth.shape
    timestamps = arrays["frame_timestamps_ns"][observation_index]
    payloads = {
        "head_rgb": rgb[0].tobytes(),
        "head_depth": depth.astype(np.float32, copy=False).tobytes(),
        "left_wrist_rgb": rgb[1].tobytes(),
        "right_wrist_rgb": rgb[2].tobytes(),
    }
    cameras = tuple(
        CameraFrame(
            name,
            max(0, int(timestamps[index])),
            observation_index,
            width,
            height,
            "depth32f" if name == "head_depth" else "rgb8",
            payload=payloads[name],
        )
        for index, name in enumerate(DUAL_ARM_CAMERA_IDS)
    )
    proprioception = DualArmProprioception(
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        0.0,
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0),
    )
    observation = DualArmObservation(
        max(0, int(timestamps[0])),
        observation_index,
        str(metadata["task_id"]),
        NaturalLanguageInstruction(
            str(metadata["instruction"]), str(metadata["locale"])
        ),
        proprioception,
        cameras,
        tuple(
            FrameCameraCalibration(
                name,
                tuple(float(value) for value in arrays["intrinsics"][observation_index, index]),
                tuple(
                    float(value)
                    for value in arrays["robot_from_camera"][observation_index, index].reshape(-1)
                ),
            )
            for index, name in enumerate(DUAL_ARM_CAMERA_IDS)
        ),
    )
    result = preprocessor.preprocess(observation)
    expected_fingerprint = str(metadata["preprocess_fingerprint"])
    if result.preprocess_fingerprint != expected_fingerprint:
        raise ValueError("trajectory preprocessing fingerprint differs")
    source = str(arrays["observation_source_sha256"][observation_index])
    validity = arrays["camera_validity"][observation_index].astype(np.bool_)
    return replace(result, source_sha256=source, camera_validity=validity)


def materialize_visual_features(
    dataset_path: Path,
    cache: FoundationFeatureCache,
    preprocessor: HighResolutionVisionPreprocessor,
    provider: FrozenVisionFeatureProvider,
    output_path: Path,
) -> FoundationFeatureIndex:
    manifest = verify_autonomous_trajectory_dataset(dataset_path)
    entries = 0
    for shard in manifest["shards"]:
        with np.load(dataset_path / shard["path"], allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
        for index in range(int(shard["observation_count"])):
            frame = trajectory_vision_frame(arrays, index, shard, preprocessor)
            key = FoundationCacheKey(
                "visual",
                frame.source_sha256,
                provider.model_lock.lock_sha256,
                preprocessor.fingerprint,
            )
            if not cache.contains(key):
                valid = frame.camera_validity[[0, 2, 3]]
                features = provider.encode_vision(
                    frame.teacher_rgb, valid, frame.source_sha256
                )
                cache.store_visual(key, features)
            entries += 1
    index = FoundationFeatureIndex(
        "visual",
        provider.model_lock.role,
        file_sha256(dataset_path / "manifest.json"),
        provider.model_lock.lock_sha256,
        preprocessor.fingerprint,
        provider.model_lock.output_dimension,
        entries,
    )
    _atomic_json(output_path, index.to_dict())
    return index


def materialize_language_features(
    dataset_path: Path,
    cache: FoundationFeatureCache,
    provider: FrozenLanguageFeatureProvider,
    output_path: Path,
) -> FoundationFeatureIndex:
    manifest = verify_autonomous_trajectory_dataset(dataset_path)
    unique: dict[str, tuple[str, str]] = {}
    for shard in manifest["shards"]:
        text, locale = str(shard["instruction"]), str(shard["locale"])
        unique[language_source_sha256(text, locale)] = (text, locale)
    for source, (text, locale) in unique.items():
        key = FoundationCacheKey(
            "language",
            source,
            provider.model_lock.lock_sha256,
            LANGUAGE_PREPROCESS_SHA256,
        )
        if not cache.contains(key):
            cache.store_language(key, provider.encode_language(text, locale))
    index = FoundationFeatureIndex(
        "language",
        provider.model_lock.role,
        file_sha256(dataset_path / "manifest.json"),
        provider.model_lock.lock_sha256,
        LANGUAGE_PREPROCESS_SHA256,
        provider.model_lock.output_dimension,
        len(unique),
    )
    _atomic_json(output_path, index.to_dict())
    return index


def load_feature_index(path: Path) -> FoundationFeatureIndex:
    value = json.loads(path.read_text(encoding="utf-8"))
    return FoundationFeatureIndex(**value)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
