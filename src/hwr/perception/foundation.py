"""Engine-independent contracts for frozen foundation feature providers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np


FOUNDATION_MODEL_SCHEMA = "hwr.foundation-model-lock/v1"
FOUNDATION_FEATURE_SCHEMA = "hwr.foundation-feature/v1"
FoundationRole = Literal["dense_vision", "vision_language", "language"]


def _sha256(value: str, name: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} requires a SHA-256 digest")
    return digest


def _readonly_float32(value: np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WeightArtifact:
    """One immutable local file belonging to a frozen model revision."""

    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("weight artifact path must be relative and contained")
        if self.size_bytes <= 0:
            raise ValueError("weight artifact size must be positive")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "weight artifact"))


@dataclass(frozen=True)
class FoundationModelLock:
    """Auditable identity and local files for one frozen feature provider."""

    model_id: str
    revision: str
    role: FoundationRole
    license_id: str
    output_dimension: int
    artifacts: tuple[WeightArtifact, ...]
    schema_version: str = FOUNDATION_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if not self.model_id or not self.license_id:
            raise ValueError("foundation model identity and license are required")
        if not self.revision or self.revision in {"main", "master", "latest"}:
            raise ValueError("foundation model revision must be immutable")
        if self.role not in {"dense_vision", "vision_language", "language"}:
            raise ValueError("foundation model role is invalid")
        if self.output_dimension <= 0:
            raise ValueError("foundation model output dimension must be positive")
        artifacts = tuple(self.artifacts)
        if not artifacts or len({value.relative_path for value in artifacts}) != len(artifacts):
            raise ValueError("foundation model artifacts must be non-empty and unique")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def lock_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def verify_local_files(self, root: Path) -> tuple[str, ...]:
        errors: list[str] = []
        root = root.resolve()
        for artifact in self.artifacts:
            path = (root / artifact.relative_path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                errors.append(f"missing:{artifact.relative_path}")
                continue
            if path.stat().st_size != artifact.size_bytes:
                errors.append(f"size:{artifact.relative_path}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact.sha256:
                errors.append(f"sha256:{artifact.relative_path}")
        return tuple(errors)


@dataclass(frozen=True)
class DenseVisualFeatures:
    """Spatial foundation features; camera and patch axes are never pooled away."""

    values: np.ndarray
    valid: np.ndarray
    encoder_lock_sha256: str
    source_sha256: str
    schema_version: str = FOUNDATION_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        values = _readonly_float32(self.values, "dense visual features")
        valid = np.ascontiguousarray(self.valid, dtype=np.bool_)
        if values.ndim != 4 or min(values.shape) <= 0:
            raise ValueError("visual features must have camera, row, column, channel axes")
        if valid.shape != values.shape[:3]:
            raise ValueError("visual feature validity must match camera and patch axes")
        valid.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(
            self, "encoder_lock_sha256", _sha256(self.encoder_lock_sha256, "encoder lock")
        )
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "feature source"))


@dataclass(frozen=True)
class SemanticLanguageFeatures:
    """A frozen semantic embedding with no generated text or symbolic plan."""

    values: np.ndarray
    encoder_lock_sha256: str
    source_sha256: str
    schema_version: str = FOUNDATION_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        values = _readonly_float32(self.values, "language features")
        if values.ndim != 1 or not values.size:
            raise ValueError("language features must be a non-empty vector")
        norm = float(np.linalg.norm(values))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("language features require a non-zero norm")
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "encoder_lock_sha256", _sha256(self.encoder_lock_sha256, "encoder lock")
        )
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "feature source"))


@runtime_checkable
class FrozenVisionFeatureProvider(Protocol):
    """Adapter boundary implemented by offline frozen vision models."""

    @property
    def model_lock(self) -> FoundationModelLock: ...

    def encode(self, rgb: np.ndarray, camera_valid: np.ndarray) -> DenseVisualFeatures: ...


@runtime_checkable
class FrozenLanguageFeatureProvider(Protocol):
    """Adapter boundary implemented by an offline frozen semantic encoder."""

    @property
    def model_lock(self) -> FoundationModelLock: ...

    def encode(self, text: str, locale: str) -> SemanticLanguageFeatures: ...
