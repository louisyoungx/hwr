"""Content-addressed cache for rebuildable frozen foundation features."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from hwr.perception.foundation import DenseVisualFeatures, SemanticLanguageFeatures


FOUNDATION_CACHE_SCHEMA = "hwr.foundation-feature-cache/v1"
FeatureKind = Literal["visual", "language"]


def _sha256(value: str, name: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} requires a SHA-256 digest")
    return digest


@dataclass(frozen=True)
class FoundationCacheKey:
    kind: FeatureKind
    source_sha256: str
    encoder_lock_sha256: str
    preprocess_sha256: str
    schema_version: str = FOUNDATION_CACHE_SCHEMA

    def __post_init__(self) -> None:
        if self.kind not in {"visual", "language"}:
            raise ValueError("foundation cache kind is invalid")
        for name in ("source_sha256", "encoder_lock_sha256", "preprocess_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

    @property
    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class FoundationFeatureCache:
    """Persist derived features without treating them as original observations."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, key: FoundationCacheKey) -> Path:
        return self.root / key.kind / key.digest[:2] / f"{key.digest}.npz"

    def contains(self, key: FoundationCacheKey) -> bool:
        return self.path_for(key).is_file()

    def discard(self, key: FoundationCacheKey) -> bool:
        """Remove one rebuildable cache entry and report whether it existed."""
        path = self.path_for(key)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def store_visual(
        self, key: FoundationCacheKey, features: DenseVisualFeatures
    ) -> Path:
        self._check_identity(key, "visual", features.source_sha256, features.encoder_lock_sha256)
        return self._store(
            key,
            values=features.values,
            valid=features.valid,
        )

    def load_visual(self, key: FoundationCacheKey) -> DenseVisualFeatures:
        if key.kind != "visual":
            raise ValueError("visual feature load requires a visual cache key")
        arrays = self._load(key, ("values", "valid"))
        return DenseVisualFeatures(
            arrays["values"],
            arrays["valid"],
            key.encoder_lock_sha256,
            key.source_sha256,
        )

    def store_language(
        self, key: FoundationCacheKey, features: SemanticLanguageFeatures
    ) -> Path:
        self._check_identity(
            key, "language", features.source_sha256, features.encoder_lock_sha256
        )
        return self._store(key, values=features.values)

    def load_language(self, key: FoundationCacheKey) -> SemanticLanguageFeatures:
        if key.kind != "language":
            raise ValueError("language feature load requires a language cache key")
        arrays = self._load(key, ("values",))
        return SemanticLanguageFeatures(
            arrays["values"], key.encoder_lock_sha256, key.source_sha256
        )

    def _check_identity(
        self,
        key: FoundationCacheKey,
        kind: FeatureKind,
        source_sha256: str,
        encoder_lock_sha256: str,
    ) -> None:
        if key.kind != kind:
            raise ValueError(f"{kind} features require a {kind} cache key")
        if key.source_sha256 != source_sha256:
            raise ValueError("foundation cache source identity differs")
        if key.encoder_lock_sha256 != encoder_lock_sha256:
            raise ValueError("foundation cache encoder identity differs")

    def _store(self, key: FoundationCacheKey, **arrays: np.ndarray) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._load(key, tuple(arrays))
            return path
        metadata = np.asarray(
            json.dumps(
                {
                    "schema_version": FOUNDATION_CACHE_SCHEMA,
                    "cache_key": key.digest,
                    "kind": key.kind,
                    "source_sha256": key.source_sha256,
                    "encoder_lock_sha256": key.encoder_lock_sha256,
                    "preprocess_sha256": key.preprocess_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, metadata=metadata, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _load(
        self, key: FoundationCacheKey, expected_arrays: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        path = self.path_for(key)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            with np.load(path, allow_pickle=False) as stored:
                if set(stored.files) != {"metadata", *expected_arrays}:
                    raise ValueError("foundation cache fields are invalid")
                metadata = json.loads(str(stored["metadata"].item()))
                arrays = {name: stored[name].copy() for name in expected_arrays}
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"foundation cache entry is corrupt: {path.name}") from error
        expected = {
            "schema_version": FOUNDATION_CACHE_SCHEMA,
            "cache_key": key.digest,
            "kind": key.kind,
            "source_sha256": key.source_sha256,
            "encoder_lock_sha256": key.encoder_lock_sha256,
            "preprocess_sha256": key.preprocess_sha256,
        }
        if metadata != expected:
            raise ValueError("foundation cache metadata does not match its key")
        return arrays
