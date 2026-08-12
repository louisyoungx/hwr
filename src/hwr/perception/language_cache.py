"""Deployment-time semantic language lookup with no model inference or generation."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from hwr.perception.foundation import (
    SemanticLanguageFeatures,
    language_source_sha256,
)


@runtime_checkable
class LanguageFeatureResolver(Protocol):
    encoder_lock_sha256: str
    output_dimension: int

    def resolve(self, text: str, locale: str) -> SemanticLanguageFeatures: ...


class StaticLanguageFeatureResolver:
    """Immutable content-hash map populated before the control process starts."""

    def __init__(
        self,
        features: Mapping[tuple[str, str], np.ndarray],
        *,
        encoder_lock_sha256: str,
        output_dimension: int,
    ) -> None:
        if len(encoder_lock_sha256) != 64 or output_dimension <= 0:
            raise ValueError("language resolver identity or dimension is invalid")
        self.encoder_lock_sha256 = encoder_lock_sha256
        self.output_dimension = output_dimension
        values: dict[str, SemanticLanguageFeatures] = {}
        for (locale, text), embedding in features.items():
            source = language_source_sha256(text, locale)
            feature = SemanticLanguageFeatures(
                embedding, encoder_lock_sha256, source
            )
            if feature.values.shape != (output_dimension,):
                raise ValueError("language resolver embedding dimension changed")
            if source in values and not np.array_equal(values[source].values, feature.values):
                raise ValueError("language resolver has conflicting content hashes")
            values[source] = feature
        if not values:
            raise ValueError("language resolver requires at least one embedding")
        self._values = values

    def resolve(self, text: str, locale: str) -> SemanticLanguageFeatures:
        source = language_source_sha256(text, locale)
        try:
            return self._values[source]
        except KeyError as error:
            raise KeyError(
                "instruction embedding was not prepared before deployment"
            ) from error
