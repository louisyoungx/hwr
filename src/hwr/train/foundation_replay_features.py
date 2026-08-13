"""Feature-index helpers for the bounded foundation replay."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from hwr.data.foundation_cache import FoundationCacheKey, FoundationFeatureCache
from hwr.data.foundation_features import load_feature_index
from hwr.perception.foundation import language_source_sha256
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.perception.language_cache import StaticLanguageFeatureResolver


def language_resolver_from_replay(
    shards: Iterable[Mapping[str, object]],
    cache: FoundationFeatureCache,
    index_path: Path,
) -> StaticLanguageFeatureResolver:
    if not index_path.is_file():
        raise RuntimeError("language features must be materialized before Actor collection")
    index = load_feature_index(index_path)
    features = {}
    for shard in shards:
        text, locale = str(shard["instruction"]), str(shard["locale"])
        source = language_source_sha256(text, locale)
        key = FoundationCacheKey(
            "language", source, index.encoder_lock_sha256, index.preprocess_sha256
        )
        features[(locale, text)] = cache.load_language(key).values.copy()
    return StaticLanguageFeatureResolver(
        features,
        encoder_lock_sha256=index.encoder_lock_sha256,
        output_dimension=index.output_dimension,
    )


def discard_visual_feature_sources(
    sources: tuple[str, ...],
    cache: FoundationFeatureCache,
    preprocessor: HighResolutionVisionPreprocessor,
    index_paths: Iterable[Path],
) -> None:
    if not sources:
        return
    for path in index_paths:
        if not path.is_file():
            continue
        index = load_feature_index(path)
        for source in sources:
            cache.discard(
                FoundationCacheKey(
                    "visual",
                    source,
                    index.encoder_lock_sha256,
                    preprocessor.fingerprint,
                )
            )
