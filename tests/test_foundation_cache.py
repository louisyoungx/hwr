from __future__ import annotations

import numpy as np
import pytest

from hwr.data.foundation_cache import FoundationCacheKey, FoundationFeatureCache
from hwr.perception import DenseVisualFeatures, SemanticLanguageFeatures


SOURCE = "1" * 64
ENCODER = "2" * 64
PREPROCESS = "3" * 64


def test_visual_foundation_cache_round_trip_is_content_addressed(tmp_path) -> None:
    cache = FoundationFeatureCache(tmp_path)
    key = FoundationCacheKey("visual", SOURCE, ENCODER, PREPROCESS)
    features = DenseVisualFeatures(
        np.arange(72, dtype=np.float32).reshape(3, 2, 3, 4),
        np.ones((3, 2, 3), dtype=np.bool_),
        ENCODER,
        SOURCE,
    )

    path = cache.store_visual(key, features)
    loaded = cache.load_visual(key)

    assert path == cache.path_for(key)
    assert cache.contains(key)
    assert np.array_equal(loaded.values, features.values)
    assert np.array_equal(loaded.valid, features.valid)
    assert not loaded.values.flags.writeable
    assert cache.store_visual(key, features) == path
    assert cache.discard(key) is True
    assert cache.discard(key) is False
    assert not cache.contains(key)


def test_language_foundation_cache_round_trip_and_identity_checks(tmp_path) -> None:
    cache = FoundationFeatureCache(tmp_path)
    key = FoundationCacheKey("language", SOURCE, ENCODER, PREPROCESS)
    features = SemanticLanguageFeatures(
        np.asarray([0.1, 0.3, -0.7], dtype=np.float32), ENCODER, SOURCE
    )

    cache.store_language(key, features)
    loaded = cache.load_language(key)

    assert np.array_equal(loaded.values, features.values)
    wrong = FoundationCacheKey("language", "4" * 64, ENCODER, PREPROCESS)
    with pytest.raises(ValueError, match="source identity"):
        cache.store_language(wrong, features)


def test_foundation_cache_rejects_corrupt_or_cross_kind_entries(tmp_path) -> None:
    cache = FoundationFeatureCache(tmp_path)
    key = FoundationCacheKey("visual", SOURCE, ENCODER, PREPROCESS)
    cache.path_for(key).parent.mkdir(parents=True)
    cache.path_for(key).write_bytes(b"not-an-npz")

    with pytest.raises(ValueError, match="corrupt"):
        cache.load_visual(key)
    with pytest.raises(ValueError, match="visual cache key"):
        cache.load_visual(FoundationCacheKey("language", SOURCE, ENCODER, PREPROCESS))


def test_visual_memory_cache_is_small_lru_and_discard_invalidates_it(tmp_path) -> None:
    cache = FoundationFeatureCache(tmp_path, visual_memory_cache_entries=1)
    keys = [
        FoundationCacheKey("visual", str(index) * 64, ENCODER, PREPROCESS)
        for index in (4, 5)
    ]
    for index, key in enumerate(keys):
        cache.store_visual(
            key,
            DenseVisualFeatures(
                np.full((1, 1, 1, 1), index, np.float32),
                np.ones((1, 1, 1), np.bool_),
                ENCODER,
                key.source_sha256,
            ),
        )

    first = cache.load_visual(keys[0])
    assert cache.load_visual(keys[0]) is first
    cache.load_visual(keys[1])
    assert cache.load_visual(keys[0]) is not first
    assert cache.discard(keys[0]) is True
