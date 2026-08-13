"""Foundation feature materialization kept outside the online control loop."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from hwr.data.foundation_cache import FoundationFeatureCache
from hwr.data.foundation_features import (
    materialize_language_features,
    materialize_visual_features,
)
from hwr.data.foundation_loading import FoundationPreparedFeatures
from hwr.perception.foundation import (
    FrozenLanguageFeatureProvider,
    FrozenVisionFeatureProvider,
)
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.train.accelerator_memory import release_unused_accelerator_memory


def materialize_foundation_replay_features(
    training_replay: Path,
    audit_replay: Path,
    cache: FoundationFeatureCache,
    preprocessor: HighResolutionVisionPreprocessor,
    output: Path,
    audit_output: Path,
    *,
    vision_language_factory: Callable[[], FrozenVisionFeatureProvider],
    dense_vision_factory: Callable[[], FrozenVisionFeatureProvider],
    language_factory: Callable[[], FrozenLanguageFeatureProvider],
) -> tuple[FoundationPreparedFeatures, FoundationPreparedFeatures]:
    output.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)
    vision_language = vision_language_factory()
    training_vl = materialize_visual_features(
        training_replay,
        cache,
        preprocessor,
        vision_language,
        output / "vision-language.json",
    )
    audit_vl = materialize_visual_features(
        audit_replay,
        cache,
        preprocessor,
        vision_language,
        audit_output / "vision-language.json",
    )
    del vision_language
    release_unused_accelerator_memory()
    dense_vision = dense_vision_factory()
    training_dense = materialize_visual_features(
        training_replay,
        cache,
        preprocessor,
        dense_vision,
        output / "dense-vision.json",
    )
    audit_dense = materialize_visual_features(
        audit_replay,
        cache,
        preprocessor,
        dense_vision,
        audit_output / "dense-vision.json",
    )
    del dense_vision
    release_unused_accelerator_memory()
    language = language_factory()
    training_language = materialize_language_features(
        training_replay, cache, language, output / "language.json"
    )
    audit_language = materialize_language_features(
        audit_replay, cache, language, audit_output / "language.json"
    )
    del language
    release_unused_accelerator_memory()
    return (
        FoundationPreparedFeatures(training_vl, training_dense, training_language),
        FoundationPreparedFeatures(audit_vl, audit_dense, audit_language),
    )
