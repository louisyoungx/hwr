from __future__ import annotations

import hashlib
import json

import pytest

from scripts.fetch_foundation_models import LOCK_SCHEMA, _load_sources, _lock_model
from scripts.verify_foundation_models import (
    _require_language_quality,
    _require_vision_quality,
)


def test_committed_foundation_sources_are_pinned_and_license_identified() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    models = _load_sources(root / "configs/foundation/model-sources.json")

    assert {model["adapter"] for model in models} == {
        "siglip2",
        "dinov3_vit",
        "qwen3_embedding",
    }
    assert all(len(model["revision"]) == 40 for model in models)
    assert all(model["license_id"] for model in models)
    dense = next(model for model in models if model["role"] == "dense_vision")
    assert dense["model_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert dense["license_id"] == "DINOv3-License-2025-08-19"
    assert dense["gated"] is True
    assert dense["access_url"].startswith("https://huggingface.co/facebook/")
    assert dense["license_url"].startswith(
        "https://github.com/facebookresearch/dinov3/"
    )
    assert "LICENSE.md" in dense["required_files"]
    assert all(model["representation_id"].endswith("/v1") for model in models)
    assert all("model.safetensors" in model["required_files"] for model in models)


def test_lock_model_hashes_every_required_file(tmp_path) -> None:
    model_root = tmp_path / "models"
    local = model_root / "fixture"
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"fixture-weight")
    model = {
        "adapter": "fixture",
        "local_name": "fixture",
        "model_id": "fixture/model",
        "revision": "a" * 40,
        "role": "dense_vision",
        "license_id": "Apache-2.0",
        "output_dimension": 4,
        "representation_id": "fixture-grid/v1",
        "required_files": ["config.json", "model.safetensors"],
    }

    lock = _lock_model(model, local, model_root)

    assert lock["artifacts"][1]["sha256"] == hashlib.sha256(b"fixture-weight").hexdigest()
    assert lock["artifacts"][0]["relative_path"] == "fixture/config.json"
    assert lock["representation_id"] == "fixture-grid/v1"


def test_source_loader_rejects_moving_or_duplicate_model_definition(tmp_path) -> None:
    source = tmp_path / "models.json"
    source.write_text(json.dumps({"schema_version": LOCK_SCHEMA, "models": []}))
    with pytest.raises(ValueError, match="schema mismatch"):
        _load_sources(source)


def test_foundation_inference_gate_rejects_degenerate_continuous_features() -> None:
    vision = {
        "feature_shape": [3, 14, 14, 384],
        "valid_patch_count": 392,
        "minimum_valid_norm": 1.0,
        "maximum_valid_norm": 1.0,
        "invalid_features_zero": True,
        "mean_patch_variation": 0.0,
        "mean_valid_image_difference": 0.0,
    }
    language = {
        "feature_shape": [1024],
        "feature_norm": 1.0,
        "paraphrase_cosine": 0.80,
        "different_intent_cosine": 0.81,
    }

    with pytest.raises(RuntimeError, match="non-degeneracy"):
        _require_vision_quality(vision, 384)
    with pytest.raises(RuntimeError, match="semantic gate"):
        _require_language_quality(language, 1024)


def test_foundation_inference_gate_accepts_spatial_and_semantic_signal() -> None:
    _require_vision_quality(
        {
            "feature_shape": [3, 14, 14, 384],
            "valid_patch_count": 392,
            "minimum_valid_norm": 0.9999,
            "maximum_valid_norm": 1.0001,
            "invalid_features_zero": True,
            "mean_patch_variation": 0.02,
            "mean_valid_image_difference": 0.03,
        },
        384,
    )
    _require_language_quality(
        {
            "feature_shape": [1024],
            "feature_norm": 1.0,
            "paraphrase_cosine": 0.90,
            "different_intent_cosine": 0.70,
        },
        1024,
    )
