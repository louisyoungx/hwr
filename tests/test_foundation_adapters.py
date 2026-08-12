from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hwr.adapters.foundation.locks import load_foundation_model_locks
from hwr.adapters.foundation.runtime import validate_vision_input
from hwr.perception import language_source_sha256


def test_committed_foundation_locks_match_runtime_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    locks = load_foundation_model_locks(
        root / "configs/foundation/model-locks.json", root / "models/foundation"
    )
    runtime = json.loads(
        (root / "configs/foundation/runtime-v1.json").read_text(encoding="utf-8")
    )

    configured = {
        runtime["dense_vision_model"],
        runtime["vision_language_model"],
        runtime["language_model"],
    }
    assert configured == set(locks)
    assert {value.model_lock.role for value in locks.values()} == {
        "dense_vision",
        "vision_language",
        "language",
    }
    assert runtime["teacher_image_size"] >= 224
    assert runtime["student_image_size"] >= 160
    assert runtime["visual_history"] >= 4


def test_vision_adapter_input_rejects_invalid_range_and_digest() -> None:
    rgb = np.zeros((3, 224, 224, 3), dtype=np.float32)
    valid = np.ones(3, dtype=np.bool_)

    output, mask = validate_vision_input(rgb, valid, "a" * 64)
    assert output.shape == rgb.shape
    assert mask.tolist() == [True, True, True]
    with pytest.raises(ValueError, match="normalized"):
        validate_vision_input(rgb - 1.0, valid, "a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_vision_input(rgb, valid, "short")


def test_language_source_identity_normalizes_whitespace_but_preserves_locale() -> None:
    first = language_source_sha256("双手  搬运托盘", "zh-CN")
    second = language_source_sha256(" 双手 搬运托盘 ", "zh-CN")
    english = language_source_sha256("双手 搬运托盘", "en-US")

    assert first == second
    assert first != english
