from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hwr.adapters.foundation.locks import load_foundation_model_locks
from hwr.adapters.foundation.dinov3 import _patch_grid
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


def test_dinov3_fast_image_processor_runtime_is_installed() -> None:
    import torch
    import torchvision
    from transformers.models.dinov3_vit.image_processing_dinov3_vit_fast import (
        DINOv3ViTImageProcessorFast,
    )

    processor = DINOv3ViTImageProcessorFast()
    import transformers

    images = np.linspace(0.0, 1.0, 3 * 224 * 224 * 3, dtype=np.float32).reshape(
        3, 224, 224, 3
    )
    batch = processor(
        images=list(images),
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
        do_rescale=False,
    )

    assert processor.__class__.__name__ == "DINOv3ViTImageProcessorFast"
    assert torch.__version__.split(".")[:2] == ["2", "13"]
    assert torchvision.__version__.split(".")[:2] == ["0", "28"]
    assert transformers.__version__.split(".")[:2] == ["4", "57"]
    assert tuple(batch["pixel_values"].shape) == (3, 3, 224, 224)


def test_dinov3_patch_grid_strips_class_and_register_tokens() -> None:
    import torch

    prefix = torch.zeros((2, 5, 3), dtype=torch.float32)
    patches = torch.arange(2 * 196 * 3, dtype=torch.float32).reshape(2, 196, 3) + 1.0
    output = SimpleNamespace(last_hidden_state=torch.cat((prefix, patches), dim=1))
    pixels = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    config = SimpleNamespace(patch_size=16, num_register_tokens=4, hidden_size=3)

    grid = _patch_grid(output, pixels, config)

    assert tuple(grid.shape) == (2, 14, 14, 3)
    assert torch.allclose(grid[:, 0, 0], torch.nn.functional.normalize(patches[:, 0], dim=-1))
    assert torch.allclose(torch.linalg.vector_norm(grid, dim=-1), torch.ones((2, 14, 14)))


def test_dinov3_patch_grid_rejects_inconsistent_token_count() -> None:
    import torch

    output = SimpleNamespace(last_hidden_state=torch.zeros((1, 200, 3)))
    pixels = torch.zeros((1, 3, 224, 224))
    config = SimpleNamespace(patch_size=16, num_register_tokens=4, hidden_size=3)

    with pytest.raises(ValueError, match="token count"):
        _patch_grid(output, pixels, config)
