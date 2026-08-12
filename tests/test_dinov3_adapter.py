from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hwr.adapters.foundation.dinov3 import (
    DINOV3_CONVNEXT_REPRESENTATION,
    Dinov3ConvNextDenseVisionProvider,
)
from hwr.adapters.foundation.locks import LockedFoundationModel
from hwr.perception.foundation import FoundationModelLock, WeightArtifact


def _model_lock(representation: str = DINOV3_CONVNEXT_REPRESENTATION):
    return FoundationModelLock(
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "1" * 40,
        "dense_vision",
        "DINOv3-License",
        384,
        representation,
        (WeightArtifact("fixture/model.safetensors", "2" * 64, 1),),
    )


class _Processor:
    def __call__(self, *, images, **kwargs):
        del kwargs
        return {
            "pixel_values": torch.from_numpy(
                np.stack(images).transpose(0, 3, 1, 2).copy()
            )
        }


class _Model:
    def __call__(self, *, pixel_values, output_hidden_states, return_dict):
        assert output_hidden_states and return_dict
        batch = pixel_values.shape[0]
        stage_three = torch.arange(1, 385, dtype=torch.float32).reshape(
            1, 384, 1, 1
        ).expand(batch, 384, 14, 14)
        return SimpleNamespace(
            hidden_states=(
                pixel_values,
                torch.zeros(batch, 96, 56, 56),
                torch.zeros(batch, 192, 28, 28),
                stage_three,
                torch.zeros(batch, 768, 7, 7),
            )
        )


def test_dinov3_adapter_returns_normalized_stride_16_dense_grid() -> None:
    provider = Dinov3ConvNextDenseVisionProvider.__new__(
        Dinov3ConvNextDenseVisionProvider
    )
    provider.locked = SimpleNamespace(model_lock=_model_lock())
    provider.device = torch.device("cpu")
    provider.processor = _Processor()
    provider.model = _Model()
    rgb = np.zeros((3, 224, 224, 3), dtype=np.float32)
    valid = np.asarray((True, True, False), dtype=np.bool_)

    features = provider.encode_vision(rgb, valid, "a" * 64)

    assert features.values.shape == (3, 14, 14, 384)
    np.testing.assert_allclose(
        np.linalg.norm(features.values[features.valid], axis=-1), 1.0, atol=1e-6
    )
    assert np.all(features.values[~features.valid] == 0.0)


def test_dinov3_adapter_rejects_unlocked_feature_selector(tmp_path) -> None:
    locked = LockedFoundationModel(
        "dinov3_convnext", "fixture", tmp_path, _model_lock("final-grid-l2/v1")
    )

    with pytest.raises(ValueError, match="representation identity"):
        Dinov3ConvNextDenseVisionProvider(locked, device="cpu")
