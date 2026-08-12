from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hwr.adapters.foundation.dinov3 import (
    DINOV3_VIT_REPRESENTATION,
    Dinov3ViTDenseVisionProvider,
)
from hwr.adapters.foundation.locks import LockedFoundationModel
from hwr.perception.foundation import FoundationModelLock, WeightArtifact


def _model_lock(representation: str = DINOV3_VIT_REPRESENTATION):
    return FoundationModelLock(
        "facebook/dinov3-vits16-pretrain-lvd1689m",
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
    config = SimpleNamespace(
        model_type="dinov3_vit",
        patch_size=16,
        hidden_size=384,
        num_register_tokens=4,
    )

    def __call__(self, *, pixel_values, return_dict):
        assert return_dict
        batch = pixel_values.shape[0]
        prefix = torch.zeros(batch, 5, 384)
        patches = torch.arange(1, 385, dtype=torch.float32).reshape(
            1, 1, 384
        ).expand(batch, 196, 384)
        return SimpleNamespace(last_hidden_state=torch.cat((prefix, patches), dim=1))


def test_dinov3_adapter_returns_normalized_stride_16_dense_grid() -> None:
    provider = Dinov3ViTDenseVisionProvider.__new__(Dinov3ViTDenseVisionProvider)
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
        "dinov3_vit", "fixture", tmp_path, _model_lock("pooled-token/v1")
    )

    with pytest.raises(ValueError, match="representation identity"):
        Dinov3ViTDenseVisionProvider(locked, device="cpu")
