"""Frozen DINOv3 ConvNeXt dense spatial feature adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from hwr.adapters.foundation.locks import LockedFoundationModel
from hwr.adapters.foundation.runtime import freeze_model, select_device, validate_vision_input
from hwr.perception.foundation import DenseVisualFeatures, FoundationModelLock


DINOV3_CONVNEXT_REPRESENTATION = "convnext-stage3-grid-l2/v1"


def _stage_three_grid(
    output: Any, pixel_values: Any, output_dimension: int
) -> Any:
    """Select the stride-16 DINOv3 grid instead of the coarse pooled output."""
    import torch

    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) < 4:
        raise ValueError("DINOv3 did not return all ConvNeXt stage features")
    feature_map = hidden_states[-2]
    expected_grid = (pixel_values.shape[-2] // 16, pixel_values.shape[-1] // 16)
    if feature_map.ndim != 4 or tuple(feature_map.shape[-2:]) != expected_grid:
        raise ValueError("DINOv3 stage-three feature grid has an invalid stride")
    if feature_map.shape[1] != output_dimension:
        raise ValueError("DINOv3 stage-three dimension differs from its model lock")
    values = feature_map.permute(0, 2, 3, 1)
    return torch.nn.functional.normalize(values, dim=-1)


class Dinov3ConvNextDenseVisionProvider:
    """Expose only normalized continuous DINOv3 stage-three feature grids."""

    def __init__(self, locked: LockedFoundationModel, *, device: str = "auto") -> None:
        model_lock = locked.model_lock
        if locked.adapter != "dinov3_convnext" or model_lock.role != "dense_vision":
            raise ValueError("DINOv3 ConvNeXt adapter requires a dense-vision lock")
        if model_lock.representation_id != DINOV3_CONVNEXT_REPRESENTATION:
            raise ValueError("DINOv3 ConvNeXt representation identity differs")
        locked.verify()
        from transformers import AutoImageProcessor, AutoModel

        self.locked = locked
        self.device = select_device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            locked.local_path, local_files_only=True, use_fast=False
        )
        model = AutoModel.from_pretrained(
            locked.local_path, local_files_only=True, trust_remote_code=False
        )
        if getattr(model.config, "model_type", None) != "dinov3_convnext":
            raise ValueError("DINOv3 ConvNeXt checkpoint architecture differs")
        if int(model.config.hidden_sizes[-2]) != model_lock.output_dimension:
            raise ValueError("DINOv3 stage-three configuration dimension differs")
        self.model = freeze_model(model, self.device)

    @property
    def model_lock(self) -> FoundationModelLock:
        return self.locked.model_lock

    def encode_vision(
        self, rgb: np.ndarray, camera_valid: np.ndarray, source_sha256: str
    ) -> DenseVisualFeatures:
        import torch

        images, valid = validate_vision_input(rgb, camera_valid, source_sha256)
        inputs = self.processor(
            images=list(images),
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
            do_rescale=False,
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            output = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        values = _stage_three_grid(
            output, pixel_values, self.model_lock.output_dimension
        ).float().cpu().numpy()
        values[~valid] = 0.0
        patch_valid = np.broadcast_to(valid[:, None, None], values.shape[:3]).copy()
        return DenseVisualFeatures(
            values, patch_valid, self.model_lock.lock_sha256, source_sha256
        )
