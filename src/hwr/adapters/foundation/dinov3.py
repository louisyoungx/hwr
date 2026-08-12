"""Frozen DINOv3 ViT dense spatial feature adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from hwr.adapters.foundation.locks import LockedFoundationModel
from hwr.adapters.foundation.runtime import freeze_model, select_device, validate_vision_input
from hwr.perception.foundation import DenseVisualFeatures, FoundationModelLock


DINOV3_VIT_REPRESENTATION = "vit-patch-grid-l2/v1"


def _patch_grid(output: Any, pixel_values: Any, model_config: Any) -> Any:
    """Discard class/register tokens while preserving every stride-16 patch."""
    import torch

    tokens = output.last_hidden_state
    patch_size = int(model_config.patch_size)
    grid_height = pixel_values.shape[-2] // patch_size
    grid_width = pixel_values.shape[-1] // patch_size
    prefix = 1 + int(model_config.num_register_tokens)
    patch_tokens = tokens[:, prefix:]
    if patch_tokens.shape[1] != grid_height * grid_width:
        raise ValueError("DINOv3 patch token count does not match the image grid")
    if patch_tokens.shape[-1] != int(model_config.hidden_size):
        raise ValueError("DINOv3 patch dimension differs from its configuration")
    values = patch_tokens.reshape(
        len(pixel_values), grid_height, grid_width, model_config.hidden_size
    )
    return torch.nn.functional.normalize(values, dim=-1)


class Dinov3ViTDenseVisionProvider:
    """Expose only normalized continuous DINOv3 final-layer patch grids."""

    def __init__(self, locked: LockedFoundationModel, *, device: str = "auto") -> None:
        model_lock = locked.model_lock
        if locked.adapter != "dinov3_vit" or model_lock.role != "dense_vision":
            raise ValueError("DINOv3 ViT adapter requires a dense-vision lock")
        if model_lock.representation_id != DINOV3_VIT_REPRESENTATION:
            raise ValueError("DINOv3 ViT representation identity differs")
        locked.verify()
        from transformers import AutoImageProcessor, AutoModel

        self.locked = locked
        self.device = select_device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            locked.local_path, local_files_only=True, use_fast=True
        )
        model = AutoModel.from_pretrained(
            locked.local_path, local_files_only=True, trust_remote_code=False
        )
        if getattr(model.config, "model_type", None) != "dinov3_vit":
            raise ValueError("DINOv3 ViT checkpoint architecture differs")
        if int(model.config.hidden_size) != model_lock.output_dimension:
            raise ValueError("DINOv3 patch configuration dimension differs")
        if int(model.config.patch_size) != 16:
            raise ValueError("DINOv3 dense teacher requires a stride-16 patch model")
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
            output = self.model(pixel_values=pixel_values, return_dict=True)
        values = _patch_grid(
            output, pixel_values, self.model.config
        ).float().cpu().numpy()
        values[~valid] = 0.0
        patch_valid = np.broadcast_to(valid[:, None, None], values.shape[:3]).copy()
        return DenseVisualFeatures(
            values, patch_valid, self.model_lock.lock_sha256, source_sha256
        )
