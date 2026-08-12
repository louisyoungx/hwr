"""Frozen DINOv2 dense spatial feature adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from hwr.adapters.foundation.locks import LockedFoundationModel
from hwr.adapters.foundation.runtime import freeze_model, select_device, validate_vision_input
from hwr.perception.foundation import DenseVisualFeatures, FoundationModelLock


class Dinov2DenseVisionProvider:
    def __init__(self, locked: LockedFoundationModel, *, device: str = "auto") -> None:
        if locked.adapter != "dinov2" or locked.model_lock.role != "dense_vision":
            raise ValueError("DINOv2 adapter requires a dense vision lock")
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
        tokens = output.last_hidden_state[:, 1:]
        patch_size = int(self.model.config.patch_size)
        grid_height = pixel_values.shape[-2] // patch_size
        grid_width = pixel_values.shape[-1] // patch_size
        if tokens.shape[1] != grid_height * grid_width:
            raise ValueError("DINOv2 patch token count does not match the image grid")
        values = tokens.reshape(len(images), grid_height, grid_width, -1)
        values = torch.nn.functional.normalize(values, dim=-1).float().cpu().numpy()
        values[~valid] = 0.0
        patch_valid = np.broadcast_to(valid[:, None, None], values.shape[:3]).copy()
        if values.shape[-1] != self.model_lock.output_dimension:
            raise ValueError("DINOv2 feature dimension differs from its model lock")
        return DenseVisualFeatures(
            values, patch_valid, self.model_lock.lock_sha256, source_sha256
        )
