"""Frozen SigLIP2 joint dense vision and multilingual text adapter."""

from __future__ import annotations

import numpy as np

from hwr.adapters.foundation.locks import LockedFoundationModel
from hwr.adapters.foundation.runtime import freeze_model, select_device, validate_vision_input
from hwr.perception.foundation import (
    DenseVisualFeatures,
    FoundationModelLock,
    SemanticLanguageFeatures,
    language_source_sha256,
)


class Siglip2VisionLanguageProvider:
    def __init__(self, locked: LockedFoundationModel, *, device: str = "auto") -> None:
        if locked.adapter != "siglip2" or locked.model_lock.role != "vision_language":
            raise ValueError("SigLIP2 adapter requires a vision-language lock")
        locked.verify()
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        self.locked = locked
        self.device = select_device(device)
        self.image_processor = AutoImageProcessor.from_pretrained(
            locked.local_path, local_files_only=True, use_fast=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            locked.local_path, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            locked.local_path, local_files_only=True, trust_remote_code=False
        )
        self.model = freeze_model(model, self.device)
        self.maximum_text_tokens = int(self.model.config.text_config.max_position_embeddings)

    @property
    def model_lock(self) -> FoundationModelLock:
        return self.locked.model_lock

    def encode_vision(
        self, rgb: np.ndarray, camera_valid: np.ndarray, source_sha256: str
    ) -> DenseVisualFeatures:
        import torch

        images, valid = validate_vision_input(rgb, camera_valid, source_sha256)
        inputs = self.image_processor(
            images=list(images),
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
        )
        pixels = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            output = self.model.vision_model(pixel_values=pixels, return_dict=True)
        tokens = output.last_hidden_state
        grid = round(tokens.shape[1] ** 0.5)
        if grid * grid != tokens.shape[1]:
            raise ValueError("SigLIP2 patch tokens do not form a spatial grid")
        values = tokens.reshape(len(images), grid, grid, -1)
        values = torch.nn.functional.normalize(values, dim=-1).float().cpu().numpy()
        values[~valid] = 0.0
        patch_valid = np.broadcast_to(valid[:, None, None], values.shape[:3]).copy()
        if values.shape[-1] != self.model_lock.output_dimension:
            raise ValueError("SigLIP2 feature dimension differs from its model lock")
        return DenseVisualFeatures(
            values, patch_valid, self.model_lock.lock_sha256, source_sha256
        )

    def encode_language(self, text: str, locale: str) -> SemanticLanguageFeatures:
        import torch

        source_sha256 = language_source_sha256(text, locale)
        normalized = " ".join(text.split())
        inputs = self.tokenizer(
            [normalized],
            padding="max_length",
            truncation=True,
            max_length=self.maximum_text_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            values = self.model.get_text_features(**inputs)
        values = torch.nn.functional.normalize(values, dim=-1).float().cpu().numpy()[0]
        if values.shape != (self.model_lock.output_dimension,):
            raise ValueError("SigLIP2 language dimension differs from its model lock")
        return SemanticLanguageFeatures(
            values, self.model_lock.lock_sha256, source_sha256
        )
