"""Frozen Qwen3 multilingual semantic embedding adapter."""

from __future__ import annotations

import numpy as np

from hwr.adapters.foundation.locks import LockedFoundationModel
from hwr.adapters.foundation.runtime import freeze_model, select_device
from hwr.perception.foundation import (
    FoundationModelLock,
    SemanticLanguageFeatures,
    language_source_sha256,
)


class Qwen3LanguageProvider:
    def __init__(
        self,
        locked: LockedFoundationModel,
        *,
        device: str = "auto",
        maximum_tokens: int = 512,
    ) -> None:
        if locked.adapter != "qwen3_embedding" or locked.model_lock.role != "language":
            raise ValueError("Qwen3 adapter requires a language lock")
        if maximum_tokens <= 0:
            raise ValueError("Qwen3 maximum token count must be positive")
        locked.verify()
        from transformers import AutoModel, AutoTokenizer

        self.locked = locked
        self.maximum_tokens = maximum_tokens
        self.device = select_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            locked.local_path, local_files_only=True, padding_side="left"
        )
        model = AutoModel.from_pretrained(
            locked.local_path, local_files_only=True, trust_remote_code=False
        )
        self.model = freeze_model(model, self.device)

    @property
    def model_lock(self) -> FoundationModelLock:
        return self.locked.model_lock

    def encode_language(self, text: str, locale: str) -> SemanticLanguageFeatures:
        import torch

        source_sha256 = language_source_sha256(text, locale)
        normalized = " ".join(text.split())
        inputs = self.tokenizer(
            [f"[{locale}] {normalized}"],
            padding=True,
            truncation=True,
            max_length=self.maximum_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=False, return_dict=True)
        attention = inputs["attention_mask"]
        if bool(torch.all(attention[:, -1])):
            pooled = output.last_hidden_state[:, -1]
        else:
            indices = attention.sum(dim=1) - 1
            pooled = output.last_hidden_state[
                torch.arange(len(indices), device=self.device), indices
            ]
        values = torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy()[0]
        if values.shape != (self.model_lock.output_dimension,):
            raise ValueError("Qwen3 language dimension differs from its model lock")
        return SemanticLanguageFeatures(
            values, self.model_lock.lock_sha256, source_sha256
        )
