"""Shared tensor helpers confined to third-party foundation adapters."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def validate_vision_input(
    rgb: np.ndarray, camera_valid: np.ndarray, source_sha256: str
) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(rgb, dtype=np.float32)
    valid = np.asarray(camera_valid, dtype=np.bool_)
    if images.ndim != 4 or images.shape[-1] != 3 or min(images.shape) <= 0:
        raise ValueError("foundation vision input must be camera-height-width-RGB")
    if valid.shape != (images.shape[0],):
        raise ValueError("foundation camera validity shape is invalid")
    if not np.isfinite(images).all() or np.any(images < 0.0) or np.any(images > 1.0):
        raise ValueError("foundation RGB values must be finite and normalized")
    if len(source_sha256) != 64:
        raise ValueError("foundation vision source requires a SHA-256 identity")
    return images, valid


def select_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS foundation inference was requested but is unavailable")
    return device


def freeze_model(model: Any, device: Any) -> Any:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def tensor_sha256(value: np.ndarray, valid: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(value, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(valid, dtype=np.bool_).tobytes())
    return digest.hexdigest()
