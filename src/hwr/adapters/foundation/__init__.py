"""Offline third-party foundation model adapters."""

from hwr.adapters.foundation.dinov2 import Dinov2DenseVisionProvider
from hwr.adapters.foundation.dinov3 import Dinov3ConvNextDenseVisionProvider
from hwr.adapters.foundation.locks import LockedFoundationModel, load_foundation_model_locks
from hwr.adapters.foundation.qwen3 import Qwen3LanguageProvider
from hwr.adapters.foundation.siglip2 import Siglip2VisionLanguageProvider

__all__ = [
    "Dinov2DenseVisionProvider",
    "Dinov3ConvNextDenseVisionProvider",
    "LockedFoundationModel",
    "Qwen3LanguageProvider",
    "Siglip2VisionLanguageProvider",
    "load_foundation_model_locks",
]
