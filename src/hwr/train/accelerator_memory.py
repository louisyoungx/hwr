"""Task-independent accelerator cache maintenance for long training runs."""

from __future__ import annotations

import gc

import torch


ACCELERATOR_CACHE_RELEASE_INTERVAL = 10


def release_unused_accelerator_memory() -> None:
    """Return allocator caches without touching live model or optimizer tensors."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_accelerator_memory_after_step(step: int) -> bool:
    """Release caches at a fixed, task-agnostic optimizer-step cadence."""
    if step <= 0:
        raise ValueError("accelerator memory step must be positive")
    if step % ACCELERATOR_CACHE_RELEASE_INTERVAL:
        return False
    release_unused_accelerator_memory()
    return True
