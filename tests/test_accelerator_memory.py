from __future__ import annotations

import pytest

from hwr.train import accelerator_memory


def test_release_unused_accelerator_memory_releases_available_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(accelerator_memory.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        accelerator_memory.torch.backends.mps, "is_available", lambda: True
    )
    monkeypatch.setattr(
        accelerator_memory.torch.mps,
        "empty_cache",
        lambda: calls.append("mps"),
    )
    monkeypatch.setattr(accelerator_memory.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        accelerator_memory.torch.cuda,
        "empty_cache",
        lambda: calls.append("cuda"),
    )

    accelerator_memory.release_unused_accelerator_memory()

    assert calls == ["gc", "mps", "cuda"]


def test_release_accelerator_memory_uses_fixed_task_agnostic_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        accelerator_memory,
        "release_unused_accelerator_memory",
        lambda: calls.append(1),
    )

    assert accelerator_memory.release_accelerator_memory_after_step(9) is False
    assert accelerator_memory.release_accelerator_memory_after_step(10) is True
    assert calls == [1]
    with pytest.raises(ValueError, match="must be positive"):
        accelerator_memory.release_accelerator_memory_after_step(0)
