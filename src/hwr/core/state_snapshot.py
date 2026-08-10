"""Engine-independent state snapshots for automatic initial-state curricula."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalStateSnapshot:
    """Opaque generalized positions owned and interpreted by a runtime adapter."""

    task_id: str
    backend_fingerprint: str
    generalized_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.backend_fingerprint:
            raise ValueError("snapshot task and backend identities are required")
        if not self.generalized_positions or not all(
            math.isfinite(value) for value in self.generalized_positions
        ):
            raise ValueError("snapshot generalized positions must be finite")
