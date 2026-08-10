"""Engine-independent state snapshots for automatic initial-state curricula."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalStateSnapshot:
    """Opaque dynamical state owned and interpreted by a runtime adapter.

    The optional vectors preserve the physical continuation of a visited state.
    They are reset state, not policy inputs or labeled future actions.
    """

    task_id: str
    backend_fingerprint: str
    generalized_positions: tuple[float, ...]
    generalized_velocities: tuple[float, ...] = ()
    generalized_accelerations: tuple[float, ...] = ()
    actuator_controls: tuple[float, ...] = ()
    solver_state: tuple[float, ...] = ()
    runtime_state: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id or not self.backend_fingerprint:
            raise ValueError("snapshot task and backend identities are required")
        if not self.generalized_positions or not all(
            math.isfinite(value) for value in self.generalized_positions
        ):
            raise ValueError("snapshot generalized positions must be finite")
        for name, values in (
            ("generalized velocities", self.generalized_velocities),
            ("generalized accelerations", self.generalized_accelerations),
            ("actuator controls", self.actuator_controls),
            ("solver state", self.solver_state),
            ("runtime state", self.runtime_state),
        ):
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"snapshot {name} must be finite")
