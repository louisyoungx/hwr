"""Runtime contracts shared by simulation and real robot backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from hwr.core.embodied import ActionChunk, DualArmActionFrame, DualArmObservation
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.core.types import ActionFrame, EpisodeEvent, EpisodeResult, ObservationFrame


@dataclass(frozen=True)
class LegalEnvironmentTransform:
    """A simulator-declared, action-preserving data augmentation.

    The identifier selects a transform from the platform-wide augmentation
    registry.  It carries no task name, reward hint, target pose, or action.
    """

    transform_id: str

    def __post_init__(self) -> None:
        normalized = "_".join(self.transform_id.strip().lower().split())
        if not normalized:
            raise ValueError("environment transform identifier is required")
        object.__setattr__(self, "transform_id", normalized)


@dataclass(frozen=True)
class StepOutcome:
    observation: ObservationFrame
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    events: tuple[EpisodeEvent, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)


class RuntimeBackend(Protocol):
    """Canonical dual-arm interface implemented by simulation and future hardware."""

    def reset(self, *, seed: int, task_id: str) -> DualArmObservation: ...

    def observe(self) -> DualArmObservation: ...

    def apply(self, action: DualArmActionFrame) -> "RuntimeStepOutcome": ...

    def result(self) -> EpisodeResult | None: ...

    def close(self) -> None: ...


class SnapshotRuntimeBackend(RuntimeBackend, Protocol):
    """Optional simulation extension used only to vary training resets."""

    def reset(
        self,
        *,
        seed: int,
        task_id: str,
        initial_state: PhysicalStateSnapshot | None = None,
    ) -> DualArmObservation: ...

    def capture_state_snapshot(self) -> PhysicalStateSnapshot: ...

    def legal_environment_transforms(
        self,
    ) -> tuple[LegalEnvironmentTransform, ...]: ...


@dataclass(frozen=True)
class RuntimeStepOutcome:
    observation: DualArmObservation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    events: tuple[EpisodeEvent, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)


class LegacyRuntimeBackend(Protocol):
    """V1 single-arm compatibility interface; prohibited for formal training."""

    def reset(self, *, seed: int, task_id: str) -> ObservationFrame: ...

    def observe(self) -> ObservationFrame: ...

    def apply(self, action: ActionFrame) -> StepOutcome: ...

    def result(self) -> EpisodeResult | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    observation_history: int
    action_horizon: int
    control_hz: float
    arm_dof: int
    required_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if min(self.observation_history, self.action_horizon, self.arm_dof) < 0:
            raise ValueError("policy dimensions must be non-negative")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")


class Policy(Protocol):
    """Canonical deployable policy: raw observation in, 16-D action chunk out."""

    def spec(self) -> PolicySpec: ...

    def reset(self, *, task_id: str, seed: int) -> None: ...

    def infer(self, observations: Sequence[DualArmObservation]) -> ActionChunk: ...

    def close(self) -> None: ...


class LegacyPolicy(Protocol):
    """V1 single-arm policy compatibility; prohibited for formal training."""

    def spec(self) -> PolicySpec: ...

    def reset(self, *, task_id: str, seed: int) -> None: ...

    def infer(self, observations: Sequence[ObservationFrame]) -> tuple[ActionFrame, ...]: ...

    def close(self) -> None: ...
