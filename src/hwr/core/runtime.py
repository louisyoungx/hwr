"""Runtime contracts shared by simulation and real robot backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from hwr.core.types import ActionFrame, EpisodeEvent, EpisodeResult, ObservationFrame


@dataclass(frozen=True)
class StepOutcome:
    observation: ObservationFrame
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    events: tuple[EpisodeEvent, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)


class RuntimeBackend(Protocol):
    """Common environment interface implemented by simulation and hardware."""

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
    def spec(self) -> PolicySpec: ...

    def reset(self, *, task_id: str, seed: int) -> None: ...

    def infer(self, observations: Sequence[ObservationFrame]) -> tuple[ActionFrame, ...]: ...

    def close(self) -> None: ...

