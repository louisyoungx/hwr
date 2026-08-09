"""Clock abstractions used by both simulation and real robot runtimes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    """Monotonic nanosecond clock."""

    def now_ns(self) -> int:
        """Return the current monotonic time in nanoseconds."""


class SystemClock:
    """Clock backed by the operating system monotonic clock."""

    def now_ns(self) -> int:
        return time.monotonic_ns()


@dataclass
class DeterministicClock:
    """Manually advanced clock for simulation and deterministic tests."""

    current_ns: int = 0

    def now_ns(self) -> int:
        return self.current_ns

    def advance_ns(self, delta_ns: int) -> int:
        if delta_ns < 0:
            raise ValueError("clock cannot move backwards")
        self.current_ns += delta_ns
        return self.current_ns

    def advance_seconds(self, seconds: float) -> int:
        if seconds < 0:
            raise ValueError("clock cannot move backwards")
        return self.advance_ns(round(seconds * 1_000_000_000))

