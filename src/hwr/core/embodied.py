"""Vendor-neutral contracts for language-conditioned dual-arm control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


DUAL_ARM_ACTION_DIM = 16


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class NaturalLanguageInstruction:
    """Unparsed instruction supplied by a person or a dataset."""

    text: str
    locale: str = "zh-CN"

    def __post_init__(self) -> None:
        text = " ".join(self.text.split())
        if not text:
            raise ValueError("instruction text is required")
        if not self.locale:
            raise ValueError("instruction locale is required")
        object.__setattr__(self, "text", text)


@dataclass(frozen=True)
class FrozenLanguageEmbedding:
    """Output of a locally versioned, frozen language encoder."""

    encoder_id: str
    weights_sha256: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.encoder_id:
            raise ValueError("language encoder id is required")
        digest = self.weights_sha256.lower()
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("language encoder weights require a SHA-256 digest")
        object.__setattr__(self, "weights_sha256", digest)
        object.__setattr__(self, "values", _finite(self.values, "language embedding"))


@dataclass(frozen=True)
class DualArmAction:
    """One simultaneous mobile-base and dual-arm command."""

    base_linear: float
    base_angular: float
    left_arm: tuple[float, ...]
    right_arm: tuple[float, ...]
    left_gripper: float
    right_gripper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_linear) or not math.isfinite(self.base_angular):
            raise ValueError("base commands must be finite")
        left = _finite(self.left_arm, "left arm command")
        right = _finite(self.right_arm, "right arm command")
        if len(left) != 6 or len(right) != 6:
            raise ValueError("dual-arm action requires six commands per arm")
        grippers = (float(self.left_gripper), float(self.right_gripper))
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in grippers):
            raise ValueError("gripper targets must be finite values between zero and one")
        object.__setattr__(self, "left_arm", left)
        object.__setattr__(self, "right_arm", right)
        object.__setattr__(self, "left_gripper", grippers[0])
        object.__setattr__(self, "right_gripper", grippers[1])

    def vector(self) -> tuple[float, ...]:
        return (
            self.base_linear,
            self.base_angular,
            *self.left_arm,
            *self.right_arm,
            self.left_gripper,
            self.right_gripper,
        )

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "DualArmAction":
        vector = tuple(float(value) for value in values)
        if len(vector) != DUAL_ARM_ACTION_DIM:
            raise ValueError(f"dual-arm action vector must have {DUAL_ARM_ACTION_DIM} values")
        return cls(
            base_linear=vector[0],
            base_angular=vector[1],
            left_arm=vector[2:8],
            right_arm=vector[8:14],
            left_gripper=vector[14],
            right_gripper=vector[15],
        )


@dataclass(frozen=True)
class ActionChunk:
    """A policy prediction containing several future executable actions."""

    actions: tuple[DualArmAction, ...]
    valid_steps: int

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("action chunk cannot be empty")
        if not 1 <= self.valid_steps <= len(actions):
            raise ValueError("valid action steps must fit inside the chunk")
        object.__setattr__(self, "actions", actions)

    def vectors(self) -> tuple[tuple[float, ...], ...]:
        return tuple(action.vector() for action in self.actions)
