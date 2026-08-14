"""Bounded, metric-gated warmup for newly admitted foundation Actors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from hwr.train.foundation_metrics import mean_metrics
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig


ACTOR_WARMUP_SCHEMA = "hwr.foundation-actor-warmup/v1"


@dataclass(frozen=True)
class ActorWarmupCriteria:
    minimum_updates: int
    maximum_updates: int
    window_updates: int
    stable_windows: int
    maximum_gradient_norm: float
    maximum_return_relative_range: float
    minimum_motion_entropy: float
    minimum_gripper_entropy: float

    def __post_init__(self) -> None:
        if min(
            self.minimum_updates,
            self.maximum_updates,
            self.window_updates,
            self.stable_windows,
        ) <= 0:
            raise ValueError("Actor warmup dimensions must be positive")
        if not self.window_updates <= self.minimum_updates <= self.maximum_updates:
            raise ValueError("Actor warmup update bounds are invalid")
        if self.minimum_updates % self.window_updates:
            raise ValueError("Actor warmup minimum must contain complete windows")
        if self.maximum_updates % self.window_updates:
            raise ValueError("Actor warmup maximum must contain complete windows")
        if self.stable_windows > self.minimum_updates // self.window_updates:
            raise ValueError("Actor warmup lacks minimum stability windows")
        if self.maximum_gradient_norm <= 0.0:
            raise ValueError("Actor warmup gradient limit must be positive")
        if not 0.0 <= self.maximum_return_relative_range <= 1.0:
            raise ValueError("Actor warmup return range is invalid")

    @classmethod
    def from_config(
        cls, config: FoundationOnlineTrainingConfig
    ) -> ActorWarmupCriteria:
        return cls(
            config.actor_warmup_minimum_updates,
            config.actor_warmup_maximum_updates,
            config.actor_warmup_window_updates,
            config.actor_warmup_stable_windows,
            config.actor_warmup_maximum_gradient_norm,
            config.actor_warmup_maximum_return_relative_range,
            config.actor_warmup_minimum_motion_entropy,
            config.actor_warmup_minimum_gripper_entropy,
        )


@dataclass(frozen=True)
class FoundationActorWarmupResult:
    update_count: int
    metrics: dict[str, float]
    assessment: dict[str, object]


def assess_actor_warmup(
    windows: Sequence[Mapping[str, float]],
    actor_kind: str,
    criteria: ActorWarmupCriteria,
    *,
    update_count: int,
) -> dict[str, object]:
    if actor_kind not in {"exploration", "task"}:
        raise ValueError("unknown foundation Actor warmup kind")
    selected = tuple(windows[-criteria.stable_windows :])
    prefix = "exploration" if actor_kind == "exploration" else "imagination"
    names = {
        "actor_gradient_norm": f"{prefix}/actor_gradient_norm",
        "value_gradient_norm": f"{prefix}/value_gradient_norm",
        "return": f"{prefix}/{'return' if actor_kind == 'exploration' else 'imagined_return'}",
        "motion_entropy": f"{prefix}/motion_entropy",
        "gripper_entropy": f"{prefix}/gripper_entropy",
    }
    complete = len(selected) == criteria.stable_windows and all(
        all(name in window for name in names.values()) for window in selected
    )
    values = {
        name: np.asarray([window.get(key, np.nan) for window in selected], np.float64)
        for name, key in names.items()
    }
    finite = complete and all(np.isfinite(value).all() for value in values.values())
    relative_range = _relative_range(values["return"]) if finite else float("inf")
    checks = {
        "minimum_updates": update_count >= criteria.minimum_updates,
        "complete_stability_windows": complete,
        "finite_metrics": finite,
        "bounded_actor_gradient": finite
        and float(values["actor_gradient_norm"].max())
        <= criteria.maximum_gradient_norm,
        "bounded_value_gradient": finite
        and float(values["value_gradient_norm"].max())
        <= criteria.maximum_gradient_norm,
        "stable_imagined_return": finite
        and relative_range <= criteria.maximum_return_relative_range,
        "motion_entropy_not_collapsed": finite
        and float(values["motion_entropy"].min())
        >= criteria.minimum_motion_entropy,
        "gripper_entropy_not_collapsed": finite
        and float(values["gripper_entropy"].min())
        >= criteria.minimum_gripper_entropy,
    }
    return {
        "schema_version": ACTOR_WARMUP_SCHEMA,
        "actor_kind": actor_kind,
        "passed": all(checks.values()),
        "update_count": update_count,
        "criteria": asdict(criteria),
        "checks": checks,
        "return_relative_range": relative_range,
        "windows": [dict(value) for value in selected],
        "task_semantic_fields": [],
    }


def _relative_range(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).all():
        return float("inf")
    scale = max(abs(float(values.mean())), 1.0)
    return float(np.ptp(values) / scale)
