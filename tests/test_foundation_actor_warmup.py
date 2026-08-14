from __future__ import annotations

from hwr.train.foundation_actor_warmup import (
    ActorWarmupCriteria,
    assess_actor_warmup,
)


def _window(return_value: float, *, motion_entropy: float = 1.0):
    return {
        "exploration/actor_gradient_norm": 2.0,
        "exploration/value_gradient_norm": 3.0,
        "exploration/return": return_value,
        "exploration/motion_entropy": motion_entropy,
        "exploration/gripper_entropy": 0.0,
    }


def _criteria() -> ActorWarmupCriteria:
    return ActorWarmupCriteria(
        minimum_updates=4,
        maximum_updates=8,
        window_updates=2,
        stable_windows=2,
        maximum_gradient_norm=10.0,
        maximum_return_relative_range=0.1,
        minimum_motion_entropy=0.5,
        minimum_gripper_entropy=-0.5,
    )


def test_actor_warmup_requires_stable_return_and_noncollapsed_entropy() -> None:
    result = assess_actor_warmup(
        (_window(-3.0), _window(-3.1)),
        "exploration",
        _criteria(),
        update_count=4,
    )

    assert result["passed"] is True
    assert result["checks"]["stable_imagined_return"] is True
    assert result["checks"]["motion_entropy_not_collapsed"] is True


def test_actor_warmup_rejects_unstable_or_collapsed_policy() -> None:
    unstable = assess_actor_warmup(
        (_window(-1.0), _window(-3.0)),
        "exploration",
        _criteria(),
        update_count=4,
    )
    collapsed = assess_actor_warmup(
        (_window(-3.0, motion_entropy=0.1),) * 2,
        "exploration",
        _criteria(),
        update_count=4,
    )

    assert unstable["passed"] is False
    assert unstable["checks"]["stable_imagined_return"] is False
    assert collapsed["passed"] is False
    assert collapsed["checks"]["motion_entropy_not_collapsed"] is False
