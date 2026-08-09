from __future__ import annotations

from hwr.eval import (
    MultiObjectStabilityCriterion,
    PlacementSample,
    StabilityConfig,
    StablePlacementCriterion,
    TargetVolume,
)


def test_stable_placement_requires_full_two_second_window() -> None:
    criterion = StablePlacementCriterion(
        TargetVolume(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        StabilityConfig(control_hz=20.0, hold_seconds=2.0),
    )
    values = [
        criterion.update(
            position=(0.5, 0.5, 0.5),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )
        for _ in range(40)
    ]

    assert not any(values[:-1])
    assert values[-1]


def test_motion_or_leaving_target_resets_stability_window() -> None:
    criterion = StablePlacementCriterion(
        TargetVolume(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        StabilityConfig(control_hz=10.0, hold_seconds=2.0),
    )
    for _ in range(19):
        assert not criterion.update(
            position=(0.5, 0.5, 0.5),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )
    assert not criterion.update(
        position=(1.2, 0.5, 0.5),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )
    assert criterion.stable_steps == 0
    assert not criterion.update(
        position=(0.5, 0.5, 0.5),
        linear_velocity=(0.1, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )
    assert criterion.stable_steps == 0


def test_multi_object_gate_requires_same_two_second_window() -> None:
    config = StabilityConfig(control_hz=10, hold_seconds=2.0)
    target = TargetVolume(-0.1, 0.1, -0.1, 0.1, 0.0, 0.2)
    criterion = MultiObjectStabilityCriterion(("cup", "plate"), config)
    still = PlacementSample((0, 0, 0.1), (0, 0, 0), (0, 0, 0), target)
    moving = PlacementSample((0, 0, 0.1), (0.05, 0, 0), (0, 0, 0), target)

    for _ in range(19):
        assert criterion.update({"cup": still, "plate": still}) is False
    assert criterion.update({"cup": still, "plate": moving}) is False
    assert criterion.stable_steps == 0
    for _ in range(19):
        assert criterion.update({"cup": still, "plate": still}) is False
    assert criterion.update({"cup": still, "plate": still}) is True
