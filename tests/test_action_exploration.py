from __future__ import annotations

import numpy as np

from hwr.train import TemporalActionExplorer, TemporalExplorationConfig


def test_gripper_exploration_is_persistent_and_independent_of_observations() -> None:
    explorer = TemporalActionExplorer(
        TemporalExplorationConfig(
            noise_standard_deviation=0.0,
            action_smoothing=0.0,
            gripper_epsilon=1.0,
            gripper_hold_steps=5,
        ),
        np.random.default_rng(4),
    )
    policy = np.full(16, 0.5)

    actions = [explorer.perturb(policy) for _ in range(5)]

    assert all(np.array_equal(action[14:], actions[0][14:]) for action in actions)
    assert set(actions[0][14:]).issubset({0.0, 1.0})
    assert explorer.audit()["observation_fields"] == []
    assert explorer.audit()["action_labels"] is False


def test_continuous_noise_is_correlated_bounded_and_resettable() -> None:
    explorer = TemporalActionExplorer(
        TemporalExplorationConfig(
            noise_standard_deviation=1.0,
            noise_correlation=0.9,
            action_smoothing=0.5,
            gripper_epsilon=0.0,
        ),
        np.random.default_rng(8),
    )
    first = explorer.perturb(np.zeros(16))
    second = explorer.perturb(np.zeros(16))
    explorer.reset()

    assert np.all(np.abs(first[:2]) <= (0.45, 1.0))
    assert np.all(np.abs(second[2:14]) <= 1.2)
    assert not np.array_equal(first[:14], second[:14])
    assert explorer.audit()["privileged_fields"] == []
