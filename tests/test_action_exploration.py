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

    assert np.all(np.abs(first[:2]) <= (0.18, 0.50))
    assert np.all(np.abs(second[2:14]) <= 0.35)
    assert not np.array_equal(first[:14], second[:14])
    assert explorer.audit()["privileged_fields"] == []


def test_policy_gripper_sample_is_held_for_physical_servo_time() -> None:
    explorer = TemporalActionExplorer(
        TemporalExplorationConfig(
            noise_standard_deviation=0.0,
            action_smoothing=0.0,
            gripper_epsilon=0.0,
            policy_gripper_hold_steps=3,
        ),
        np.random.default_rng(12),
    )
    requested = [
        np.asarray((*([0.0] * 14), value, 1.0 - value))
        for value in (0.1, 0.2, 0.3, 0.9)
    ]

    actions = [explorer.perturb(value) for value in requested]

    assert all(np.array_equal(action[14:], actions[0][14:]) for action in actions[:3])
    assert np.array_equal(actions[3][14:], requested[3][14:])
    assert explorer.audit()["policy_gripper_hold_steps"] == 3


def test_reflection_coupling_is_embodiment_only_and_preserves_independent_mode() -> None:
    coupled = TemporalActionExplorer(
        TemporalExplorationConfig(
            reflection_coupled_probability=1.0,
            paired_gripper_probability=1.0,
        ),
        np.random.default_rng(10),
    )
    random_action = coupled.sample_random()
    perturbed = coupled.perturb(np.zeros(16))
    signs = np.asarray((1, -1, 1, -1, 1, -1))

    assert np.allclose(random_action[2:8], random_action[8:14] * signs)
    assert random_action[14] == random_action[15]
    assert np.allclose(perturbed[2:8], perturbed[8:14] * signs)
    assert coupled.audit()["task_conditioned"] is False

    independent = TemporalActionExplorer(
        TemporalExplorationConfig(
            reflection_coupled_probability=0.0,
            paired_gripper_probability=0.0,
        ),
        np.random.default_rng(10),
    ).sample_random()
    assert not np.allclose(independent[2:8], independent[8:14] * signs)


def test_global_random_bursts_escape_a_local_policy_without_task_inputs() -> None:
    explorer = TemporalActionExplorer(
        TemporalExplorationConfig(
            noise_standard_deviation=0.0,
            action_smoothing=0.0,
            gripper_epsilon=0.0,
            global_random_burst_probability=1.0,
            global_random_burst_steps=3,
        ),
        np.random.default_rng(19),
    )
    policy = np.zeros(16)

    first, second, third, fourth = (
        explorer.perturb(policy) for _ in range(4)
    )

    assert np.array_equal(first, second)
    assert np.array_equal(second, third)
    assert not np.array_equal(third, fourth)
    assert explorer.audit()["observation_fields"] == []
    assert explorer.audit()["task_conditioned"] is False
