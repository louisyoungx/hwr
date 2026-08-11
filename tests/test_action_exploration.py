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
    policy = np.asarray((*([0.0] * 14), 0.25, 0.75))

    first, second, third, fourth = (
        explorer.perturb(policy) for _ in range(4)
    )

    assert np.array_equal(first, second)
    assert np.array_equal(second, third)
    assert not np.array_equal(third, fourth)
    assert np.array_equal(first[14:], policy[14:])
    assert np.array_equal(fourth[14:], policy[14:])
    assert explorer.audit()["observation_fields"] == []
    assert explorer.audit()["task_conditioned"] is False
    assert explorer.audit()["global_random_bursts"]["grippers"] == "policy-held"


def test_actuator_dwell_holds_random_grippers_without_motion_or_task_inputs() -> None:
    explorer = TemporalActionExplorer(
        TemporalExplorationConfig(
            noise_standard_deviation=1.0,
            actuator_dwell_probability=1.0,
            actuator_dwell_steps=3,
            paired_gripper_probability=1.0,
        ),
        np.random.default_rng(23),
    )

    actions = [explorer.perturb(np.ones(16)) for _ in range(3)]

    assert all(np.array_equal(action, actions[0]) for action in actions)
    assert np.array_equal(actions[0][:14], np.zeros(14))
    assert actions[0][14] == actions[0][15]
    assert actions[0][14] in (0.0, 1.0)
    assert explorer.audit()["actuator_dwell"] == {
        "probability": 1.0,
        "initial_probability": 0.0,
        "hold_steps": 3,
        "closed_probability": 0.5,
        "motion": "zero",
        "grippers": "paired-or-independent-bernoulli-binary",
    }
    assert explorer.audit()["observation_fields"] == []
    assert explorer.audit()["task_conditioned"] is False


def test_initial_dwell_can_close_both_grippers_before_policy_motion() -> None:
    explorer = TemporalActionExplorer(
        TemporalExplorationConfig(
            noise_standard_deviation=0.0,
            action_smoothing=0.0,
            gripper_epsilon=0.0,
            paired_gripper_probability=1.0,
            actuator_dwell_probability=0.0,
            actuator_initial_dwell_probability=1.0,
            actuator_dwell_closed_probability=1.0,
            actuator_dwell_steps=3,
        ),
        np.random.default_rng(31),
    )

    first, second, third, fourth = (
        explorer.perturb(np.ones(16)) for _ in range(4)
    )

    assert all(np.array_equal(action[:14], np.zeros(14)) for action in (first, second, third))
    assert all(np.array_equal(action[14:], np.ones(2)) for action in (first, second, third))
    assert not np.array_equal(fourth[:14], np.zeros(14))
