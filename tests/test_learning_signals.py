from __future__ import annotations

import numpy as np

from hwr.train.learning_signals import (
    failure_boundary_step,
    reward_improvement_speeds,
)


def test_reward_improvement_is_local_instead_of_an_episode_wide_label() -> None:
    values = reward_improvement_speeds((-1.0, -0.5, -0.5, -0.8), smoothing=0.5)

    np.testing.assert_allclose(values, (0.0, 0.5, 0.25, -0.175))
    assert len(set(values)) > 1


def test_only_environment_termination_creates_a_failure_boundary() -> None:
    costs = (0.0, 0.0, 1.0)

    assert failure_boundary_step(costs, terminated_failure=True) == 1
    assert failure_boundary_step(costs, terminated_failure=False) == -1
