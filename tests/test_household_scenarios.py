from __future__ import annotations

import pytest

from hwr.eval import evaluate_policy
from hwr.scenarios import ExpertPolicy, PickPlaceExpert, household_task_registry
from hwr.sim import Household2DEnv, RobotSpec


@pytest.mark.parametrize("task_id", sorted(household_task_registry()))
def test_expert_completes_household_scenario(task_id: str) -> None:
    robot = RobotSpec()
    task = household_task_registry()[task_id]
    report = evaluate_policy(
        task,
        lambda: Household2DEnv(robot, task),
        ExpertPolicy(PickPlaceExpert(robot)),
        seeds=range(10),
    )

    assert report.success_rate == 1.0
    assert report.average_collisions == 0.0

