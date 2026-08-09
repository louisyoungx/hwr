from __future__ import annotations

from hwr.eval import evaluate_policy
from hwr.scenarios import ExpertPolicy, PickPlaceExpert, debug_pick_place_task
from hwr.sim import Household2DEnv, RobotSpec


def test_closed_loop_evaluator_reports_success() -> None:
    robot = RobotSpec()
    task = debug_pick_place_task()
    report = evaluate_policy(
        task,
        lambda: Household2DEnv(robot, task),
        ExpertPolicy(PickPlaceExpert(robot)),
        seeds=range(3),
    )

    assert report.success_rate == 1.0
    assert report.average_collisions == 0.0
    assert report.reasons == {"completed": 3}

