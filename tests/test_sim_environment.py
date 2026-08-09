from __future__ import annotations

from hwr.scenarios import PickPlaceExpert, debug_pick_place_task
from hwr.sim import Household2DEnv, RobotSpec


def test_reset_is_deterministic() -> None:
    task = debug_pick_place_task()
    environment = Household2DEnv(RobotSpec(), task)

    first = environment.reset(seed=42, task_id=task.task_id)
    second = environment.reset(seed=42, task_id=task.task_id)

    assert first == second


def test_rule_expert_completes_debug_task() -> None:
    robot = RobotSpec()
    task = debug_pick_place_task()
    environment = Household2DEnv(robot, task)
    expert = PickPlaceExpert(robot)
    observation = environment.reset(seed=7, task_id=task.task_id)

    for _ in range(task.max_steps):
        outcome = environment.apply(expert.action(observation))
        observation = outcome.observation
        if outcome.terminated or outcome.truncated:
            break

    assert environment.result() is not None
    assert environment.result().success
    assert environment.result().metrics["collisions"] == 0

