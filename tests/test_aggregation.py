from __future__ import annotations

from hwr.data import BehaviorDataset, aggregate_policy_dataset, generate_expert_dataset
from hwr.scenarios import ExpertPolicy, PickPlaceExpert, debug_pick_place_task
from hwr.sim import Household2DEnv, RobotSpec


def test_aggregation_adds_learner_visited_samples(tmp_path) -> None:
    robot = RobotSpec()
    task = debug_pick_place_task()
    factory = lambda: Household2DEnv(robot, task)
    expert = PickPlaceExpert(robot)
    base_path = generate_expert_dataset(
        tmp_path,
        "base",
        task,
        factory,
        expert,
        seeds=range(2),
    )
    base = BehaviorDataset.load(base_path)
    aggregated_path = aggregate_policy_dataset(
        tmp_path,
        "aggregated",
        base,
        task,
        factory,
        expert,
        ExpertPolicy(expert),
        seeds=(20,),
        expert_action_probability=0.0,
    )
    aggregated = BehaviorDataset.load(aggregated_path)

    assert len(aggregated) > len(base)
    assert aggregated.manifest["episode_count"] == base.manifest["episode_count"] + 1
    assert aggregated.manifest["metadata"]["parent_checksum"] == base.manifest["checksum"]

