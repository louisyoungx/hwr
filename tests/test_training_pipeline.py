from __future__ import annotations

import numpy as np

from hwr.data import BehaviorDataset, generate_expert_dataset
from hwr.policy import NeuralPolicy
from hwr.scenarios import PickPlaceExpert, debug_pick_place_task
from hwr.sim import Household2DEnv, RobotSpec
from hwr.train import TrainingConfig, load_policy, save_training_result, train_behavior_policy


def test_dataset_training_and_model_registry(tmp_path) -> None:
    robot = RobotSpec()
    task = debug_pick_place_task()
    dataset_path = generate_expert_dataset(
        tmp_path / "datasets",
        "debug",
        task,
        lambda: Household2DEnv(robot, task),
        PickPlaceExpert(robot),
        seeds=range(4),
    )
    dataset = BehaviorDataset.load(dataset_path)
    result = train_behavior_policy(
        dataset,
        TrainingConfig(epochs=3, batch_size=128, hidden_dims=(32, 32), device="cpu"),
    )
    model_path = save_training_result(
        tmp_path / "models",
        "debug-policy",
        "v1",
        result,
        dataset_manifest=dataset.manifest,
        control_hz=robot.control_hz,
    )
    policy = load_policy(model_path)

    observation = Household2DEnv(robot, task).reset(seed=10, task_id=task.task_id)
    action = policy.infer((observation,))[0]

    assert len(dataset) > 100
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert action.policy_version == "debug-policy:v1"
    assert np.isfinite(action.arm_command).all()

