from __future__ import annotations

from hwr.train import TaskPartitionedGoalReplayBuffer
from tests.test_goal_replay import _episode


def test_task_partitioned_replay_balances_scenes_despite_unequal_history() -> None:
    replay = TaskPartitionedGoalReplayBuffer(96, ("basket", "drawer", "tray"), seed=9)
    replay.add_episode("basket", _episode())
    for _ in range(4):
        replay.add_episode("drawer", _episode(legal_transforms=()))
    batch = replay.sample(12, failure_fraction=0.0, discovery_fraction=0.0)

    assert replay.task_sizes() == {"basket": 16, "drawer": 32, "tray": 0}
    assert batch.rewards.shape == (12,)
    assert replay.episode_count == 5
    assert replay.hindsight_count == 20


def test_task_partitioned_replay_round_trips_all_partition_state() -> None:
    replay = TaskPartitionedGoalReplayBuffer(96, ("basket", "drawer", "tray"), seed=4)
    replay.add_episode("tray", _episode())
    restored = TaskPartitionedGoalReplayBuffer(
        96, ("basket", "drawer", "tray"), seed=100
    )

    restored.load_state_dict(replay.state_dict())

    assert restored.task_sizes() == replay.task_sizes()
    assert restored.augmentation_count == replay.augmentation_count
    assert restored.discovery_size == replay.discovery_size
    assert restored.progress_size == replay.progress_size
    assert restored.safety_size == replay.safety_size


def test_task_partitioned_replay_can_shrink_during_audited_fork() -> None:
    replay = TaskPartitionedGoalReplayBuffer(96, ("basket", "drawer", "tray"), seed=5)
    for _ in range(3):
        replay.add_episode("tray", _episode())
    restored = TaskPartitionedGoalReplayBuffer(
        48, ("basket", "drawer", "tray"), seed=6
    )

    restored.load_state_dict(replay.state_dict())

    assert restored.task_sizes()["tray"] == 16
    assert restored.size == 16


def test_task_partitioned_replay_discards_only_requested_task() -> None:
    replay = TaskPartitionedGoalReplayBuffer(96, ("basket", "drawer", "tray"), seed=5)
    replay.add_episode("basket", _episode())
    replay.add_episode("tray", _episode())

    discarded = replay.discard_tasks(("tray",))

    assert discarded["tray"]["size"] == 16
    assert discarded["tray"]["episode_count"] == 1
    assert replay.task_sizes() == {"basket": 16, "drawer": 0, "tray": 0}
