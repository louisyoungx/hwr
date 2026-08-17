from __future__ import annotations

import pytest

from hwr.train.foundation_batch_arms import (
    BATCH_ARMS,
    audit_batch_arm_schedule,
    build_batch_arm_schedule,
)


class _Loader:
    def __init__(self) -> None:
        self.windows = []
        for task, reason, sources in (
            ("a", "timeout", ("a0", "a1")),
            ("b", "collision", ("b0", "b1", "b2")),
            ("c", "timeout", ("c0",)),
        ):
            for source in sources:
                for slot in range(4):
                    self.windows.append(
                        {
                            "task_id": task,
                            "episode_id": f"{source}-{slot}",
                            "transition_start": 0,
                            "transition_stop": 16,
                            "transition_count": 16,
                            "metadata": {
                                "result_reason": (
                                    "severe_collision"
                                    if reason == "collision"
                                    else "formal_household_timeout"
                                ),
                                "visual_supervision": slot < 2,
                                "sequence_reservoir": {
                                    "source_episode_id": source,
                                },
                            },
                        }
                    )

    def __len__(self):
        return len(self.windows)

    def window_metadata(self, index: int):
        return self.windows[index]


def test_batch_arm_schedule_changes_only_second_sample_source() -> None:
    loader = _Loader()

    schedule = build_batch_arm_schedule(loader, seed=17, updates=200)
    audit = audit_batch_arm_schedule(loader, schedule)

    assert audit["passed"]
    assert audit["eligible_window_count"] == 20
    assert audit["excluded_window_count"] == 4
    assert set(audit["arms"]) == set(BATCH_ARMS)
    assert audit["arms"]["duplicate"]["source_episodes_per_batch_min"] == 1
    assert audit["arms"]["duplicate"]["unique_windows_per_batch_max"] == 1
    assert audit["arms"]["same_source"]["source_episodes_per_batch_max"] == 1
    assert audit["arms"]["same_source"]["unique_windows_per_batch_min"] == 2
    assert audit["arms"]["cross_source"]["source_episodes_per_batch_min"] == 2
    assert audit["arms"]["cross_source"]["unique_windows_per_batch_min"] == 2
    assert all(
        loader.window_metadata(step.anchor_index)["metadata"]["visual_supervision"]
        is True
        for update, step in enumerate(schedule.steps)
        if update % 4 == 0
    )


def test_batch_arm_schedule_is_seed_deterministic() -> None:
    loader = _Loader()

    first = build_batch_arm_schedule(loader, seed=29, updates=40)
    second = build_batch_arm_schedule(loader, seed=29, updates=40)

    assert first == second
    assert [step.indices("duplicate") for step in first.steps] == [
        (step.anchor_index, step.anchor_index) for step in first.steps
    ]


def test_batch_arm_schedule_rejects_unknown_arm() -> None:
    schedule = build_batch_arm_schedule(_Loader(), seed=3, updates=2)

    with pytest.raises(ValueError, match="unknown batch arm"):
        schedule.steps[0].indices("unknown")
