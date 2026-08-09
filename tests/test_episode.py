from __future__ import annotations

import json

import pytest

from hwr.core.types import (
    ActionFrame,
    EpisodeEvent,
    EpisodeMetadata,
    EpisodeResult,
    ObservationFrame,
    StepRecord,
)
from hwr.data.episode import EpisodeReader, EpisodeRecorder


def _action(timestamp_ns: int) -> ActionFrame:
    return ActionFrame(
        created_at_ns=timestamp_ns,
        valid_from_ns=timestamp_ns,
        valid_until_ns=timestamp_ns + 100,
        source="expert",
        arm_command=(0.1, 0.2),
    )


def test_episode_record_and_replay(tmp_path) -> None:
    metadata = EpisodeMetadata("episode-1", "tidy_table", "robot/v1", "task/v1", "sim", 7, 0)
    recorder = EpisodeRecorder(tmp_path, metadata)
    observation = ObservationFrame(0, 0, "tidy_table", "approach", joint_position=(0.0, 0.0))
    recorder.append_step(StepRecord(observation, _action(0), _action(0)))
    recorder.append_event(EpisodeEvent(0, "reset", "runtime", {"seed": 7}))
    episode_path = recorder.close(EpisodeResult(True, "completed", 100, {"steps": 1}))

    reader = EpisodeReader(episode_path)

    assert reader.metadata == metadata
    assert reader.result.success
    assert list(reader.steps())[0].observation == observation
    assert list(reader.events())[0].details == {"seed": 7}


def test_episode_reader_detects_tampering(tmp_path) -> None:
    metadata = EpisodeMetadata("episode-2", "tidy_table", "robot/v1", "task/v1", "sim", 8, 0)
    recorder = EpisodeRecorder(tmp_path, metadata)
    observation = ObservationFrame(0, 0, "tidy_table", "approach")
    recorder.append_step(StepRecord(observation, _action(0), _action(0)))
    episode_path = recorder.close(EpisodeResult(False, "timeout", 100))
    with (episode_path / "steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tampered": True}) + "\n")

    with pytest.raises(ValueError, match="checksum"):
        EpisodeReader(episode_path)

