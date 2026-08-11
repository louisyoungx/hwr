from __future__ import annotations

import numpy as np

from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.train import (
    LearningFrontierCandidate,
    LearningFrontierConfig,
    LearningSignal,
    TaskAgnosticLearningFrontier,
)


def _snapshot(task_id: str, value: float) -> PhysicalStateSnapshot:
    return PhysicalStateSnapshot(task_id, "test-backend", (value, value + 1.0))


def _candidate(step: int, **signals: float) -> LearningFrontierCandidate:
    return LearningFrontierCandidate(
        _snapshot("task", float(step)),
        (float(step), 1.0),
        LearningSignal(
            signals.get("novelty", 0.1),
            signals.get("td_error", 0.1),
            signals.get("improvement", 0.0),
            signals.get("boundary", 0.0),
            safe=bool(signals.get("safe", 1.0)),
        ),
        source_episode=3,
        source_step=step,
    )


def test_frontier_uses_only_generic_learning_signals() -> None:
    frontier = TaskAgnosticLearningFrontier(
        ("task",), LearningFrontierConfig(candidates_per_episode=2)
    )

    added = frontier.consider_episode(
        "task",
        (
            _candidate(1, novelty=0.9),
            _candidate(2, td_error=1.2),
            _candidate(3, boundary=1.0),
            _candidate(4, safe=0.0, novelty=10.0),
        ),
    )

    assert added == 2
    assert len(frontier.entries["task"]) == 2
    assert all(item.source_step != 4 for item in frontier.entries["task"])
    assert frontier.audit()["distance_thresholds"] is False
    assert frontier.audit()["task_semantic_fields"] == []
    assert frontier.audit()["action_outputs"] is False


def test_frontier_selection_and_state_roundtrip_are_reproducible() -> None:
    config = LearningFrontierConfig(reset_probability=1.0)
    frontier = TaskAgnosticLearningFrontier(("task",), config)
    frontier.consider_episode("task", (_candidate(1), _candidate(2, boundary=1.0)))
    restored = TaskAgnosticLearningFrontier(("task",), config)
    restored.load_state_dict(frontier.state_dict())

    selected = restored.select("task", np.random.default_rng(5))

    assert selected is not None
    assert selected.snapshot.task_id == "task"
    assert restored.state_dict() == frontier.state_dict() | {"reset_count": 1}


def test_frontier_discards_legacy_distance_candidates_on_load() -> None:
    frontier = TaskAgnosticLearningFrontier(("task",))
    legacy = {
        "task_ids": ("task",),
        "entries": {"task": [{"outcome": {"left_reach_distance": 0.05}}]},
    }

    frontier.load_state_dict(legacy)

    assert frontier.entries["task"] == []
    assert frontier.legacy_discarded_entry_count == 1


def test_frontier_novelty_is_geometry_agnostic_cosine_distance() -> None:
    frontier = TaskAgnosticLearningFrontier(("task",))
    frontier.consider_episode("task", (_candidate(1),))

    same = frontier.state_novelty("task", (1.0, 1.0))
    different = frontier.state_novelty("task", (-1.0, 1.0))

    np.testing.assert_allclose(same, 0.0, atol=1.0e-12)
    assert different > same
