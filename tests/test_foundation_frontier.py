from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.train.foundation_frontier import (
    FoundationLearningFrontierController,
    PreparedFoundationFrontierCollection,
)
from hwr.train.foundation_learning_signals import (
    EpisodeLearningEvidence,
    EpisodeLearningSignals,
    EpisodeWindowLearningSignal,
)
from hwr.train.learning_frontier import (
    LearningFrontierConfig,
    PreparedLearningFrontierReset,
)


def _snapshot(step: int) -> PhysicalStateSnapshot:
    return PhysicalStateSnapshot(
        "task",
        "fixture-backend",
        (float(step),),
        runtime_state=(float(step),),
    )


class _SnapshotBackend:
    def __init__(self) -> None:
        self.snapshot = _snapshot(0)

    def reset(self, *, seed: int, task_id: str, initial_state=None):
        del seed
        assert task_id == "task"
        self.snapshot = initial_state or _snapshot(0)
        return SimpleNamespace(task_id=task_id)

    def capture_state_snapshot(self):
        return self.snapshot


def test_foundation_frontier_uses_learned_signals_and_reproduces_reset() -> None:
    controller = FoundationLearningFrontierController(
        ("task",),
        LearningFrontierConfig(reset_probability=1.0),
        seed=5,
        episode_seed_base=7,
    )
    prepared = PreparedFoundationFrontierCollection(
        None,
        [_snapshot(0), _snapshot(1)],
        None,
        PreparedLearningFrontierReset(None, 7, False, False, False),
    )
    controller.remember("episode", 0, prepared)
    episode = SimpleNamespace(
        episode_id="episode",
        task_id="task",
        arrays={
            "reward": np.asarray([0.0, 1.0], np.float32),
            "safety_intervention": np.zeros(2, np.float32),
            "terminated": np.asarray([False, True]),
            "truncated": np.asarray([False, False]),
        },
        metadata={"success": True},
    )
    evidence = EpisodeLearningEvidence(
        EpisodeLearningSignals(0.4, 0.8, 1),
        (EpisodeWindowLearningSignal(0, (1.0, 0.0), 0.8),),
    )

    result = controller.consider(episode, evidence)
    reset = controller.prepare_collection(
        _SnapshotBackend(),
        task_id="task",
        episode_index=1,
        episode_seed=11,
        resets_enabled=True,
    )

    assert result.entries_added == 1
    assert reset.entry is not None
    assert reset.reset is not None and reset.reset.reproduced is True
    assert controller.audit()["task_semantic_fields"] == []
    assert controller.audit()["policy_inputs"] is False


def test_foundation_frontier_state_roundtrip_includes_independent_rng() -> None:
    config = LearningFrontierConfig(reset_probability=0.2)
    controller = FoundationLearningFrontierController(
        ("task",), config, seed=5, episode_seed_base=7
    )
    restored = FoundationLearningFrontierController(
        ("task",), config, seed=999, episode_seed_base=7
    )

    restored.load_state_dict(controller.state_dict())

    assert restored.state_dict() == controller.state_dict()
