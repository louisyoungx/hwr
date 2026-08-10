from __future__ import annotations

import numpy as np

from hwr.core import PhysicalStateSnapshot
from hwr.train import (
    FrontierCurriculumConfig,
    FrontierOutcome,
    OutcomeFrontierCurriculum,
)


def _snapshot(task_id: str, value: float) -> PhysicalStateSnapshot:
    return PhysicalStateSnapshot(task_id, "test-backend/v1", (value, value + 1.0))


def test_frontier_rejects_ordinary_and_unsafe_states_without_action_outputs() -> None:
    frontier = OutcomeFrontierCurriculum(("tray",))

    assert not frontier.consider(
        "tray",
        _snapshot("tray", 0.0),
        FrontierOutcome(0.10, 0.11, False, False),
        source_episode=0,
        source_step=1,
    )
    assert not frontier.consider(
        "tray",
        _snapshot("tray", 1.0),
        FrontierOutcome(0.02, 0.03, False, False, severe_collision=True),
        source_episode=0,
        source_step=2,
    )
    assert frontier.consider(
        "tray",
        _snapshot("tray", 2.0),
        FrontierOutcome(0.09, 0.099, False, False),
        source_episode=1,
        source_step=3,
    )
    assert not frontier.consider(
        "tray",
        _snapshot("tray", 3.0),
        FrontierOutcome(0.09, 0.101, False, False),
        source_episode=1,
        source_step=4,
    )
    audit = frontier.audit()
    assert audit["action_outputs"] is False
    assert audit["actor_input_fields"] == []
    assert audit["task_stages"] is False


def test_frontier_preserves_side_diversity_and_round_trips_checkpoint() -> None:
    config = FrontierCurriculumConfig(
        capacity_per_task=4, reset_probability=1.0
    )
    frontier = OutcomeFrontierCurriculum(("tray",), config)
    outcomes = (
        FrontierOutcome(0.03, 0.10, True, False),
        FrontierOutcome(0.10, 0.03, False, True),
        FrontierOutcome(0.03, 0.04, False, False),
        FrontierOutcome(0.02, 0.02, True, True),
    )
    for index, outcome in enumerate(outcomes):
        assert frontier.consider(
            "tray",
            _snapshot("tray", float(index)),
            outcome,
            source_episode=index,
            source_step=index + 10,
        )

    restored = OutcomeFrontierCurriculum(("tray",), config)
    restored.load_state_dict(frontier.state_dict())
    signatures = {entry.signature for entry in restored.entries["tray"]}
    selected = restored.select("tray", np.random.default_rng(7))

    assert signatures == {1, 2, 3, 7}
    assert selected is not None
    assert selected.snapshot.task_id == "tray"
    assert restored.reset_count == 1


def test_frontier_prioritizes_worst_side_progress_over_far_single_contact() -> None:
    config = FrontierCurriculumConfig(
        capacity_per_task=8, reset_probability=1.0
    )
    frontier = OutcomeFrontierCurriculum(("tray",), config)
    candidates = (
        FrontierOutcome(0.03, 0.40, True, False),
        FrontierOutcome(0.05, 0.09, False, False),
        FrontierOutcome(0.07, 0.10, True, False),
        FrontierOutcome(0.06, 0.11, False, False),
    )
    for index, outcome in enumerate(candidates):
        assert frontier.consider(
            "tray",
            _snapshot("tray", float(index)),
            outcome,
            source_episode=index,
            source_step=index,
        )

    ranked = frontier.entries["tray"]
    selected_sources = {
        frontier.select("tray", np.random.default_rng(seed)).source_episode
        for seed in range(20)
    }

    assert ranked[0].source_episode == 2
    assert ranked[-1].source_episode == 0
    assert selected_sources <= {1, 2}
