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


def _complete_snapshot(task_id: str, value: float) -> PhysicalStateSnapshot:
    return PhysicalStateSnapshot(
        task_id,
        "test-backend/v1",
        (value, value + 1.0),
        runtime_state=(value + 2.0,),
    )


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


def test_frontier_rejects_unsupported_or_moving_instantaneous_near_states() -> None:
    frontier = OutcomeFrontierCurriculum(("tray",))

    assert not frontier.consider(
        "tray",
        _snapshot("tray", 0.0),
        FrontierOutcome(
            0.04,
            0.05,
            False,
            False,
            support_contact=False,
        ),
        source_episode=0,
        source_step=0,
    )
    assert not frontier.consider(
        "tray",
        _snapshot("tray", 1.0),
        FrontierOutcome(
            0.04,
            0.05,
            False,
            False,
            support_contact=True,
            payload_linear_speed=0.051,
        ),
        source_episode=0,
        source_step=1,
    )
    assert frontier.consider(
        "tray",
        _snapshot("tray", 2.0),
        FrontierOutcome(
            0.04,
            0.05,
            True,
            False,
            support_contact=False,
            payload_linear_speed=0.01,
            payload_angular_speed=0.02,
        ),
        source_episode=0,
        source_step=2,
        contact_stability_steps=40,
    )

    audit = frontier.audit()
    assert audit["physical_stability_filter"] == {
        "requires_support_or_arm_contact": True,
        "maximum_payload_linear_speed": 0.05,
        "maximum_payload_angular_speed": 0.15,
    }


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
            contact_stability_steps=40,
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
            contact_stability_steps=40,
        )

    ranked = frontier.entries["tray"]
    selected_sources = [
        frontier.select("tray", np.random.default_rng(seed)).source_episode
        for seed in range(512)
    ]

    assert ranked[0].source_episode == 1
    assert ranked[-1].source_episode == 0
    assert set(selected_sources) == {0, 1, 2, 3}
    assert selected_sources.count(1) > selected_sources.count(3)
    assert selected_sources.count(2) > selected_sources.count(0)


def test_frontier_contact_cannot_outrank_better_bilateral_reach() -> None:
    config = FrontierCurriculumConfig(
        capacity_per_task=8, reset_probability=1.0
    )
    frontier = OutcomeFrontierCurriculum(("tray",), config)
    candidates = (
        FrontierOutcome(0.083, 0.123, True, False),
        FrontierOutcome(0.052, 0.109, False, False),
    )
    for index, outcome in enumerate(candidates):
        assert frontier.consider(
            "tray",
            _snapshot("tray", float(index)),
            outcome,
            source_episode=index,
            source_step=index,
            contact_stability_steps=40,
        )

    audit = frontier.audit()

    assert frontier.entries["tray"][0].source_episode == 1
    assert audit["contact_affects_score"] is False


def test_frontier_selection_replays_each_discovered_contact_signature() -> None:
    frontier = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(capacity_per_task=8, reset_probability=1.0),
    )
    candidates = (
        FrontierOutcome(0.03, 0.14, True, False),
        FrontierOutcome(0.04, 0.15, True, False),
        FrontierOutcome(0.28, 0.04, False, True),
        FrontierOutcome(0.30, 0.03, False, True),
    )
    for index, outcome in enumerate(candidates):
        assert frontier.consider(
            "tray",
            _snapshot("tray", float(index)),
            outcome,
            source_episode=index,
            source_step=index,
            contact_stability_steps=40,
        )

    selected = [
        frontier.select("tray", np.random.default_rng(seed)).signature
        for seed in range(256)
    ]
    selected_signatures = set(selected)

    assert selected_signatures == {1, 2}
    assert selected.count(1) > selected.count(2)
    assert frontier.audit()["selection"] == (
        "quality_weighted_signature_and_source_with_diversity_floor"
    )
    assert frontier.audit()["signature_uniform_fraction"] == 0.2


def test_frontier_limits_duplicate_frames_from_one_episode_and_prunes_legacy() -> None:
    config = FrontierCurriculumConfig(
        capacity_per_task=8,
        reset_probability=1.0,
        maximum_entries_per_source_signature=2,
    )
    frontier = OutcomeFrontierCurriculum(("tray",), config)
    for index in range(4):
        assert frontier.consider(
            "tray",
            _snapshot("tray", float(index)),
            FrontierOutcome(0.04, 0.08 - index * 0.005, True, False),
            source_episode=7,
            source_step=index,
            contact_stability_steps=40,
        )
    assert sum(
        item.source_episode == 7 for item in frontier.entries["tray"]
    ) == 2

    legacy = frontier.state_dict()
    legacy["config"].pop("maximum_entries_per_source_signature")
    legacy["entries"]["tray"] = legacy["entries"]["tray"] * 2
    restored = OutcomeFrontierCurriculum(("tray",), config)
    restored.load_state_dict(legacy)

    assert len(restored.entries["tray"]) == 2
    assert restored.audit()["maximum_entries_per_source_signature"] == 2

    selection = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(capacity_per_task=16, reset_probability=1.0),
    )
    for source_episode, distance in ((7, 0.07), (8, 0.14)):
        for index in range(2):
            assert selection.consider(
                "tray",
                _snapshot("tray", float(source_episode * 10 + index)),
                FrontierOutcome(0.04, distance + index * 0.005, True, False),
                source_episode=source_episode,
                source_step=index,
                contact_stability_steps=40,
            )
    selected_sources = [
        selection.select("tray", np.random.default_rng(seed)).source_episode
        for seed in range(256)
    ]
    assert set(selected_sources) == {7, 8}
    assert selected_sources.count(7) > selected_sources.count(8)


def test_frontier_requires_two_seconds_of_contact_before_snapshotting() -> None:
    frontier = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(minimum_contact_stability_steps=40),
    )
    outcome = FrontierOutcome(0.04, 0.05, True, True)

    assert not frontier.consider(
        "tray",
        _snapshot("tray", 0.0),
        outcome,
        source_episode=3,
        source_step=38,
        contact_stability_steps=39,
    )
    assert frontier.consider(
        "tray",
        _snapshot("tray", 1.0),
        outcome,
        source_episode=3,
        source_step=39,
        contact_stability_steps=40,
    )
    assert frontier.entries["tray"][0].contact_stability_steps == 40
    assert frontier.audit()["minimum_contact_stability_steps"] == 40

    legacy = frontier.state_dict()
    legacy["config"].pop("minimum_contact_stability_steps")
    legacy["entries"]["tray"][0].pop("contact_stability_steps")
    restored = OutcomeFrontierCurriculum(("tray",), frontier.config)
    restored.load_state_dict(legacy)
    assert restored.entries["tray"] == []


def test_complete_snapshot_replaces_and_suppresses_stronger_legacy_state() -> None:
    frontier = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(capacity_per_task=4, reset_probability=1.0),
    )
    assert frontier.consider(
        "tray",
        _snapshot("tray", 1.0),
        FrontierOutcome(0.02, 0.03, True, False),
        source_episode=1,
        source_step=10,
        contact_stability_steps=40,
    )
    assert frontier.consider(
        "tray",
        _complete_snapshot("tray", 2.0),
        FrontierOutcome(0.04, 0.05, True, False),
        source_episode=2,
        source_step=20,
        contact_stability_steps=40,
    )

    selected = frontier.select("tray", np.random.default_rng(7))

    assert selected is not None
    assert selected.source_episode == 2
    assert selected.snapshot.runtime_state
    assert all(item.snapshot.runtime_state for item in frontier.entries["tray"])
    assert frontier.audit()["snapshot_migration"] == (
        "complete_state_replaces_and_suppresses_legacy_within_signature"
    )
