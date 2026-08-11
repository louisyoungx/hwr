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
    assert not frontier.consider(
        "tray",
        _complete_snapshot("tray", 1.5),
        FrontierOutcome(
            0.04,
            0.05,
            True,
            True,
            target_distance=1.0,
            initial_target_distance=0.8,
            task_progress_observed=True,
        ),
        source_episode=1,
        source_step=41,
        contact_stability_steps=40,
    )

    audit = frontier.audit()
    assert audit["physical_stability_filter"] == {
        "requires_support_or_arm_contact": True,
        "maximum_candidate_target_distance_meters": 1.1,
        "maximum_target_regression_meters": 0.15,
        "maximum_payload_linear_speed": 0.05,
        "maximum_payload_angular_speed": 0.15,
    }


def test_frontier_accepts_start_support_without_claiming_target_support() -> None:
    frontier = OutcomeFrontierCurriculum(("tray",))
    metrics = {
        "left_reach_distance": 0.08,
        "right_reach_distance": 0.09,
        "left_contact": 0.0,
        "right_contact": 0.0,
        "severe_collisions": 0.0,
        "support_contact": 0.0,
        "physical_support_contact": 1.0,
        "payload_linear_speed": 0.0,
        "payload_angular_speed": 0.0,
        "target_distance": 0.8,
        "initial_target_distance": 0.8,
        "articulation_position": 0.0,
    }

    outcome = frontier.outcome_from_metrics(metrics)

    assert outcome.support_contact is True
    assert frontier.qualifies(outcome)


def test_frontier_audits_each_physical_qualification_failure() -> None:
    frontier = OutcomeFrontierCurriculum(("tray",))
    rejected = FrontierOutcome(
        0.11,
        0.12,
        False,
        False,
        support_contact=False,
        payload_linear_speed=0.06,
        payload_angular_speed=0.16,
        target_distance=1.20,
        initial_target_distance=0.80,
        task_progress_observed=True,
    )
    qualified = FrontierOutcome(0.08, 0.09, False, False)

    assert not frontier.observe("tray", rejected)
    assert frontier.observe("tray", qualified)

    expected = {
        "observed": 2,
        "qualified": 1,
        "severe_collision": 0,
        "unsupported": 1,
        "payload_linear_speed": 1,
        "payload_angular_speed": 1,
        "target_beyond_workspace": 1,
        "target_regression": 1,
        "not_near": 1,
    }
    assert frontier.audit()["qualification_counts"]["tray"] == expected

    restored = OutcomeFrontierCurriculum(("tray",))
    restored.load_state_dict(frontier.state_dict())
    assert restored.audit()["qualification_counts"]["tray"] == expected


def test_frontier_discards_only_changed_task_state() -> None:
    frontier = OutcomeFrontierCurriculum(("basket", "tray"))
    for task_id in frontier.task_ids:
        outcome = FrontierOutcome(0.08, 0.09, False, False)
        assert frontier.observe(task_id, outcome)
        assert frontier.consider(
            task_id,
            _snapshot(task_id, 1.0),
            outcome,
            source_episode=1,
            source_step=2,
        )

    discarded = frontier.discard_tasks(("tray",))

    assert discarded["tray"]["entry_count"] == 1
    assert frontier.entries["tray"] == []
    assert len(frontier.entries["basket"]) == 1
    assert frontier.audit()["qualification_counts"]["tray"]["observed"] == 0


def test_frontier_rejects_observed_states_that_regress_far_from_task_target() -> None:
    frontier = OutcomeFrontierCurriculum(("tray",))

    assert not frontier.consider(
        "tray",
        _complete_snapshot("tray", 1.0),
        FrontierOutcome(
            0.04,
            0.05,
            True,
            True,
            target_distance=1.11,
            initial_target_distance=0.80,
            task_progress_observed=True,
        ),
        source_episode=1,
        source_step=40,
        contact_stability_steps=40,
    )
    assert frontier.consider(
        "tray",
        _complete_snapshot("tray", 2.0),
        FrontierOutcome(0.04, 0.05, True, True),
        source_episode=2,
        source_step=40,
        contact_stability_steps=40,
    )

    checkpoint = frontier.state_dict()
    checkpoint["entries"]["tray"][0]["outcome"].update(
        target_distance=1.60,
        task_progress_observed=True,
    )
    restored = OutcomeFrontierCurriculum(("tray",))
    restored.load_state_dict(checkpoint)
    assert restored.entries["tray"] == []


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


def test_frontier_reset_probability_can_change_on_audited_fork() -> None:
    source = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(reset_probability=1.0),
    )
    outcome = FrontierOutcome(0.08, 0.09, False, False)
    assert source.consider(
        "tray",
        _snapshot("tray", 1.0),
        outcome,
        source_episode=3,
        source_step=12,
    )

    fork = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(reset_probability=0.0),
    )
    fork.load_state_dict(source.state_dict())

    assert len(fork.entries["tray"]) == 1
    assert fork.config.reset_probability == 0.0
    assert fork.select("tray", np.random.default_rng(7)) is None


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


def test_frontier_prefers_autonomous_physical_task_progress_at_equal_reach() -> None:
    frontier = OutcomeFrontierCurriculum(
        ("drawer",),
        FrontierCurriculumConfig(capacity_per_task=4, reset_probability=1.0),
    )
    for source, target_distance, articulation in (
        (1, 0.70, 0.02),
        (2, 0.30, 0.22),
    ):
        assert frontier.consider(
            "drawer",
            _complete_snapshot("drawer", float(source)),
            FrontierOutcome(
                0.04,
                0.05,
                True,
                True,
                target_distance=target_distance,
                articulation_position=articulation,
                initial_target_distance=0.70,
                task_progress_observed=True,
            ),
            source_episode=source,
            source_step=40,
            contact_stability_steps=40,
        )

    assert frontier.entries["drawer"][0].source_episode == 2
    assert "target_and_articulation" in frontier.audit()["score"]


def test_frontier_outcome_migrates_checkpoint_without_progress_fields() -> None:
    frontier = OutcomeFrontierCurriculum(("tray",))
    value = FrontierOutcome(0.04, 0.05, True, True)
    fields = value.__dict__.copy()
    fields.pop("target_distance")
    fields.pop("articulation_position")

    migrated = FrontierOutcome(**fields)

    assert migrated.target_distance == 10.0
    assert migrated.articulation_position == 0.0
    assert migrated.initial_target_distance == 10.0
    assert migrated.task_progress_observed is False


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
    assert frontier.report_reset_outcome(selected, 40) is True
    assert all(item.snapshot.runtime_state for item in frontier.entries["tray"])
    assert frontier.audit()["snapshot_migration"] == (
        "complete_state_replaces_and_suppresses_legacy_within_signature"
    )
    assert frontier.audit()["reset_validation_success_count"] == 1


def test_failed_complete_reset_is_removed_and_falls_back_to_legacy() -> None:
    frontier = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(capacity_per_task=8, reset_probability=1.0),
    )
    legacy = FrontierOutcome(0.03, 0.04, True, True)
    complete = FrontierOutcome(0.04, 0.05, True, True)
    assert frontier.consider(
        "tray",
        _snapshot("tray", 1.0),
        legacy,
        source_episode=1,
        source_step=10,
        contact_stability_steps=40,
    )
    assert frontier.consider(
        "tray",
        _complete_snapshot("tray", 2.0),
        complete,
        source_episode=2,
        source_step=20,
        contact_stability_steps=40,
    )
    selected = frontier.select("tray", np.random.default_rng(9))
    assert selected is not None and selected.source_episode == 2

    assert frontier.report_reset_outcome(selected, 0) is False
    fallback = frontier.select("tray", np.random.default_rng(9))

    assert fallback is not None and fallback.source_episode == 1
    assert frontier.audit()["reset_validation_failure_count"] == 1


def test_frontier_preserves_first_stable_frame_and_best_later_frame() -> None:
    frontier = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(
            capacity_per_task=8,
            maximum_entries_per_source_signature=2,
            minimum_contact_stability_steps=20,
        ),
    )
    for step, distance in ((20, 0.08), (21, 0.07), (22, 0.06)):
        assert frontier.consider(
            "tray",
            _complete_snapshot("tray", float(step)),
            FrontierOutcome(distance, distance, True, True),
            source_episode=3,
            source_step=step,
            contact_stability_steps=step,
        )

    retained = frontier.entries["tray"]

    assert {item.source_step for item in retained} == {20, 22}
