from __future__ import annotations

from hwr.train.foundation_interaction_coverage import summarize_interaction_coverage


def test_interaction_coverage_reports_staged_physical_evidence_by_task(
    tmp_path,
) -> None:
    audits = (
        {
            "left_contact_steps": 3,
            "right_contact_steps": 0,
            "simultaneous_contact_steps": 0,
            "maximum_controlled_rigid_displacement": 0.0,
            "maximum_controlled_articulation_displacement": 0.0,
            "severe_collision_count": 1,
        },
        {
            "left_contact_steps": 3,
            "right_contact_steps": 4,
            "simultaneous_contact_steps": 2,
            "maximum_controlled_rigid_displacement": 0.02,
            "maximum_controlled_articulation_displacement": 0.0,
            "severe_collision_count": 0,
        },
    )
    manifest = {
        "shards": [
            {
                "task_id": "task-a/v1",
                "transition_count": 16,
                "metadata": {
                    "interaction_audit": audit,
                    "interaction_evidence_retained": True,
                },
            }
            for audit in audits
        ]
    }

    report = summarize_interaction_coverage(
        tmp_path,
        manifest,
        minimum_displacement=0.01,
        minimum_transitions=16,
    )

    task = report["partitions"]["task-a/v1"]
    assert task["unilateral_contact_episode_count"] == 2
    assert task["bilateral_contact_episode_count"] == 1
    assert task["controlled_motion_episode_count"] == 1
    assert task["severe_collision_positive_episode_count"] == 1
    assert task["severe_collision_negative_episode_count"] == 1


def test_interaction_coverage_excludes_episodes_too_short_for_training(
    tmp_path,
) -> None:
    manifest = {
        "shards": [{
            "task_id": "task-a/v1",
            "transition_count": 15,
            "metadata": {
                "interaction_audit": {
                    "left_contact_steps": 1,
                    "severe_collision_count": 1,
                },
                "interaction_evidence_retained": True,
            },
        }]
    }

    report = summarize_interaction_coverage(
        tmp_path,
        manifest,
        minimum_displacement=0.01,
        minimum_transitions=16,
    )

    task = report["partitions"]["task-a/v1"]
    assert task["episode_count"] == 0
    assert task["ineligible_short_episode_count"] == 1
    assert task["unilateral_contact_episode_count"] == 0
    assert task["severe_collision_positive_episode_count"] == 0


def test_interaction_coverage_counts_sequence_windows_once_per_source(tmp_path) -> None:
    audit = {"left_contact_steps": 1, "severe_collision_count": 0}
    manifest = {
        "shards": [
            {
                "episode_id": f"excerpt-{index}",
                "task_id": "task-a/v1",
                "transition_count": 16,
                "metadata": {
                    "interaction_audit": audit,
                    "interaction_evidence_retained": True,
                    "sequence_reservoir": {"source_episode_id": "source-1"},
                },
            }
            for index in range(2)
        ]
    }

    report = summarize_interaction_coverage(
        tmp_path,
        manifest,
        minimum_displacement=0.01,
        minimum_transitions=16,
    )

    task = report["partitions"]["task-a/v1"]
    assert task["episode_count"] == 1
    assert task["unilateral_contact_episode_count"] == 1
    assert task["severe_collision_negative_episode_count"] == 1
