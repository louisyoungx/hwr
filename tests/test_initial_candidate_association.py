from __future__ import annotations

from hwr.eval.initial_candidate_association import (
    analyze_episode_records,
    classify_episode,
)
from hwr.eval.target_selection import Candidate


def _candidate(classification: str) -> dict[str, object]:
    return {
        "candidate_index": 0,
        "candidate": {
            **Candidate(
                (1.0, 0.0, 0.5),
                (-1.0, 0.0, 0.0),
                0.1,
                0.1,
                30,
                2,
                0,
                20,
                20,
            ).__dict__,
        },
        "classification": classification,
    }


def test_episode_classification_distinguishes_selector_failure() -> None:
    record = classify_episode(
        task_id="task",
        planned_episode_id="episode",
        selected_index=0,
        candidate_records=(
            _candidate("stage_incompatible"),
            _candidate("stage_compatible"),
        ),
    )

    assert record["classification"] == "relevant_exists_but_distractor_selected"


def test_empty_candidate_is_retained_in_denominator() -> None:
    record = classify_episode(
        task_id="task",
        planned_episode_id="episode",
        selected_index=-1,
        candidate_records=(),
    )

    assert record["classification"] == "no_relevant_final_candidate"
    assert record["subtype"] == "candidate_set_empty"


def test_high_association_gate_requires_each_task_and_low_mixed_count() -> None:
    tasks = ("a", "b", "c")
    records = []
    for task in tasks:
        records.extend(
            {
                "task_id": task,
                "planned_episode_id": f"{task}-{index}",
                "classification": (
                    "stage_compatible_selected"
                    if index < 6 else "no_relevant_final_candidate"
                ),
            }
            for index in range(8)
        )

    report = analyze_episode_records(records, tasks)

    assert report["decision"] == (
        "accepted as initial-association stopping-gate evidence"
    )
    assert report["sample_unit"] == "Episode"


def test_low_association_gate_stops_b1_route() -> None:
    tasks = ("a", "b", "c")
    records = [
        {
            "task_id": task,
            "planned_episode_id": f"{task}-{index}",
            "classification": (
                "stage_compatible_selected"
                if index < 2 else "no_relevant_final_candidate"
            ),
        }
        for task in tasks
        for index in range(8)
    ]

    report = analyze_episode_records(records, tasks)

    assert report["decision"] == (
        "accepted as selector-relevance stopping evidence"
    )
    assert report["capability_claim_allowed"] is False


def test_duplicate_episode_fails_closed() -> None:
    records = [
        {
            "task_id": task,
            "planned_episode_id": "duplicate",
            "classification": "stage_compatible_selected",
        }
        for task in ("a", "b", "c")
        for _ in range(8)
    ]

    report = analyze_episode_records(records, ("a", "b", "c"))

    assert report["decision"] == "invalid"
