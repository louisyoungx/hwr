from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import hwr.train.foundation_holdout as foundation_holdout
from hwr.train.foundation_holdout import (
    ACTION_EXECUTION_VALIDATION_PHASE,
    COLLISION_VALIDATION_PHASE,
    SYSTEM_IDENTIFICATION_CORRELATIONS,
    SYSTEM_IDENTIFICATION_PHASE,
    _collision_balance_target,
    _episode_collision_class,
    _collect_holdout_attempt,
    _holdout_collection_config,
    _holdout_motion_correlation,
    causality_batches_by_task,
    causality_window_manifest,
    select_causality_windows,
)
from hwr.train.foundation_exploration import RandomRLExplorationConfig
from hwr.train.foundation_collection import AutonomousCollectionConfig


class _Loader:
    def __init__(self) -> None:
        self.windows = SimpleNamespace(transitions=2)
        self.metadata = []
        for task in ("task-a/v1", "task-b/v1"):
            for episode_index in range(2):
                for start in range(4):
                    self.metadata.append(
                        {
                            "task_id": task,
                            "episode_id": f"{task}-episode-{episode_index}",
                            "seed": 10 + episode_index,
                            "transition_start": start,
                            "transition_stop": start + 2,
                            "metadata": {
                                "holdout_phase": SYSTEM_IDENTIFICATION_PHASE
                            },
                        }
                    )

    def __len__(self):
        return len(self.metadata)

    def window_metadata(self, index):
        return self.metadata[index]

    def build(self, indices, *, include_visual_targets=True):
        assert include_visual_targets is False
        return tuple(indices)


def test_causality_window_selection_is_deterministic_balanced_and_nonoverlapping() -> None:
    loader = _Loader()

    first = select_causality_windows(
        loader, ("task-a/v1", "task-b/v1"), windows_per_task=4, selection_seed=7
    )
    second = select_causality_windows(
        loader, ("task-a/v1", "task-b/v1"), windows_per_task=4, selection_seed=7
    )
    manifest = causality_window_manifest(loader, first)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "task-a/v1": 4,
        "task-b/v1": 4,
    }
    assert all(value["transition_start"] % 2 == 0 for value in manifest)
    assert {
        (value["task_id"], value["episode_id"]) for value in manifest
    } == {
        (task, f"{task}-episode-{episode}")
        for task in ("task-a/v1", "task-b/v1")
        for episode in range(2)
    }


def test_causality_batches_are_built_lazily_per_task() -> None:
    loader = _Loader()
    selected = select_causality_windows(
        loader, ("task-a/v1", "task-b/v1"), windows_per_task=4, selection_seed=7
    )

    batches = causality_batches_by_task(loader, selected, batch_size=2)

    assert list(batches["task-a/v1"]) == [
        selected["task-a/v1"][:2],
        selected["task-a/v1"][2:],
    ]


def test_causality_window_selection_rejects_an_undercovered_episode() -> None:
    loader = _Loader()

    with pytest.raises(ValueError, match="lacks windows"):
        select_causality_windows(
            loader,
            ("task-a/v1", "task-b/v1"),
            windows_per_task=6,
            selection_seed=7,
        )


def test_formal_holdout_slots_balance_collision_outcomes() -> None:
    targets = tuple(_collision_balance_target(index, 16) for index in range(16))

    assert targets.count("positive") == 8
    assert targets.count("negative") == 8
    assert _collision_balance_target(0, 1) is None
    assert SYSTEM_IDENTIFICATION_CORRELATIONS == (0.0, 0.5, 0.9, 0.96)


def test_holdout_collection_bounds_unbalanced_and_negative_episodes() -> None:
    system = _holdout_collection_config(
        "abc123",
        maximum_steps=6000,
        minimum_transitions=128,
        balance_kind=None,
        balance_target=None,
    )
    negative = _holdout_collection_config(
        "abc123",
        maximum_steps=6000,
        minimum_transitions=16,
        balance_kind="safety_intervention",
        balance_target="negative",
    )
    positive = _holdout_collection_config(
        "abc123",
        maximum_steps=6000,
        minimum_transitions=16,
        balance_kind="safety_intervention",
        balance_target="positive",
    )

    assert system.maximum_steps == 128
    assert negative.maximum_steps == 16
    assert positive.maximum_steps == 6000
    assert positive.minimum_stop_steps == 16
    assert positive.stop_after_safety_intervention is True
    assert positive.stop_after_severe_collision is False


def test_collision_class_uses_retained_physical_audit() -> None:
    metadata = {
        "result_reason": "severe_collision_evidence",
        "interaction_audit": {"severe_collision_count": 1.0},
    }

    assert _episode_collision_class(metadata) == "positive"
    assert _episode_collision_class(
        {
            "result_reason": ACTION_EXECUTION_VALIDATION_PHASE,
            "interaction_audit": {"severe_collision_count": 0.0},
        }
    ) == "negative"


def test_only_system_identification_cycles_excitation_correlations() -> None:
    exploration = RandomRLExplorationConfig(0.96, 0.05)

    assert tuple(
        _holdout_motion_correlation(
            exploration,
            holdout_phase=SYSTEM_IDENTIFICATION_PHASE,
            episode_index=index,
        )
        for index in range(4)
    ) == SYSTEM_IDENTIFICATION_CORRELATIONS
    for phase in (
        ACTION_EXECUTION_VALIDATION_PHASE,
        COLLISION_VALIDATION_PHASE,
    ):
        assert all(
            _holdout_motion_correlation(
                exploration,
                holdout_phase=phase,
                episode_index=index,
            )
            == 0.96
            for index in range(16)
        )


def test_positive_holdout_search_replays_deterministic_tail(monkeypatch) -> None:
    episodes = []

    @dataclass(frozen=True)
    class Episode:
        arrays: dict[str, list[float]]
        metadata: dict[str, object]

    class Collector:
        def __init__(self, config):
            self.config = config

        def collect(self, environment, source, *, task_id, seed):
            del environment, source, task_id, seed
            positive = self.config.stop_after_safety_intervention
            steps = 4
            labels = [0.0, 0.0, 0.0, float(positive)]
            episode = Episode(
                arrays={
                    "executed_action": [0] * min(
                        steps, self.config.retained_transition_capacity or steps
                    ),
                    "safety_intervention": labels[
                        -(self.config.retained_transition_capacity or steps) :
                    ],
                },
                metadata={
                    "collection_transition_count": steps,
                    "collection_stop_reason": (
                        "safety_intervention_evidence"
                        if positive
                        else "environment"
                    ),
                },
            )
            episodes.append((self.config, episode))
            return episode

    collector = Collector(
        AutonomousCollectionConfig(
            "fixture/v1",
            "abc123",
            maximum_steps=8,
            stop_after_safety_intervention=True,
            minimum_stop_steps=2,
            retained_transition_capacity=2,
        )
    )
    monkeypatch.setattr(
        foundation_holdout,
        "AutonomousEpisodeCollector",
        lambda preprocessor, config: Collector(config),
    )
    monkeypatch.setattr(
        foundation_holdout,
        "_episode_balance_class",
        lambda episode, kind: (
            "positive"
            if any(episode.arrays["safety_intervention"])
            else "negative"
        ),
    )
    monkeypatch.setattr(
        foundation_holdout, "RandomRLActionSource", lambda *args: object()
    )

    episode = _collect_holdout_attempt(
        collector,
        object(),
        object(),
        object(),
        RandomRLExplorationConfig(),
        task_id="fixture/v1",
        seed=1,
        balance_kind="safety_intervention",
        balance_target="positive",
        minimum_transitions=2,
    )

    assert len(episodes) == 2
    assert episodes[0][0].camera_render_start_transition == 8
    assert episodes[1][0].camera_render_start_transition == 0
    assert episode.metadata["search_transition_count"] == 4
    assert episode.metadata["camera_warmup_transitions"] == 2
    assert episode.arrays["safety_intervention"][-1] == 1.0
