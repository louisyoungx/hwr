from __future__ import annotations

from types import SimpleNamespace

from hwr.train.foundation_holdout import (
    causality_batches_by_task,
    causality_window_manifest,
    select_causality_windows,
)


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
