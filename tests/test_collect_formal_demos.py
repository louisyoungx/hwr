from __future__ import annotations

from pathlib import Path

from hwr.apps import collect_formal_demos


def test_collection_cli_accepts_explicit_seed_list(tmp_path: Path) -> None:
    arguments = collect_formal_demos.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--task-id",
            "clear_dining_table_3d/v1",
            "--seed-list",
            "301",
            "1000",
            "1406",
        ]
    )

    assert arguments.seed_list == [301, 1000, 1406]


def test_collection_uses_explicit_seeds_for_one_task(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate(root, dataset_id, task, environment_factory, expert_factory, seeds, **kwargs):
        del task, environment_factory, expert_factory, kwargs
        captured.update(root=root, dataset_id=dataset_id, seeds=list(seeds))
        return tmp_path / "generated"

    monkeypatch.setattr(collect_formal_demos, "generate_visual_expert_dataset", fake_generate)
    monkeypatch.setattr(
        collect_formal_demos,
        "verify_visual_dataset",
        lambda path: {"path": str(path)},
    )
    arguments = collect_formal_demos.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--task-id",
            "clear_dining_table_3d/v1",
            "--seed-list",
            "301",
            "1000",
            "1406",
        ]
    )

    report = collect_formal_demos.collect(arguments)

    assert captured == {
        "root": tmp_path,
        "dataset_id": "clear_dining_table_3d_v1-expert-s301",
        "seeds": [301, 1000, 1406],
    }
    assert report["datasets"]["clear_dining_table_3d/v1"] == {
        "path": str(tmp_path / "generated")
    }
