from __future__ import annotations

import json

from hwr.apps.evaluate_bimanual_rl import _unseen_seeds, build_parser


def test_evaluation_cli_defaults_to_full_unseen_seed_gate(tmp_path) -> None:
    arguments = build_parser().parse_args([str(tmp_path / "training")])

    assert arguments.seed_count == 20
    assert arguments.video_seed_count == 1
    assert arguments.video_width == 640
    assert arguments.video_height == 480


def test_evaluation_seed_generator_excludes_every_training_seed(tmp_path) -> None:
    run_path = tmp_path / "training"
    run_path.mkdir()
    (run_path / "manifest.json").write_text(
        json.dumps({"training_config": {"seed": 100}}), encoding="utf-8"
    )
    (run_path / "episodes.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"seed": 500}),
                json.dumps({"seed": 105229}),
            )
        ),
        encoding="utf-8",
    )

    seeds = _unseen_seeds(run_path, 3, 500)

    assert seeds == (209958, 314687, 419416)
    assert not set(seeds) & {500, 105229}
