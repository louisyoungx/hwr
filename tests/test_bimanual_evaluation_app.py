from __future__ import annotations

import json

from hwr.apps.evaluate_bimanual_rl import (
    _evaluation_manifest,
    _unseen_seeds,
    build_parser,
)
from hwr.eval import plan_episode_seeds


def test_evaluation_cli_defaults_to_full_unseen_seed_gate(tmp_path) -> None:
    arguments = build_parser().parse_args([str(tmp_path / "training")])

    assert arguments.seed_count == 20
    assert arguments.video_seed_count == 1
    assert arguments.video_width == 640
    assert arguments.video_height == 480
    assert arguments.seed_salt_file is None


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


def test_evaluation_manifest_reveals_compatible_seed_lineage(tmp_path) -> None:
    run = tmp_path / "training"
    output = tmp_path / "evaluation"
    run.mkdir()
    output.mkdir()
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "model-manifest.json").write_text(
        json.dumps({"actor_sha256": "a" * 64}), encoding="utf-8"
    )
    (output / "report.json").write_text("{}", encoding="utf-8")
    (output / "acceptance.json").write_text("{}", encoding="utf-8")
    salt = "manifest-fixture"
    planned = plan_episode_seeds(
        "plan-a",
        "task-a/v1",
        "none",
        2,
        salt,
        environment_seeds=(31, 32),
    ) + plan_episode_seeds(
        "plan-a",
        "task-b/v1",
        "none",
        2,
        salt,
        environment_seeds=(31, 32),
    )

    manifest = _evaluation_manifest(
        output,
        run,
        (31, 32),
        (),
        "plan-a",
        salt,
        planned,
    )

    lineage = manifest["seed_lineage"]
    assert manifest["schema_version"] == "hwr.bimanual-evaluation-run/v2"
    assert lineage["plan_id"] == "plan-a"
    assert lineage["environment_seed_mode"] == "compatibility"
    assert lineage["reveal"]["salt"] == salt
    assert [
        (episode["environment_seed"], episode["policy_rng_seed"])
        for episode in lineage["episodes"]
    ] == [
        (episode.environment_seed, episode.policy_rng_seed)
        for episode in planned
    ]
