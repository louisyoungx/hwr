from __future__ import annotations

import json
from pathlib import Path

from hwr.train import (
    BimanualRLTrainingConfig,
    BimanualTrainingRunner,
    load_bimanual_actor,
    load_default_bimanual_training_catalogs,
    resume_bimanual_training_run,
    save_bimanual_live_progress,
    save_bimanual_training_run,
    verify_bimanual_training_run,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_result():
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config = BimanualRLTrainingConfig(
        episodes=1,
        episode_step_limit=2,
        replay_capacity=32,
        batch_size=4,
        learning_starts=100,
        initial_random_episodes=1,
        raw_image_width=16,
        raw_image_height=12,
        image_width=8,
        image_height=6,
        point_count=8,
        language_dim=16,
        hidden_dim=32,
    )
    return BimanualTrainingRunner(tasks, bindings, config).train()


def test_training_run_saves_verified_no_demonstration_lineage(tmp_path) -> None:
    result = _small_result()

    path = save_bimanual_training_run(
        tmp_path, "smoke-run", result, source_commit="a" * 40
    )
    manifest = verify_bimanual_training_run(path)
    lineage = json.loads((path / "lineage.json").read_text(encoding="utf-8"))
    replay = json.loads(
        (path / "replay-manifest.json").read_text(encoding="utf-8")
    )
    model = json.loads(
        (path / "model-manifest.json").read_text(encoding="utf-8")
    )
    actor = load_bimanual_actor(path)

    assert manifest["record_count"] == 1
    assert lineage["action_label_sources"] == []
    assert lineage["expert_policies"] == []
    assert lineage["behavior_cloning"] is False
    assert replay["action_labels"] is False
    assert replay["safety_event_size"] == 0
    assert set(replay["task_partition_sizes"]) == {
        "carry_dining_tray/v1",
        "carry_living_room_basket/v1",
        "hold_drawer_place_item/v1",
    }
    assert replay["task_sampling"]["task_stages"] is False
    assert replay["action_exploration"]["task_conditioned"] is False
    assert replay["action_exploration"]["global_random_bursts"] == {
        "probability": 0.01,
        "hold_steps": 8,
    }
    assert replay["hindsight_transition_count"] == 2
    assert model["contains_critic"] is False
    assert actor.config.action_dim == 16

    result.records.append(result.records[0])
    progress = save_bimanual_live_progress(path, result)
    assert len(progress.read_text(encoding="utf-8").splitlines()) == 2
    assert verify_bimanual_training_run(path)["record_count"] == 1


def test_training_run_resumes_at_next_episode_with_replay_and_rng(tmp_path) -> None:
    result = _small_result()
    path = save_bimanual_training_run(
        tmp_path, "resume-run", result, source_commit="b" * 40
    )
    manifest_path = path / "manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["training_config"].pop("discovery_replay_fraction")
    legacy_manifest["training_config"].pop("safety_replay_fraction")
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config_values = result.config.to_dict()
    config_values["episodes"] = 2
    runner = BimanualTrainingRunner(
        tasks, bindings, BimanualRLTrainingConfig(**config_values)
    )

    resume_bimanual_training_run(path, runner)
    resumed = runner.train()

    assert [record.episode for record in resumed.records] == [0, 1]
    assert resumed.replay.episode_count == 2
    assert resumed.environment_steps == sum(record.steps for record in resumed.records)
