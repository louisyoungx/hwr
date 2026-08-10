from __future__ import annotations

import json
from pathlib import Path

from hwr.train import (
    BimanualRLTrainingConfig,
    BimanualTrainingRunner,
    load_bimanual_actor,
    load_default_bimanual_training_catalogs,
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
    assert replay["hindsight_transition_count"] == 2
    assert model["contains_critic"] is False
    assert actor.config.action_dim == 16
