from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hwr.adapters.mujoco import (
    MujocoBimanualBackendFactory,
    load_default_bimanual_training_catalogs,
)
from hwr.train import (
    BimanualRLTrainingConfig,
    BimanualTrainingRunner,
    fork_bimanual_training_run,
    load_bimanual_actor,
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
    return BimanualTrainingRunner(
        tasks, MujocoBimanualBackendFactory(bindings), config
    ).train()


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
    checkpoint = torch.load(
        path / "training-checkpoint.pt", map_location="cpu", weights_only=False
    )

    assert manifest["record_count"] == 1
    assert lineage["action_label_sources"] == []
    assert lineage["expert_policies"] == []
    assert lineage["behavior_cloning"] is False
    assert replay["action_labels"] is False
    assert replay["safety_event_size"] == 0
    assert replay["physical_progress_size"] == 0
    assert replay["physical_progress_criteria"]["action_labels"] is False
    assert set(replay["task_partition_sizes"]) == {
        "carry_dining_tray/v1",
        "carry_living_room_basket/v1",
        "hold_drawer_place_item/v1",
    }
    assert replay["task_sampling"]["task_stages"] is False
    assert replay["task_sampling"]["reach_metric"] == (
        "minimum_over_time_of_worst_side_distance"
    )
    assert replay["frontier_curriculum"]["task_stages"] is False
    assert replay["frontier_curriculum"]["action_outputs"] is False
    assert replay["action_exploration"]["task_conditioned"] is False
    assert replay["action_exploration"]["global_random_bursts"] == {
        "probability": 0.01,
        "hold_steps": 8,
        "motion": "task-agnostic-uniform",
        "grippers": "policy-held",
    }
    assert replay["action_exploration"]["actuator_dwell"] == {
        "probability": 0.0,
        "initial_probability": 0.0,
        "hold_steps": 240,
        "closed_probability": 0.5,
        "motion": "zero",
        "grippers": "paired-or-independent-bernoulli-binary",
    }
    assert replay["random_streams"]["shared"] is False
    assert "task_rng_state" in checkpoint
    assert "frontier_rng_state" in checkpoint
    assert "exploration_rng_state" in checkpoint
    assert "numpy_rng_state" not in checkpoint
    assert replay["hindsight_transition_count"] == 2
    assert model["contains_critic"] is False
    assert actor.config.action_dim == 16
    assert manifest["rl_config"]["actor_learning_rate"] == (
        result.config.actor_learning_rate
    )
    assert manifest["rl_config"]["visual_temporal_contrastive_weight"] == 0.05
    assert manifest["critic_config"]["privileged_state_dim"] == 62

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
    legacy_manifest["training_config"].pop("progress_replay_fraction")
    legacy_manifest["training_config"].pop("safety_replay_fraction")
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config_values = result.config.to_dict()
    config_values["episodes"] = 2
    runner = BimanualTrainingRunner(
        tasks,
        MujocoBimanualBackendFactory(bindings),
        BimanualRLTrainingConfig(**config_values),
    )

    resume_bimanual_training_run(path, runner)
    resumed = runner.train()

    assert [record.episode for record in resumed.records] == [0, 1]
    assert resumed.replay.episode_count == 2
    assert resumed.environment_steps == sum(record.steps for record in resumed.records)


def test_training_run_rejects_a_different_critic_state_layout(tmp_path) -> None:
    result = _small_result()
    path = save_bimanual_training_run(
        tmp_path, "critic-layout", result, source_commit="c" * 40
    )
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["critic_config"]["privileged_state_dim"] = 60
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config_values = result.config.to_dict()
    config_values["episodes"] = 2
    runner = BimanualTrainingRunner(
        tasks,
        MujocoBimanualBackendFactory(bindings),
        BimanualRLTrainingConfig(**config_values),
    )

    with pytest.raises(ValueError, match="Critic architecture"):
        resume_bimanual_training_run(path, runner)


def test_training_fork_records_parent_hashes_and_only_exploration_changes(
    tmp_path,
) -> None:
    parent = _small_result()
    parent_path = save_bimanual_training_run(
        tmp_path, "parent-run", parent, source_commit="d" * 40
    )
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config_values = parent.config.to_dict()
    config_values.update(
        episodes=2,
        actuator_dwell_probability=0.001,
        actuator_dwell_steps=260,
    )
    runner = BimanualTrainingRunner(
        tasks,
        MujocoBimanualBackendFactory(bindings),
        BimanualRLTrainingConfig(**config_values),
    )

    provenance = fork_bimanual_training_run(parent_path, runner)
    result = runner.train()
    fork_path = save_bimanual_training_run(
        tmp_path,
        "fork-run",
        result,
        source_commit="e" * 40,
        parent_training_run=provenance,
    )
    manifest = verify_bimanual_training_run(fork_path)
    lineage = json.loads((fork_path / "lineage.json").read_text(encoding="utf-8"))

    assert provenance["fork_record_count"] == 1
    assert set(provenance["config_changes"]) == {
        "actuator_dwell_probability",
        "actuator_dwell_steps",
        "episodes",
    }
    assert len(provenance["parent_checkpoint_sha256"]) == 64
    assert manifest["parent_training_run"] == provenance
    assert lineage["initialization"] == "audited-no-demonstration-checkpoint"
    assert lineage["parent_training_run"] == provenance


def test_training_fork_rejects_non_exploration_config_changes(tmp_path) -> None:
    parent = _small_result()
    parent_path = save_bimanual_training_run(
        tmp_path, "fixed-parent", parent, source_commit="f" * 40
    )
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    config_values = parent.config.to_dict()
    config_values["replay_capacity"] = 64
    runner = BimanualTrainingRunner(
        tasks,
        MujocoBimanualBackendFactory(bindings),
        BimanualRLTrainingConfig(**config_values),
    )

    with pytest.raises(ValueError, match="replay_capacity"):
        fork_bimanual_training_run(parent_path, runner)
