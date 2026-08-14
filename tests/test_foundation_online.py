from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from hwr.core.embodied import (
    DualArmObservation,
    DualArmProprioception,
    NaturalLanguageInstruction,
)
from hwr.core.runtime import LegalEnvironmentTransform, RuntimeStepOutcome
from hwr.core.types import CameraFrame, EpisodeResult
from hwr.perception.contracts import (
    DUAL_ARM_CAMERA_IDS,
    CameraCalibration,
    PinholeIntrinsics,
)
from hwr.perception.foundation import (
    DenseVisualFeatures,
    FoundationModelLock,
    SemanticLanguageFeatures,
    WeightArtifact,
    language_source_sha256,
)
from hwr.perception.high_resolution import (
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
)
from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
)
from hwr.policy.latent_actions import LatentActionScaling
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.policy.latent_value import LatentValueModel
from hwr.train.foundation_online import (
    FoundationOnlineTrainingRunner,
    FoundationProviderFactories,
    FoundationTaskInterface,
)
from hwr.train.foundation_collection import (
    AutonomousCollectionConfig,
    AutonomousEpisodeCollector,
)
from hwr.train.foundation_exploration import RandomRLActionSource
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_setup import FoundationLearningStack
from hwr.train.foundation_trainer import (
    FoundationTrainerConfig,
    FoundationWorldModelTrainer,
)
from hwr.train.imagination_rl import ImaginationRLConfig
from hwr.train.intrinsic_exploration import IntrinsicExplorationConfig
from hwr.world_model import (
    ActionConditionedWorldModel,
    WorldModelConfig,
    WorldModelLoss,
    WorldModelLossConfig,
)


TASK_IDS = ("fixture-a/v1", "fixture-b/v1", "fixture-c/v1")


class _VisionProvider:
    def __init__(self, role: str, dimension: int, marker: str) -> None:
        self._lock = FoundationModelLock(
            f"fixture/{marker}", marker * 40, role, "Apache-2.0", dimension,
            "fixture-grid/v1",
            (WeightArtifact(f"{marker}.bin", marker * 64, 1),),
        )

    @property
    def model_lock(self):
        return self._lock

    def encode_vision(self, rgb, camera_valid, source_sha256):
        values = np.full((3, 2, 2, self._lock.output_dimension), 0.25, np.float32)
        valid = np.broadcast_to(camera_valid[:, None, None], (3, 2, 2)).copy()
        return DenseVisualFeatures(
            values, valid, self._lock.lock_sha256, source_sha256
        )


class _LanguageProvider:
    def __init__(self) -> None:
        self._lock = FoundationModelLock(
            "fixture/language", "c" * 40, "language", "Apache-2.0", 6,
            "fixture-language/v1",
            (WeightArtifact("c.bin", "c" * 64, 1),),
        )

    @property
    def model_lock(self):
        return self._lock

    def encode_language(self, text, locale):
        return SemanticLanguageFeatures(
            np.arange(1, 7, dtype=np.float32),
            self._lock.lock_sha256,
            language_source_sha256(text, locale),
        )


def _observation(task_id: str, sequence: int) -> DualArmObservation:
    size = 160
    timestamp = sequence * 50_000_000
    rgb = np.full((size, size, 3), sequence, np.uint8).tobytes()
    depth = np.ones((size, size), np.float32).tobytes()
    cameras = tuple(
        CameraFrame(
            name,
            timestamp,
            sequence,
            size,
            size,
            "depth32f" if name == "head_depth" else "rgb8",
            payload=depth if name == "head_depth" else rgb,
        )
        for name in DUAL_ARM_CAMERA_IDS
    )
    proprioception = DualArmProprioception(
        (0.0,) * 6, (0.0,) * 6, (0.0,) * 6, (0.0,) * 6,
        0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0),
    )
    return DualArmObservation(
        timestamp,
        sequence,
        task_id,
        NaturalLanguageInstruction(f"执行 {task_id} 的双臂任务"),
        proprioception,
        cameras,
    )


class _Backend:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.sequence = 0
        self._result = None

    def reset(self, *, seed: int, task_id: str):
        del seed
        if task_id != self.task_id:
            raise ValueError("fixture task differs")
        self.sequence = 0
        self._result = None
        return _observation(task_id, 0)

    def observe(self):
        return _observation(self.task_id, self.sequence)

    def apply(self, frame):
        self.sequence += 1
        terminal = self.sequence == 2
        if terminal:
            self._result = EpisodeResult(True, "fixture_success", self.sequence)
        return RuntimeStepOutcome(
            _observation(self.task_id, self.sequence),
            reward=float(self.sequence),
            terminated=terminal,
            info={"applied_action": replace(frame), "safety_intervened": False},
        )

    def result(self):
        return self._result

    def task_audit(self):
        interaction = int(self.sequence > 0)
        return {
            "left_contact_steps": interaction,
            "right_contact_steps": interaction,
            "simultaneous_contact_steps": interaction,
            "severe_collision_count": 0,
            "metrics": {
                "maximum_controlled_target_progress": 0.1 * interaction,
                "maximum_controlled_articulation_progress": 0.0,
            },
        }

    def legal_environment_transforms(self):
        return (LegalEnvironmentTransform("lateral_reflection"),)

    def close(self):
        pass


def _preprocessor() -> HighResolutionVisionPreprocessor:
    size = 160
    calibrations = {
        name: CameraCalibration(
            f"fixture-{name}",
            name,
            PinholeIntrinsics(size, size, 100.0, 100.0, 80.0, 80.0),
            tuple(np.eye(4).reshape(-1)),
        )
        for name in DUAL_ARM_CAMERA_IDS
    }
    return HighResolutionVisionPreprocessor(HighResolutionVisionConfig(), calibrations)


def _stack() -> FoundationLearningStack:
    visual_config = VisualStudentConfig(
        image_size=160,
        visual_history=2,
        backbone_dimensions=(8, 12, 16, 24),
        backbone_depths=(1, 1, 1, 1),
        feature_dimension=8,
        state_queries=2,
        attention_heads=2,
        fusion_layers=1,
        temporal_layers=1,
        formal=False,
    )
    world_config = WorldModelConfig(
        visual_dimension=8,
        language_dimension=6,
        proprioception_dimension=31,
        action_dimension=16,
        observation_embedding_dimension=12,
        deterministic_dimension=10,
        stochastic_variables=3,
        stochastic_classes=4,
        hidden_dimension=16,
        prior_ensemble=2,
        reward_bins=11,
        formal=False,
    )
    student = VisualStudentModel(visual_config)
    visual_objective = VisualFoundationObjectives(
        VisualObjectiveConfig(
            student_dimension=8,
            vision_language_dimension=7,
            dense_vision_dimension=5,
        )
    )
    world = ActionConditionedWorldModel(world_config)
    actor = LatentActor(
        LatentActorConfig(
            world_config.feature_dimension,
            hidden_dimension=16,
            hidden_layers=2,
            formal=False,
        )
    )
    value = LatentValueModel(
        world_config.feature_dimension, bins=11, hidden_dimension=16, hidden_layers=2
    )
    trainer = FoundationWorldModelTrainer(
        student,
        visual_objective,
        world,
        WorldModelLoss(world_config, WorldModelLossConfig()),
        actor,
        value,
        ImaginationRLConfig(horizon=2, value_bins=11, value_symlog_limit=5.0),
        IntrinsicExplorationConfig(
            horizon=2, value_bins=11, value_symlog_limit=5.0
        ),
        LatentActionScaling(),
        FoundationTrainerConfig(),
    )
    return FoundationLearningStack(trainer, LatentActionScaling())


def _diagnostic(passed: bool) -> dict[str, object]:
    ratio = 1.2 if passed else 1.0
    physical = {
        "passed": passed,
        "components": {
            "visual_latent": {"passed": passed},
            "proprioception": {"passed": passed},
        },
    }
    statistics = {
        "count": 1,
        "ratio_p05": ratio,
        "lower_bound_passed": passed,
        "robust_passed": passed,
    }
    return {
        "schema_version": "hwr.foundation-action-causality/v6",
        "action_source": "actual_executed_action",
        "safety_action_source": "actor_proposal",
        "counterfactual_pairing": "proposal-executed-pair/v1",
        "report": {"shuffled_to_true_ratio": ratio},
        "assessment": {
            "passed": passed,
            "shuffled_to_true_ratio": ratio,
            "horizon_count": 2,
            "worse_horizon_count": 2 if passed else 0,
            "worse_horizon_fraction": 1.0 if passed else 0.0,
        },
        "shuffle_statistics": statistics,
        "one_step_action_utilization": {
            "assessment": physical,
            "shuffle_statistics": statistics,
        },
        "partitions": {
            task_id: {
                "assessment": {"passed": passed},
                "shuffle_statistics": statistics,
                "one_step_action_utilization": {
                    "assessment": physical,
                    "shuffle_statistics": statistics,
                },
            }
            for task_id in TASK_IDS
        },
    }


def _passing_action_probe(*args, **kwargs):
    del args, kwargs
    return {
        "state_only_to_state_action_ratio": 1.2,
        "bootstrap": {"ratio_p05": 1.1},
        "partitions": {
            task_id: {
                "state_only_to_state_action_ratio": 1.2,
                "bootstrap": {"ratio_p05": 1.1},
            }
            for task_id in TASK_IDS
        },
    }


def _deployment_failure_with_physical_causality() -> dict[str, object]:
    diagnostic = _diagnostic(False)
    physical = _diagnostic(True)
    diagnostic["one_step_action_utilization"] = physical[
        "one_step_action_utilization"
    ]
    for task_id in TASK_IDS:
        diagnostic["partitions"][task_id]["one_step_action_utilization"] = (
            physical["partitions"][task_id]["one_step_action_utilization"]
        )
    return diagnostic


def _runner(tmp_path, config: FoundationOnlineTrainingConfig):
    tasks = {name: FoundationTaskInterface(name, 2) for name in TASK_IDS}
    providers = FoundationProviderFactories(
        lambda: _VisionProvider("vision_language", 7, "a"),
        lambda: _VisionProvider("dense_vision", 5, "b"),
        _LanguageProvider,
    )
    return FoundationOnlineTrainingRunner(
        tasks,
        lambda task_id, width, height: _Backend(task_id),
        _preprocessor(),
        providers,
        _stack(),
        config,
        tmp_path / "run",
        source_commit="abc123",
        development_ready_sha256="d" * 64,
        execution={"device": "cpu", "foundation_device": "cpu"},
    )


def _config(*, episodes: int = 6) -> FoundationOnlineTrainingConfig:
    return FoundationOnlineTrainingConfig(
            episodes=episodes,
            minimum_actor_readiness_episodes=min(3, episodes),
            actor_readiness_consecutive_passes=1,
            actor_warmup_minimum_updates=1,
            actor_warmup_maximum_updates=1,
            actor_warmup_window_updates=1,
            actor_warmup_stable_windows=1,
            actor_warmup_maximum_return_relative_range=1.0,
            actor_warmup_minimum_motion_entropy=-100.0,
            actor_warmup_minimum_gripper_entropy=-100.0,
            minimum_active_action_dimension_fraction=0.01,
            minimum_action_effective_rank=0.01,
            minimum_collision_positive_episodes_per_task=0,
            minimum_collision_validation_positive_episodes_per_task=0,
            minimum_collision_validation_negative_episodes_per_task=1,
            minimum_collision_validation_recall=0.0,
            minimum_collision_validation_pr_auc=0.0,
            maximum_collision_validation_brier_score=1.0,
            maximum_collision_validation_false_positive_rate=1.0,
            minimum_collision_validation_terminal_alignment=0.0,
            minimum_collision_validation_action_sensitivity_ratio=1.0,
            collision_validation_holdout_episodes_per_task=1,
            collision_validation_holdout_transitions_per_episode=2,
            minimum_action_execution_positive_episodes_per_task=0,
            minimum_action_execution_negative_episodes_per_task=1,
            minimum_action_execution_recall=0.0,
            minimum_action_execution_pr_auc=0.0,
            maximum_action_execution_brier_score=1.0,
            maximum_intervention_action_normalized_rmse=100.0,
            maximum_identity_action_normalized_rmse=100.0,
            maximum_action_execution_out_of_bounds_rate=1.0,
            action_execution_holdout_episodes_per_task=1,
            action_execution_holdout_transitions_per_episode=2,
            calibration_early_stop_episodes=episodes,
            collection_episodes_per_cycle=3,
            updates_per_cycle=1,
            batch_size=1,
            sequence_transitions=2,
            camera_width=160,
            camera_height=160,
            replay_transition_capacity=6,
            published_checkpoint_retention=1,
            causality_holdout_episodes_per_task=1,
            causality_audit_windows_per_task=1,
            causality_audit_batch_size=1,
            seed=7,
        )


def test_online_runner_uses_one_loop_for_random_then_current_rl_actions(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "hwr.train.foundation_admission.evaluate_foundation_action_causality_audit",
        lambda trainer, batches, criteria, shuffle_seed, shuffle_repeats: _diagnostic(True),
    )
    monkeypatch.setattr(
        "hwr.train.foundation_admission.evaluate_foundation_data_action_probe",
        _passing_action_probe,
    )
    config = _config(episodes=9)
    runner = _runner(tmp_path, config)

    result = runner.train()

    assert result.update_count == 3
    assert {record.task_id for record in result.records} == set(TASK_IDS)
    assert [record.action_source for record in result.records[:3]] == [
        "random_rl_exploration"
    ] * 3
    assert [record.action_source for record in result.records[3:6]] == [
        "intrinsic_rl_actor"
    ] * 3
    assert [record.action_source for record in result.records[6:]] == [
        "rl_actor"
    ] * 3
    assert all(record.state_novelty >= 0.0 for record in result.records)
    assert all(record.td_error >= 0.0 for record in result.records)
    assert len({record.td_error for record in result.records[:3]}) > 1
    assert result.latest_checkpoint.is_dir()
    assert result.latest_deployment.is_dir()
    assert result.latest_action_causality_report.is_file()
    latest = json.loads((tmp_path / "run/latest.json").read_text())
    run_manifest = json.loads((tmp_path / "run/run-manifest.json").read_text())
    recovery = json.loads(
        (result.latest_checkpoint / "recovery/manifest.json").read_text()
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "run/episodes.jsonl").read_text().splitlines()
    ]
    checkpoint = json.loads(
        (result.latest_checkpoint / "manifest.json").read_text()
    )
    deployment = json.loads(
        (result.latest_deployment / "manifest.json").read_text()
    )
    assert latest["action_causality_sha256"] == checkpoint[
        "training_diagnostics"
    ]["action_causality_report_sha256"]
    assert run_manifest["schema_version"] == "hwr.foundation-online-run/v4"
    assert run_manifest["execution"] == {
        "device": "cpu",
        "foundation_device": "cpu",
    }
    assert run_manifest["development_ready"] == {
        "schema_version": "hwr.foundation-development-ready/v3",
        "sha256": "d" * 64,
        "path": "development-ready.json",
    }
    assert run_manifest["lineage"]["expert_policies"] == []
    assert run_manifest["lineage"]["teacher_actions"] is False
    assert run_manifest["lineage"]["action_search"] is False
    assert recovery["schema_version"] == "hwr.foundation-runner-recovery/v6"
    assert all("safety_intervention_rate" in record for record in records)
    assert all("safety_cost_rate" not in record for record in records)
    assert deployment["training_diagnostics"] == checkpoint[
        "training_diagnostics"
    ]
    assert runner.store.manifest["transition_count"] <= 6
    assert runner.causality_store.manifest["episode_count"] == 9
    phases = {
        shard["metadata"]["holdout_phase"]
        for shard in runner.causality_store.manifest["shards"]
    }
    assert phases == {
        "system_identification",
        "action_execution_validation",
        "collision_validation",
    }
    assert runner.causality_store.manifest["transition_count"] == 18
    assert len(list((tmp_path / "run/checkpoints").glob("update-*"))) == 1
    assert len(list((tmp_path / "run/deployments").glob("update-*"))) == 1
    assert runner.task_sampler.audit()["distance_thresholds"] is False
    cycle_metrics = json.loads(
        (tmp_path / "run/metrics/cycle-000003.json").read_text()
    )
    assert cycle_metrics["training"]["trainer/visual_gradient_norm"] >= 0.0
    assert cycle_metrics["action_coverage"]["transition_count"] == 6
    assert cycle_metrics["episodes"]["count"] == 3
    assert cycle_metrics["action_causality"]["passed"] is True
    assert cycle_metrics["actor_readiness"]["unlocked"] is True
    assert checkpoint["training_diagnostics"]["task_actor_update_count"] == 2
    assert cycle_metrics["learning_frontier"]["task_semantic_fields"] == []
    assert all("frontier_entries_added" in record for record in records)

    resumed = _runner(tmp_path, config)
    resumed.resume_latest()
    assert resumed.result().latest_deployment == result.latest_deployment

    checkpoint_manifest = result.latest_checkpoint / "manifest.json"
    drifted = json.loads(checkpoint_manifest.read_text())
    drifted["training_diagnostics"]["action_causality_report_sha256"] = "0" * 64
    checkpoint_manifest.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint diagnostic provenance"):
        _runner(tmp_path, config).resume_latest()


def test_online_runner_never_exports_a_failed_causality_deployment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "hwr.train.foundation_admission.evaluate_foundation_action_causality_audit",
        lambda trainer, batches, criteria, shuffle_seed, shuffle_repeats: (
            _deployment_failure_with_physical_causality()
        ),
    )
    monkeypatch.setattr(
        "hwr.train.foundation_admission.evaluate_foundation_data_action_probe",
        _passing_action_probe,
    )
    runner = _runner(tmp_path, _config(episodes=3))

    with pytest.raises(RuntimeError, match="no causality-qualified deployment"):
        runner.train()

    latest = json.loads((tmp_path / "run/latest.json").read_text())
    checkpoint = tmp_path / "run" / latest["training_checkpoint"]
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert "deployment" not in latest
    assert manifest["training_diagnostics"]["action_causality_passed"] is False
    assert not (tmp_path / "run/deployments").exists()
    resumed = _runner(tmp_path, _config(episodes=3))
    resumed.resume_latest()
    with pytest.raises(RuntimeError, match="no causality-qualified deployment"):
        resumed.result()


def test_resume_rolls_replay_back_to_last_atomic_checkpoint(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "hwr.train.foundation_admission.evaluate_foundation_action_causality_audit",
        lambda trainer, batches, criteria, shuffle_seed, shuffle_repeats: _diagnostic(True),
    )
    monkeypatch.setattr(
        "hwr.train.foundation_admission.evaluate_foundation_data_action_probe",
        _passing_action_probe,
    )
    config = _config()
    runner = _runner(tmp_path, config)
    result = runner.train()
    snapshot = json.loads(
        (result.latest_checkpoint / "recovery/replay-manifest.json").read_text()
    )
    holdout_snapshot = json.loads(
        (result.latest_checkpoint / "recovery/causality-manifest.json").read_text()
    )
    collector = AutonomousEpisodeCollector(
        runner.preprocessor,
        AutonomousCollectionConfig("fixture-env/v1", "abc123", maximum_steps=2),
    )
    extra = collector.collect(
        _Backend(TASK_IDS[0]),
        RandomRLActionSource(runner.stack.action_scaling),
        task_id=TASK_IDS[0],
        seed=999,
    )
    extra_path = runner.store.append(extra)
    extra_holdout_path = runner.causality_store.append(extra)
    archive = tmp_path / "run/recovery/replay-prune-archive"
    runner.store.prune_to_task_capacities(
        {task_id: 2 for task_id in TASK_IDS}, recovery_archive=archive
    )

    resumed = _runner(tmp_path, config)
    resumed.resume_latest()

    assert resumed.store.manifest == snapshot
    assert resumed.causality_store.manifest == holdout_snapshot
    assert not extra_path.exists()
    assert not extra_holdout_path.exists()
    assert not archive.exists()
    assert len(resumed.records) == len(result.records)
    assert resumed.completed_cycles == 2
    recovery = json.loads(
        (tmp_path / "run/recovery/last-resume.json").read_text()
    )
    assert recovery["restored_archived_shards"] == 1
    assert len(recovery["discarded_uncheckpointed_shards"]) == 2


def test_resume_restores_next_torch_random_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "hwr.train.foundation_admission.evaluate_foundation_action_causality_audit",
        lambda trainer, batches, criteria, shuffle_seed, shuffle_repeats: _diagnostic(True),
    )
    monkeypatch.setattr(
        "hwr.train.foundation_admission.evaluate_foundation_data_action_probe",
        _passing_action_probe,
    )
    torch.manual_seed(101)
    config = _config(episodes=6)
    runner = _runner(tmp_path, config)
    runner.train()
    expected = torch.rand(16)

    torch.manual_seed(999)
    resumed = _runner(tmp_path, config)
    resumed.resume_latest()

    torch.testing.assert_close(torch.rand(16), expected)
