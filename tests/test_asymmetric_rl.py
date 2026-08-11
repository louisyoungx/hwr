from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from hwr.data import load_vla_dataset
from hwr.policy import PrivilegedCriticConfig, VLAActorConfig, VLAActorModel
from hwr.policy.vla_actions import bounded_vla_actions
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS
from hwr.policy.vla_runtime import VLANormalization
from hwr.train import (
    AsymmetricActorCriticTrainer,
    AsymmetricRLBatch,
    AsymmetricRLConfig,
    AsymmetricReplayBuffer,
    load_asymmetric_training_checkpoint,
    load_deployable_vla_actor,
    save_asymmetric_training_checkpoint,
    save_vla_actor_checkpoint,
)
from hwr.train.asymmetric_rl import _motion_slew_penalty
from tests.vla_fixtures import actor_input, build_dataset


def _actor_inputs(batch: int = 4) -> dict[str, torch.Tensor]:
    values = {
        "head_rgb": torch.randn(batch, 2, 8, 8, 3),
        "head_depth": torch.rand(batch, 2, 8, 8),
        "head_depth_valid": torch.ones(batch, 2, 8, 8, dtype=torch.bool),
        "head_points": torch.randn(batch, 2, 8, 6),
        "head_point_valid": torch.ones(batch, 2, 8, dtype=torch.bool),
        "left_wrist_rgb": torch.randn(batch, 2, 8, 8, 3),
        "right_wrist_rgb": torch.randn(batch, 2, 8, 8, 3),
        "camera_validity": torch.ones(batch, 2, 4, dtype=torch.bool),
        "proprioception": torch.randn(batch, 37),
        "instruction_embedding": torch.randn(batch, 12),
        "action_history": torch.randn(batch, 2, 16),
    }
    assert frozenset(values) == VLA_POLICY_INPUT_FIELDS
    return values


def _batch() -> AsymmetricRLBatch:
    stop = torch.zeros(4, 3)
    stop[:, -1] = 1.0
    return AsymmetricRLBatch(
        actor_inputs=_actor_inputs(),
        next_actor_inputs=_actor_inputs(),
        privileged_state=torch.randn(4, 10),
        next_privileged_state=torch.randn(4, 10),
        action_chunks=torch.randn(4, 3, 16),
        stop_decisions=stop,
        rewards=torch.tensor((1.0, 0.2, -0.1, 0.5)),
        done=torch.tensor((1.0, 0.0, 0.0, 1.0)),
    )


def _trainer() -> AsymmetricActorCriticTrainer:
    actor = VLAActorModel(
        VLAActorConfig(
            visual_history=2,
            action_history=2,
            proprioception_dim=37,
            language_dim=12,
            point_count=8,
            action_chunk_size=3,
            hidden_dim=32,
            attention_heads=4,
            transformer_layers=1,
        )
    )
    return AsymmetricActorCriticTrainer(
        actor,
        PrivilegedCriticConfig(10, 3, hidden_dim=32),
        AsymmetricRLConfig(
            policy_delay=1,
            actor_warmup_updates=0,
            behavior_regularization=0.01,
        ),
    )


def test_asymmetric_update_changes_actor_using_separate_privileged_critic() -> None:
    trainer = _trainer()
    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]

    metrics = trainer.update(_batch())

    assert metrics["actor_updated"] == 1.0
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["conservative_loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["actor_reward_value"])
    assert np.isfinite(metrics["actor_safety_risk"])
    assert np.isfinite(metrics["reward_critic_disagreement"])
    assert np.isfinite(metrics["safety_critic_disagreement"])
    assert 0.0 <= metrics["actor_motion_mean_ratio"] <= 1.0
    assert 0.0 <= metrics["actor_motion_max_ratio"] <= 1.0
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, trainer.actor.parameters(), strict=True)
    )
    assert not any("critic" in name for name, _ in trainer.actor.named_parameters())


def test_actor_update_includes_label_free_temporal_visual_objective() -> None:
    trainer = _trainer()
    trainer.config = replace(
        trainer.config,
        visual_temporal_contrastive_weight=0.05,
    )

    metrics = trainer.update(_batch())

    assert metrics["visual_contrastive_loss"] > 0.0
    assert np.isfinite(metrics["visual_contrastive_loss"])


def test_actor_optimizes_the_pessimistic_twin_critic_value() -> None:
    class ConflictingCritic(torch.nn.Module):
        def forward(self, state, action):
            del state
            return torch.full_like(action[:, 0], 10.0), action[:, 0]

    trainer = _trainer()
    trainer.config = AsymmetricRLConfig(
        policy_delay=1,
        actor_warmup_updates=0,
        behavior_regularization=0.0,
        entropy_temperature=0.0,
        gripper_entropy_temperature=0.0,
        action_magnitude_penalty=0.0,
        action_slew_penalty=0.0,
        safety_actor_penalty=0.0,
    )
    trainer.critic = ConflictingCritic()
    batch = trainer._to_device(_batch())
    torch.manual_seed(17)
    with torch.no_grad():
        output = trainer.actor(batch.actor_inputs)
        expected = -trainer._sample_action(output).values[:, 0, 0].mean()

    torch.manual_seed(17)
    loss, metrics = trainer._update_actor(batch)

    assert float(loss.detach()) == pytest.approx(float(expected), abs=1e-6)
    assert metrics["actor_reward_value"] == pytest.approx(float(-expected))


def test_maximum_entropy_actor_and_action_regularization_are_enabled() -> None:
    config = _trainer().config
    defaults = AsymmetricRLConfig()

    assert config.entropy_temperature > 0
    assert 0 < config.gripper_entropy_temperature < config.entropy_temperature
    assert (
        config.minimum_motion_log_standard_deviation
        < config.initial_motion_log_standard_deviation
        < config.maximum_motion_log_standard_deviation
    )
    assert (
        config.minimum_gripper_log_standard_deviation
        < config.initial_gripper_log_standard_deviation
        < config.maximum_gripper_log_standard_deviation
    )
    assert (
        config.initial_gripper_log_standard_deviation
        > config.initial_motion_log_standard_deviation
    )
    assert config.action_magnitude_penalty > 0
    assert config.action_slew_penalty > 0
    assert config.reward_scale == 0.25
    assert config.conservative_critic_weight == 0.05
    assert config.conservative_action_samples > 1
    assert defaults.actor_learning_rate < defaults.critic_learning_rate
    assert defaults.final_actor_learning_rate < defaults.actor_learning_rate
    assert defaults.actor_learning_rate < defaults.safety_learning_rate
    assert defaults.policy_delay >= 5


def test_actor_uses_one_learning_rate_for_all_action_dimensions() -> None:
    trainer = _trainer()
    groups = trainer.actor_optimizer.param_groups
    parameter_ids = {id(parameter) for parameter in groups[0]["params"]}

    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(trainer.config.actor_learning_rate)
    assert id(trainer.actor.action_head.weight) in parameter_ids
    assert id(trainer.actor_log_standard_deviation) in parameter_ids


def test_actor_learning_rate_decays_after_calibrated_critic_updates() -> None:
    trainer = _trainer()
    trainer.update_count = trainer.config.actor_learning_rate_decay_updates - 1
    trainer._schedule_actor_learning_rate()

    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(
        trainer.config.actor_learning_rate
    )

    trainer.update_count += 1
    trainer._schedule_actor_learning_rate()

    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(
        trainer.config.final_actor_learning_rate
    )


def test_motion_slew_penalty_does_not_hold_grippers_open() -> None:
    previous = torch.zeros(2, 1, 16)
    bounded = torch.zeros(2, 1, 16)
    scales = torch.ones(16)
    bounded[..., 14:] = 1.0

    gripper_only = _motion_slew_penalty(previous, bounded, scales)
    bounded[..., 2] = 1.0
    with_motion = _motion_slew_penalty(previous, bounded, scales)

    assert torch.equal(gripper_only, torch.zeros(2))
    assert torch.all(with_motion > 0.0)


def test_stochastic_actor_samples_bounded_actions_and_finite_density() -> None:
    trainer = _trainer()
    inputs = _actor_inputs(batch=2)
    output = trainer.actor(inputs)

    first = trainer._sample_action(output)
    second = trainer._sample_action(output)
    deterministic = trainer._sample_action(output, deterministic=True)
    repeated = trainer._sample_action(output, deterministic=True)

    assert not torch.equal(first.values, second.values)
    assert torch.isfinite(first.log_probability).all()
    assert torch.equal(
        first.log_probability,
        first.motion_log_probability + first.gripper_log_probability,
    )
    assert torch.all(first.values[..., 0].abs() <= trainer.config.base_linear_scale)
    assert torch.all(first.values[..., 1].abs() <= trainer.config.base_angular_scale)
    assert torch.all(first.values[..., 2:14].abs() <= trainer.config.arm_velocity_scale)
    assert torch.all((0.0 <= first.values[..., 14:]) & (first.values[..., 14:] <= 1.0))
    assert torch.equal(deterministic.values, repeated.values)

    expanded = type(output)(
        output.action_chunks.repeat(1024, 1, 1),
        output.stop_logits.repeat(1024, 1),
    )
    grippers = trainer._sample_action(expanded).values[..., 14:]
    assert (grippers < 0.2).float().mean() > 0.01
    assert (grippers > 0.8).float().mean() > 0.01


def test_entropy_update_learns_action_distribution_scale() -> None:
    trainer = _trainer()
    before = trainer.actor_log_standard_deviation.detach().clone()

    metrics = trainer.update(_batch())

    assert metrics["actor_entropy"] != 0.0
    assert np.isfinite(metrics["actor_motion_log_standard_deviation"])
    assert np.isfinite(metrics["actor_gripper_log_standard_deviation"])
    assert not torch.equal(before, trainer.actor_log_standard_deviation)


def test_unsupervised_actor_does_not_optimize_unused_stop_head() -> None:
    trainer = _trainer()
    trainer.config = AsymmetricRLConfig(
        policy_delay=1,
        actor_warmup_updates=0,
        behavior_regularization=0.0,
    )
    before = [value.detach().clone() for value in trainer.actor.stop_head.parameters()]

    trainer.update(_batch())

    assert all(
        torch.equal(previous, current)
        for previous, current in zip(
            before, trainer.actor.stop_head.parameters(), strict=True
        )
    )
def test_actor_waits_for_critic_only_warmup() -> None:
    trainer = _trainer()
    trainer.config = AsymmetricRLConfig(
        policy_delay=1,
        actor_warmup_updates=2,
        behavior_regularization=0.0,
    )
    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]

    first = trainer.update(_batch())
    second = trainer.update(_batch())

    assert first["actor_updated"] == second["actor_updated"] == 0.0
    assert all(
        torch.equal(previous, current)
        for previous, current in zip(
            before, trainer.actor.parameters(), strict=True
        )
    )
    assert trainer.update(_batch())["actor_updated"] == 1.0


def test_privileged_safety_critic_learns_self_observed_interventions() -> None:
    trainer = _trainer()
    batch = _batch()
    batch = replace(
        batch,
        proposed_action_chunks=batch.action_chunks.clone(),
        safety_costs=torch.tensor((0.0, 1.0, 1.0, 0.0)),
    )
    before = [
        parameter.detach().clone()
        for parameter in trainer.safety_critic.parameters()
    ]

    metrics = trainer.update(batch)

    assert metrics["safety_loss"] > 0.0
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(
            before, trainer.safety_critic.parameters(), strict=True
        )
    )
    assert not any("safety" in name for name, _ in trainer.actor.named_parameters())


def test_asymmetric_actor_rejects_privileged_field_even_during_rl() -> None:
    trainer = _trainer()
    batch = _batch()
    actor_inputs = dict(batch.actor_inputs)
    actor_inputs["object_truth"] = batch.privileged_state
    leaky = AsymmetricRLBatch(
        actor_inputs=actor_inputs,
        next_actor_inputs=batch.next_actor_inputs,
        privileged_state=batch.privileged_state,
        next_privileged_state=batch.next_privileged_state,
        action_chunks=batch.action_chunks,
        stop_decisions=batch.stop_decisions,
        rewards=batch.rewards,
        done=batch.done,
    )

    with pytest.raises(ValueError, match="non-deployment"):
        trainer.update(leaky)


def test_asymmetric_training_state_restores_for_continuation() -> None:
    trainer = _trainer()
    trainer.update(_batch())
    state = copy.deepcopy(trainer.state_dict())
    restored = _trainer()

    restored.load_state_dict(state)

    assert restored.update_count == 1
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            trainer.actor.parameters(), restored.actor.parameters(), strict=True
        )
    )
    assert torch.equal(
        trainer.actor_log_standard_deviation,
        restored.actor_log_standard_deviation,
    )


def test_asymmetric_replay_and_training_checkpoint_resume(tmp_path) -> None:
    trainer = _trainer()
    replay = AsymmetricReplayBuffer(8, seed=7)
    source = replace(_batch(), bootstrap_discounts=torch.full((4,), 0.75))
    replay.add(source)
    replay_state = replay.state_dict()
    assert all(
        tensor.shape[0] == replay.size
        for tensor in replay_state["storage"].values()
    )
    trainer.update(replay.sample(3))
    path = save_asymmetric_training_checkpoint(
        tmp_path / "resume", trainer, replay, run_metadata={"scene": "kitchen"}
    )
    restored_trainer = _trainer()
    restored_replay = AsymmetricReplayBuffer(8, seed=99)

    manifest = load_asymmetric_training_checkpoint(
        path, restored_trainer, restored_replay
    )

    assert restored_trainer.update_count == trainer.update_count
    assert restored_replay.size == replay.size == 4
    assert torch.all(restored_replay.all().bootstrap_discounts == 0.75)
    assert manifest["replay_size"] == 4
    assert restored_replay.sample(2).action_chunks.shape == (2, 3, 16)


def test_rl_export_contains_actor_only_and_reloads(tmp_path) -> None:
    trainer = _trainer()
    trainer.update(_batch())
    dataset = load_vla_dataset(build_dataset(tmp_path / "datasets"))
    normalization = VLANormalization(
        proprioception_mean=(0.0,) * 37,
        proprioception_std=(1.0,) * 37,
        action_mean=(0.0,) * 16,
        action_std=(1.0,) * 16,
    )
    path = save_vla_actor_checkpoint(
        tmp_path / "models",
        "rl-actor",
        "v1",
        trainer.actor,
        normalization,
        dataset_manifest=dataset.manifest,
        training_metadata={
            "training_kind": "asymmetric_rl",
            "rl_updates": trainer.update_count,
        },
    )

    actor = load_deployable_vla_actor(path)
    chunk = actor.predict(actor_input(4))

    assert len(chunk.actions) == 3
    assert set(path.iterdir()) == {path / "actor.pt", path / "manifest.json"}
