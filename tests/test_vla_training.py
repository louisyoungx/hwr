from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from hwr.data import load_vla_dataset
from hwr.policy.bimanual_input import actor_input_tensors
from hwr.policy.vla_model import VLAActorConfig, VLAActorModel
from hwr.policy.vla_runtime import vla_input_tensors
from hwr.train import (
    VLABehaviorTrainingConfig,
    load_deployable_vla_actor,
    save_vla_behavior_result,
    train_vla_behavior_cloning,
)
from tests.vla_fixtures import actor_input, build_dataset


def test_vla_actor_action_head_starts_randomized_around_zero() -> None:
    scale = 1.0e-3
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
            action_head_init_scale=scale,
        )
    )

    weights = actor.action_head.weight.detach()
    bias = actor.action_head.bias.detach()
    assert torch.count_nonzero(weights) > 0
    assert weights.abs().max() <= scale
    assert abs(float(weights.mean())) < scale / 4.0
    assert torch.count_nonzero(bias) == 0


def test_isolated_gripper_head_does_not_backpropagate_into_shared_motion() -> None:
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
            isolated_gripper_head=True,
        )
    )

    output = actor(actor_input_tensors(actor_input(2)))
    output.action_chunks[..., 14:].sum().backward()

    assert actor.gripper_head.weight.grad is not None
    assert torch.count_nonzero(actor.motion_head.weight.grad) == 0
    assert torch.count_nonzero(actor.output_norm.weight.grad) == 0
    assert torch.count_nonzero(actor.head_encoder.network[0].weight.grad) == 0


@pytest.mark.parametrize("scale", [0.0, float("inf"), 0.02])
def test_vla_actor_rejects_invalid_action_head_initialization(scale: float) -> None:
    with pytest.raises(ValueError, match="initialization scale"):
        VLAActorConfig(2, 2, 37, 12, 8, 3, action_head_init_scale=scale)


def test_vla_behavior_clone_saves_reloads_and_predicts_action_chunk(tmp_path) -> None:
    dataset = load_vla_dataset(build_dataset(tmp_path / "datasets"))
    result = train_vla_behavior_cloning(
        dataset,
        VLABehaviorTrainingConfig(
            epochs=2,
            batch_size=4,
            device="cpu",
            hidden_dim=32,
            attention_heads=4,
            transformer_layers=1,
        ),
    )
    path = save_vla_behavior_result(
        tmp_path / "models",
        "housework-vla",
        "v1",
        result,
        dataset_manifest=dataset.manifest,
    )
    actor = load_deployable_vla_actor(path)

    chunk = actor.predict(actor_input(9))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    assert len(result.history) == 2
    assert np.isfinite(result.best_validation_loss)
    assert len(chunk.actions) == 3
    assert 1 <= chunk.valid_steps <= 3
    assert len(chunk.actions[0].vector()) == 16
    assert not any("critic" in key.lower() for key in manifest)
    assert set(path.iterdir()) == {path / "actor.pt", path / "manifest.json"}


def test_vla_actor_rejects_privileged_tensor_field(tmp_path) -> None:
    dataset = load_vla_dataset(build_dataset(tmp_path / "datasets"))
    result = train_vla_behavior_cloning(
        dataset,
        VLABehaviorTrainingConfig(
            epochs=1,
            batch_size=6,
            device="cpu",
            hidden_dim=32,
            attention_heads=4,
            transformer_layers=1,
        ),
    )
    inputs = dict(dataset.inputs)
    inputs["object_truth"] = np.zeros((len(dataset), 3), dtype=np.float32)

    with pytest.raises(ValueError, match="non-deployment"):
        vla_input_tensors(inputs, result.normalization)
