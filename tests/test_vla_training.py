from __future__ import annotations

import json

import numpy as np
import pytest

from hwr.data import load_vla_dataset
from hwr.policy.vla_runtime import vla_input_tensors
from hwr.train import (
    VLABehaviorTrainingConfig,
    load_deployable_vla_actor,
    save_vla_behavior_result,
    train_vla_behavior_cloning,
)
from tests.vla_fixtures import actor_input, build_dataset


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
