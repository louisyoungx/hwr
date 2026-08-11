from __future__ import annotations

import torch
import pytest

from hwr.policy import VLAActorConfig, VLAActorModel
from hwr.train.visual_self_supervision import temporal_visual_contrastive_loss
from tests.test_asymmetric_rl import _actor_inputs


def _actor() -> VLAActorModel:
    return VLAActorModel(
        VLAActorConfig(
            visual_history=2,
            action_history=2,
            proprioception_dim=37,
            language_dim=12,
            point_count=8,
            action_chunk_size=1,
            hidden_dim=32,
            attention_heads=4,
            transformer_layers=1,
        )
    )


def test_temporal_visual_contrast_uses_only_actor_inputs_and_updates_encoder() -> None:
    actor = _actor()
    target = _actor().eval()
    current = _actor_inputs(batch=4)
    successor = _actor_inputs(batch=4)

    loss = temporal_visual_contrastive_loss(
        actor,
        target,
        current,
        successor,
        torch.ones(4),
        temperature=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss > 0.0
    assert torch.count_nonzero(actor.head_encoder.network[0].weight.grad) > 0


def test_temporal_visual_contrast_excludes_non_actor_rows() -> None:
    loss = temporal_visual_contrastive_loss(
        _actor(),
        _actor().eval(),
        _actor_inputs(batch=4),
        _actor_inputs(batch=4),
        torch.zeros(4),
        temperature=0.1,
    )

    assert loss == 0.0


def test_visual_self_supervision_rejects_privileged_fields() -> None:
    inputs = _actor_inputs(batch=2)
    inputs["desired_goal"] = torch.zeros(2, 12)

    with pytest.raises(ValueError, match="non-deployment"):
        _actor().visual_representation(inputs)
