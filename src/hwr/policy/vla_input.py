"""Deployment-only visual-language-action Actor input assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hwr.core.embodied import DUAL_ARM_ACTION_DIM, FrozenLanguageEmbedding
from hwr.perception.contracts import ProcessedVision


VLA_POLICY_INPUT_SCHEMA = "hwr.vla-actor-input/v1"
VLA_POLICY_INPUT_FIELDS = frozenset(
    {
        "head_rgb",
        "head_depth",
        "head_depth_valid",
        "head_points",
        "head_point_valid",
        "wrist_rgb",
        "camera_validity",
        "proprioception",
        "instruction_embedding",
        "action_history",
    }
)


@dataclass(frozen=True)
class VLAActorInput:
    head_rgb: np.ndarray
    head_depth: np.ndarray
    head_depth_valid: np.ndarray
    head_points: np.ndarray
    head_point_valid: np.ndarray
    wrist_rgb: np.ndarray
    camera_validity: np.ndarray
    proprioception: np.ndarray
    instruction_embedding: np.ndarray
    action_history: np.ndarray
    preprocess_fingerprint: str
    language_encoder_id: str
    language_weights_sha256: str
    schema_version: str = VLA_POLICY_INPUT_SCHEMA

    def named_arrays(self) -> dict[str, np.ndarray]:
        values = {name: getattr(self, name) for name in VLA_POLICY_INPUT_FIELDS}
        if frozenset(values) != VLA_POLICY_INPUT_FIELDS:
            raise ValueError("VLA Actor input whitelist changed unexpectedly")
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("VLA Actor input contains non-finite values")
        return values


def build_vla_actor_input(
    visual_history: Sequence[ProcessedVision],
    language: FrozenLanguageEmbedding,
    *,
    proprioception: Sequence[float],
    action_history: Sequence[Sequence[float]],
) -> VLAActorInput:
    """Build tensors without carrying task ids, stages, truth, or symbolic plans."""
    history = tuple(visual_history)
    if not history:
        raise ValueError("VLA Actor requires visual history")
    fingerprints = {value.preprocess_fingerprint for value in history}
    if len(fingerprints) != 1:
        raise ValueError("visual history uses different preprocessing configurations")
    proprio = np.asarray(proprioception, dtype=np.float32)
    actions = np.asarray(action_history, dtype=np.float32)
    if proprio.ndim != 1 or not len(proprio) or not np.isfinite(proprio).all():
        raise ValueError("proprioception must be a finite vector")
    if (
        actions.ndim != 2
        or actions.shape[1] != DUAL_ARM_ACTION_DIM
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            f"action history must have shape (history, {DUAL_ARM_ACTION_DIM})"
        )
    return VLAActorInput(
        head_rgb=np.stack([value.head_rgb for value in history]),
        head_depth=np.stack([value.head_depth for value in history]),
        head_depth_valid=np.stack([value.head_depth_valid for value in history]),
        head_points=np.stack([value.head_points for value in history]),
        head_point_valid=np.stack([value.head_point_valid for value in history]),
        wrist_rgb=np.stack([value.wrist_rgb for value in history]),
        camera_validity=np.stack([value.camera_validity for value in history]),
        proprioception=proprio,
        instruction_embedding=np.asarray(language.values, dtype=np.float32),
        action_history=actions,
        preprocess_fingerprint=next(iter(fingerprints)),
        language_encoder_id=language.encoder_id,
        language_weights_sha256=language.weights_sha256,
    )
