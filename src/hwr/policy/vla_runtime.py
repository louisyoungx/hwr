"""Normalization and deployment wrapper for the VLA Actor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from hwr.core.embodied import ActionChunk, DualArmAction
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS, VLAActorInput
from hwr.policy.vla_model import VLAActorModel


VLA_INPUT_ORDER = tuple(sorted(VLA_POLICY_INPUT_FIELDS))


@dataclass(frozen=True)
class VLANormalization:
    proprioception_mean: tuple[float, ...]
    proprioception_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "VLANormalization":
        return cls(
            proprioception_mean=tuple(value["proprioception_mean"]),
            proprioception_std=tuple(value["proprioception_std"]),
            action_mean=tuple(value["action_mean"]),
            action_std=tuple(value["action_std"]),
        )


def vla_input_tensors(
    inputs: Mapping[str, np.ndarray],
    normalization: VLANormalization,
    indices: np.ndarray | slice | None = None,
) -> dict[str, torch.Tensor]:
    if frozenset(inputs) != VLA_POLICY_INPUT_FIELDS:
        raise ValueError("VLA tensor conversion received non-deployment fields")
    select = slice(None) if indices is None else indices
    values = {
        name: torch.from_numpy(inputs[name][select].astype(np.float32, copy=True))
        for name in VLA_INPUT_ORDER
    }
    mean = np.asarray(normalization.proprioception_mean, dtype=np.float32)
    standard_deviation = np.asarray(normalization.proprioception_std, dtype=np.float32)
    values["proprioception"] = torch.from_numpy(
        (inputs["proprioception"][select] - mean) / standard_deviation
    ).to(torch.float32)
    return values


class DeployableVLAActor:
    def __init__(
        self,
        model: VLAActorModel,
        normalization: VLANormalization,
        *,
        preprocess_fingerprint: str,
        language_encoder_id: str,
        language_weights_sha256: str,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.normalization = normalization
        self.preprocess_fingerprint = preprocess_fingerprint
        self.language_encoder_id = language_encoder_id
        self.language_weights_sha256 = language_weights_sha256
        self.device = torch.device(device)

    def predict(self, actor_input: VLAActorInput) -> ActionChunk:
        self._check_identity(actor_input)
        arrays = {name: value[None] for name, value in actor_input.named_arrays().items()}
        tensors = {
            name: value.to(self.device)
            for name, value in vla_input_tensors(arrays, self.normalization).items()
        }
        with torch.inference_mode():
            output = self.model(tensors)
        normalized = output.action_chunks[0].cpu().numpy()
        mean = np.asarray(self.normalization.action_mean, dtype=np.float32)
        standard_deviation = np.asarray(self.normalization.action_std, dtype=np.float32)
        vectors = normalized * standard_deviation + mean
        vectors[:, 14:] = np.clip(vectors[:, 14:], 0.0, 1.0)
        stop = torch.sigmoid(output.stop_logits[0]).cpu().numpy()
        candidates = np.flatnonzero(stop >= 0.5)
        valid_steps = int(candidates[0] + 1) if len(candidates) else len(vectors)
        actions = tuple(DualArmAction.from_vector(vector) for vector in vectors)
        return ActionChunk(actions, valid_steps)

    def _check_identity(self, actor_input: VLAActorInput) -> None:
        actual = (
            actor_input.preprocess_fingerprint,
            actor_input.language_encoder_id,
            actor_input.language_weights_sha256,
        )
        expected = (
            self.preprocess_fingerprint,
            self.language_encoder_id,
            self.language_weights_sha256,
        )
        if actual != expected:
            raise ValueError("Actor input preprocessing or language weights differ from checkpoint")
