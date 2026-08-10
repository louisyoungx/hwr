"""Deployable runtime wrapper for the no-demonstration bimanual Actor."""

from __future__ import annotations

from typing import Sequence

import torch

from hwr.core.embodied import ActionChunk, DualArmAction, DualArmObservation
from hwr.core.runtime import PolicySpec
from hwr.perception import FrozenNgramLanguageEncoder
from hwr.policy.bimanual_input import (
    BimanualActorInputPipeline,
    BimanualInputConfig,
    actor_input_tensors,
)
from hwr.policy.vla_actions import VLAActionScaling, bounded_vla_actions
from hwr.policy.vla_model import VLAActorModel


class BimanualVLAActorPolicy:
    """Stateful observation preprocessing and Actor inference for deployment."""

    def __init__(
        self,
        model: VLAActorModel,
        input_config: BimanualInputConfig,
        language_encoder: FrozenNgramLanguageEncoder,
        action_scaling: VLAActionScaling,
        *,
        policy_id: str,
        preprocess_fingerprint: str,
        control_hz: float = 20.0,
        device: str = "cpu",
    ) -> None:
        if not policy_id or control_hz <= 0:
            raise ValueError("bimanual deployment policy identity is invalid")
        self.model = model.to(device).eval()
        self.pipeline = BimanualActorInputPipeline(
            input_config, language_encoder
        )
        if self.pipeline.preprocessor.fingerprint != preprocess_fingerprint:
            raise ValueError("deployment visual preprocessing differs from checkpoint")
        if model.config.visual_history != input_config.visual_history:
            raise ValueError("deployment visual history differs from Actor")
        if model.config.action_history != input_config.action_history:
            raise ValueError("deployment action history differs from Actor")
        if model.config.language_dim != language_encoder.config.dimension:
            raise ValueError("deployment language dimensions differ from Actor")
        self.action_scaling = action_scaling
        self.policy_id = policy_id
        self.control_hz = float(control_hz)
        self.device = torch.device(device)
        self._task_id: str | None = None

    def spec(self) -> PolicySpec:
        return PolicySpec(
            self.policy_id,
            self.model.config.visual_history,
            self.model.config.action_chunk_size,
            self.control_hz,
            12,
        )

    def reset(self, *, task_id: str, seed: int) -> None:
        del seed
        if not task_id:
            raise ValueError("deployment task identity is required")
        self.pipeline.reset()
        self._task_id = task_id

    def infer(self, observations: Sequence[DualArmObservation]) -> ActionChunk:
        if self._task_id is None:
            raise RuntimeError("deployment policy must be reset before inference")
        if not observations or observations[-1].task_id != self._task_id:
            raise ValueError("deployment observation task differs from policy reset")
        actor_input = self.pipeline.build(observations[-1])
        tensors = actor_input_tensors(actor_input, device=self.device)
        with torch.inference_mode():
            output = self.model(tensors)
            vectors = bounded_vla_actions(output, self.action_scaling)[0].cpu()
        actions = tuple(
            DualArmAction.from_vector(vector.tolist()) for vector in vectors
        )
        return ActionChunk(actions, len(actions))

    def record_applied_action(self, action: DualArmAction) -> None:
        if self._task_id is None:
            raise RuntimeError("deployment policy must be reset before feedback")
        self.pipeline.record_action(action)

    def close(self) -> None:
        self._task_id = None
        self.pipeline.reset()
