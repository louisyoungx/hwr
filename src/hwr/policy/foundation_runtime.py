"""Deployable visual-world-model Actor without teachers, Critics, or rewards."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from hwr.core.embodied import ActionChunk, DualArmAction, DualArmObservation
from hwr.core.runtime import PolicySpec
from hwr.perception.high_resolution import HighResolutionVisionPreprocessor
from hwr.perception.language_cache import LanguageFeatureResolver
from hwr.perception.student import VisualStudentModel
from hwr.perception.student_input import (
    VisualStudentInputAssembler,
    visual_student_tensors,
)
from hwr.policy.latent_actions import LatentActionScaling, scale_latent_action
from hwr.policy.latent_actor import LatentActor
from hwr.world_model.deploy import DeployableWorldModelStateFilter
from hwr.world_model.rssm import RSSMState


class FoundationWorldModelPolicy:
    """Stateful posterior filtering followed by deterministic learned action."""

    def __init__(
        self,
        visual_student: VisualStudentModel,
        world_model: DeployableWorldModelStateFilter,
        actor: LatentActor,
        preprocessor: HighResolutionVisionPreprocessor,
        language_resolver: LanguageFeatureResolver,
        action_scaling: LatentActionScaling,
        *,
        policy_id: str,
        control_hz: float = 20.0,
        device: str = "cpu",
    ) -> None:
        if not policy_id or control_hz <= 0.0:
            raise ValueError("foundation deployment policy identity is invalid")
        if visual_student.config.feature_dimension != world_model.config.visual_dimension:
            raise ValueError("deployment visual and world model dimensions differ")
        if world_model.config.feature_dimension != actor.config.latent_dimension:
            raise ValueError("deployment world model and Actor dimensions differ")
        if world_model.config.language_dimension != language_resolver.output_dimension:
            raise ValueError("deployment language and world model dimensions differ")
        if visual_student.config.image_size != preprocessor.config.student_image_size:
            raise ValueError("deployment visual student and preprocessing sizes differ")
        self.device = torch.device(device)
        self.visual_student = visual_student.to(self.device).eval()
        self.world_model = world_model.to(self.device).eval()
        self.actor = actor.to(self.device).eval()
        self.preprocessor = preprocessor
        self.language_resolver = language_resolver
        self.action_scaling = action_scaling
        self.policy_id = policy_id
        self.control_hz = float(control_hz)
        self.input_assembler = VisualStudentInputAssembler(
            visual_history=visual_student.config.visual_history,
            image_size=visual_student.config.image_size,
        )
        self._task_id: str | None = None
        self._state: RSSMState | None = None
        self._pending_action: torch.Tensor | None = None
        self._awaiting_feedback = False

    def spec(self) -> PolicySpec:
        return PolicySpec(
            self.policy_id,
            self.visual_student.config.visual_history,
            1,
            self.control_hz,
            12,
            required_features=("head_rgb", "head_depth", "left_wrist_rgb", "right_wrist_rgb"),
        )

    def reset(self, *, task_id: str, seed: int) -> None:
        del seed
        if not task_id:
            raise ValueError("foundation deployment task identity is required")
        self.input_assembler.reset()
        self._task_id = task_id
        self._state = None
        self._pending_action = None
        self._awaiting_feedback = False

    def infer(self, observations: Sequence[DualArmObservation]) -> ActionChunk:
        return self._infer(observations, deterministic=True)

    def infer_stochastic(
        self, observations: Sequence[DualArmObservation]
    ) -> ActionChunk:
        """Sample only from the current RL Actor during autonomous collection."""
        return self._infer(observations, deterministic=False)

    def _infer(
        self,
        observations: Sequence[DualArmObservation],
        *,
        deterministic: bool,
    ) -> ActionChunk:
        if self._task_id is None:
            raise RuntimeError("foundation deployment policy must be reset")
        if self._awaiting_feedback:
            raise RuntimeError("actual executed action feedback is required before inference")
        if not observations or observations[-1].task_id != self._task_id:
            raise ValueError("foundation deployment observation task differs from reset")
        observation = observations[-1]
        frame = self.preprocessor.preprocess(observation)
        student_input = self.input_assembler.build(frame)
        tensors = visual_student_tensors(student_input, device=self.device)
        language = self.language_resolver.resolve(
            observation.instruction.text, observation.instruction.locale
        )
        if language.encoder_lock_sha256 != self.language_resolver.encoder_lock_sha256:
            raise ValueError("deployment language resolver returned a different encoder")
        language_tensor = torch.from_numpy(language.values.copy())[None].to(self.device)
        proprioception = torch.tensor(
            observation.proprioception.vector(), dtype=torch.float32, device=self.device
        )[None]
        with torch.inference_mode():
            visual = self.visual_student(tensors).pooled_state
            self._state = self.world_model.posterior_step(
                visual,
                language_tensor,
                proprioception,
                previous=self._state,
                executed_action=self._pending_action,
                sample=False,
            )
            latent = self.world_model.features(self._state)
            normalized = (
                self.actor.deterministic(latent)
                if deterministic
                else self.actor.sample(latent).action
            )
            vector = scale_latent_action(normalized, self.action_scaling)[0].cpu()
        self._pending_action = None
        self._awaiting_feedback = True
        return ActionChunk((DualArmAction.from_vector(vector.tolist()),), 1)

    def record_applied_action(self, action: DualArmAction) -> None:
        if self._task_id is None or not self._awaiting_feedback:
            raise RuntimeError("foundation policy has no proposal awaiting feedback")
        self._pending_action = torch.tensor(
            action.vector(), dtype=torch.float32, device=self.device
        )[None]
        self._awaiting_feedback = False

    def close(self) -> None:
        self.input_assembler.reset()
        self._task_id = None
        self._state = None
        self._pending_action = None
        self._awaiting_feedback = False
