"""Stateful assembly of four-camera Actor inputs for training and deployment."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from hwr.core.embodied import DUAL_ARM_ACTION_DIM, DualArmAction, DualArmObservation
from hwr.perception import (
    CameraCalibration,
    DualArmVisionPreprocessor,
    FrozenNgramLanguageEncoder,
    PinholeIntrinsics,
    VisionPreprocessConfig,
)
from hwr.perception.contracts import DUAL_ARM_CAMERA_IDS, DualArmProcessedVision
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS, VLAActorInput, build_vla_actor_input


@dataclass(frozen=True)
class BimanualInputConfig:
    raw_width: int
    raw_height: int
    image_width: int = 32
    image_height: int = 24
    point_count: int = 32
    visual_history: int = 1
    action_history: int = 1
    camera_fovy_degrees: float = 66.0

    def __post_init__(self) -> None:
        dimensions = (
            self.raw_width,
            self.raw_height,
            self.image_width,
            self.image_height,
            self.point_count,
            self.visual_history,
            self.action_history,
        )
        if min(dimensions) <= 0 or not 1.0 < self.camera_fovy_degrees < 179.0:
            raise ValueError("bimanual Actor input configuration is invalid")


def default_four_camera_calibrations(
    config: BimanualInputConfig,
) -> dict[str, CameraCalibration]:
    focal = config.raw_height / (
        2 * math.tan(math.radians(config.camera_fovy_degrees) / 2)
    )
    intrinsics = PinholeIntrinsics(
        config.raw_width,
        config.raw_height,
        focal,
        focal,
        (config.raw_width - 1) / 2,
        (config.raw_height - 1) / 2,
    )
    identity = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    return {
        name: CameraCalibration("bimanual-default/v1", name, intrinsics, identity)
        for name in DUAL_ARM_CAMERA_IDS
    }


class BimanualActorInputPipeline:
    """Keep only deployable visual and action history between control steps."""

    def __init__(
        self,
        config: BimanualInputConfig,
        language_encoder: FrozenNgramLanguageEncoder,
    ) -> None:
        self.config = config
        self.language_encoder = language_encoder
        self.preprocessor = DualArmVisionPreprocessor(
            VisionPreprocessConfig(
                config.image_width,
                config.image_height,
                config.point_count,
            ),
            default_four_camera_calibrations(config),
        )
        self._vision: deque[DualArmProcessedVision] = deque(
            maxlen=config.visual_history
        )
        self._actions: deque[tuple[float, ...]] = deque(maxlen=config.action_history)

    def reset(self) -> None:
        self._vision.clear()
        self._actions.clear()

    def build(self, observation: DualArmObservation) -> VLAActorInput:
        vision = self.preprocessor.preprocess(observation)
        self._vision.append(vision)
        while len(self._vision) < self.config.visual_history:
            self._vision.appendleft(vision)
        while len(self._actions) < self.config.action_history:
            self._actions.appendleft((0.0,) * DUAL_ARM_ACTION_DIM)
        language = self.language_encoder.encode(observation.instruction)
        return build_vla_actor_input(
            tuple(self._vision),
            language,
            proprioception=observation.proprioception.vector(),
            action_history=tuple(self._actions),
        )

    def record_action(self, action: DualArmAction) -> None:
        self._actions.append(action.vector())


def actor_input_tensors(
    actor_input: VLAActorInput, *, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    arrays = actor_input.named_arrays()
    if frozenset(arrays) != VLA_POLICY_INPUT_FIELDS:
        raise ValueError("Actor input tensorization received non-deployment fields")
    return {
        name: torch.from_numpy(np.asarray(value, dtype=np.float32).copy())[None].to(device)
        for name, value in arrays.items()
    }


def stack_actor_inputs(inputs: list[VLAActorInput]) -> Mapping[str, torch.Tensor]:
    if not inputs:
        raise ValueError("cannot stack an empty Actor input sequence")
    return {
        name: torch.from_numpy(
            np.stack([value.named_arrays()[name] for value in inputs]).astype(
                np.float32, copy=False
            )
        )
        for name in VLA_POLICY_INPUT_FIELDS
    }
