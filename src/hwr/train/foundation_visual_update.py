"""Memory-bounded visual distillation update for the unified trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from hwr.perception.student import VisualStudentModel
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualTeacherTargets,
)
from hwr.train.accelerator_memory import release_unused_accelerator_memory
from hwr.train.foundation_batch import FoundationTrainingBatch


@dataclass(frozen=True)
class VisualUpdateResult:
    pooled_state: torch.Tensor
    losses: Mapping[str, float]
    microbatch_count: int
    gradient_norm: float
    deployment_gradient_norms: Mapping[str, float]


def optimize_visual_student(
    student: VisualStudentModel,
    objective: VisualFoundationObjectives,
    batch: FoundationTrainingBatch,
    optimizer: torch.optim.Optimizer,
    *,
    microbatch_observations: int,
    maximum_gradient_norm: float,
) -> VisualUpdateResult:
    """Accumulate one visual update without retaining every activation at once."""
    if min(microbatch_observations, maximum_gradient_norm) <= 0:
        raise ValueError("visual microbatch and gradient norm must be positive")
    if batch.visual_targets is None:
        raise ValueError("visual update requires frozen teacher targets")
    observation_total = batch.sequence_batch_size * batch.observation_count
    optimizer.zero_grad(set_to_none=True)
    pooled: list[torch.Tensor] = []
    aggregate: dict[str, float] = {}
    microbatch_count = 0
    for start in range(0, observation_total, microbatch_observations):
        stop = min(start + microbatch_observations, observation_total)
        weight = (stop - start) / observation_total
        inputs = {
            name: value[start:stop]
            for name, value in batch.student_inputs.items()
        }
        output = student(inputs)
        targets = _slice_targets(batch.visual_targets, start, stop)
        losses = objective(output, targets)
        (losses["total"] * weight).backward()
        pooled.append(output.pooled_state.detach())
        for name, value in losses.items():
            aggregate[name] = aggregate.get(name, 0.0) + float(value.detach()) * weight
        microbatch_count += 1
        del inputs, output, targets, losses
        if stop < observation_total:
            release_unused_accelerator_memory()
    parameters = [*student.parameters(), *objective.parameters()]
    deployment_gradient_norms = {
        name: _module_gradient_norm(module)
        for name, module in (
            ("camera_fusion", student.camera_fusion),
            ("temporal_fusion", student.temporal_fusion),
            ("output_norm", student.output_norm),
        )
    }
    gradient_norm = nn.utils.clip_grad_norm_(parameters, maximum_gradient_norm)
    optimizer.step()
    return VisualUpdateResult(
        torch.cat(pooled, dim=0),
        aggregate,
        microbatch_count,
        float(gradient_norm.detach().cpu()),
        deployment_gradient_norms,
    )


def encode_visual_student_bounded(
    student: VisualStudentModel,
    batch: FoundationTrainingBatch,
    *,
    microbatch_observations: int,
) -> VisualUpdateResult:
    """Encode a world-model batch without teacher I/O or visual gradients."""
    if microbatch_observations <= 0:
        raise ValueError("visual inference microbatch must be positive")
    observation_total = batch.sequence_batch_size * batch.observation_count
    training = student.training
    student.eval()
    pooled: list[torch.Tensor] = []
    microbatch_count = 0
    try:
        with torch.no_grad():
            for start in range(0, observation_total, microbatch_observations):
                stop = min(start + microbatch_observations, observation_total)
                inputs = {
                    name: value[start:stop]
                    for name, value in batch.student_inputs.items()
                }
                pooled.append(student(inputs).pooled_state.detach())
                microbatch_count += 1
                if stop < observation_total:
                    release_unused_accelerator_memory()
    finally:
        student.train(training)
    return VisualUpdateResult(
        torch.cat(pooled, dim=0), {}, microbatch_count, 0.0, {}
    )


def _module_gradient_norm(module: nn.Module) -> float:
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt().cpu())


def _slice_targets(
    targets: VisualTeacherTargets, start: int, stop: int
) -> VisualTeacherTargets:
    correspondences = targets.correspondences
    selected = (
        (correspondences[:, 0] >= start)
        & (correspondences[:, 0] < stop)
        & (correspondences[:, 5] >= start)
        & (correspondences[:, 5] < stop)
    )
    local_correspondences = correspondences[selected].clone()
    local_correspondences[:, 0] -= start
    local_correspondences[:, 5] -= start
    return VisualTeacherTargets(
        targets.vision_language[start:stop],
        targets.vision_language_valid[start:stop],
        targets.dense_vision[start:stop],
        targets.dense_vision_valid[start:stop],
        targets.rgb[start:stop],
        targets.reconstruction_mask[start:stop],
        targets.head_depth_m[start:stop],
        targets.head_depth_valid[start:stop],
        local_correspondences,
    )
