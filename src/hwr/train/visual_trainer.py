"""Deterministic local trainer for the formal multi-view visual policy."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hwr.data.visual_loading import LoadedVisualDataset
from hwr.policy.visual_model import (
    HouseholdVisualPolicyModel,
    VisualModelConfig,
    VisualModelOutput,
)
from hwr.policy.visual_policy import VisualNormalization, visual_input_tensors


@dataclass(frozen=True)
class VisualTrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.25
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size) <= 0 or self.learning_rate <= 0:
            raise ValueError("visual training configuration must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class VisualTrainingResult:
    model: HouseholdVisualPolicyModel
    normalization: VisualNormalization
    model_config: VisualModelConfig
    training_config: VisualTrainingConfig
    history: list[dict[str, float]]
    best_validation_loss: float
    device: str
    phase_action_mask: tuple[tuple[bool, ...], ...]
    phase_step_limits: tuple[tuple[int, int], ...]


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalization(dataset: LoadedVisualDataset, indices: np.ndarray) -> VisualNormalization:
    proprioception = dataset.inputs["proprioception"][indices]
    actions = dataset.actions[indices, :8]
    return VisualNormalization(
        proprioception_mean=tuple(float(value) for value in proprioception.mean(axis=0)),
        proprioception_std=tuple(
            float(value) for value in np.maximum(proprioception.std(axis=0), 0.02)
        ),
        action_mean=tuple(float(value) for value in actions.mean(axis=0)),
        action_std=tuple(float(value) for value in np.maximum(actions.std(axis=0), 0.05)),
    )


def _tensor_dataset(
    dataset: LoadedVisualDataset,
    indices: np.ndarray,
    normalization: VisualNormalization,
) -> TensorDataset:
    inputs = visual_input_tensors(dataset.inputs, normalization, indices)
    actions = dataset.actions[indices].astype(np.float32, copy=True)
    mean = np.asarray(normalization.action_mean, dtype=np.float32)
    std = np.asarray(normalization.action_std, dtype=np.float32)
    actions[:, :8] = (actions[:, :8] - mean) / std
    phases = dataset.phase_indices[indices].astype(np.int64, copy=True)
    return TensorDataset(*inputs, torch.from_numpy(actions), torch.from_numpy(phases))


def _phase_step_limits(
    dataset: LoadedVisualDataset,
) -> tuple[tuple[int, int], ...]:
    stride = int(dataset.manifest.get("metadata", {}).get("sample_stride", 1))
    episodes = np.unique(dataset.episode_ids)
    limits = []
    for phase in range(len(dataset.phase_names)):
        durations = []
        for episode in episodes:
            mask = (dataset.episode_ids == episode) & (dataset.phase_indices == phase)
            steps = dataset.step_indices[mask]
            if len(steps):
                durations.append(int(steps[-1] - steps[0] + stride))
        if not durations:
            raise ValueError(f"phase {phase} has no episode timing evidence")
        minimum = max(1, int(min(durations) * 0.85))
        maximum = max(minimum + 1, int(max(durations) * 1.25 + 0.5))
        limits.append((minimum, maximum))
    return tuple(limits)


def _loss(
    output: VisualModelOutput,
    target: torch.Tensor,
    phases: torch.Tensor,
    phase_weights: torch.Tensor,
) -> torch.Tensor:
    batch_indices = torch.arange(phases.shape[0], device=phases.device)
    prediction = output.actions[batch_indices, phases]
    continuous = nn.functional.smooth_l1_loss(
        prediction[:, :8].contiguous(), target[:, :8].contiguous()
    )
    gripper = nn.functional.binary_cross_entropy_with_logits(
        prediction[:, 8], target[:, 8]
    )
    phase = nn.functional.cross_entropy(output.phase_logits, phases, weight=phase_weights)
    return continuous + 0.30 * gripper + 0.50 * phase


def _evaluate(
    model: HouseholdVisualPolicyModel,
    loader: DataLoader,
    device: torch.device,
    phase_weights: torch.Tensor,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            values = tuple(item.to(device) for item in batch)
            loss = _loss(model(*values[:-2]), values[-2], values[-1], phase_weights)
            total += float(loss.cpu()) * values[-2].shape[0]
            count += values[-2].shape[0]
    return total / max(1, count)


def train_visual_policy(
    dataset: LoadedVisualDataset,
    config: VisualTrainingConfig,
) -> VisualTrainingResult:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device_name = _select_device(config.device)
    device = torch.device(device_name)
    train_indices, validation_indices = dataset.split_by_episode(
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    normalization = _normalization(dataset, train_indices)
    train_data = _tensor_dataset(dataset, train_indices, normalization)
    validation_data = _tensor_dataset(dataset, validation_indices, normalization)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size)
    image_width, image_height = dataset.manifest["image_size"]
    model_config = VisualModelConfig(
        image_width=int(image_width),
        image_height=int(image_height),
        action_history=int(dataset.manifest["action_history"]),
        phase_count=len(dataset.phase_names),
    )
    model = HouseholdVisualPolicyModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    phase_counts = np.bincount(
        dataset.phase_indices[train_indices], minlength=len(dataset.phase_names)
    )
    phase_weights = torch.from_numpy(
        np.sqrt(
            phase_counts.sum() / np.maximum(phase_counts, 1) / len(phase_counts)
        ).astype(np.float32)
    ).to(device)
    phase_action_mask = tuple(
        tuple(
            bool(value)
            for value in np.max(
                np.abs(
                    dataset.actions[
                        train_indices[
                            dataset.phase_indices[train_indices] == phase_index
                        ],
                        :8,
                    ]
                ),
                axis=0,
            )
            > 1e-4
        )
        for phase_index in range(len(dataset.phase_names))
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(config.epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            values = tuple(item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(
                model(*values[:-2]), values[-2], values[-1], phase_weights
            )
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu()) * values[-2].shape[0]
            train_count += values[-2].shape[0]
        validation_loss = _evaluate(model, validation_loader, device, phase_weights)
        train_loss = train_total / max(1, train_count)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("visual training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu")
    return VisualTrainingResult(
        model,
        normalization,
        model_config,
        config,
        history,
        best_loss,
        device_name,
        phase_action_mask,
        _phase_step_limits(dataset),
    )
