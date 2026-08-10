"""Local behavior-cloning warm start for the end-to-end VLA Actor."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hwr.data.vla_loading import LoadedVLADataset
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS
from hwr.policy.vla_model import VLAActorConfig, VLAActorModel
from hwr.policy.vla_runtime import VLA_INPUT_ORDER, VLANormalization, vla_input_tensors


@dataclass(frozen=True)
class VLABehaviorTrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    validation_fraction: float = 0.25
    seed: int = 0
    device: str = "auto"
    hidden_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 2

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.learning_rate) <= 0:
            raise ValueError("VLA behavior training values must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("VLA validation fraction must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VLABehaviorTrainingResult:
    model: VLAActorModel
    model_config: VLAActorConfig
    training_config: VLABehaviorTrainingConfig
    normalization: VLANormalization
    best_validation_loss: float
    history: tuple[dict[str, float], ...]
    device: str


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalization(dataset: LoadedVLADataset, indices: np.ndarray) -> VLANormalization:
    proprioception = dataset.inputs["proprioception"][indices].astype(np.float32)
    chunks = dataset.action_chunks[indices]
    valid = dataset.valid_steps[indices]
    actions = np.concatenate(
        [chunk[:steps] for chunk, steps in zip(chunks, valid, strict=True)], axis=0
    )
    return VLANormalization(
        proprioception_mean=tuple(float(value) for value in proprioception.mean(axis=0)),
        proprioception_std=tuple(
            float(value) for value in np.maximum(proprioception.std(axis=0), 0.02)
        ),
        action_mean=tuple(float(value) for value in actions.mean(axis=0)),
        action_std=tuple(float(value) for value in np.maximum(actions.std(axis=0), 0.05)),
    )


def _model_config(
    dataset: LoadedVLADataset,
    training: VLABehaviorTrainingConfig,
) -> VLAActorConfig:
    shapes = dataset.manifest["input_shapes"]
    return VLAActorConfig(
        visual_history=int(shapes["head_rgb"][0]),
        action_history=int(shapes["action_history"][0]),
        proprioception_dim=int(shapes["proprioception"][0]),
        language_dim=int(shapes["instruction_embedding"][0]),
        point_count=int(shapes["head_points"][1]),
        action_chunk_size=int(dataset.manifest["action_chunk_size"]),
        hidden_dim=training.hidden_dim,
        attention_heads=training.attention_heads,
        transformer_layers=training.transformer_layers,
    )


def _tensor_dataset(
    dataset: LoadedVLADataset,
    indices: np.ndarray,
    normalization: VLANormalization,
) -> TensorDataset:
    inputs = vla_input_tensors(dataset.inputs, normalization, indices)
    chunks = dataset.action_chunks[indices].astype(np.float32, copy=True)
    mean = np.asarray(normalization.action_mean, dtype=np.float32)
    standard_deviation = np.asarray(normalization.action_std, dtype=np.float32)
    chunks = (chunks - mean) / standard_deviation
    tensors = [inputs[name] for name in VLA_INPUT_ORDER]
    return TensorDataset(
        *tensors,
        torch.from_numpy(chunks),
        torch.from_numpy(dataset.valid_steps[indices].astype(np.int64)),
    )


def _unpack(batch: tuple[torch.Tensor, ...], device: torch.device):
    values = tuple(value.to(device) for value in batch)
    inputs = dict(zip(VLA_INPUT_ORDER, values[: len(VLA_INPUT_ORDER)], strict=True))
    return inputs, values[-2], values[-1]


def vla_behavior_loss(
    model: VLAActorModel,
    inputs: dict[str, torch.Tensor],
    target_chunks: torch.Tensor,
    valid_steps: torch.Tensor,
) -> torch.Tensor:
    output = model(inputs)
    chunk_size = output.action_chunks.shape[1]
    steps = torch.arange(chunk_size, device=valid_steps.device)[None]
    valid = steps < valid_steps[:, None]
    element_loss = nn.functional.smooth_l1_loss(
        output.action_chunks, target_chunks, reduction="none"
    ).mean(dim=2)
    action_loss = (element_loss * valid).sum() / valid.sum().clamp_min(1)
    stop_targets = steps == (valid_steps[:, None] - 1)
    stop_loss = nn.functional.binary_cross_entropy_with_logits(
        output.stop_logits, stop_targets.to(output.stop_logits.dtype)
    )
    adjacent = output.action_chunks[:, 1:] - output.action_chunks[:, :-1]
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    smoothness = (
        adjacent.square().mean(dim=2) * adjacent_valid
    ).sum() / adjacent_valid.sum().clamp_min(1)
    return action_loss + 0.1 * stop_loss + 0.01 * smoothness


def _evaluate(
    model: VLAActorModel, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs, chunks, valid = _unpack(batch, device)
            loss = vla_behavior_loss(model, inputs, chunks, valid)
            total += float(loss.cpu()) * chunks.shape[0]
            count += chunks.shape[0]
    return total / max(1, count)


def train_vla_behavior_cloning(
    dataset: LoadedVLADataset,
    config: VLABehaviorTrainingConfig,
) -> VLABehaviorTrainingResult:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device_name = _select_device(config.device)
    device = torch.device(device_name)
    train_indices, validation_indices = dataset.split_by_episode(
        validation_fraction=config.validation_fraction, seed=config.seed
    )
    normalization = _normalization(dataset, train_indices)
    model_config = _model_config(dataset, config)
    model = VLAActorModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        _tensor_dataset(dataset, train_indices, normalization),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        _tensor_dataset(dataset, validation_indices, normalization),
        batch_size=config.batch_size,
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(config.epochs):
        model.train()
        training_total = 0.0
        training_count = 0
        for batch in train_loader:
            inputs, chunks, valid = _unpack(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = vla_behavior_loss(model, inputs, chunks, valid)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_total += float(loss.detach().cpu()) * chunks.shape[0]
            training_count += chunks.shape[0]
        validation_loss = _evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": training_total / max(1, training_count),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return VLABehaviorTrainingResult(
        model=model,
        model_config=model_config,
        training_config=config,
        normalization=normalization,
        best_validation_loss=best_loss,
        history=tuple(history),
        device=device_name,
    )
