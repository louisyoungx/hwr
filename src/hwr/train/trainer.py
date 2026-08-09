"""Deterministic local trainer for the reference behavior cloning policy."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hwr.data.dataset import BehaviorDataset
from hwr.policy.model import BehaviorMLP, ModelConfig
from hwr.policy.neural import Normalization


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    seed: int = 0
    hidden_dims: tuple[int, ...] = (128, 128)
    device: str = "auto"

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size) <= 0 or self.learning_rate <= 0:
            raise ValueError("training epochs, batch size, and learning rate must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrainingResult:
    model: BehaviorMLP
    normalization: Normalization
    model_config: ModelConfig
    training_config: TrainingConfig
    history: list[dict[str, float]]
    best_validation_loss: float
    device: str


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalization(observations: np.ndarray, actions: np.ndarray) -> Normalization:
    observation_mean = observations.mean(axis=0)
    observation_std = np.maximum(observations.std(axis=0), 1e-6)
    continuous_actions = actions[:, :4]
    action_mean = continuous_actions.mean(axis=0)
    action_std = np.maximum(continuous_actions.std(axis=0), 1e-6)
    return Normalization(
        observation_mean=tuple(float(value) for value in observation_mean),
        observation_std=tuple(float(value) for value in observation_std),
        action_mean=tuple(float(value) for value in action_mean),
        action_std=tuple(float(value) for value in action_std),
    )


def _tensor_dataset(
    dataset: BehaviorDataset,
    indices: np.ndarray,
    normalization: Normalization,
) -> TensorDataset:
    observation_mean = np.asarray(normalization.observation_mean, dtype=np.float32)
    observation_std = np.asarray(normalization.observation_std, dtype=np.float32)
    action_mean = np.asarray(normalization.action_mean, dtype=np.float32)
    action_std = np.asarray(normalization.action_std, dtype=np.float32)
    observations = (dataset.observations[indices] - observation_mean) / observation_std
    actions = dataset.actions[indices].copy()
    actions[:, :4] = (actions[:, :4] - action_mean) / action_std
    return TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions))


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    continuous = nn.functional.smooth_l1_loss(prediction[:, :4], target[:, :4])
    gripper = nn.functional.binary_cross_entropy_with_logits(prediction[:, 4], target[:, 4])
    return continuous + 0.25 * gripper


def _validation_loss(model: BehaviorMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for observation, action in loader:
            observation = observation.to(device)
            action = action.to(device)
            batch_loss = _loss(model(observation), action)
            total += float(batch_loss.cpu()) * observation.shape[0]
            count += observation.shape[0]
    return total / max(1, count)


def train_behavior_policy(
    dataset: BehaviorDataset,
    config: TrainingConfig,
) -> TrainingResult:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device_name = _select_device(config.device)
    device = torch.device(device_name)
    train_indices, validation_indices = dataset.split_by_episode(
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    normalization = _normalization(dataset.observations[train_indices], dataset.actions[train_indices])
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
    model_config = ModelConfig(dataset.observations.shape[1], hidden_dims=config.hidden_dims)
    model = BehaviorMLP(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(config.epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for observation, action in train_loader:
            observation = observation.to(device)
            action = action.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = _loss(model(observation), action)
            batch_loss.backward()
            optimizer.step()
            train_total += float(batch_loss.detach().cpu()) * observation.shape[0]
            train_count += observation.shape[0]
        validation_loss = _validation_loss(model, validation_loader, device)
        train_loss = train_total / max(1, train_count)
        history.append(
            {"epoch": float(epoch + 1), "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training produced no model state")
    model.load_state_dict(best_state)
    model.to("cpu")
    return TrainingResult(
        model=model,
        normalization=normalization,
        model_config=model_config,
        training_config=config,
        history=history,
        best_validation_loss=best_loss,
        device=device_name,
    )

