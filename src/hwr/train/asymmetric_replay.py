"""Bounded replay storage for asymmetric Actor-Critic transitions."""

from __future__ import annotations

from typing import Mapping

import torch

from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS
from hwr.train.asymmetric_rl import AsymmetricRLBatch


class AsymmetricReplayBuffer:
    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self._storage: dict[str, torch.Tensor] = {}
        self._generator = torch.Generator().manual_seed(seed)

    def add(self, batch: AsymmetricRLBatch) -> None:
        values = self._flatten_batch(batch)
        batch_size = batch.rewards.shape[0]
        if not self._storage:
            self._storage = {
                name: torch.empty(
                    (self.capacity, *value.shape[1:]), dtype=value.dtype
                )
                for name, value in values.items()
            }
        if set(values) != set(self._storage) or any(
            value.shape[1:] != self._storage[name].shape[1:]
            for name, value in values.items()
        ):
            raise ValueError("replay transition shapes changed")
        for row in range(batch_size):
            for name, value in values.items():
                self._storage[name][self.position].copy_(value[row].detach().cpu())
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> AsymmetricRLBatch:
        if batch_size <= 0 or self.size < batch_size:
            raise ValueError("replay buffer does not contain the requested batch")
        indices = torch.randint(
            self.size, (batch_size,), generator=self._generator
        )
        values = {name: value[indices].clone() for name, value in self._storage.items()}
        return self._unflatten_batch(values)

    def all(self) -> AsymmetricRLBatch:
        if self.size == 0:
            raise ValueError("replay buffer is empty")
        values = {
            name: value[: self.size].clone()
            for name, value in self._storage.items()
        }
        return self._unflatten_batch(values)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "storage": {
                name: value[: self.size].clone()
                for name, value in self._storage.items()
            },
            "generator_state": self._generator.get_state(),
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if int(value["capacity"]) != self.capacity:
            raise ValueError("replay checkpoint capacity differs")
        self.size = int(value["size"])
        self.position = int(value["position"])
        self._storage = {}
        for name, tensor in value["storage"].items():
            if tensor.shape[0] not in (self.size, self.capacity):
                raise ValueError("replay checkpoint storage length is invalid")
            expanded = torch.empty(
                (self.capacity, *tensor.shape[1:]), dtype=tensor.dtype
            )
            expanded[: self.size].copy_(tensor[: self.size])
            self._storage[name] = expanded
        self._generator.set_state(value["generator_state"])

    def _flatten_batch(self, batch: AsymmetricRLBatch) -> dict[str, torch.Tensor]:
        if frozenset(batch.actor_inputs) != VLA_POLICY_INPUT_FIELDS or frozenset(
            batch.next_actor_inputs
        ) != VLA_POLICY_INPUT_FIELDS:
            raise ValueError("replay Actor fields violate the deployment whitelist")
        values = {
            f"actor__{name}": value for name, value in batch.actor_inputs.items()
        }
        values.update(
            {
                f"next_actor__{name}": value
                for name, value in batch.next_actor_inputs.items()
            }
        )
        if batch.proposed_action_chunks is not None and batch.safety_costs is not None:
            values["proposed_action_chunks"] = batch.proposed_action_chunks
            values["safety_costs"] = batch.safety_costs
        values.update(
            {
                "privileged_state": batch.privileged_state,
                "next_privileged_state": batch.next_privileged_state,
                "action_chunks": batch.action_chunks,
                "stop_decisions": batch.stop_decisions,
                "rewards": batch.rewards,
                "done": batch.done,
                "actor_weights": (
                    batch.actor_weights
                    if batch.actor_weights is not None
                    else torch.ones_like(batch.rewards)
                ),
            }
        )
        batch_sizes = {value.shape[0] for value in values.values()}
        if len(batch_sizes) != 1:
            raise ValueError("replay transition batch sizes differ")
        return values

    def _unflatten_batch(
        self, values: Mapping[str, torch.Tensor]
    ) -> AsymmetricRLBatch:
        actor = {
            name: values[f"actor__{name}"] for name in VLA_POLICY_INPUT_FIELDS
        }
        next_actor = {
            name: values[f"next_actor__{name}"] for name in VLA_POLICY_INPUT_FIELDS
        }
        return AsymmetricRLBatch(
            actor_inputs=actor,
            next_actor_inputs=next_actor,
            privileged_state=values["privileged_state"],
            next_privileged_state=values["next_privileged_state"],
            action_chunks=values["action_chunks"],
            stop_decisions=values["stop_decisions"],
            rewards=values["rewards"],
            done=values["done"],
            actor_weights=values["actor_weights"],
            proposed_action_chunks=values.get("proposed_action_chunks"),
            safety_costs=values.get("safety_costs"),
        )
