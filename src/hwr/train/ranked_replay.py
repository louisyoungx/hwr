"""Globally ranked retention for a bounded task-agnostic replay stratum."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from hwr.train.asymmetric_replay import (
    AsymmetricReplayBuffer,
    select_batch_rows,
)
from hwr.train.asymmetric_rl import AsymmetricRLBatch


class RankedReplayRetention:
    """Keep the highest-scoring rows seen across episode boundaries."""

    def __init__(self, replay: AsymmetricReplayBuffer) -> None:
        self.replay = replay
        self._scores = torch.full((replay.capacity,), -torch.inf)

    def add(self, batch: AsymmetricRLBatch, scores: torch.Tensor) -> int:
        values = scores.detach().cpu().to(torch.float32).flatten()
        if values.shape != batch.rewards.shape:
            raise ValueError("ranked replay scores must align with replay batch")
        if not torch.isfinite(values).all():
            raise ValueError("ranked replay scores must be finite")
        order = torch.argsort(values, descending=True, stable=True)
        ordered = select_batch_rows(batch, order)
        ordered_scores = values[order]
        retained = self._append_available(ordered, ordered_scores)
        for row in range(retained, ordered_scores.numel()):
            minimum = int(torch.argmin(self._scores[: self.replay.size]))
            score = ordered_scores[row]
            if score <= self._scores[minimum]:
                break
            index = torch.tensor((row,), dtype=torch.int64)
            self.replay.replace_rows(
                torch.tensor((minimum,), dtype=torch.int64),
                select_batch_rows(ordered, index),
            )
            self._scores[minimum] = score
            retained += 1
        return retained

    def score_state(self) -> torch.Tensor:
        return self._scores[: self.replay.size].clone()

    def load_scores(self, values: torch.Tensor) -> None:
        scores = values.detach().cpu().to(torch.float32).flatten()
        if scores.numel() != self.replay.size or not torch.isfinite(scores).all():
            raise ValueError("ranked replay checkpoint scores are invalid")
        self._scores.fill_(-torch.inf)
        self._scores[: scores.numel()].copy_(scores)

    def rebuild(
        self,
        stores: Sequence[AsymmetricReplayBuffer],
        estimate: Callable[[AsymmetricRLBatch], torch.Tensor],
        *,
        chunk_size: int = 64,
    ) -> int:
        """Build an exact global Top-K from retained autonomous stores."""
        for store in stores:
            if not store.size:
                continue
            batch = store.chronological()
            chunks = []
            for indices in torch.arange(store.size).split(chunk_size):
                chunks.append(estimate(select_batch_rows(batch, indices)))
            scores = torch.cat(chunks).to(torch.float32)
            eligible = torch.ones(scores.shape, dtype=torch.bool)
            if batch.safety_costs is not None:
                eligible &= batch.safety_costs <= 0.0
            indices = torch.nonzero(eligible).flatten()
            order = torch.argsort(scores[indices], descending=True, stable=True)
            selected = indices[order[: self.replay.capacity]]
            self.add(select_batch_rows(batch, selected), scores[selected])
        return self.replay.size

    def refresh(
        self,
        estimate: Callable[[AsymmetricRLBatch], torch.Tensor],
        *,
        chunk_size: int = 64,
    ) -> int:
        """Refresh non-stationary scores for all retained rows in place."""
        if not self.replay.size:
            return 0
        batch = self.replay.all()
        scores = []
        for indices in torch.arange(self.replay.size).split(chunk_size):
            scores.append(estimate(select_batch_rows(batch, indices)))
        self.load_scores(torch.cat(scores))
        return self.replay.size

    def _append_available(
        self, batch: AsymmetricRLBatch, scores: torch.Tensor
    ) -> int:
        count = min(self.replay.capacity - self.replay.size, scores.numel())
        if count <= 0:
            return 0
        indices = torch.arange(count)
        positions = self.replay.add(select_batch_rows(batch, indices))
        self._scores[positions] = scores[:count]
        return count
