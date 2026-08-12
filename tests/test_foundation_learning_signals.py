from __future__ import annotations

import torch
from torch import nn

from hwr.train.foundation_learning_signals import (
    observed_one_step_td_error,
    posterior_state_change_novelty,
)


class _ZeroValue(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_zeros(*features.shape[:-1], 5)


def test_posterior_novelty_is_scale_free_state_change() -> None:
    unchanged = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    changed = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])

    unchanged_novelty = posterior_state_change_novelty(unchanged)
    changed_novelty = posterior_state_change_novelty(changed)

    torch.testing.assert_close(unchanged_novelty, torch.zeros(1, 2))
    torch.testing.assert_close(changed_novelty, torch.ones(1, 2))


def test_observed_td_error_uses_episode_rewards_not_global_training_mean() -> None:
    features = torch.zeros(2, 3, 4)
    rewards = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    continues = torch.ones_like(rewards)

    error = observed_one_step_td_error(
        features,
        rewards,
        continues,
        _ZeroValue(),
        _ZeroValue(),
        discount=0.99,
        symlog_limit=5.0,
    )

    torch.testing.assert_close(error, rewards)
    assert error[0].mean() != error[1].mean()
