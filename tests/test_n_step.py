from __future__ import annotations

import pytest

from hwr.train import build_n_step_targets


def test_n_step_returns_accumulate_until_horizon_or_terminal() -> None:
    targets = build_n_step_targets(
        (1.0, 2.0, 3.0, 4.0),
        (0.0, 0.0, 0.0, 1.0),
        horizon=3,
        discount=0.9,
    )

    assert targets.rewards == pytest.approx((5.23, 7.94, 6.6, 4.0))
    assert targets.next_indices == (2, 3, 3, 3)
    assert targets.bootstrap_discounts == pytest.approx((0.9**3, 0.0, 0.0, 0.0))
    assert targets.done == (0.0, 1.0, 1.0, 1.0)


def test_n_step_targets_reject_invalid_shapes_and_horizon() -> None:
    with pytest.raises(ValueError, match="equal non-zero"):
        build_n_step_targets((1.0,), (), horizon=2, discount=0.9)
    with pytest.raises(ValueError, match="horizon"):
        build_n_step_targets((1.0,), (1.0,), horizon=0, discount=0.9)
