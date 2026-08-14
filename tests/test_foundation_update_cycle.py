from __future__ import annotations

import numpy as np

from hwr.train.foundation_update_cycle import ShardLocalWindowSampler


class _Loader:
    shards = (0, 1, 1, 1)

    def __len__(self):
        return len(self.shards)

    def window_shard_index(self, index: int) -> int:
        return self.shards[index]


class _CollisionLoader(_Loader):
    def window_metadata(self, index: int):
        return {
            "transition_stop": index + 1,
            "transition_count": 4,
            "metadata": {
                "result_reason": "severe_collision" if index == 3 else "timeout"
            },
        }


class _MultiEpisodeCollisionLoader:
    def __len__(self):
        return 4

    def window_shard_index(self, index: int) -> int:
        return index

    def window_metadata(self, index: int):
        return {
            "transition_stop": 4,
            "transition_count": 4,
            "metadata": {"result_reason": "severe_collision"},
        }


def test_shard_local_sampler_is_uniform_per_window_without_batch_thrashing() -> None:
    loader = _Loader()
    sampler = ShardLocalWindowSampler(loader)
    rng = np.random.default_rng(7)
    batches = [sampler.sample(rng, 4) for _ in range(4000)]

    assert all(len({loader.window_shard_index(index) for index in batch}) == 1 for batch in batches)
    large_shard_fraction = np.mean(
        [loader.window_shard_index(batch[0]) == 1 for batch in batches]
    )
    assert 0.72 < large_shard_fraction < 0.78


def test_shard_local_sampler_reserves_collision_terminal_batches() -> None:
    sampler = ShardLocalWindowSampler(_CollisionLoader())

    batch = sampler.sample(
        np.random.default_rng(7), 3, severe_collision_fraction=1.0
    )

    assert batch == (3, 3, 3)


def test_collision_batch_uses_distinct_episodes_when_available() -> None:
    loader = _MultiEpisodeCollisionLoader()
    sampler = ShardLocalWindowSampler(loader)

    batch = sampler.sample(
        np.random.default_rng(7), 3, severe_collision_fraction=1.0
    )

    assert len(set(batch)) == 3
    assert len({loader.window_shard_index(index) for index in batch}) == 3
