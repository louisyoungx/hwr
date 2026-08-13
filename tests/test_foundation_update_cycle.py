from __future__ import annotations

import numpy as np

from hwr.train.foundation_update_cycle import ShardLocalWindowSampler


class _Loader:
    shards = (0, 1, 1, 1)

    def __len__(self):
        return len(self.shards)

    def window_shard_index(self, index: int) -> int:
        return self.shards[index]


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
