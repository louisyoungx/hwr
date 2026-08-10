"""Episode-level physical metrics for bimanual training."""

from __future__ import annotations


def bilateral_near_statistics(
    states: list[tuple[float, ...]], threshold: float = 0.10
) -> tuple[int, int]:
    total = 0
    current = 0
    longest = 0
    for state in states:
        near = max(state[24], state[25]) <= threshold
        total += int(near)
        current = current + 1 if near else 0
        longest = max(longest, current)
    return total, longest
