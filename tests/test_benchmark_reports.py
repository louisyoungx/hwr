from __future__ import annotations

from scripts.verify_benchmarks import verify_benchmarks


def test_tracked_benchmarks_meet_release_thresholds() -> None:
    assert verify_benchmarks() == []

