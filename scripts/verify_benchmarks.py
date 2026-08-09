#!/usr/bin/env python3
"""Verify tracked household benchmark reports against release thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = Path("benchmarks/thresholds.json")
DEFAULT_RESULTS = Path("benchmarks/results")


def load_reports(results_root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(results_root.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        task_id = report["task_id"]
        if task_id in reports:
            raise ValueError(f"duplicate benchmark report for {task_id}")
        reports[task_id] = report
    return reports


def verify_benchmarks(
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    results_root: Path = DEFAULT_RESULTS,
) -> list[str]:
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    reports = load_reports(results_root)
    errors: list[str] = []
    for task_id in thresholds["required_tasks"]:
        if task_id not in reports:
            errors.append(f"missing benchmark report for {task_id}")
            continue
        evaluation = reports[task_id]["evaluation"]
        if evaluation["episode_count"] < thresholds["minimum_episode_count"]:
            errors.append(f"{task_id}: too few evaluation episodes")
        if evaluation["success_rate"] < thresholds["minimum_success_rate"]:
            errors.append(
                f"{task_id}: success rate {evaluation['success_rate']:.3f} is below "
                f"{thresholds['minimum_success_rate']:.3f}"
            )
        if evaluation["average_collisions"] > thresholds["maximum_average_collisions"]:
            errors.append(
                f"{task_id}: average collisions {evaluation['average_collisions']:.3f} exceeds "
                f"{thresholds['maximum_average_collisions']:.3f}"
            )
    return errors


def main() -> int:
    errors = verify_benchmarks()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    reports = load_reports(DEFAULT_RESULTS)
    print(f"Benchmark verification passed: {len(reports)} household scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

