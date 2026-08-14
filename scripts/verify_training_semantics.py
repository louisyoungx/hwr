#!/usr/bin/env python3
"""Run executable training-semantics checks without starting a training run."""

from __future__ import annotations

import json
from pathlib import Path

from hwr.train.development_semantics import verify_training_semantics


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            verify_training_semantics(root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
