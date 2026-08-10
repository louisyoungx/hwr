"""Compact formal expert stages into macro-phase visual datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hwr.data import compact_visual_dataset, verify_visual_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    path = compact_visual_dataset(
        arguments.source, arguments.output_root, arguments.dataset_id
    )
    print(json.dumps(verify_visual_dataset(path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
