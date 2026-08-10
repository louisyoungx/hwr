"""Train all three physical bimanual tasks locally without demonstrations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from hwr.train import (
    BimanualRLTrainingConfig,
    BimanualTrainingRunner,
    load_default_bimanual_training_catalogs,
)
from hwr.train.bimanual_registry import save_bimanual_training_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/bimanual-rl"))
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--episode-steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-width", type=int, default=64)
    parser.add_argument("--raw-height", type=int, default=48)
    parser.add_argument("--image-width", type=int, default=32)
    parser.add_argument("--image-height", type=int, default=24)
    parser.add_argument("--point-count", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--exploration-noise", type=float, default=0.18)
    return parser


def _source_commit(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    tasks, bindings = load_default_bimanual_training_catalogs(root)
    config = BimanualRLTrainingConfig(
        episodes=arguments.episodes,
        episode_step_limit=arguments.episode_steps,
        batch_size=arguments.batch_size,
        learning_starts=arguments.learning_starts,
        seed=arguments.seed,
        device=arguments.device,
        raw_image_width=arguments.raw_width,
        raw_image_height=arguments.raw_height,
        image_width=arguments.image_width,
        image_height=arguments.image_height,
        point_count=arguments.point_count,
        hidden_dim=arguments.hidden_dim,
        exploration_noise=arguments.exploration_noise,
    )
    result = BimanualTrainingRunner(tasks, bindings, config).train()
    output_root = (
        arguments.output_root
        if arguments.output_root.is_absolute()
        else root / arguments.output_root
    )
    path = save_bimanual_training_run(
        output_root,
        arguments.run_id,
        result,
        source_commit=_source_commit(root),
    )
    return {
        "run_path": str(path),
        "episodes": len(result.records),
        "successes": sum(record.success for record in result.records),
        "updates": result.trainer.update_count,
        "replay_size": result.replay.size,
    }


def main(argv: Sequence[str] | None = None) -> int:
    value = run(build_parser().parse_args(argv))
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
