"""Run the frozen R0001-P19 free-nats gradient dead-zone diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

from hwr.apps.evaluate_posterior_overshooting import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INPUT_RUN,
    EXPECTED_CHECKPOINT_ARTIFACT,
    EXPECTED_CHECKPOINT_MANIFEST,
    SELECTION_SEED,
    select_source_episode_windows,
)
from hwr.eval.free_nats_deadzone import (
    aggregate_free_nats_deadzone,
    evaluate_free_nats_deadzone,
)
from hwr.train.foundation_batch_replay import load_frozen_batch_replay_inputs
from hwr.train.foundation_registry import load_foundation_training_checkpoint
from hwr.train.foundation_setup import build_foundation_learning_stack
from hwr.train.foundation_visual_update import encode_visual_student_bounded


DEFAULT_OUTPUT = Path(
    "runs/research-loop/0003/r0003-p19-free-nats-deadzone-s20261319"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    _require_clean_source(root)
    source_commit = _source_commit(root)
    input_run = _resolve(root, arguments.input_run)
    checkpoint = _resolve(root, arguments.checkpoint)
    output = _resolve(root, arguments.output)
    if output.exists():
        raise FileExistsError(output)
    _require_checkpoint(checkpoint)
    inputs = load_frozen_batch_replay_inputs(
        root, input_run, device=str(arguments.device)
    )
    selected = select_source_episode_windows(
        inputs.training_loader, seed=SELECTION_SEED
    )
    stack = build_foundation_learning_stack(
        root / "configs/foundation",
        device=str(arguments.device),
        seed=SELECTION_SEED,
    )
    load_foundation_training_checkpoint(checkpoint, stack.trainer)
    trainer = stack.trainer
    trainer.visual_student.eval()
    trainer.world_model.eval()
    reports = []
    output.mkdir(parents=True)
    try:
        for source, index in selected.items():
            batch = inputs.training_loader.build(
                (index,), include_visual_targets=False
            )
            visual = encode_visual_student_bounded(
                trainer.visual_student,
                batch,
                microbatch_observations=(
                    trainer.config.visual_inference_microbatch_observations
                ),
            ).pooled_state.reshape(
                1,
                batch.observation_count,
                trainer.world_model.config.visual_dimension,
            )
            actions = batch.executed_actions.detach().clone().requires_grad_(True)
            observed = trainer.world_model.observe(
                visual,
                batch.language_features,
                batch.proprioception,
                batch.actor_proposals,
                actions,
            )
            report = evaluate_free_nats_deadzone(
                trainer.world_model, observed.sequence, actions
            )
            report.update(
                {
                    "source_episode_id": source,
                    "window_index": index,
                }
            )
            reports.append(report)
            _write_json(output / "episodes" / f"{source}.json", report)
        aggregate = aggregate_free_nats_deadzone(reports)
        aggregate.update(
            {
                "proposal_id": "R0001-P19",
                "source_commit": source_commit,
                "selection_seed": SELECTION_SEED,
                "input_run": str(input_run),
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_sha256": _sha256(
                    checkpoint / "manifest.json"
                ),
                "checkpoint_artifact_sha256": _sha256(
                    checkpoint / "training-state.pt"
                ),
            }
        )
        _write_json(output / "report.json", aggregate)
        _write_manifest(output, source_commit)
    except BaseException:
        _write_json(
            output / "failure.json",
            {
                "schema_version": "hwr.free-nats-deadzone-failure/v1",
                "proposal_id": "R0001-P19",
                "source_commit": source_commit,
                "completed_episode_count": len(reports),
            },
        )
        _write_manifest(output, source_commit)
        raise
    return {
        "output": str(output),
        "decision": aggregate["assessment"]["decision"],
        "episode_count": len(reports),
        "report_sha256": _sha256(output / "report.json"),
    }


def _require_checkpoint(path: Path) -> None:
    if (
        _sha256(path / "manifest.json") != EXPECTED_CHECKPOINT_MANIFEST
        or _sha256(path / "training-state.pt") != EXPECTED_CHECKPOINT_ARTIFACT
    ):
        raise ValueError("P19 frozen checkpoint identity differs")


def _write_manifest(output: Path, source_commit: str) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "hwr.free-nats-deadzone-artifacts/v1",
            "proposal_id": "R0001-P19",
            "source_commit": source_commit,
            "artifacts": {
                str(path.relative_to(output)): {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in paths
            },
        },
    )


def _source_commit(root: Path) -> str:
    value = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise RuntimeError("P19 diagnostic requires a Git source commit")
    return value


def _require_clean_source(root: Path) -> None:
    value = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if value:
        raise RuntimeError("P19 diagnostic requires clean committed source")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
