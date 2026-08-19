"""Run the frozen R0001-P06 posterior overshooting preflight."""

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

from hwr.eval.posterior_overshooting import (
    aggregate_posterior_overshooting,
    evaluate_posterior_overshooting,
)
from hwr.train.foundation_batch_replay import load_frozen_batch_replay_inputs
from hwr.train.foundation_registry import load_foundation_training_checkpoint
from hwr.train.foundation_setup import build_foundation_learning_stack
from hwr.train.foundation_visual_update import encode_visual_student_bounded
from hwr.train.foundation_sequence_reservoir import source_episode_id


DEFAULT_INPUT_RUN = Path(
    "runs/foundation-world-model/r0001-p01-baseline-v4-s20260812"
)
DEFAULT_CHECKPOINT = DEFAULT_INPUT_RUN / "checkpoints/update-000001600"
DEFAULT_OUTPUT = Path(
    "runs/research-loop/0003/r0003-p06-posterior-overshooting-s20261306"
)
SELECTION_SEED = 20_261_306
EXPECTED_CHECKPOINT_MANIFEST = (
    "72f9361762d7ff5086f086b9ae1db05396caa3cf91822ece20686095df4ad75b"
)
EXPECTED_CHECKPOINT_ARTIFACT = (
    "ef24bdfcca3cc46274bdfebc1d8b1a4afc81c73abff3aa4128e393e6da2109c6"
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
    for parameter in trainer.world_model.parameters():
        parameter.requires_grad_(False)
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
            observed = trainer.world_model.observe(
                visual,
                batch.language_features,
                batch.proprioception,
                batch.actor_proposals,
                batch.executed_actions,
            )
            report = evaluate_posterior_overshooting(
                trainer.world_model,
                observed.sequence,
                batch.executed_actions,
            )
            report.update(
                {
                    "source_episode_id": source,
                    "window_index": index,
                    "window": _window_identity(
                        inputs.training_loader.window_metadata(index)
                    ),
                }
            )
            reports.append(report)
            _write_json(output / "episodes" / f"{source}.json", report)
        aggregate = aggregate_posterior_overshooting(reports)
        aggregate.update(
            {
                "proposal_id": "R0001-P06",
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
                "selected_windows": [
                    report["window"] for report in reports
                ],
            }
        )
        _write_json(output / "report.json", aggregate)
        _write_manifest(output, source_commit)
    except BaseException:
        _write_json(
            output / "failure.json",
            {
                "schema_version": "hwr.posterior-overshooting-failure/v1",
                "proposal_id": "R0001-P06",
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


def select_source_episode_windows(loader, *, seed: int) -> dict[str, int]:
    grouped: dict[str, list[tuple[bytes, int]]] = {}
    for index in range(len(loader)):
        metadata = loader.window_metadata(index)
        source = source_episode_id(metadata)
        if not source:
            raise ValueError("overshooting source Episode identity is missing")
        payload = (
            f"{seed}:{source}:{metadata['episode_id']}:"
            f"{metadata['transition_start']}"
        )
        grouped.setdefault(source, []).append(
            (hashlib.sha256(payload.encode()).digest(), index)
        )
    selected = {
        source: min(values)[1] for source, values in sorted(grouped.items())
    }
    if len(selected) != 24:
        raise ValueError("overshooting preflight requires 24 source Episodes")
    return selected


def _window_identity(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "episode_id": str(metadata["episode_id"]),
        "task_id": str(metadata["task_id"]),
        "seed": int(metadata["seed"]),
        "transition_start": int(metadata["transition_start"]),
        "transition_stop": int(metadata["transition_stop"]),
    }


def _require_checkpoint(path: Path) -> None:
    if (
        _sha256(path / "manifest.json") != EXPECTED_CHECKPOINT_MANIFEST
        or _sha256(path / "training-state.pt") != EXPECTED_CHECKPOINT_ARTIFACT
    ):
        raise ValueError("P06 frozen checkpoint identity differs")


def _write_manifest(output: Path, source_commit: str) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "hwr.posterior-overshooting-artifacts/v1",
            "proposal_id": "R0001-P06",
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
        raise RuntimeError("P06 preflight requires a Git source commit")
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
        raise RuntimeError("P06 preflight requires clean committed source")


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
