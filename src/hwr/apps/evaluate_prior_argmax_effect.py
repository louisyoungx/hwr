"""Run the frozen R0001-P23 prior argmax-effect diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.apps.evaluate_action_input_contribution import (
    EXPECTED_WINDOW_SELECTION,
    _require_checkpoint,
    _require_replay_manifest,
    _selected_windows,
    _selection_sha256,
    _sha256,
)
from hwr.apps.evaluate_posterior_overshooting import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INPUT_RUN,
    SELECTION_SEED,
    _window_identity,
    select_source_episode_windows,
)
from hwr.eval.prior_argmax_effect import (
    _criteria,
    aggregate_prior_argmax_effect,
    evaluate_prior_argmax_effect,
)
from hwr.train.foundation_batch_replay import load_frozen_batch_replay_inputs
from hwr.train.foundation_registry import load_foundation_training_checkpoint
from hwr.train.foundation_setup import build_foundation_learning_stack
from hwr.train.foundation_visual_update import encode_visual_student_bounded


DEFAULT_OUTPUT = Path(
    "runs/research-loop/0003/r0003-p23-prior-argmax-s20261323"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="mps")
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    _require_clean_source(root)
    source_commit = _source_commit(root)
    input_run = _resolve(root, arguments.input_run)
    checkpoint = _resolve(root, arguments.checkpoint)
    output = _resolve(root, arguments.output)
    _require_frozen_invocation(
        root, input_run, checkpoint, output, str(arguments.device)
    )
    if output.exists():
        raise FileExistsError(output)
    _require_checkpoint(checkpoint)
    replay_manifest = input_run / "replay/autonomous/manifest.json"
    _require_replay_manifest(replay_manifest)
    inputs = load_frozen_batch_replay_inputs(
        root, input_run, device=str(arguments.device)
    )
    selected = select_source_episode_windows(
        inputs.training_loader, seed=SELECTION_SEED
    )
    selected_windows = _selected_windows(inputs.training_loader, selected)
    selection_sha256 = _selection_sha256(selected_windows)
    if selection_sha256 != EXPECTED_WINDOW_SELECTION:
        raise ValueError("P23 frozen window selection identity differs")
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
    failure_stage = "episode_evaluation"
    current_source = None
    current_window = None
    output.mkdir(parents=True)
    try:
        for source, index in selected.items():
            current_source = source
            window = _window_identity(
                inputs.training_loader.window_metadata(index)
            )
            current_window = window
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
            observation_embeddings = trainer.world_model.encode_observations(
                visual,
                batch.language_features,
                batch.proprioception,
            )
            sequence = trainer.world_model.rssm.observe(
                observation_embeddings,
                batch.executed_actions,
            )
            report = evaluate_prior_argmax_effect(
                trainer.world_model,
                sequence,
                batch.executed_actions,
            )
            report.update(
                {
                    "source_episode_id": source,
                    "window_index": index,
                    "window": window,
                }
            )
            _write_json(output / "episodes" / f"{source}.json", report)
            reports.append(report)
        failure_stage = "aggregate"
        aggregate = aggregate_prior_argmax_effect(reports)
        aggregate.update(
            {
                "proposal_id": "R0001-P23",
                "source_commit": source_commit,
                "device": str(arguments.device),
                "selection_seed": SELECTION_SEED,
                "window_selection_sha256": selection_sha256,
                "input_run": str(input_run),
                "input_replay_manifest_sha256": _sha256(replay_manifest),
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_sha256": _sha256(
                    checkpoint / "manifest.json"
                ),
                "checkpoint_artifact_sha256": _sha256(
                    checkpoint / "training-state.pt"
                ),
                "selected_windows": selected_windows,
                "invocation": {
                    "module": "hwr.apps.evaluate_prior_argmax_effect",
                    "device": str(arguments.device),
                    "input_run": str(input_run),
                    "checkpoint": str(checkpoint),
                    "output": str(output),
                },
            }
        )
        failure_stage = "report_write"
        _write_json(output / "report.json", aggregate)
        failure_stage = "manifest_write"
        _write_manifest(output, source_commit)
    except BaseException as error:
        _write_json(
            output / "failure.json",
            {
                "schema_version": "hwr.prior-argmax-effect-failure/v1",
                "proposal_id": "R0001-P23",
                "source_commit": source_commit,
                "device": str(arguments.device),
                "failure_stage": failure_stage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "completed_episode_count": len(reports),
                "current_source_episode_id": current_source,
                "current_window": current_window,
                "selection_seed": SELECTION_SEED,
                "window_selection_sha256": selection_sha256,
                "input_run": str(input_run),
                "input_replay_manifest_sha256": _sha256(replay_manifest),
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_sha256": _sha256(
                    checkpoint / "manifest.json"
                ),
                "checkpoint_artifact_sha256": _sha256(
                    checkpoint / "training-state.pt"
                ),
                "criteria": _criteria(),
                "invocation": {
                    "module": "hwr.apps.evaluate_prior_argmax_effect",
                    "device": str(arguments.device),
                    "input_run": str(input_run),
                    "checkpoint": str(checkpoint),
                    "output": str(output),
                },
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


def _write_manifest(output: Path, source_commit: str) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "hwr.prior-argmax-effect-artifacts/v1",
            "proposal_id": "R0001-P23",
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
        raise RuntimeError("P23 diagnostic requires a Git source commit")
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
        raise RuntimeError("P23 diagnostic requires clean committed source")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _require_frozen_invocation(
    root: Path,
    input_run: Path,
    checkpoint: Path,
    output: Path,
    device: str,
) -> None:
    expected = (
        root / DEFAULT_INPUT_RUN,
        root / DEFAULT_CHECKPOINT,
        root / DEFAULT_OUTPUT,
        "mps",
    )
    if (input_run, checkpoint, output, device) != expected:
        raise ValueError("P23 invocation differs from frozen experiment")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
