"""Run the frozen R0001-P24 decoder gain diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

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
from hwr.eval.decoder_gain import (
    _criteria,
    aggregate_decoder_gain,
    build_decoder_branches,
    build_decoder_calibration,
    build_true_decoder_feature,
    deserialize_decoder_calibration,
    evaluate_decoder_gain,
    serialize_decoder_calibration,
)
from hwr.train.foundation_batch_replay import load_frozen_batch_replay_inputs
from hwr.train.foundation_registry import load_foundation_training_checkpoint
from hwr.train.foundation_setup import build_foundation_learning_stack
from hwr.train.foundation_visual_update import encode_visual_student_bounded


DEFAULT_OUTPUT = Path(
    "runs/research-loop/0003/r0003-p24-decoder-gain-s20261324-r1"
)
ORIGINAL_OUTPUT = Path(
    "runs/research-loop/0003/r0003-p24-decoder-gain-s20261324"
)
ORIGINAL_SOURCE_COMMIT = "107b4c7e68fe407b79910daed3c62e0dc2ecee3e"
ORIGINAL_REPORT_SHA256 = (
    "fcf1b5dad3b93316054a5c884e13c8c35d0417d83a6ae745cabe3dda988f6cb5"
)
ORIGINAL_CALIBRATION_SHA256 = (
    "16d4d6be2390415e215c5f02a61325171d38cfbebad4e0da67ab26c90b085337"
)
ORIGINAL_MANIFEST_SHA256 = (
    "2ce984233c4ce3a3c3aa932f9e8d3f1765fe5a315c5e4e13cc0a17928a521dda"
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
    recovery_of = _require_recovery_artifacts(root)
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
        raise ValueError("P24 frozen window selection identity differs")
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
    failure_stage = "calibration_true_pass"
    current_source = None
    current_window = None
    output.mkdir(parents=True)
    try:
        windows = []
        true_features = []
        for source, index in selected.items():
            current_source = source
            current_window = _window_identity(
                inputs.training_loader.window_metadata(index)
            )
            batch, sequence = _load_sequence(
                trainer, inputs.training_loader, index
            )
            true_features.append(
                build_true_decoder_feature(
                    trainer.world_model, sequence, batch.executed_actions
                )
            )
            windows.append((source, index, current_window))
        calibration = build_decoder_calibration(
            trainer.world_model, true_features
        )
        failure_stage = "calibration_write"
        _write_json(
            output / "calibration.json",
            serialize_decoder_calibration(calibration),
        )
        calibration_sha256 = _sha256(output / "calibration.json")
        failure_stage = "calibration_reload"
        calibration = deserialize_decoder_calibration(
            json.loads(
                (output / "calibration.json").read_text(encoding="utf-8")
            ),
            device="cpu",
        )
        failure_stage = "shift_episode_evaluation"
        for (source, index, window), expected_true in zip(
            windows, true_features, strict=True
        ):
            current_source = source
            current_window = window
            batch, sequence = _load_sequence(
                trainer, inputs.training_loader, index
            )
            branch = build_decoder_branches(
                trainer.world_model, sequence, batch.executed_actions
            )
            if not torch.equal(branch.true_feature, expected_true):
                raise ValueError("P24 true feature changed after calibration")
            report = evaluate_decoder_gain(
                trainer.world_model, branch, calibration
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
        aggregate = aggregate_decoder_gain(reports)
        aggregate.update(
            {
                "proposal_id": "R0001-P24",
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
                "calibration_sha256": calibration_sha256,
                "recovery_of": recovery_of,
                "invocation": {
                    "module": "hwr.apps.evaluate_decoder_gain",
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
                "schema_version": "hwr.decoder-gain-failure/v1",
                "proposal_id": "R0001-P24",
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
                "calibration_sha256": (
                    _sha256(output / "calibration.json")
                    if (output / "calibration.json").is_file()
                    else None
                ),
                "criteria": _criteria(),
                "recovery_of": recovery_of,
                "invocation": {
                    "module": "hwr.apps.evaluate_decoder_gain",
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


def _load_sequence(trainer, loader, index: int):
    batch = loader.build((index,), include_visual_targets=False)
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
    embeddings = trainer.world_model.encode_observations(
        visual,
        batch.language_features,
        batch.proprioception,
    )
    sequence = trainer.world_model.rssm.observe(
        embeddings, batch.executed_actions
    )
    return batch, sequence


def _write_manifest(output: Path, source_commit: str) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "hwr.decoder-gain-artifacts/v1",
            "proposal_id": "R0001-P24",
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
        raise RuntimeError("P24 diagnostic requires a Git source commit")
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
        raise RuntimeError("P24 diagnostic requires clean committed source")


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
        raise ValueError("P24 invocation differs from frozen experiment")


def _require_recovery_artifacts(root: Path) -> dict[str, object]:
    original = root / ORIGINAL_OUTPUT
    report = original / "report.json"
    calibration = original / "calibration.json"
    manifest = original / "manifest.json"
    if (
        _sha256(report) != ORIGINAL_REPORT_SHA256
        or _sha256(calibration) != ORIGINAL_CALIBRATION_SHA256
        or _sha256(manifest) != ORIGINAL_MANIFEST_SHA256
    ):
        raise ValueError("P24 recovery artifact identity differs")
    value = json.loads(report.read_text(encoding="utf-8"))
    if value.get("source_commit") != ORIGINAL_SOURCE_COMMIT:
        raise ValueError("P24 recovery source commit differs")
    return {
        "run": str(original),
        "source_commit": ORIGINAL_SOURCE_COMMIT,
        "report_sha256": ORIGINAL_REPORT_SHA256,
        "calibration_sha256": ORIGINAL_CALIBRATION_SHA256,
        "manifest_sha256": ORIGINAL_MANIFEST_SHA256,
    }


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
