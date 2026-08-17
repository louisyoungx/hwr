"""Run the frozen R0001-P05 replay batch-arm experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from hwr.train.accelerator_memory import release_unused_accelerator_memory
from hwr.train.foundation_batch_arms import (
    BATCH_ARMS,
    audit_batch_arm_schedule,
    build_batch_arm_schedule,
)
from hwr.train.foundation_batch_replay import (
    evaluate_frozen_batch_arm,
    frozen_input_identity,
    load_batch_replay_checkpoint,
    load_frozen_batch_replay_inputs,
    module_state_sha256,
    save_batch_replay_checkpoint,
    train_frozen_batch_arm,
)
from hwr.train.foundation_setup import build_foundation_learning_stack


DEFAULT_INPUT_RUN = Path(
    "runs/foundation-world-model/r0001-p01-baseline-v4-s20260812"
)
DEFAULT_OUTPUT_ROOT = Path("runs/research-loop/0003")
DEFAULT_RUN_PREFIX = "r0003-p05-batch-arms-s20261205"
FORMAL_SEEDS = (20_261_205, 202_716_734, 202_821_463)
FORMAL_UPDATES = 1_600
AUDIT_INTERVAL = 200
RUN_SCHEMA = "hwr.foundation-batch-arm-run/v1"
EXPECTED_REPLAY_SHA256 = (
    "c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985"
)
EXPECTED_AUDIT_SHA256 = (
    "8e1f0b521aac0c6a5b2f65cf7031fefd693890eb0ae37928fd5881ffc11b9907"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--arm", choices=BATCH_ARMS)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    smoke = bool(arguments.smoke)
    if smoke and (arguments.arm is not None or arguments.seed is not None or arguments.resume):
        raise ValueError("batch-arm smoke selects all arms and the first frozen seed")
    if not smoke and (arguments.arm is None or arguments.seed not in FORMAL_SEEDS):
        raise ValueError("formal batch-arm run requires one frozen arm and seed")
    if not smoke:
        _require_clean_source(root)
    source_commit = _source_commit(root)
    input_run = _resolve(root, Path(arguments.input_run))
    output_root = _resolve(root, Path(arguments.output_root))
    if smoke:
        run_path = output_root / f"{arguments.run_prefix}-smoke"
        if run_path.exists():
            raise FileExistsError(run_path)
        run_path.mkdir(parents=True)
        report = _run_smoke(
            root,
            input_run,
            run_path,
            device=str(arguments.device),
            source_commit=source_commit,
        )
    else:
        arm = str(arguments.arm)
        seed = int(arguments.seed)
        run_path = output_root / f"{arguments.run_prefix}-{arm}-seed-{seed}"
        report = _run_formal(
            root,
            input_run,
            run_path,
            arm=arm,
            seed=seed,
            device=str(arguments.device),
            resume=bool(arguments.resume),
            source_commit=source_commit,
        )
    return {
        "run_path": str(run_path),
        "mode": report["mode"],
        "decision": report["decision"],
        "source_commit": source_commit,
    }


def _run_smoke(
    root: Path,
    input_run: Path,
    run_path: Path,
    *,
    device: str,
    source_commit: str,
) -> dict[str, object]:
    inputs = load_frozen_batch_replay_inputs(root, input_run, device=device)
    identity = frozen_input_identity(inputs)
    _require_frozen_inputs(identity)
    schedule = build_batch_arm_schedule(
        inputs.training_loader,
        seed=FORMAL_SEEDS[0],
        updates=2,
        visual_update_interval=4,
    )
    schedule_audit = audit_batch_arm_schedule(inputs.training_loader, schedule)
    if not schedule_audit["passed"]:
        raise RuntimeError("batch-arm smoke schedule audit failed")
    arm_results = {}
    for arm in BATCH_ARMS:
        stack = build_foundation_learning_stack(
            root / "configs/foundation", device=device, seed=FORMAL_SEEDS[0]
        )
        trainer = stack.trainer
        frozen_before = _frozen_hash(trainer)
        learned_before = _learned_hash(trainer)
        started = time.perf_counter()
        metrics = train_frozen_batch_arm(
            trainer,
            inputs,
            schedule,
            arm,
            start_update=0,
            stop_update=2,
            progress_interval=2,
        )
        frozen_after = _frozen_hash(trainer)
        learned_after = _learned_hash(trainer)
        arm_results[arm] = {
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "frozen_hash_before": frozen_before,
            "frozen_hash_after": frozen_after,
            "learned_hash_before": learned_before,
            "learned_hash_after": learned_after,
            "frozen_components_unchanged": frozen_before == frozen_after,
            "learned_components_changed": learned_before != learned_after,
        }
        del stack
        release_unused_accelerator_memory()
    passed = all(
        value["frozen_components_unchanged"] and value["learned_components_changed"]
        for value in arm_results.values()
    )
    report = {
        "schema_version": RUN_SCHEMA,
        "proposal_id": "R0001-P05",
        "mode": "smoke",
        "decision": "smoke_passed" if passed else "smoke_failed",
        "source_commit": source_commit,
        "input_identity": identity,
        "schedule": schedule.to_dict(),
        "schedule_audit": schedule_audit,
        "arms": arm_results,
    }
    _write_json(run_path / "report.json", report)
    return report


def _run_formal(
    root: Path,
    input_run: Path,
    run_path: Path,
    *,
    arm: str,
    seed: int,
    device: str,
    resume: bool,
    source_commit: str,
) -> dict[str, object]:
    if run_path.exists() != resume:
        raise FileExistsError(run_path) if run_path.exists() else FileNotFoundError(run_path)
    if not resume:
        run_path.mkdir(parents=True)
    inputs = load_frozen_batch_replay_inputs(root, input_run, device=device)
    identity = frozen_input_identity(inputs)
    _require_frozen_inputs(identity)
    schedule = build_batch_arm_schedule(
        inputs.training_loader,
        seed=seed,
        updates=FORMAL_UPDATES,
        visual_update_interval=4,
    )
    schedule_audit = audit_batch_arm_schedule(inputs.training_loader, schedule)
    if not schedule_audit["passed"]:
        raise RuntimeError("formal batch-arm schedule audit failed")
    schedule_path = run_path / "schedule.json"
    if not resume:
        _write_json(schedule_path, schedule.to_dict())
        _write_json(run_path / "schedule-audit.json", schedule_audit)
        _write_json(
            run_path / "run.json",
            {
                "schema_version": RUN_SCHEMA,
                "proposal_id": "R0001-P05",
                "source_commit": source_commit,
                "arm": arm,
                "seed": seed,
                "target_updates": FORMAL_UPDATES,
                "audit_interval": AUDIT_INTERVAL,
                "device": device,
                "input_identity": identity,
            },
        )
    else:
        expected_run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
        if (
            expected_run.get("source_commit") != source_commit
            or expected_run.get("arm") != arm
            or int(expected_run.get("seed", -1)) != seed
            or expected_run.get("input_identity") != identity
            or json.loads(schedule_path.read_text(encoding="utf-8"))
            != schedule.to_dict()
        ):
            raise ValueError("resumed batch-arm run identity differs")
    schedule_sha = _file_sha256(schedule_path)
    stack = build_foundation_learning_stack(
        root / "configs/foundation", device=device, seed=seed
    )
    trainer = stack.trainer
    frozen_initial = _frozen_hash(trainer)
    checkpoint = run_path / "checkpoint.pt"
    if resume:
        load_batch_replay_checkpoint(
            checkpoint,
            trainer,
            arm=arm,
            seed=seed,
            schedule_sha256=schedule_sha,
            input_identity=identity,
        )
    start_update = trainer.update_count
    audits = _existing_audits(run_path)
    started = time.perf_counter()

    def progress(update: int, metrics: Mapping[str, float], elapsed: float) -> None:
        audit = evaluate_frozen_batch_arm(trainer, inputs)
        value = {
            "update": update,
            "training_metrics": dict(metrics),
            "elapsed_seconds": elapsed,
            **audit,
        }
        _write_json(run_path / f"audits/update-{update:07d}.json", value)
        audits[str(update)] = value
        save_batch_replay_checkpoint(
            checkpoint,
            trainer,
            arm=arm,
            seed=seed,
            schedule_sha256=schedule_sha,
            input_identity=identity,
        )
        _write_json(
            run_path / "progress.json",
            {
                "schema_version": RUN_SCHEMA,
                "arm": arm,
                "seed": seed,
                "update": update,
                "target_updates": FORMAL_UPDATES,
                "elapsed_seconds": elapsed,
            },
        )

    metrics = train_frozen_batch_arm(
        trainer,
        inputs,
        schedule,
        arm,
        start_update=start_update,
        stop_update=FORMAL_UPDATES,
        progress_interval=AUDIT_INTERVAL,
        progress=progress,
    )
    final_audit = audits[str(FORMAL_UPDATES)]
    frozen_final = _frozen_hash(trainer)
    report = {
        "schema_version": RUN_SCHEMA,
        "proposal_id": "R0001-P05",
        "mode": "formal",
        "decision": "completed",
        "source_commit": source_commit,
        "arm": arm,
        "seed": seed,
        "updates": FORMAL_UPDATES,
        "input_identity": identity,
        "schedule_sha256": schedule_sha,
        "schedule_audit": schedule_audit,
        "training_metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_hash_initial": frozen_initial,
        "frozen_hash_final": frozen_final,
        "frozen_components_unchanged": frozen_initial == frozen_final,
        "final_audit": final_audit,
        "audit_updates": sorted(int(value) for value in audits),
    }
    if not report["frozen_components_unchanged"]:
        raise RuntimeError("batch-arm run changed Actor or value components")
    _write_json(run_path / "report.json", report)
    return report


def _frozen_hash(trainer) -> str:
    return module_state_sha256(
        trainer.actor,
        trainer.value,
        trainer.exploration_actor,
        trainer.exploration_value,
        trainer.imagination.slow_value,
        trainer.intrinsic_exploration.slow_value,
    )


def _learned_hash(trainer) -> str:
    return module_state_sha256(
        trainer.visual_student,
        trainer.visual_objective,
        trainer.world_model,
    )


def _require_frozen_inputs(identity: Mapping[str, object]) -> None:
    if (
        identity.get("replay_manifest_sha256") != EXPECTED_REPLAY_SHA256
        or identity.get("audit_manifest_sha256") != EXPECTED_AUDIT_SHA256
        or identity.get("window_count") != 168
        or identity.get("source_episode_count") != 24
    ):
        raise ValueError("P05 frozen replay identity differs")


def _existing_audits(run_path: Path) -> dict[str, object]:
    values = {}
    for path in sorted((run_path / "audits").glob("update-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        values[str(int(value["update"]))] = value
    return values


def _source_commit(root: Path) -> str:
    value = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise RuntimeError("batch-arm experiment requires a Git source commit")
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
        raise RuntimeError("formal batch-arm experiment requires clean committed source")


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


def _file_sha256(path: Path) -> str:
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
