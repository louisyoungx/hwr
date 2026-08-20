"""Run the frozen R0001-P32-E1 retained-Replay information diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.data.autonomous_trajectory import (
    ALLOWED_ACTION_SOURCES,
    AUTONOMOUS_TRAJECTORY_SCHEMA,
    TRAJECTORY_ARRAY_FIELDS,
    _validate_arrays,
)
from hwr.eval.paired_action_intervention import (
    clopper_pearson_lower,
    clopper_pearson_upper,
)
from hwr.eval.replay_conditional_information import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    POWER_BASE_SEED,
    POWER_BOOTSTRAP_SAMPLES,
    POWER_TRIALS,
    PROPOSAL_ID,
    BootstrapPlan,
    NestedDesign,
    ReplayDataset,
    _RidgeDesign,
    _seed,
    build_fold_manifest,
    evaluate_replay_conditional_information,
    summarize_errors,
    target_deltas,
)

DEFAULT_INPUT_RUN = Path(
    "runs/foundation-world-model/r0001-p01-baseline-v4-s20260812"
)
DEFAULT_OUTPUT = Path("runs/research-loop/0007/r0007-p32-replay-conditional-e1-s20263201")
REPLAY_RELATIVE = Path("replay/autonomous")
EXPECTED_MANIFEST_SHA256 = (
    "c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985"
)
EXPECTED_MANIFEST_BYTES = 450_509
INPUT_CONTRACT_SCHEMA = "hwr.autonomous-trajectory-dataset/v2"
FROZEN_PARENT_COMMIT = "a722f3522cdb8f12c1a78c56ce8c1d7c873e9190"
FROZEN_DOCUMENT_COMMIT = "4ef5f3b728e3e13aa18552ea6cb744121ccce71f"
FORMAL_COMMAND = (
    ".venv/bin/python -m hwr.apps.evaluate_replay_conditional_information "
    "--input-run runs/foundation-world-model/r0001-p01-baseline-v4-s20260812 "
    "--output runs/research-loop/0007/"
    "r0007-p32-replay-conditional-e1-s20263201"
)
RUN_SCHEMA = "hwr.replay-conditional-information-run/v1"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--power-trials", type=int, default=POWER_TRIALS)
    parser.add_argument(
        "--power-bootstrap-samples",
        type=int,
        default=POWER_BOOTSTRAP_SAMPLES,
    )
    return parser

def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    formal = str(arguments.mode) == "formal"
    bootstrap_samples = int(arguments.bootstrap_samples)
    power_trials = int(arguments.power_trials)
    power_bootstrap = int(arguments.power_bootstrap_samples)
    _validate_run_configuration(
        formal, bootstrap_samples, power_trials, power_bootstrap
    )
    source_commit = _source_commit(root)
    if formal:
        _require_formal_source(root)
    input_run = _resolve(root, Path(arguments.input_run))
    output = _resolve(root, Path(arguments.output))
    if not formal and output == root / DEFAULT_OUTPUT:
        output = output.with_name(
            f"{output.name}-smoke-{power_trials}x{power_bootstrap}"
        )
    if output.exists():
        raise FileExistsError(output)
    dataset, input_identity = load_replay_dataset(input_run, formal=formal)
    folds = build_fold_manifest(dataset, formal=formal)
    output.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(output / "folds.json", folds)
        report, design = evaluate_replay_conditional_information(
            dataset,
            folds,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        power = run_exact_pipeline_power(
            dataset,
            design,
            trials=power_trials,
            bootstrap_samples=power_bootstrap,
            base_seed=POWER_BASE_SEED,
        )
        report.update(
            {
                "mode": "formal" if formal else "smoke",
                "source_commit": source_commit,
                "input": input_identity,
                "power": power,
            }
        )
        report["assessment"] = assess_replay_conditional_information(
            report, power
        )
        _write_json(output / "report.json", report)
        _write_manifest(
            output,
            source_commit=source_commit,
            input_identity=input_identity,
            mode=report["mode"],
            constants=_formal_constants(
                bootstrap_samples, power_trials, power_bootstrap
            ),
        )
    except BaseException as error:
        (output / "report.json").unlink(missing_ok=True)
        _write_json(
            output / "failure.json",
            {
                "schema_version": "hwr.replay-conditional-information-failure/v1",
                "proposal_id": PROPOSAL_ID,
                "source_commit": source_commit,
                "mode": "formal" if formal else "smoke",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        _write_manifest(
            output,
            source_commit=source_commit,
            input_identity=input_identity,
            mode="formal" if formal else "smoke",
            constants=_formal_constants(
                bootstrap_samples, power_trials, power_bootstrap
            ),
        )
        raise
    return {
        "output": str(output),
        "mode": report["mode"],
        "decision": report["assessment"]["decision"],
        "report_sha256": _sha256(output / "report.json"),
    }

def load_replay_dataset(
    input_run: Path, *, formal: bool
) -> tuple[ReplayDataset, dict[str, object]]:
    replay = input_run.resolve() / REPLAY_RELATIVE
    manifest_path = replay / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if formal and (
        manifest_sha256 != EXPECTED_MANIFEST_SHA256
        or len(manifest_bytes) != EXPECTED_MANIFEST_BYTES
    ):
        raise ValueError("P32 frozen Replay manifest identity differs")
    manifest = json.loads(manifest_bytes)
    _validate_manifest_header(manifest, formal=formal)
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "task_ids",
            "source_ids",
            "shard_ids",
            "shard_offsets",
            "state",
            "action",
            "actor_proposal",
            "rate_target",
            "configuration_target",
            "safety_rewrite",
        )
    }
    source_ranges: dict[str, list[tuple[int, int, int, int]]] = {}
    source_tasks: dict[str, set[str]] = {}
    source_commits: set[str] = set()
    seen_paths: set[str] = set()
    seen_episode_ids: set[str] = set()
    shard_identities = []
    for shard in manifest["shards"]:
        relative = str(shard["path"])
        path = _contained_member(replay, relative)
        episode_id = str(shard["episode_id"])
        if relative in seen_paths or episode_id in seen_episode_ids:
            raise ValueError("P32 Replay shard identity is duplicated")
        seen_paths.add(relative)
        seen_episode_ids.add(episode_id)
        if _sha256(path) != str(shard.get("sha256", "")):
            raise ValueError(f"P32 Replay shard hash differs: {relative}")
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
        _validate_shard_arrays(arrays)
        task_id, source_id, start, stop, slot, source_count = _shard_lineage(shard)
        if stop - start != 16 or int(shard["transition_count"]) != 16:
            raise ValueError("P32 Replay absolute transition range differs")
        if int(shard["observation_count"]) != 17:
            raise ValueError("P32 Replay observation count differs")
        source_ranges.setdefault(source_id, []).append(
            (start, stop, slot, source_count)
        )
        source_tasks.setdefault(source_id, set()).add(task_id)
        source_commits.add(str(shard.get("source_commit", "")))
        rate, configuration = target_deltas(arrays["proprioception"])
        count = len(arrays["executed_action"])
        rows["task_ids"].append(np.full(count, task_id))
        rows["source_ids"].append(np.full(count, source_id))
        rows["shard_ids"].append(np.full(count, episode_id))
        rows["shard_offsets"].append(np.arange(count, dtype=np.int64))
        rows["state"].append(arrays["proprioception"][:-1])
        rows["action"].append(arrays["executed_action"])
        rows["actor_proposal"].append(arrays["actor_proposal"])
        rows["rate_target"].append(rate)
        rows["configuration_target"].append(configuration)
        rows["safety_rewrite"].append(arrays["safety_intervention"] > 0.0)
        shard_identities.append(
            {
                "path": relative,
                "sha256": shard["sha256"],
                "task_id": task_id,
                "source_id": source_id,
                "start": start,
                "stop": stop,
            }
        )
    _validate_source_ranges(source_ranges, source_tasks, formal=formal)
    if len(source_commits) != 1 or not next(iter(source_commits), ""):
        raise ValueError("P32 Replay shard source commits differ")
    dataset = ReplayDataset(
        **{name: np.concatenate(values) for name, values in rows.items()}
    )
    return dataset, {
        "run_path": str(input_run.resolve()),
        "replay_path": str(replay),
        "contract_schema": INPUT_CONTRACT_SCHEMA,
        "storage_schema": manifest["schema_version"],
        "manifest_sha256": manifest_sha256,
        "manifest_bytes": len(manifest_bytes),
        "source_commit": next(iter(source_commits)),
        "source_episode_count": len(source_ranges),
        "shard_count": len(manifest["shards"]),
        "transition_count": len(dataset.state),
        "task_source_counts": {
            task: len(
                {
                    source
                    for source, tasks in source_tasks.items()
                    if tasks == {task}
                }
            )
            for task in sorted({next(iter(tasks)) for tasks in source_tasks.values()})
        },
        "shard_identity_sha256": hashlib.sha256(
            json.dumps(
                shard_identities,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }

def run_exact_pipeline_power(
    dataset: ReplayDataset,
    design: NestedDesign,
    *,
    trials: int = POWER_TRIALS,
    bootstrap_samples: int = POWER_BOOTSTRAP_SAMPLES,
    base_seed: int = POWER_BASE_SEED,
) -> dict[str, object]:
    if trials <= 0 or bootstrap_samples <= 0 or base_seed < 0:
        raise ValueError("exact-pipeline power configuration is invalid")
    conditions = ("zero_action_residual", "random_target", "planted")
    passes = {name: 0 for name in conditions}
    trial_reports = []
    minimum_rank = min(fold.effective_rank for fold in design.folds)
    for trial in range(trials):
        targets, calibration = _power_targets(
            dataset, design, trial, base_seed
        )
        first = design.apply(targets[conditions[0]])
        plan = BootstrapPlan.build(
            dataset,
            first,
            bootstrap_samples,
            _seed(base_seed, "power-bootstrap", trial),
        )
        condition_reports = {}
        for name in conditions:
            errors = first if name == conditions[0] else design.apply(targets[name])
            summary = summarize_errors(
                dataset, errors, plan, dataset.masks()["all"]
            )
            passed = _signal_gate(
                summary["aggregate"], summary["tasks"], minimum_rank
            )
            passes[name] += int(passed)
            condition_reports[name] = {
                "passed": passed,
                "ratio": summary["aggregate"]["ratio"],
                "mean_log_ratio_p05": summary["aggregate"]["bootstrap"][
                    "mean_log_ratio_p05"
                ],
            }
        trial_reports.append(
            {
                "trial": trial,
                "conditions": condition_reports,
                "planted_training_oracle_ratios": calibration,
            }
        )
    results = {
        name: {
            "passes": passes[name],
            "trials": trials,
            "empirical_rate": passes[name] / trials,
        }
        for name in conditions
    }
    for name in conditions[:2]:
        results[name]["clopper_pearson_95_upper"] = clopper_pearson_upper(
            passes[name], trials, 0.95
        )
    results["planted"]["clopper_pearson_95_lower"] = clopper_pearson_lower(
        passes["planted"], trials, 0.95
    )
    passed = (
        all(
            results[name]["clopper_pearson_95_upper"] <= 0.05
            for name in conditions[:2]
        )
        and results["planted"]["clopper_pearson_95_lower"] >= 0.80
    )
    return {
        "schema_version": "hwr.replay-conditional-information-power/v1",
        "pipeline": "exact_nested_source_task_stratified_pipeline",
        "trial_count": trials,
        "bootstrap_samples_per_trial": bootstrap_samples,
        "base_seed": base_seed,
        "planted_oracle_ratio": 1.10,
        "outer_test_used_for_calibration": False,
        "conditions": results,
        "passed": passed,
        "trials": trial_reports,
    }

def _power_targets(
    dataset: ReplayDataset,
    design: NestedDesign,
    trial: int,
    base_seed: int,
) -> tuple[dict[str, dict[int, np.ndarray]], list[dict[str, float]]]:
    names = ("zero_action_residual", "random_target", "planted")
    result = {name: {} for name in names}
    calibration = []
    for fold in design.folds:
        actual = dataset.rate_target
        nuisance = _RidgeDesign.build(
            dataset.state[fold.train], dataset.state
        )
        state_signal = nuisance.predict(actual[fold.train])
        residual = actual[fold.train] - state_signal[fold.train]
        residual_rms = np.sqrt(np.mean(np.square(residual), axis=0))
        residual_rms[residual_rms < 1.0e-8] = 1.0
        target_scale = actual[fold.train].std(axis=0)
        target_scale[target_scale < 1.0e-8] = 1.0
        noise = _training_scaled(
            _source_gaussian(
                dataset, base_seed, "zero-noise", trial, fold.outer_fold
            ),
            fold.train,
            residual_rms,
        )
        random_target = _training_scaled(
            _source_gaussian(
                dataset, base_seed, "random-target", trial, fold.outer_fold
            ),
            fold.train,
            target_scale,
        )
        action_residual = np.zeros_like(dataset.action)
        action_residual[fold.train] = fold.action_oof
        action_residual[fold.test] = fold.action_test
        standardized = (
            action_residual - fold.residual.mean
        ) / fold.residual.scale
        rng = np.random.default_rng(
            _seed(base_seed, "planted-direction", trial, fold.outer_fold)
        )
        direction, _ = np.linalg.qr(rng.standard_normal((16, 16)))
        signal = _training_scaled(
            standardized @ direction, fold.train, residual_rms
        )
        scaled_noise = noise[fold.train] / target_scale
        scaled_signal = signal[fold.train] / target_scale
        noise_mse = float(np.square(scaled_noise).mean())
        signal_mse = float(np.square(scaled_signal).mean())
        cross = float((scaled_noise * scaled_signal).mean())
        coefficient = (
            -cross
            + math.sqrt(cross * cross + signal_mse * 0.10 * noise_mse)
        ) / signal_mse
        planted = state_signal + noise + coefficient * signal
        oracle = float(
            np.square((noise[fold.train] + coefficient * signal[fold.train])
            / target_scale).mean()
            / np.square(noise[fold.train] / target_scale).mean()
        )
        result["zero_action_residual"][fold.outer_fold] = state_signal + noise
        result["random_target"][fold.outer_fold] = random_target
        result["planted"][fold.outer_fold] = planted
        calibration.append({"outer_fold": fold.outer_fold, "ratio": oracle})
    return result, calibration

def assess_replay_conditional_information(
    report: Mapping[str, object], power: Mapping[str, object]
) -> dict[str, object]:
    rate = report["target_families"]["rate"]["strata"]
    configuration = report["target_families"]["configuration"]["strata"]["all"]
    controller = report["controller_context_guard"]["strata"]["all"]
    rank = float(report["effective_rank"]["minimum"])
    main = rate["all"]
    main_gates = {
        "aggregate_ratio_at_least_1_05": main["aggregate"]["ratio"] >= 1.05,
        "aggregate_log_ratio_p05_positive": _positive_p05(main),
        "every_task_ratio_above_one": all(
            value["ratio"] > 1.0 for value in main["tasks"].values()
        ),
        "candidate_absolute_mse_improves": (
            main["aggregate"]["candidate_mse"]
            < main["aggregate"]["control_mse"]
        ),
        "minimum_effective_rank_at_least_six": rank >= 6.0,
        "zero_action_null_fpr_upper_at_most_0_05": power["conditions"]["zero_action_residual"]["clopper_pearson_95_upper"] <= 0.05,
        "random_target_null_fpr_upper_at_most_0_05": power["conditions"]["random_target"]["clopper_pearson_95_upper"] <= 0.05,
        "planted_power_lower_at_least_0_80": power["conditions"]["planted"]["clopper_pearson_95_lower"] >= 0.80,
    }
    guard_gates = {
        "controller_context": _guard_pass(controller, 1.02),
        "no_rewrite": _guard_pass(rate["no_rewrite"], 1.0),
        "shard_interior": _guard_pass(rate["shard_interior"], 1.0),
        "configuration_aggregate": _guard_pass(configuration, 1.0),
        "configuration_tasks": all(
            value["ratio"] is not None and value["ratio"] > 1.0
            for value in configuration["tasks"].values()
        ),
    }
    required = (
        main,
        rate["no_rewrite"],
        rate["shard_interior"],
        configuration,
        controller,
    )
    support = all(
        value["aggregate"]["source_level_measurable"] for value in required
    )
    if rank < 6.0 or not power["passed"] or not support:
        decision = "inconclusive"
    elif all(main_gates.values()) and all(guard_gates.values()):
        decision = (
            "accepted as retained-Replay conditional information evidence"
        )
    else:
        decision = "rejected"
    return {
        "decision": decision,
        "main_gates": main_gates,
        "mechanism_guards": guard_gates,
        "required_strata_source_level_measurable": support,
        "capability_claim_allowed": False,
        "plant_causality_claim_allowed": False,
        "production_utilization_claim_allowed": False,
        "closed_loop_success_available": False,
    }

def _validate_manifest_header(
    manifest: Mapping[str, object], *, formal: bool
) -> None:
    if (
        manifest.get("schema_version") != AUTONOMOUS_TRAJECTORY_SCHEMA
        or frozenset(manifest.get("array_fields", ()))
        != TRAJECTORY_ARRAY_FIELDS
        or frozenset(manifest.get("allowed_action_sources", ()))
        != ALLOWED_ACTION_SOURCES
        or not isinstance(manifest.get("shards"), list)
    ):
        raise ValueError("P32 Replay manifest schema differs")
    if formal and (
        int(manifest.get("episode_count", -1)) != 168
        or int(manifest.get("transition_count", -1)) != 2_688
        or len(manifest["shards"]) != 168
    ):
        raise ValueError("P32 frozen Replay manifest counts differ")

def _validate_shard_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if frozenset(arrays) != TRAJECTORY_ARRAY_FIELDS:
        raise ValueError("P32 Replay shard fields differ")
    _validate_arrays(arrays)
    if (
        arrays["proprioception"].shape != (17, 37)
        or arrays["actor_proposal"].shape != (16, 16)
        or arrays["executed_action"].shape != (16, 16)
        or arrays["safety_intervention"].shape != (16,)
    ):
        raise ValueError("P32 Replay shard shape differs")

def _shard_lineage(
    shard: Mapping[str, object],
) -> tuple[str, str, int, int, int, int]:
    task_id = str(shard.get("task_id", ""))
    metadata = shard.get("metadata")
    reservoir = (
        metadata.get("sequence_reservoir")
        if isinstance(metadata, Mapping)
        else None
    )
    if not task_id or not isinstance(reservoir, Mapping):
        raise ValueError("P32 Replay shard lineage is incomplete")
    source_id = str(reservoir.get("source_episode_id", ""))
    start = int(reservoir.get("transition_start", -1))
    stop = int(reservoir.get("transition_stop", -1))
    slot = int(reservoir.get("slot", -1))
    source_count = int(reservoir.get("source_transition_count", -1))
    if (
        not source_id
        or start < 0
        or stop <= start
        or stop > source_count
        or slot < 0
        or int(reservoir.get("slot_count", -1)) != 7
    ):
        raise ValueError("P32 Replay absolute source range is invalid")
    return task_id, source_id, start, stop, slot, source_count

def _validate_source_ranges(
    ranges: Mapping[str, Sequence[tuple[int, int, int, int]]],
    tasks: Mapping[str, set[str]],
    *,
    formal: bool,
) -> None:
    if any(len(values) != 7 for values in ranges.values()):
        raise ValueError("P32 Replay source does not contain seven shards")
    for source, values in ranges.items():
        ordered = sorted(values)
        if (
            len(tasks[source]) != 1
            or {slot for _, _, slot, _ in values} != set(range(7))
            or len({source_count for _, _, _, source_count in values}) != 1
            or any(
                ordered[index][1] > ordered[index + 1][0]
                for index in range(len(ordered) - 1)
            )
        ):
            raise ValueError("P32 Replay source ranges overlap or drifted")
    task_counts = {
        task: sum(task in values for values in tasks.values())
        for task in sorted({next(iter(values)) for values in tasks.values()})
    }
    if formal and task_counts != {
        "clear_dining_table_3d/v1": 6,
        "store_kitchen_items_3d/v1": 6,
        "tidy_living_room_3d/v1": 12,
    }:
        raise ValueError("P32 frozen Replay source/task counts differ")

def _source_gaussian(
    dataset: ReplayDataset,
    base_seed: int,
    domain: str,
    trial: int,
    outer_fold: int,
) -> np.ndarray:
    values = np.empty((len(dataset.state), 16), np.float64)
    for source in dataset.source_order:
        rows = np.flatnonzero(dataset.source_ids == source)
        task = str(np.unique(dataset.task_ids[rows]).item())
        rng = np.random.default_rng(
            _seed(base_seed, domain, trial, outer_fold, task, source)
        )
        values[rows] = rng.standard_normal((len(rows), 16))
    return values

def _training_scaled(
    values: np.ndarray, training: np.ndarray, desired: np.ndarray
) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(values[training]), axis=0))
    scale[scale < 1.0e-8] = 1.0
    return values / scale * desired

def _signal_gate(
    aggregate: Mapping[str, object],
    tasks: Mapping[str, Mapping[str, object]],
    minimum_rank: float,
) -> bool:
    return bool(
        aggregate["ratio"] >= 1.05
        and aggregate["candidate_mse"] < aggregate["control_mse"]
        and aggregate["bootstrap"]["mean_log_ratio_p05"] > 0.0
        and all(value["ratio"] > 1.0 for value in tasks.values())
        and minimum_rank >= 6.0
    )


def _positive_p05(summary: Mapping[str, object]) -> bool:
    value = summary["aggregate"]["bootstrap"]["mean_log_ratio_p05"]
    return value is not None and value > 0.0


def _guard_pass(summary: Mapping[str, object], threshold: float) -> bool:
    ratio = summary["aggregate"]["ratio"]
    return ratio is not None and (ratio > threshold if threshold == 1.0 else ratio >= threshold) and _positive_p05(summary)


def _validate_run_configuration(
    formal: bool,
    bootstrap_samples: int,
    power_trials: int,
    power_bootstrap_samples: int,
) -> None:
    values = (bootstrap_samples, power_trials, power_bootstrap_samples)
    if any(value <= 0 for value in values):
        raise ValueError("P32 run dimensions must be positive")
    if formal and values != (
        BOOTSTRAP_SAMPLES,
        POWER_TRIALS,
        POWER_BOOTSTRAP_SAMPLES,
    ):
        raise ValueError("formal P32 rejects trial or bootstrap overrides")


def _require_formal_source(root: Path) -> None:
    _require_clean_source(root)
    for commit in (FROZEN_PARENT_COMMIT, FROZEN_DOCUMENT_COMMIT):
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            cwd=root,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("P32 frozen document commit is not an ancestor")
    history = tuple(
        f"docs/research-loop/{index:04d}" for index in range(1, 7)
    )
    result = subprocess.run(
        ("git", "diff", "--quiet", FROZEN_PARENT_COMMIT, "HEAD", "--", *history),
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("P32 historical research-loop documents drifted")


def _require_clean_source(root: Path) -> None:
    result = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("formal P32 requires clean committed source")


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("P32 requires a committed source identity")
    return commit


def _formal_constants(
    bootstrap_samples: int,
    power_trials: int,
    power_bootstrap: int,
) -> dict[str, object]:
    return {
        "ridge": 1.0e-3,
        "outer_folds": 3,
        "inner_folds": 2,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "power_trials": power_trials,
        "power_bootstrap_samples": power_bootstrap,
        "power_base_seed": POWER_BASE_SEED,
        "rate_indices": [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 29, 30],
        "configuration_indices": [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28],
        "controller_history_steps": 4,
        "planted_oracle_ratio": 1.10,
    }


def _write_manifest(
    output: Path,
    *,
    source_commit: str,
    input_identity: Mapping[str, object],
    mode: str,
    constants: Mapping[str, object],
) -> None:
    paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": RUN_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "mode": mode,
            "source_commit": source_commit,
            "frozen_parent_commit": FROZEN_PARENT_COMMIT,
            "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
            "stable_command": FORMAL_COMMAND,
            "input": dict(input_identity),
            "formal_constants": dict(constants),
            "claim_boundaries": {
                "capability_claim_allowed": False,
                "plant_causality_claim_allowed": False,
                "production_utilization_claim_allowed": False,
                "closed_loop_success_available": False,
            },
            "artifacts": {
                path.name: {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in paths
            },
        },
    )


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else root / path


def _contained_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("P32 Replay shard escaped or is missing")
    return path


def _write_json(path: Path, value: Mapping[str, object]) -> None:
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
