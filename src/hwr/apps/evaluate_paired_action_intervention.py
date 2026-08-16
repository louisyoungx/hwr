"""Run the frozen R0001-P17 paired physical action intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.adapters.mujoco import (
    MujocoFormalHouseholdDualArmBackend,
    load_default_formal_household_catalogs,
)
from hwr.eval.paired_action_collection import collect_paired_episode
from hwr.eval.paired_action_intervention import (
    PAIRED_INJECTION_TRIALS,
    PAIRED_PERMUTATIONS,
    PairedEpisodeEffect,
    analyze_paired_effects,
    blind_injection_power,
    clopper_pearson_upper,
)
from hwr.policy.latent_actions import LatentActionScaling


DEFAULT_OUTPUT_ROOT = Path("runs/research-loop/0003")
DEFAULT_RUN_ID = "r0003-p17-paired-action-s20261017"
PRECHECK_SEED_BASE = 20_261_101
FORMAL_SEED_BASE = 620_261_101
EPISODE_COUNT = 64
SEED_STRIDE = 104_729
RUN_SCHEMA = "hwr.paired-action-intervention-run/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--episode-count",
        type=int,
        default=EPISODE_COUNT,
        help="smoke override; formal/precheck contract requires 64",
    )
    parser.add_argument(
        "--injection-trials",
        type=int,
        default=PAIRED_INJECTION_TRIALS,
        help="smoke override; precheck contract requires 1000",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=PAIRED_PERMUTATIONS,
        help="smoke override; formal contract requires 999",
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    formal = bool(arguments.formal)
    episode_count = int(arguments.episode_count)
    injection_trials = int(arguments.injection_trials)
    permutations = int(arguments.permutations)
    if min(episode_count, injection_trials, permutations) <= 0:
        raise ValueError("paired intervention dimensions must be positive")
    contract = (
        episode_count == EPISODE_COUNT
        and injection_trials == PAIRED_INJECTION_TRIALS
        and permutations == PAIRED_PERMUTATIONS
    )
    if formal and not contract:
        raise ValueError("formal paired intervention requires the frozen contract")
    if contract:
        _require_clean_source(root)
    source_commit = _source_commit(root)
    base_run_id = str(arguments.run_id)
    run_id = (
        base_run_id + ("-formal" if formal else "-preflight")
        if contract
        else base_run_id
        + (
            f"-smoke-e{episode_count}-i{injection_trials}-p{permutations}"
        )
    )
    output_root = _resolve(root, Path(arguments.output_root))
    output_path = output_root / run_id
    output_path.mkdir(parents=True, exist_ok=False)
    try:
        if formal:
            preflight = _require_preflight(
                output_root, base_run_id, source_commit=source_commit
            )
        else:
            preflight = None
        report = _collect_and_analyze(
            root,
            output_path,
            formal=formal,
            episode_count=episode_count,
            injection_trials=injection_trials,
            permutations=permutations,
        )
        report.update(
            {
                "source_commit": source_commit,
                "run_id": run_id,
                "mode": (
                    "formal" if formal else "preflight" if contract else "smoke"
                ),
                "preflight_identity": preflight,
            }
        )
        _write_json(output_path / "report.json", report)
        manifest = _manifest(output_path, source_commit, report)
        _write_json(output_path / "manifest.json", manifest)
    except BaseException:
        _write_json(
            output_path / "failure.json",
            {
                "schema_version": "hwr.paired-action-intervention-failure/v1",
                "source_commit": source_commit,
                "formal": formal,
                "episode_count": episode_count,
            },
        )
        raise
    return {
        "output_path": str(output_path),
        "mode": report["mode"],
        "decision": report["decision"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
    }


def _collect_and_analyze(
    root: Path,
    output_path: Path,
    *,
    formal: bool,
    episode_count: int,
    injection_trials: int,
    permutations: int,
) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    task_ids = tuple(sorted(tasks))
    seed_base = FORMAL_SEED_BASE if formal else PRECHECK_SEED_BASE
    effects: list[PairedEpisodeEffect] = []
    episodes = []
    scaling = LatentActionScaling()
    for task_index, task_id in enumerate(task_ids):
        backend = MujocoFormalHouseholdDualArmBackend(
            tasks[task_id],
            bindings[task_id],
            camera_width=32,
            camera_height=24,
        )
        try:
            for episode_index in range(episode_count):
                seed = seed_base + episode_index * SEED_STRIDE
                backend.reset(seed=seed, task_id=task_id)
                snapshot = backend.capture_state_snapshot()
                artifact = (
                    output_path
                    / "episodes"
                    / f"{task_id.replace('/', '-')}-seed-{seed}.npz"
                )
                effect, audit = collect_paired_episode(
                    backend,
                    task_id=task_id,
                    task_index=task_index,
                    seed=seed,
                    episode_index=episode_index,
                    snapshot=snapshot,
                    output_path=artifact,
                    scaling=scaling,
                )
                effects.append(effect)
                episodes.append(
                    {
                        "task_id": task_id,
                        "seed": seed,
                        "episode_index": episode_index,
                        **audit,
                    }
                )
        finally:
            backend.close()
    guard = _guard_report(effects, episodes)
    injection = None
    confirmation = None
    if not formal:
        injection = blind_injection_power(
            effects,
            trials=injection_trials,
        )
    elif guard["passed"]:
        confirmation = analyze_paired_effects(
            effects, permutation_count=permutations
        )
    decision = (
        "accepted"
        if formal and guard["passed"] and confirmation["passed"]
        else "rejected"
        if formal and guard["passed"]
        else "inconclusive"
        if formal
        else "preflight_passed"
        if guard["passed"] and injection["passed"]
        else "preflight_failed"
    )
    return {
        "schema_version": "hwr.paired-action-intervention-evaluation/v1",
        "proposal_id": "R0001-P17",
        "decision": decision,
        "episode_count_per_task": episode_count,
        "seed_base": seed_base,
        "seed_stride": SEED_STRIDE,
        "permutations": permutations,
        "injection_trials": injection_trials,
        "guard": guard,
        "injection": injection,
        "confirmation": confirmation,
        "episodes": episodes,
    }


def _guard_report(
    effects: Sequence[PairedEpisodeEffect],
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = len(effects)
    sham_failures = sum(not value.sham_equal for value in effects)
    sham_upper = clopper_pearson_upper(sham_failures, count, 0.95)
    task_sham = {}
    for task_id in sorted({value.task_id for value in effects}):
        selected = [value for value in effects if value.task_id == task_id]
        failures = sum(not value.sham_equal for value in selected)
        task_sham[task_id] = {
            "episodes": len(selected),
            "failures": failures,
            "failure_rate_upper_95": clopper_pearson_upper(
                failures, len(selected), 0.95
            ),
        }
    minimum_rms = min(value.action_difference_rms for value in effects)
    minimum_cosine = min(
        (
            min(value.direction_cosines)
            if value.direction_cosines
            else 0.0
        )
        for value in effects
    )
    maximum_asymmetry = max(
        float(value["first_stage_relative_asymmetry"]) for value in episodes
    )
    total_steps = sum(int(value["all_branch_steps"]) for value in episodes)
    total_safety = sum(
        int(value["all_branch_safety_interventions"]) for value in episodes
    )
    safety_rate = total_safety / max(total_steps, 1)
    severe = max(int(value["all_branch_severe_collisions"]) for value in episodes)
    terminated = any(
        bool(value["all_branch_terminated_early"]) for value in episodes
    )
    checks = {
        "sham_upper_95_at_most_0_05": (
            sham_upper <= 0.05
            and all(
                value["failure_rate_upper_95"] <= 0.05
                for value in task_sham.values()
            )
        ),
        "minimum_action_difference_rms_at_least_0_10": minimum_rms >= 0.10,
        "minimum_direction_cosine_at_least_0_95": minimum_cosine >= 0.95,
        "maximum_first_stage_asymmetry_at_most_0_05": maximum_asymmetry <= 0.05,
        "safety_intervention_rate_at_most_0_05": safety_rate <= 0.05,
        "zero_severe_collisions": severe == 0,
        "zero_early_terminations": not terminated,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sham_failures": sham_failures,
        "sham_failure_rate_upper_95": sham_upper,
        "sham_by_task": task_sham,
        "minimum_action_difference_rms": minimum_rms,
        "minimum_direction_cosine": minimum_cosine,
        "maximum_first_stage_asymmetry": maximum_asymmetry,
        "safety_intervention_rate": safety_rate,
        "maximum_severe_collisions": severe,
        "any_early_termination": terminated,
    }


def _require_preflight(
    output_root: Path, run_id: str, *, source_commit: str
) -> dict[str, object]:
    path = output_root / f"{run_id}-preflight" / "report.json"
    report = _read_json(path)
    if (
        report.get("mode") != "preflight"
        or report.get("decision") != "preflight_passed"
        or report.get("source_commit") != source_commit
        or report.get("guard", {}).get("passed") is not True
        or report.get("injection", {}).get("passed") is not True
    ):
        raise ValueError("formal paired intervention requires passed preflight")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "source_commit": report["source_commit"],
    }


def _manifest(
    output_path: Path,
    source_commit: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    paths = [output_path / "report.json", *sorted((output_path / "episodes").glob("*.npz"))]
    return {
        "schema_version": RUN_SCHEMA,
        "proposal_id": "R0001-P17",
        "source_commit": source_commit,
        "mode": report["mode"],
        "decision": report["decision"],
        "artifacts": {
            str(path.relative_to(output_path)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
    }


def _resolve(root: Path, requested: Path) -> Path:
    return requested if requested.is_absolute() else root / requested


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean_source(root: Path) -> None:
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("paired intervention contract requires clean source")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
