"""Run the frozen R0001-P11 causal plant FIFO estimator gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.eval.causal_plant_collection import (
    collect_causal_plant_episode,
    file_sha256,
)
from hwr.eval.causal_plant_evaluation import (
    evaluate_causal_plant_estimator,
    load_confirmation_episodes,
    load_p09_episodes,
)
from hwr.policy.latent_actions import LatentActionScaling
from hwr.train.foundation_exploration import (
    RandomRLActionSource,
    RandomRLExplorationConfig,
)


DEFAULT_RUN_ID = "r0003-p11-causal-plant-s20261101"
DEFAULT_OUTPUT_ROOT = Path("runs/research-loop/0003")
DEFAULT_P09_RUN = Path(
    "runs/research-loop/0001/r0001-p09-observation-lag-s20260901"
)
CONFIRMATION_SEEDS = (
    720_261_101,
    720_365_830,
    720_470_559,
    720_575_288,
    720_680_017,
    720_784_746,
    720_889_475,
    720_994_204,
)
CONFIRMATION_CORRELATIONS = (0.50, 0.96)
CONFIRMATION_LATENCIES = (1, 2, 3)
CONFIRMATION_TRANSITIONS = 64
RUN_SCHEMA = "hwr.causal-plant-estimator-run/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--p09-run", type=Path, default=DEFAULT_P09_RUN)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="evaluate the committed P09 artifacts without physical collection",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="collect one task, rho, latency, and seed without a formal decision",
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    p09_path = _resolve(root, Path(arguments.p09_run))
    development = load_p09_episodes(p09_path)
    if arguments.development_only:
        report = evaluate_causal_plant_estimator(development, ())
        return {
            "mode": "development-only",
            "development": report["development"],
        }
    smoke = bool(arguments.smoke)
    if not smoke:
        _require_clean_source(root)
    source_commit = _source_commit(root)
    run_id = str(arguments.run_id) + ("-smoke" if smoke else "")
    run_path = _resolve(root, Path(arguments.output_root)) / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    episodes = []
    try:
        episodes = _collect(root, run_path, smoke=smoke)
        confirmation = load_confirmation_episodes(
            run_path / "episodes", episodes
        )
        report = evaluate_causal_plant_estimator(development, confirmation)
        report.update(
            {
                "run_id": run_id,
                "source_commit": source_commit,
                "mode": "smoke" if smoke else "formal",
                "contract_complete": not smoke,
                "episodes": episodes,
            }
        )
        if smoke:
            report["decision"] = "smoke_complete"
        _write_json(run_path / "report.json", report)
        manifest = _manifest(
            run_path,
            p09_path=p09_path,
            source_commit=source_commit,
            run_id=run_id,
            smoke=smoke,
        )
        _write_json(run_path / "manifest.json", manifest)
    except BaseException:
        completed = len(tuple((run_path / "episodes").glob("*.npz")))
        _write_json(
            run_path / "failure.json",
            {
                "schema_version": "hwr.causal-plant-estimator-failure/v1",
                "proposal_id": "R0001-P11",
                "source_commit": source_commit,
                "completed_episode_count": completed,
            },
        )
        raise
    return {
        "run_path": str(run_path),
        "decision": report["decision"],
        "episode_count": len(episodes),
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
    }


def _collect(
    root: Path, run_path: Path, *, smoke: bool
) -> list[dict[str, object]]:
    from hwr.adapters.mujoco import (
        MujocoFormalHouseholdDualArmBackend,
        load_default_formal_household_catalogs,
    )

    tasks, bindings = load_default_formal_household_catalogs(root)
    task_ids = tuple(sorted(tasks))
    if smoke:
        task_ids = task_ids[:1]
    correlations = CONFIRMATION_CORRELATIONS[:1] if smoke else CONFIRMATION_CORRELATIONS
    latencies = CONFIRMATION_LATENCIES[:1] if smoke else CONFIRMATION_LATENCIES
    seeds = CONFIRMATION_SEEDS[:1] if smoke else CONFIRMATION_SEEDS
    episodes = []
    for task_id in task_ids:
        backend = MujocoFormalHouseholdDualArmBackend(
            tasks[task_id],
            bindings[task_id],
            camera_width=32,
            camera_height=24,
            evaluation_profile=True,
        )
        try:
            for correlation in correlations:
                for latency in latencies:
                    for seed in seeds:
                        source = RandomRLActionSource(
                            LatentActionScaling(),
                            RandomRLExplorationConfig(
                                motion_correlation=correlation,
                                gripper_flip_probability=0.05,
                            ),
                        )
                        name = (
                            f"rho-{correlation:.2f}.{task_id.replace('/', '-')}"
                            f".lag-{latency}.seed-{seed}.npz"
                        )
                        metadata = collect_causal_plant_episode(
                            backend,
                            source,
                            task_id=task_id,
                            seed=seed,
                            correlation=correlation,
                            action_latency_steps=latency,
                            transition_count=CONFIRMATION_TRANSITIONS,
                            output_path=run_path / "episodes" / name,
                        )
                        metadata["artifact"]["path"] = name
                        episodes.append(metadata)
        finally:
            backend.close()
    return episodes


def _manifest(
    run_path: Path,
    *,
    p09_path: Path,
    source_commit: str,
    run_id: str,
    smoke: bool,
) -> dict[str, object]:
    artifacts = [
        run_path / "report.json",
        *sorted((run_path / "episodes").glob("*.npz")),
    ]
    p09_manifest = p09_path / "manifest.json"
    return {
        "schema_version": RUN_SCHEMA,
        "proposal_id": "R0001-P11",
        "run_id": run_id,
        "source_commit": source_commit,
        "mode": "smoke" if smoke else "formal",
        "p09_input": {
            "path": str(p09_path),
            "manifest_sha256": file_sha256(p09_manifest),
        },
        "formal_default_contract": {
            "task_ids": [
                "clear_dining_table_3d/v1",
                "store_kitchen_items_3d/v1",
                "tidy_living_room_3d/v1",
            ],
            "seeds": list(CONFIRMATION_SEEDS),
            "motion_correlations": list(CONFIRMATION_CORRELATIONS),
            "action_latencies": list(CONFIRMATION_LATENCIES),
            "transition_count_per_episode": CONFIRMATION_TRANSITIONS,
            "episode_count": 144,
        },
        "artifacts": {
            str(path.relative_to(run_path)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifacts
        },
    }


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
        raise RuntimeError("causal plant evaluation requires a Git source commit")
    return commit


def _require_clean_source(root: Path) -> None:
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "formal causal plant evaluation requires a clean committed worktree"
        )


def _resolve(root: Path, requested: Path) -> Path:
    return requested if requested.is_absolute() else root / requested


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
