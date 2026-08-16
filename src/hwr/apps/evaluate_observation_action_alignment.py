"""Run the frozen R0001-P09 observation-to-plant-action alignment diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hwr.eval.observation_action_alignment import (
    ALIGNMENT_BOOTSTRAP_SAMPLES,
    ALIGNMENT_BOOTSTRAP_SEED,
    ALIGNMENT_CORRELATIONS,
    ALIGNMENT_HOLDOUT_SEEDS,
    ALIGNMENT_TASK_IDS,
    ALIGNMENT_TRAINING_SEEDS,
    ALIGNMENT_TRANSITIONS,
    AlignmentEpisode,
    build_alignment_episode_plan,
    evaluate_observation_action_alignment,
)
from hwr.eval.observation_action_collection import (
    collect_alignment_episode,
    file_sha256,
)
from hwr.policy.latent_actions import LatentActionScaling
from hwr.train.foundation_exploration import (
    RandomRLActionSource,
    RandomRLExplorationConfig,
)


DEFAULT_RUN_ID = "r0001-p09-observation-lag-s20260901"
DEFAULT_OUTPUT_ROOT = Path("runs/research-loop/0001")
MANIFEST_SCHEMA = "hwr.observation-action-alignment-run/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one task with two training and two holdout seeds per rho",
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    from hwr.adapters.mujoco import (
        MujocoFormalHouseholdDualArmBackend,
        load_default_formal_household_catalogs,
    )

    root = Path(__file__).resolve().parents[3]
    plan = _plan(smoke=bool(arguments.smoke))
    run_id = str(arguments.run_id)
    if arguments.smoke and run_id == DEFAULT_RUN_ID:
        run_id = f"{DEFAULT_RUN_ID}-smoke"
    source_commit = _source_commit(root)
    if not arguments.smoke:
        _require_clean_source(root)
    output_root = _resolve_output_root(root, Path(arguments.output_root))
    run_path = output_root / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    episode_path = run_path / "episodes"
    tasks, bindings = load_default_formal_household_catalogs(root)
    episodes: list[AlignmentEpisode] = []
    backends: dict[str, MujocoFormalHouseholdDualArmBackend] = {}
    try:
        for item in plan:
            backend = backends.get(item.task_id)
            if backend is None:
                backend = MujocoFormalHouseholdDualArmBackend(
                    tasks[item.task_id],
                    bindings[item.task_id],
                    camera_width=32,
                    camera_height=24,
                )
                backends[item.task_id] = backend
            source = RandomRLActionSource(
                LatentActionScaling(),
                RandomRLExplorationConfig(
                    motion_correlation=item.correlation,
                    gripper_flip_probability=0.05,
                ),
            )
            artifact = episode_path / f"{item.episode_id}.npz"
            episodes.append(
                collect_alignment_episode(
                    backend,
                    source,
                    item,
                    artifact,
                    transition_count=ALIGNMENT_TRANSITIONS,
                    artifact_root=run_path,
                )
            )
    except BaseException:
        _write_json(
            run_path / "failure.json",
            {
                "schema_version": "hwr.observation-action-alignment-failure/v1",
                "proposal_id": "R0001-P09",
                "completed_episode_count": len(episodes),
                "expected_episode_count": len(plan),
            },
        )
        raise
    finally:
        for backend in backends.values():
            backend.close()
    report = evaluate_observation_action_alignment(
        episodes,
        plan,
        transition_count=ALIGNMENT_TRANSITIONS,
        bootstrap_samples=ALIGNMENT_BOOTSTRAP_SAMPLES,
        bootstrap_seed=ALIGNMENT_BOOTSTRAP_SEED,
    )
    report.update(
        {
            "run_id": run_id,
            "source_commit": source_commit,
            "mode": "smoke" if arguments.smoke else "formal",
        }
    )
    _write_json(run_path / "report.json", report)
    manifest = _manifest(
        run_path,
        run_id=run_id,
        source_commit=source_commit,
        smoke=bool(arguments.smoke),
        episode_count=len(episodes),
    )
    _write_json(run_path / "manifest.json", manifest)
    return {
        "run_path": str(run_path),
        "decision": report["decision"],
        "episode_count": len(episodes),
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
    }


def _plan(*, smoke: bool):
    if not smoke:
        return build_alignment_episode_plan()
    return build_alignment_episode_plan(
        task_ids=ALIGNMENT_TASK_IDS[:1],
        training_seeds=ALIGNMENT_TRAINING_SEEDS[:2],
        holdout_seeds=ALIGNMENT_HOLDOUT_SEEDS[:2],
        correlations=ALIGNMENT_CORRELATIONS,
    )


def _manifest(
    run_path: Path,
    *,
    run_id: str,
    source_commit: str,
    smoke: bool,
    episode_count: int,
) -> dict[str, object]:
    artifact_paths = [run_path / "report.json", *sorted((run_path / "episodes").glob("*.npz"))]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": "R0001-P09",
        "run_id": run_id,
        "source_commit": source_commit,
        "mode": "smoke" if smoke else "formal",
        "formal_default_contract": {
            "task_ids": list(ALIGNMENT_TASK_IDS),
            "training_seeds": list(ALIGNMENT_TRAINING_SEEDS),
            "holdout_seeds": list(ALIGNMENT_HOLDOUT_SEEDS),
            "observation_lag_schedule": [0, 1, 0, 1, 0, 1, 0, 1],
            "motion_correlations": list(ALIGNMENT_CORRELATIONS),
            "gripper_flip_probability": 0.05,
            "transition_count_per_episode": ALIGNMENT_TRANSITIONS,
            "prefix_action_count_per_episode": 1,
            "bootstrap_samples": ALIGNMENT_BOOTSTRAP_SAMPLES,
            "bootstrap_seed": ALIGNMENT_BOOTSTRAP_SEED,
        },
        "episode_count": episode_count,
        "artifacts": {
            str(path.relative_to(run_path)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
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
        raise RuntimeError("alignment diagnostic requires a Git source commit")
    return commit


def _resolve_output_root(root: Path, requested: Path) -> Path:
    return requested if requested.is_absolute() else root / requested


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
            "formal alignment diagnostic requires a clean committed worktree"
        )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
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
