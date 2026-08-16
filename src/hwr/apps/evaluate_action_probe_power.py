"""Run the frozen R0001-P16 action-probe decision-power experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.eval.action_probe_power import (
    POWER_REPORT_SCHEMA,
    POWER_TRANSITIONS,
    POWER_TRIALS,
    PowerEpisode,
    run_action_probe_power,
)


DEFAULT_INPUT = Path(
    "runs/research-loop/0001/r0001-p09-observation-lag-s20260901"
)
DEFAULT_OUTPUT = Path(
    "runs/research-loop/0001/r0001-p16-probe-power-s20260916"
)
POWER_RUN_SCHEMA = "hwr.action-probe-power-run/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--trials",
        type=int,
        default=POWER_TRIALS,
        help="smoke override; formal contract requires the default 500",
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    trials = int(arguments.trials)
    formal = trials == POWER_TRIALS
    source_commit = _source_commit(root)
    if formal:
        _require_clean_source(root)
    input_path = _resolve(root, arguments.input_run)
    output_path = _resolve(root, arguments.output)
    if not formal and output_path == root / DEFAULT_OUTPUT:
        output_path = output_path.with_name(output_path.name + f"-smoke-{trials}")
    output_path.mkdir(parents=True, exist_ok=False)
    try:
        input_identity, episodes = load_power_episodes(input_path)
        report = run_action_probe_power(episodes, trials=trials)
        report.update(
            {
                "source_commit": source_commit,
                "input_run": str(input_path),
                "input_identity": input_identity,
            }
        )
        _write_json(output_path / "report.json", report)
        manifest = _manifest(
            output_path,
            source_commit=source_commit,
            input_identity=input_identity,
            report=report,
        )
        _write_json(output_path / "manifest.json", manifest)
    except BaseException:
        _write_json(
            output_path / "failure.json",
            {
                "schema_version": "hwr.action-probe-power-failure/v1",
                "proposal_id": "R0001-P16",
                "source_commit": source_commit,
                "trials": trials,
            },
        )
        raise
    return {
        "output_path": str(output_path),
        "decision": report["decision"],
        "p14_route": report["p14_route"],
        "trials": trials,
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
    }


def load_power_episodes(
    run_path: Path,
) -> tuple[dict[str, object], tuple[PowerEpisode, ...]]:
    """Load P09 Episodes after verifying every direct artifact identity."""
    run_path = run_path.resolve()
    manifest_path = run_path / "manifest.json"
    report_path = run_path / "report.json"
    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    if (
        manifest.get("schema_version")
        != "hwr.observation-action-alignment-run/v1"
        or report.get("schema_version")
        != "hwr.observation-action-alignment/v1"
        or manifest.get("mode") != "formal"
        or report.get("mode") != "formal"
        or int(report.get("episode_count", 0)) != 96
        or manifest.get("source_commit") != report.get("source_commit")
    ):
        raise ValueError("P16 requires the complete P09 formal run")
    artifacts = manifest.get("artifacts")
    entries = report.get("episodes")
    if not isinstance(artifacts, Mapping) or not isinstance(entries, list):
        raise ValueError("P09 artifact evidence is incomplete")
    episodes = []
    for entry in entries:
        relative = str(entry["artifact"]["path"])
        identity = artifacts.get(relative)
        path = _contained_member(run_path, relative)
        if (
            not isinstance(identity, Mapping)
            or not path.is_file()
            or _sha256(path) != identity.get("sha256")
            or path.stat().st_size != int(identity.get("bytes", -1))
            or identity.get("sha256") != entry["artifact"]["sha256"]
        ):
            raise ValueError("P09 Episode artifact hash differs")
        with np.load(path, allow_pickle=False) as stored:
            visible = stored["visible_proprioception"].astype(np.float64)
            plant = stored["plant_action_with_prefix"].astype(np.float64)
        lag = int(entry["observation_latency_steps"])
        state = visible[1 : POWER_TRANSITIONS + 2]
        action = plant[1 - lag : 1 - lag + POWER_TRANSITIONS]
        content = hashlib.sha256(
            state.tobytes() + action.tobytes()
        ).hexdigest()
        episodes.append(
            PowerEpisode(
                str(entry["episode_id"]),
                str(entry["task_id"]),
                str(entry["split"]),
                float(entry["motion_correlation"]),
                state,
                action,
                content,
            )
        )
    return {
        "run_path": str(run_path),
        "source_commit": manifest["source_commit"],
        "manifest_sha256": _sha256(manifest_path),
        "report_sha256": _sha256(report_path),
        "episode_artifact_count": len(episodes),
    }, tuple(episodes)


def _manifest(
    output_path: Path,
    *,
    source_commit: str,
    input_identity: Mapping[str, object],
    report: Mapping[str, object],
) -> dict[str, object]:
    report_path = output_path / "report.json"
    return {
        "schema_version": POWER_RUN_SCHEMA,
        "proposal_id": "R0001-P16",
        "source_commit": source_commit,
        "mode": report["mode"],
        "decision": report["decision"],
        "p14_route": report["p14_route"],
        "input_identity": dict(input_identity),
        "artifacts": {
            "report.json": {
                "sha256": _sha256(report_path),
                "bytes": report_path.stat().st_size,
            }
        },
    }


def _resolve(root: Path, requested: Path) -> Path:
    return requested.resolve() if requested.is_absolute() else root / requested


def _contained_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("P09 artifact escaped its run")
    return path


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
        raise RuntimeError("P16 requires a committed source identity")
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
        raise RuntimeError("formal P16 requires a clean committed worktree")


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
