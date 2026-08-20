"""Reload and evaluate a deployable bimanual Actor on unseen MuJoCo seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from hwr.adapters.mujoco import (
    MujocoBimanualEvidenceSource,
    MujocoBimanualTaskBackend,
    load_default_bimanual_training_catalogs,
)
from hwr.eval import (
    BimanualEpisodeEvaluation,
    PlannedEpisodeSeed,
    assess_bimanual_acceptance,
    combine_bimanual_reports,
    evaluate_bimanual_policy,
    plan_episode_seeds,
    random_seed_salt,
    read_seed_salt,
    seed_lineage_manifest,
)
from hwr.perception import FrozenNgramLanguageConfig, FrozenNgramLanguageEncoder
from hwr.policy import BimanualVLAActorPolicy
from hwr.policy.bimanual_input import BimanualInputConfig
from hwr.policy.vla_actions import VLAActionScaling
from hwr.render import BimanualVideoRecorder, BimanualVideoResult
from hwr.train import (
    load_bimanual_actor,
    verify_bimanual_training_run,
)


ABLATIONS = ("none", "lock_left", "lock_right")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/bimanual-eval")
    )
    parser.add_argument("--evaluation-id")
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-salt-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-seed-count", type=int, default=1)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    return parser


class _EvaluationVideoObserver:
    def __init__(
        self,
        output_directory: Path,
        recorded_seeds: set[int],
        *,
        width: int,
        height: int,
    ) -> None:
        self.output_directory = output_directory
        self.recorded_seeds = recorded_seeds
        self.width = width
        self.height = height
        self.source: MujocoBimanualEvidenceSource | None = None
        self.recorder: BimanualVideoRecorder | None = None
        self.results: list[dict[str, Any]] = []

    def episode_started(
        self, backend, observation, *, seed: int, ablation: str
    ) -> None:
        if self.source is not None or self.recorder is not None:
            raise RuntimeError("previous evidence recording was not closed")
        if seed not in self.recorded_seeds or ablation != "none":
            return
        if not isinstance(backend, MujocoBimanualTaskBackend):
            raise TypeError("MuJoCo video observer received another backend")
        basename = f"{observation.task_id.replace('/', '_')}.seed-{seed}"
        self.source = MujocoBimanualEvidenceSource(
            backend, width=self.width, height=self.height
        )
        self.recorder = BimanualVideoRecorder(
            self.output_directory,
            basename,
            width=self.width,
            height=self.height,
            frames_per_second=round(backend.task.control_hz),
        )
        self.recorder.append(self.source.capture(observation))

    def step_recorded(self, backend, observation, *, step: int) -> None:
        del backend, step
        if self.source is not None and self.recorder is not None:
            self.recorder.append(self.source.capture(observation))

    def episode_finished(
        self, backend, record: BimanualEpisodeEvaluation
    ) -> None:
        del backend
        if self.source is None or self.recorder is None:
            return
        result = self.recorder.close()
        self.source.close()
        self.results.append(_video_result(record, result))
        self.source = None
        self.recorder = None

    def abort(self) -> None:
        if self.recorder is not None:
            self.recorder.abort()
        if self.source is not None:
            self.source.close()
        self.source = None
        self.recorder = None


def _video_result(
    record: BimanualEpisodeEvaluation, result: BimanualVideoResult
) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "seed": record.seed,
        "success": record.success,
        "frame_count": result.frame_count,
        "duration_seconds": result.duration_seconds,
        "views": {name: str(path) for name, path in result.paths.items()},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _training_seeds(run_path: Path) -> set[int]:
    lines = (run_path / "episodes.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    return {int(json.loads(line)["seed"]) for line in lines if line}


def _unseen_seeds(
    run_path: Path, count: int, requested_start: int | None
) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("evaluation seed count must be positive")
    manifest = _read_json(run_path / "manifest.json")
    training_seed = int(manifest["training_config"]["seed"])
    candidate = (
        int(requested_start)
        if requested_start is not None
        else training_seed + 1_000_000_007
    )
    training = _training_seeds(run_path)
    selected: list[int] = []
    while len(selected) < count:
        if candidate not in training:
            selected.append(candidate)
        candidate += 104729
    return tuple(selected)


def _load_deployment_policy(
    run_path: Path, *, device: str
) -> BimanualVLAActorPolicy:
    manifest = verify_bimanual_training_run(run_path)
    model_manifest = _read_json(run_path / "model-manifest.json")
    audit = _read_json(run_path / "actor-input-audit.json")
    training = manifest["training_config"]
    actor = load_bimanual_actor(run_path, device=device)
    language = FrozenNgramLanguageEncoder(
        FrozenNgramLanguageConfig(dimension=int(training["language_dim"]))
    )
    if (
        language.encoder_id != audit["language_encoder_id"]
        or language.weights_sha256 != audit["language_weights_sha256"]
    ):
        raise ValueError("evaluation language encoder differs from checkpoint")
    input_config = BimanualInputConfig(
        int(training["raw_image_width"]),
        int(training["raw_image_height"]),
        image_width=int(training["image_width"]),
        image_height=int(training["image_height"]),
        point_count=int(training["point_count"]),
        visual_history=actor.config.visual_history,
        action_history=actor.config.action_history,
    )
    scaling = VLAActionScaling(**model_manifest["rl_action_scaling"])
    actor_sha = model_manifest["actor_sha256"]
    return BimanualVLAActorPolicy(
        actor,
        input_config,
        language,
        scaling,
        policy_id=f"{manifest['run_id']}:{actor_sha[:16]}",
        preprocess_fingerprint=audit["preprocess_fingerprint"],
        device=device,
    )


def _evaluation_manifest(
    output_path: Path,
    run_path: Path,
    seeds: tuple[int, ...],
    videos: Sequence[Mapping[str, Any]],
    plan_id: str,
    salt: str,
    planned_episodes: Sequence[PlannedEpisodeSeed],
) -> dict[str, Any]:
    files = (output_path / "report.json", output_path / "acceptance.json")
    video_files = tuple(
        Path(path)
        for video in videos
        for path in video["views"].values()
    )
    return {
        "schema_version": "hwr.bimanual-evaluation-run/v2",
        "training_run": str(run_path),
        "training_manifest_sha256": _sha256(run_path / "manifest.json"),
        "actor_sha256": _read_json(run_path / "model-manifest.json")[
            "actor_sha256"
        ],
        "unseen_seeds": list(seeds),
        "seed_lineage": seed_lineage_manifest(
            plan_id, salt, planned_episodes
        ),
        "ablations": list(ABLATIONS),
        "videos": list(videos),
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (*files, *video_files)
        },
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.video_seed_count < 0:
        raise ValueError("video seed count cannot be negative")
    root = Path(__file__).resolve().parents[3]
    run_path = arguments.run_path.resolve()
    manifest = verify_bimanual_training_run(run_path)
    seeds = _unseen_seeds(run_path, arguments.seed_count, arguments.seed_start)
    evaluation_id = arguments.evaluation_id or manifest["run_id"]
    plan_id = f"bimanual-evaluation:{evaluation_id}"
    salt = (
        read_seed_salt(arguments.seed_salt_file.resolve())
        if arguments.seed_salt_file is not None
        else random_seed_salt()
    )
    output_root = (
        arguments.output_root
        if arguments.output_root.is_absolute()
        else root / arguments.output_root
    )
    output_path = output_root / evaluation_id
    output_path.mkdir(parents=True, exist_ok=False)
    tasks, bindings = load_default_bimanual_training_catalogs(root)
    policy = _load_deployment_policy(run_path, device=arguments.device)
    observer = _EvaluationVideoObserver(
        output_path / "videos",
        set(seeds[: arguments.video_seed_count]),
        width=arguments.video_width,
        height=arguments.video_height,
    )
    reports = []
    planned_episodes: list[PlannedEpisodeSeed] = []
    try:
        for task_id in sorted(tasks):
            task = tasks[task_id]

            def environment_factory(task_id=task_id):
                training = manifest["training_config"]
                return MujocoBimanualTaskBackend(
                    tasks[task_id],
                    bindings[task_id],
                    camera_width=int(training["raw_image_width"]),
                    camera_height=int(training["raw_image_height"]),
                )

            for ablation in ABLATIONS:
                episode_seeds = plan_episode_seeds(
                    plan_id,
                    task_id,
                    ablation,
                    len(seeds),
                    salt,
                    environment_seeds=seeds,
                )
                planned_episodes.extend(episode_seeds)
                reports.append(
                    evaluate_bimanual_policy(
                        task_id,
                        task.max_steps,
                        environment_factory,
                        policy,
                        episode_seeds,
                        ablation=ablation,
                        observer=observer,
                    )
                )
    finally:
        observer.abort()
        policy.close()
    report = combine_bimanual_reports(reports)
    acceptance = assess_bimanual_acceptance(
        report, {task_id: task.control_hz for task_id, task in tasks.items()}
    )
    _write_json(output_path / "report.json", report.to_dict())
    _write_json(output_path / "acceptance.json", acceptance)
    _write_json(
        output_path / "manifest.json",
        _evaluation_manifest(
            output_path,
            run_path,
            seeds,
            observer.results,
            plan_id,
            salt,
            planned_episodes,
        ),
    )
    return {
        "output_path": str(output_path),
        "passed": acceptance["passed"],
        "episode_count": len(report.episodes),
        "video_episode_count": len(observer.results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    value = run(build_parser().parse_args(argv))
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
