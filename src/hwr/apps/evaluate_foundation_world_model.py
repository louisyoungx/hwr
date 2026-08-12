"""Evaluate the stripped deterministic foundation Actor on fixed unseen seeds."""

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
from hwr.data.foundation_cache import FoundationCacheKey, FoundationFeatureCache
from hwr.data.foundation_features import load_feature_index
from hwr.eval import (
    BimanualEpisodeEvaluation,
    assess_bimanual_acceptance,
    combine_bimanual_reports,
    evaluate_bimanual_policy,
)
from hwr.perception.foundation import language_source_sha256
from hwr.perception.high_resolution import (
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
)
from hwr.perception.language_cache import StaticLanguageFeatureResolver
from hwr.policy.bimanual_input import (
    BimanualInputConfig,
    default_four_camera_calibrations,
)
from hwr.policy.foundation_runtime import FoundationWorldModelPolicy
from hwr.render import BimanualVideoRecorder, BimanualVideoResult
from hwr.train.foundation_registry import load_foundation_deployment


ABLATIONS = ("none", "lock_left", "lock_right")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/foundation-world-model-eval")
    )
    parser.add_argument("--evaluation-id")
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-seed-count", type=int, default=1)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    return parser


class _VideoObserver:
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
            raise RuntimeError("previous evaluation recording was not closed")
        if seed not in self.recorded_seeds or ablation != "none":
            return
        if not isinstance(backend, MujocoBimanualTaskBackend):
            raise TypeError("foundation video observer requires a MuJoCo backend")
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
        video = self.recorder.close()
        self.source.close()
        self.results.append(_video_result(record, video))
        self.source = None
        self.recorder = None

    def abort(self) -> None:
        if self.recorder is not None:
            self.recorder.abort()
        if self.source is not None:
            self.source.close()
        self.recorder = None
        self.source = None


def _video_result(
    record: BimanualEpisodeEvaluation, video: BimanualVideoResult
) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "seed": record.seed,
        "success": record.success,
        "frame_count": video.frame_count,
        "duration_seconds": video.duration_seconds,
        "views": {name: str(path) for name, path in video.paths.items()},
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unseen_seeds(
    run_path: Path, count: int, requested_start: int | None
) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("evaluation seed count must be positive")
    run_manifest = _read_json(run_path / "run-manifest.json")
    start = (
        int(requested_start)
        if requested_start is not None
        else int(run_manifest["training_config"]["seed"]) + 1_000_000_007
    )
    training = {
        int(json.loads(line)["seed"])
        for line in (run_path / "episodes.jsonl").read_text().splitlines()
        if line
    }
    seeds: list[int] = []
    while len(seeds) < count:
        if start not in training:
            seeds.append(start)
        start += 104729
    return tuple(seeds)


def _preprocessor(run_manifest: Mapping[str, Any]) -> HighResolutionVisionPreprocessor:
    training = run_manifest["training_config"]
    raw = BimanualInputConfig(
        int(training["camera_width"]),
        int(training["camera_height"]),
        image_width=160,
        image_height=160,
    )
    config = HighResolutionVisionConfig(
        **run_manifest["preprocessing"]["config"]
    )
    result = HighResolutionVisionPreprocessor(
        config, default_four_camera_calibrations(raw)
    )
    if result.fingerprint != run_manifest["preprocessing"]["fingerprint"]:
        raise ValueError("evaluation preprocessing differs from the training run")
    return result


def _language_resolver(run_path: Path) -> StaticLanguageFeatureResolver:
    index = load_feature_index(run_path / "features/language.json")
    cache = FoundationFeatureCache(run_path / "feature-cache")
    replay = _read_json(run_path / "replay/autonomous/manifest.json")
    values = {}
    for shard in replay["shards"]:
        text, locale = shard["instruction"], shard["locale"]
        source = language_source_sha256(text, locale)
        key = FoundationCacheKey(
            "language", source, index.encoder_lock_sha256, index.preprocess_sha256
        )
        values[(locale, text)] = cache.load_language(key).values.copy()
    return StaticLanguageFeatureResolver(
        values,
        encoder_lock_sha256=index.encoder_lock_sha256,
        output_dimension=index.output_dimension,
    )


def _policy(run_path: Path, *, device: str) -> FoundationWorldModelPolicy:
    latest = _read_json(run_path / "latest.json")
    deployment_path = run_path / latest["deployment"]
    components = load_foundation_deployment(deployment_path, device=device)
    run_manifest = _read_json(run_path / "run-manifest.json")
    return FoundationWorldModelPolicy(
        components.visual_student,
        components.state_filter,
        components.actor,
        _preprocessor(run_manifest),
        _language_resolver(run_path),
        components.action_scaling,
        policy_id=(
            f"{run_path.name}:{components.manifest['artifact_sha256'][:16]}"
        ),
        device=device,
    )


def _require_action_causality(run_path: Path) -> Path:
    latest = _read_json(run_path / "latest.json")
    path = run_path / str(latest["action_causality_report"])
    if _sha256(path) != latest.get("action_causality_sha256"):
        raise ValueError("training action causality report hash differs")
    report = _read_json(path)
    if report.get("assessment", {}).get("passed") is not True:
        raise RuntimeError("evaluation requires passed action-shuffle causality")
    checkpoint = _read_json(
        run_path / latest["training_checkpoint"] / "manifest.json"
    )
    checkpoint_diagnostics = checkpoint.get("training_diagnostics", {})
    if checkpoint_diagnostics.get("action_causality_report_sha256") != _sha256(path):
        raise ValueError("checkpoint and action causality provenance differ")
    if checkpoint_diagnostics.get("action_causality_passed") is not True:
        raise ValueError("checkpoint did not pass action causality")
    deployment = _read_json(run_path / latest["deployment"] / "manifest.json")
    diagnostics = deployment.get("training_diagnostics", {})
    if diagnostics.get("action_causality_report_sha256") != _sha256(path):
        raise ValueError("deployment and action causality provenance differ")
    if diagnostics.get("action_causality_passed") is not True:
        raise ValueError("deployment was exported before causality passed")
    return path


def _artifact_manifest(
    output_path: Path,
    run_path: Path,
    seeds: tuple[int, ...],
    videos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    causality = _require_action_causality(run_path)
    files = {
        "report.json": output_path / "report.json",
        "acceptance.json": output_path / "acceptance.json",
        "training/action-causality.json": causality,
    }
    files.update(
        {
            f"videos/{Path(path).name}": Path(path)
            for video in videos
            for path in video["views"].values()
        }
    )
    latest = _read_json(run_path / "latest.json")
    deployment_manifest = run_path / latest["deployment"] / "manifest.json"
    return {
        "schema_version": "hwr.foundation-evaluation-run/v1",
        "training_run": str(run_path),
        "training_run_manifest_sha256": _sha256(run_path / "run-manifest.json"),
        "deployment_manifest_sha256": _sha256(deployment_manifest),
        "action_causality_report": str(causality),
        "action_causality_report_sha256": _sha256(causality),
        "unseen_seeds": list(seeds),
        "ablations": list(ABLATIONS),
        "videos": list(videos),
        "artifacts": {
            name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.video_seed_count < 0:
        raise ValueError("video seed count cannot be negative")
    root = Path(__file__).resolve().parents[3]
    run_path = arguments.run_path.resolve()
    _require_action_causality(run_path)
    run_manifest = _read_json(run_path / "run-manifest.json")
    seeds = _unseen_seeds(run_path, arguments.seed_count, arguments.seed_start)
    output_root = (
        arguments.output_root
        if arguments.output_root.is_absolute()
        else root / arguments.output_root
    )
    output_path = output_root / (arguments.evaluation_id or run_path.name)
    output_path.mkdir(parents=True, exist_ok=False)
    tasks, bindings = load_default_bimanual_training_catalogs(root)
    policy = _policy(run_path, device=arguments.device)
    observer = _VideoObserver(
        output_path / "videos",
        set(seeds[: arguments.video_seed_count]),
        width=arguments.video_width,
        height=arguments.video_height,
    )
    reports = []
    try:
        for task_id in sorted(tasks):
            task = tasks[task_id]

            def environment_factory(task_id=task_id):
                training = run_manifest["training_config"]
                return MujocoBimanualTaskBackend(
                    tasks[task_id],
                    bindings[task_id],
                    camera_width=int(training["camera_width"]),
                    camera_height=int(training["camera_height"]),
                )

            for ablation in ABLATIONS:
                reports.append(
                    evaluate_bimanual_policy(
                        task_id,
                        task.max_steps,
                        environment_factory,
                        policy,
                        seeds,
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
    manifest = _artifact_manifest(
        output_path, run_path, seeds, observer.results
    )
    _write_json(output_path / "manifest.json", manifest)
    return {
        "output_path": str(output_path),
        "passed": acceptance["passed"],
        "episode_count": len(report.episodes),
        "video_count": len(observer.results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
