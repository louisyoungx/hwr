"""Evaluate the stripped deterministic foundation Actor on fixed unseen seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from hwr.adapters.mujoco import (
    BIMANUAL_EVIDENCE_VIEWS,
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
from hwr.train.foundation_registry import (
    ACTION_CAUSALITY_SCHEMA,
    DEPLOYMENT_SCHEMA,
    TRAINING_CHECKPOINT_SCHEMA,
    foundation_lineage,
    load_foundation_deployment,
    require_foundation_lineage,
)
from hwr.world_model import (
    ACTION_CAUSALITY_COMPONENTS,
    ActionCausalityCriteria,
    assess_action_causality,
    counterfactual_report_from_dict,
)


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
        candidate_seeds: set[int],
        *,
        successful_videos_per_task: int,
        width: int,
        height: int,
    ) -> None:
        self.output_directory = output_directory
        self.candidate_seeds = candidate_seeds
        self.successful_videos_per_task = successful_videos_per_task
        self.width = width
        self.height = height
        self.source: MujocoBimanualEvidenceSource | None = None
        self.recorder: BimanualVideoRecorder | None = None
        self.results: list[dict[str, Any]] = []
        self.success_counts: dict[str, int] = {}

    def episode_started(
        self, backend, observation, *, seed: int, ablation: str
    ) -> None:
        if self.source is not None or self.recorder is not None:
            raise RuntimeError("previous evaluation recording was not closed")
        task_id = observation.task_id
        if (
            seed not in self.candidate_seeds
            or ablation != "none"
            or self.success_counts.get(task_id, 0) >= self.successful_videos_per_task
        ):
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
        try:
            video = self.recorder.close()
            if video.frame_count != record.steps + 1:
                for path in video.paths.values():
                    path.unlink()
                raise RuntimeError("evaluation video omitted control-loop frames")
            if record.success:
                self.results.append(_video_result(record, video))
                self.success_counts[record.task_id] = (
                    self.success_counts.get(record.task_id, 0) + 1
                )
            else:
                for path in video.paths.values():
                    path.unlink()
        finally:
            self.source.close()
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
        "expected_frame_count": record.steps + 1,
        "uncut": video.frame_count == record.steps + 1,
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


def _run_member(run_path: Path, relative: object) -> Path:
    value = Path(str(relative))
    if value.is_absolute():
        raise ValueError("training artifact path must be relative to its run")
    root = run_path.resolve()
    result = (root / value).resolve()
    if result == root or root not in result.parents:
        raise ValueError("training artifact path escaped its run")
    return result


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
    holdout = _read_json(
        run_path / "causality-holdout/autonomous/manifest.json"
    )
    seen = training | {int(shard["seed"]) for shard in holdout["shards"]}
    seeds: list[int] = []
    while len(seeds) < count:
        if start not in seen:
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
    if latest.get("schema_version") != "hwr.foundation-online-latest/v1":
        raise ValueError("training latest schema differs")
    run_manifest = _read_json(run_path / "run-manifest.json")
    if run_manifest.get("schema_version") != "hwr.foundation-online-run/v2":
        raise ValueError("training run schema differs")
    path = _run_member(run_path, latest["action_causality_report"])
    if _sha256(path) != latest.get("action_causality_sha256"):
        raise ValueError("training action causality report hash differs")
    report = _read_json(path)
    _require_causality_structure(report, run_manifest)
    if report["assessment"]["passed"] is not True:
        raise RuntimeError("evaluation requires passed action-shuffle causality")
    checkpoint_path = _run_member(run_path, latest["training_checkpoint"])
    deployment_path = _run_member(run_path, latest["deployment"])
    checkpoint = _read_json(checkpoint_path / "manifest.json")
    deployment = _read_json(deployment_path / "manifest.json")
    if checkpoint.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("training checkpoint schema differs")
    if deployment.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("foundation deployment schema differs")
    checkpoint_artifact = checkpoint_path / str(checkpoint["artifact_file"])
    if _sha256(checkpoint_artifact) != checkpoint.get("artifact_sha256"):
        raise ValueError("training checkpoint artifact hash differs")
    if deployment.get("training_checkpoint_sha256") != _sha256(checkpoint_artifact):
        raise ValueError("deployment and training checkpoint hash differ")
    _require_causality_lineage(
        run_path, latest, run_manifest, report, checkpoint, deployment
    )
    checkpoint_diagnostics = checkpoint.get("training_diagnostics", {})
    if checkpoint_diagnostics.get("action_causality_report_sha256") != _sha256(path):
        raise ValueError("checkpoint and action causality provenance differ")
    if checkpoint_diagnostics.get("action_causality_passed") is not True:
        raise ValueError("checkpoint did not pass action causality")
    diagnostics = deployment.get("training_diagnostics", {})
    if diagnostics.get("action_causality_report_sha256") != _sha256(path):
        raise ValueError("deployment and action causality provenance differ")
    if diagnostics.get("action_causality_passed") is not True:
        raise ValueError("deployment was exported before causality passed")
    return path


def _require_causality_structure(
    report: Mapping[str, Any], run_manifest: Mapping[str, Any]
) -> None:
    expected_tasks = {
        str(task["task_id"]) for task in run_manifest.get("tasks", ())
    }
    partitions = report.get("partitions", {})
    assessment = report.get("assessment", {})
    training = run_manifest["training_config"]
    criteria = ActionCausalityCriteria(
        float(training["minimum_action_causality_ratio"]),
        float(training["minimum_action_causality_horizon_fraction"]),
    )
    _require_assessment_matches(
        report.get("report"), assessment, criteria, "aggregate"
    )
    for task_id, partition in partitions.items():
        _require_assessment_matches(
            partition.get("report"),
            partition.get("assessment", {}),
            criteria,
            task_id,
        )
    required_components = set(ACTION_CAUSALITY_COMPONENTS)
    component_assessments = assessment.get("components", {})
    if (
        report.get("schema_version") != ACTION_CAUSALITY_SCHEMA
        or report.get("action_source") != "actual_executed_action"
        or report.get("safety_action_source") != "actor_proposal"
        or report.get("counterfactual_pairing")
        != "proposal-executed-pair/v1"
        or report.get("counterfactual_transform")
        != "deterministic-global-derangement/v1"
        or report.get("partition_key") != "task_id"
        or not expected_tasks
        or set(partitions) != expected_tasks
        or assessment.get("aggregate_passed") is not True
        or assessment.get("all_components_passed") is not True
        or set(component_assessments) != required_components
        or any(
            value.get("passed") is not True
            for value in component_assessments.values()
        )
        or assessment.get("all_partitions_passed") is not True
        or any(
            value.get("assessment", {}).get("passed") is not True
            or value.get("assessment", {}).get("all_components_passed") is not True
            or set(value.get("assessment", {}).get("components", {}))
            != required_components
            for value in partitions.values()
        )
    ):
        raise ValueError("action causality partition evidence is incomplete")
    selected = report.get("window_selection", ())
    configured = int(
        run_manifest["training_config"]["causality_audit_windows_per_task"]
    )
    counts = {task_id: 0 for task_id in expected_tasks}
    intervals: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for window in selected:
        task_id = str(window.get("task_id"))
        start, stop = int(window.get("transition_start", -1)), int(
            window.get("transition_stop", -1)
        )
        if task_id not in counts or start < 0 or stop <= start:
            raise ValueError("action causality window selection is invalid")
        counts[task_id] += 1
        intervals.setdefault((task_id, str(window.get("episode_id"))), []).append(
            (start, stop)
        )
    if any(value != configured for value in counts.values()):
        raise ValueError("action causality window coverage differs from training")
    if any(_overlaps(values) for values in intervals.values()):
        raise ValueError("action causality windows overlap")


def _require_assessment_matches(
    raw_report: object,
    claimed: Mapping[str, Any],
    criteria: ActionCausalityCriteria,
    label: str,
) -> None:
    if not isinstance(raw_report, Mapping):
        raise ValueError(f"action causality raw evidence is missing: {label}")
    expected = assess_action_causality(
        counterfactual_report_from_dict(raw_report), criteria
    )
    if any(claimed.get(name) != value for name, value in expected.items()):
        raise ValueError(f"action causality assessment differs from evidence: {label}")


def _overlaps(intervals: Sequence[tuple[int, int]]) -> bool:
    ordered = sorted(intervals)
    return any(
        current[0] < previous[1]
        for previous, current in zip(ordered, ordered[1:])
    )


def _require_causality_lineage(
    run_path: Path,
    latest: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    deployment: Mapping[str, Any],
) -> None:
    source_commit = str(run_manifest.get("source_commit", ""))
    expected_lineage = foundation_lineage(source_commit)
    if not source_commit or {
        str(report.get("source_commit", "")),
        str(checkpoint.get("lineage", {}).get("source_commit", "")),
        str(deployment.get("source_commit", "")),
    } != {source_commit}:
        raise ValueError("foundation source lineage differs")
    require_foundation_lineage(
        run_manifest.get("lineage"), source_commit=source_commit
    )
    if checkpoint.get("lineage") != expected_lineage:
        raise ValueError("foundation checkpoint no-expert lineage differs")
    update_count = int(latest.get("update_count", -1))
    if update_count <= 0 or {
        int(report.get("update_count", -2)),
        int(checkpoint.get("update_count", -3)),
    } != {update_count}:
        raise ValueError("foundation update lineage differs")
    training_manifest = run_path / "replay/autonomous/manifest.json"
    audit_manifest = run_path / "causality-holdout/autonomous/manifest.json"
    training_sha = _sha256(training_manifest)
    audit_sha = _sha256(audit_manifest)
    if (
        report.get("training_data_manifest_sha256") != training_sha
        or checkpoint.get("data_manifest_sha256") != training_sha
        or report.get("audit_data_manifest_sha256") != audit_sha
        or report.get("holdout_collector") != "foundation-causality-holdout/v1"
    ):
        raise ValueError("action causality data provenance differs")


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
    deployment_manifest = _run_member(run_path, latest["deployment"]) / "manifest.json"
    return {
        "schema_version": "hwr.foundation-evaluation-run/v1",
        "training_run": str(run_path),
        "training_run_manifest_sha256": _sha256(run_path / "run-manifest.json"),
        "deployment_manifest_sha256": _sha256(deployment_manifest),
        "action_causality_report": str(causality),
        "action_causality_report_sha256": _sha256(causality),
        "unseen_seeds": list(seeds),
        "seed_exclusions": ["training_episodes", "causality_holdout"],
        "ablations": list(ABLATIONS),
        "videos": list(videos),
        "artifacts": {
            name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
    }


def _video_acceptance(
    videos: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str],
    successful_videos_per_task: int,
) -> dict[str, Any]:
    if successful_videos_per_task <= 0 or not task_ids:
        raise ValueError("successful video evidence configuration is invalid")
    counts = {
        task_id: sum(
            video.get("task_id") == task_id
            and video.get("success") is True
            and video.get("uncut") is True
            and set(video.get("views", ())) == set(BIMANUAL_EVIDENCE_VIEWS)
            for video in videos
        )
        for task_id in task_ids
    }
    passed = all(
        count >= successful_videos_per_task for count in counts.values()
    )
    return {
        "passed": passed,
        "successful_uncut_videos_per_task": counts,
        "required_per_task": successful_videos_per_task,
        "required_views": list(BIMANUAL_EVIDENCE_VIEWS),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.video_seed_count <= 0:
        raise ValueError("at least one successful video is required per task")
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
        set(seeds),
        successful_videos_per_task=arguments.video_seed_count,
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
    video_evidence = _video_acceptance(
        observer.results, tuple(sorted(tasks)), arguments.video_seed_count
    )
    acceptance["video_evidence"] = video_evidence
    acceptance["passed"] = acceptance["passed"] and video_evidence["passed"]
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
