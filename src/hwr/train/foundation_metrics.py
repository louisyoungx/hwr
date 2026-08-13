"""Small, atomic observability artifacts for foundation training runs."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.policy.latent_actions import LatentActionScaling


METRICS_SCHEMA = "hwr.foundation-training-metrics/v1"


def mean_metrics(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("foundation metric mean cannot be empty")
    names = set().union(*(set(item) for item in values))
    return {
        name: float(
            sum(item[name] for item in values if name in item)
            / sum(name in item for item in values)
        )
        for name in sorted(names)
    }


@dataclass(frozen=True)
class FoundationMetricsProgress:
    stage: str
    cycle: int
    update_count: int
    episode_count: int
    target_updates_in_cycle: int = 0
    completed_updates_in_cycle: int = 0


class FoundationMetricsStore:
    """Publish bounded status and immutable per-cycle summaries."""

    def __init__(
        self,
        run_path: Path,
        *,
        source_commit: str,
        target_episodes: int,
    ) -> None:
        if not source_commit or target_episodes <= 0:
            raise ValueError("foundation metric identity is invalid")
        self.run_path = run_path
        self.path = run_path / "metrics"
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": METRICS_SCHEMA,
            "source_commit": source_commit,
            "target_episodes": target_episodes,
        }
        manifest_path = self.path / "manifest.json"
        if manifest_path.is_file():
            if _read_json(manifest_path) != manifest:
                raise ValueError("foundation metrics manifest differs")
        else:
            _atomic_json(manifest_path, manifest)

    def publish_progress(
        self,
        progress: FoundationMetricsProgress,
        *,
        metrics: Mapping[str, float] | None = None,
    ) -> Path:
        if min(progress.cycle, progress.update_count, progress.episode_count) < 0:
            raise ValueError("foundation metric progress cannot be negative")
        payload: dict[str, object] = {
            "schema_version": METRICS_SCHEMA,
            "record_type": "progress",
            **progress.__dict__,
        }
        if metrics is not None:
            payload["metrics"] = _finite_metrics(metrics)
        return _atomic_json(self.path / "latest.json", payload)

    def publish_cycle(self, cycle: int, summary: Mapping[str, object]) -> Path:
        if cycle <= 0:
            raise ValueError("foundation metric cycle must be positive")
        payload = {
            "schema_version": METRICS_SCHEMA,
            "record_type": "cycle",
            "cycle": cycle,
            **_json_compatible(summary),
        }
        target = self.path / f"cycle-{cycle:06d}.json"
        if target.exists():
            if _read_json(target) != payload:
                raise ValueError("immutable foundation cycle metric differs")
        else:
            _atomic_json(target, payload)
        _atomic_json(self.path / "latest.json", payload)
        return target

    def rollback_after(self, cycle: int) -> tuple[Path, ...]:
        """Remove summaries beyond the recovery-complete checkpoint."""
        if cycle < 0:
            raise ValueError("foundation metric rollback cannot be negative")
        removed = []
        for path in sorted(self.path.glob("cycle-*.json")):
            value = path.stem.removeprefix("cycle-")
            if value.isdigit() and int(value) > cycle:
                path.unlink()
                removed.append(path)
        return tuple(removed)


def summarize_action_coverage(
    episodes: Sequence[object], scaling: LatentActionScaling
) -> dict[str, object]:
    """Summarize generic 16-D proposal/execution coverage without task semantics."""
    if not episodes:
        raise ValueError("action coverage requires at least one Episode")
    proposed = np.concatenate(
        [np.asarray(item.arrays["actor_proposal"], np.float64) for item in episodes]
    )
    executed = np.concatenate(
        [np.asarray(item.arrays["executed_action"], np.float64) for item in episodes]
    )
    if proposed.shape != executed.shape or proposed.ndim != 2 or proposed.shape[1] != 16:
        raise ValueError("action coverage requires paired canonical 16-D actions")
    scales = np.asarray(
        (scaling.base_linear, scaling.base_angular, *(scaling.arm_velocity,) * 12, 1.0, 1.0),
        np.float64,
    )
    normalized = executed / scales
    motion_saturated = np.abs(normalized[:, :14]) >= 0.95
    gripper_saturated = (normalized[:, 14:] <= 0.05) | (normalized[:, 14:] >= 0.95)
    saturation = np.concatenate((motion_saturated, gripper_saturated), axis=1)
    standard_deviation = normalized.std(axis=0)
    return {
        "transition_count": int(len(executed)),
        "normalized_executed_mean": normalized.mean(axis=0).tolist(),
        "normalized_executed_std": standard_deviation.tolist(),
        "normalized_executed_min": normalized.min(axis=0).tolist(),
        "normalized_executed_max": normalized.max(axis=0).tolist(),
        "saturation_fraction": saturation.mean(axis=0).tolist(),
        "active_dimension_fraction": float(np.mean(standard_deviation >= 0.05)),
        "effective_rank": _effective_rank(normalized),
        "proposal_execution_mean_absolute_delta": float(np.abs(proposed - executed).mean()),
        "proposal_execution_changed_fraction": float(np.mean(np.abs(proposed - executed) > 1.0e-6)),
        "gripper_switch_rate": _gripper_switch_rate(episodes),
    }


def summarize_replay_action_coverage(
    replay_path: Path,
    replay_manifest: Mapping[str, object],
    scaling: LatentActionScaling,
) -> dict[str, object]:
    episodes = []
    for shard in replay_manifest.get("shards", ()):
        with np.load(replay_path / str(shard["path"]), allow_pickle=False) as arrays:
            values = {
                "actor_proposal": arrays["actor_proposal"].copy(),
                "executed_action": arrays["executed_action"].copy(),
            }
        episodes.append(_ArrayEpisode(values))
    return summarize_action_coverage(episodes, scaling)


def summarize_episode_outcomes(episodes: Sequence[object]) -> dict[str, object]:
    if not episodes:
        raise ValueError("Episode outcome summary cannot be empty")
    returns = [float(np.asarray(item.arrays["reward"]).sum()) for item in episodes]
    safety = [
        float(np.asarray(item.arrays["safety_intervention"], np.float64).mean())
        for item in episodes
    ]
    return {
        "count": len(episodes),
        "success_count": sum(bool(item.metadata.get("success")) for item in episodes),
        "return_mean": float(np.mean(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "safety_intervention_rate_mean": float(np.mean(safety)),
        "by_task": _outcomes_by_task(episodes),
    }


def build_foundation_cycle_metrics(
    episodes: Sequence[object],
    metrics: Mapping[str, float],
    timings: Mapping[str, float],
    scaling: LatentActionScaling,
    *,
    update_count: int,
    episode_count: int,
    action_causality: Mapping[str, object] | None,
    actor_readiness: Mapping[str, object] | None,
    learning_frontier: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "update_count": update_count,
        "episode_count": episode_count,
        "training": dict(metrics),
        "action_coverage": summarize_action_coverage(episodes, scaling),
        "episodes": summarize_episode_outcomes(episodes),
        "timing_seconds": dict(timings),
        "action_causality": dict(action_causality or {}),
        "actor_readiness": dict(actor_readiness or {}),
        "learning_frontier": dict(learning_frontier or {}),
    }


def publish_foundation_progress(
    store: FoundationMetricsStore,
    stage: str,
    cycle: int,
    update_count: int,
    episode_count: int,
    updates_per_cycle: int,
    metrics: Mapping[str, float] | None = None,
    *,
    completed_updates: int = 0,
) -> None:
    store.publish_progress(
        FoundationMetricsProgress(
            stage,
            cycle,
            update_count,
            episode_count,
            updates_per_cycle if stage == "updating" else 0,
            completed_updates,
        ),
        metrics=metrics,
    )


def _outcomes_by_task(episodes: Sequence[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for task_id in sorted({str(item.task_id) for item in episodes}):
        selected = [item for item in episodes if item.task_id == task_id]
        result[task_id] = {
            "count": len(selected),
            "success_count": sum(bool(item.metadata.get("success")) for item in selected),
            "environment_metrics": [
                _numeric_mapping(item.metadata.get("result_metrics", {}))
                for item in selected
            ],
        }
    return result


def _effective_rank(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    covariance = np.cov(values, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    if total <= 1.0e-12:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    return float(math.exp(float(-(probabilities * np.log(probabilities)).sum())))


@dataclass(frozen=True)
class _ArrayEpisode:
    arrays: Mapping[str, np.ndarray]


def _gripper_switch_rate(episodes: Sequence[object]) -> float:
    switches = 0
    opportunities = 0
    for episode in episodes:
        values = np.asarray(episode.arrays["executed_action"], np.float64)[:, 14:]
        if len(values) < 2:
            continue
        switches += int(np.sum(np.abs(np.diff(values, axis=0)) > 0.5))
        opportunities += (len(values) - 1) * 2
    return float(switches / opportunities) if opportunities else 0.0


def _finite_metrics(values: Mapping[str, float]) -> dict[str, float]:
    result = {str(name): float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("foundation metrics contain non-finite values")
    return result


def _numeric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): float(item)
        for name, item in value.items()
        if isinstance(item, (int, float, np.integer, np.floating))
        and math.isfinite(float(item))
    }


def _json_compatible(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path
