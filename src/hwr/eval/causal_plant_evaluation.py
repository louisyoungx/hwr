"""Episode loading and frozen reporting for the R0001-P11 estimator gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.eval.causal_plant_estimator import (
    PLANT_STABLE_START,
    action_out_of_bounds_rate,
    current_proposal_baseline,
    deterministic_proposal_derangement,
    estimate_causal_plant_actions,
    normalized_action_rmse,
)


@dataclass(frozen=True)
class PlantEpisode:
    task_id: str
    seed: int
    correlation: float
    action_latency_steps: int
    proposals: np.ndarray
    applied_actions: np.ndarray
    safety_interventions: np.ndarray
    severe_collisions: int = 0
    terminated_early: bool = False
    provenance_complete: bool = True

    def __post_init__(self) -> None:
        proposal = np.asarray(self.proposals, np.float64)
        applied = np.asarray(self.applied_actions, np.float64)
        safety = np.asarray(self.safety_interventions, np.bool_)
        if (
            not self.task_id
            or self.seed < 0
            or self.correlation not in (0.50, 0.96)
            or self.action_latency_steps not in (0, 1, 2, 3)
            or proposal.ndim != 2
            or proposal.shape[1] != 16
            or applied.shape != proposal.shape
            or safety.shape != (len(proposal),)
            or not np.isfinite(proposal).all()
            or not np.isfinite(applied).all()
        ):
            raise ValueError("plant evaluation Episode is invalid")
        object.__setattr__(self, "proposals", proposal)
        object.__setattr__(self, "applied_actions", applied)
        object.__setattr__(self, "safety_interventions", safety)


def load_p09_episodes(run_path: Path) -> tuple[PlantEpisode, ...]:
    report = _read_json(run_path / "report.json")
    manifest = _read_json(run_path / "manifest.json")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, Mapping):
        raise TypeError("P09 manifest omitted artifacts")
    report_identity = manifest_artifacts.get("report.json")
    if (
        not isinstance(report_identity, Mapping)
        or _file_sha256(run_path / "report.json")
        != str(report_identity.get("sha256"))
    ):
        raise ValueError("P09 report hash differs from its manifest")
    episodes = []
    for metadata in report.get("episodes", ()):
        artifact = metadata["artifact"]
        artifact_path = run_path / str(artifact["path"])
        manifest_identity = manifest_artifacts.get(str(artifact["path"]))
        if (
            not isinstance(manifest_identity, Mapping)
            or str(manifest_identity.get("sha256")) != str(artifact["sha256"])
            or _file_sha256(artifact_path) != str(artifact["sha256"])
        ):
            raise ValueError(f"P09 artifact hash differs: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as arrays:
            proposals = arrays["actor_proposal_with_prefix"].copy()
            applied = arrays["plant_action_with_prefix"].copy()
            safety = arrays["safety_intervention_with_prefix"].copy()
        episodes.append(
            PlantEpisode(
                task_id=str(metadata["task_id"]),
                seed=int(metadata["seed"]),
                correlation=float(metadata["motion_correlation"]),
                action_latency_steps=int(metadata["action_latency_steps"]),
                proposals=proposals,
                applied_actions=applied,
                safety_interventions=safety,
                severe_collisions=int(metadata["final_severe_collision_count"]),
                terminated_early=False,
                provenance_complete=bool(metadata["provenance_complete"]),
            )
        )
    if len(episodes) != 96:
        raise ValueError("P09 development input must contain 96 Episodes")
    return tuple(episodes)


def load_confirmation_episodes(
    episode_path: Path,
    metadata: Sequence[Mapping[str, object]],
) -> tuple[PlantEpisode, ...]:
    episodes = []
    for value in metadata:
        artifact = value["artifact"]
        artifact_path = episode_path / str(artifact["path"])
        if (
            _file_sha256(artifact_path) != str(artifact["sha256"])
            or artifact_path.stat().st_size != int(artifact["bytes"])
        ):
            raise ValueError(f"confirmation artifact identity differs: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as arrays:
            proposals = arrays["actor_proposal"].copy()
            applied = arrays["applied_action"].copy()
            safety = arrays["safety_intervention"].copy()
        provenance = value.get("action_latency_diagnostic", {})
        episodes.append(
            PlantEpisode(
                task_id=str(value["task_id"]),
                seed=int(value["seed"]),
                correlation=float(value["motion_correlation"]),
                action_latency_steps=int(value["action_latency_steps"]),
                proposals=proposals,
                applied_actions=applied,
                safety_interventions=safety,
                severe_collisions=int(value["severe_collision_count"]),
                terminated_early=bool(value["terminated_early"]),
                provenance_complete=_valid_action_latency_provenance(
                    provenance, int(value["action_latency_steps"])
                ),
            )
        )
    return tuple(episodes)


def evaluate_causal_plant_estimator(
    development: Sequence[PlantEpisode],
    confirmation: Sequence[PlantEpisode],
) -> dict[str, object]:
    development_report = _dataset_report(development, confirmation=False)
    confirmation_report = _dataset_report(confirmation, confirmation=True)
    invalid = _invalid_experiment_reasons(confirmation)
    if invalid:
        decision = "inconclusive"
    elif development_report["passed"] and confirmation_report["passed"]:
        decision = "accepted"
    else:
        decision = "rejected"
    return {
        "schema_version": "hwr.causal-plant-estimator-evaluation/v1",
        "proposal_id": "R0001-P11",
        "decision": decision,
        "development": development_report,
        "confirmation": confirmation_report,
        "invalid_experiment_reasons": invalid,
    }


def _dataset_report(
    episodes: Sequence[PlantEpisode], *, confirmation: bool
) -> dict[str, object]:
    results = [_episode_result(value) for value in episodes]
    partitions = {}
    keys = sorted(
        {
            (
                value["task_id"],
                value["motion_correlation"],
                value["action_latency_steps"],
            )
            for value in results
        }
    )
    for task_id, correlation, latency in keys:
        selected = [
            value
            for value in results
            if (
                value["task_id"],
                value["motion_correlation"],
                value["action_latency_steps"],
            )
            == (task_id, correlation, latency)
        ]
        partitions[_partition_key(task_id, correlation, latency)] = (
            _partition_report(selected, confirmation=confirmation)
        )
    return {
        "passed": bool(partitions)
        and all(value["passed"] for value in partitions.values()),
        "episode_count": len(results),
        "partitions": partitions,
        "episodes": results,
    }


def _episode_result(episode: PlantEpisode) -> dict[str, object]:
    estimate = estimate_causal_plant_actions(
        episode.proposals,
        episode.applied_actions,
        episode.safety_interventions,
    )
    baseline = current_proposal_baseline(episode.proposals)
    deranged_proposals = deterministic_proposal_derangement(
        episode.proposals, seed=11_000_003 + episode.seed
    )
    deranged = estimate_causal_plant_actions(
        deranged_proposals,
        episode.applied_actions,
        episode.safety_interventions,
    )
    stable = estimate.stable & ~episode.safety_interventions
    deranged_stable = deranged.stable & ~episode.safety_interventions
    steps = np.arange(len(episode.proposals))
    cold = steps < PLANT_STABLE_START
    selected = estimate.selected_lag[stable]
    gains = estimate.selected_gain[stable]
    return {
        "task_id": episode.task_id,
        "seed": episode.seed,
        "motion_correlation": episode.correlation,
        "action_latency_steps": episode.action_latency_steps,
        "transition_count": len(episode.proposals),
        "stable_transition_count": int(stable.sum()),
        "cold_transition_count": int(cold.sum()),
        "excluded_stable_transition_count": int(
            ((steps >= PLANT_STABLE_START) & ~estimate.stable).sum()
        ),
        "intervention_stable_transition_count": int(
            (estimate.stable & episode.safety_interventions).sum()
        ),
        "stable_normalized_rmse": normalized_action_rmse(
            estimate.predicted_action, episode.applied_actions, stable
        ),
        "current_baseline_stable_normalized_rmse": normalized_action_rmse(
            baseline, episode.applied_actions, stable
        ),
        "deranged_stable_normalized_rmse": normalized_action_rmse(
            deranged.predicted_action, episode.applied_actions, deranged_stable
        ),
        "cold_normalized_rmse": normalized_action_rmse(
            estimate.predicted_action, episode.applied_actions, cold
        ),
        "out_of_bounds_rate": action_out_of_bounds_rate(
            estimate.predicted_action
        ),
        "selected_lag_accuracy": (
            float(np.mean(selected == episode.action_latency_steps))
            if len(selected)
            else 0.0
        ),
        "selected_lag_counts": {
            str(lag): int(np.sum(selected == lag)) for lag in range(4)
        },
        "gain_mean": float(np.mean(gains)) if len(gains) else float("nan"),
        "safety_intervention_count": int(episode.safety_interventions.sum()),
        "severe_collision_count": episode.severe_collisions,
        "terminated_early": episode.terminated_early,
        "provenance_complete": episode.provenance_complete,
    }


def _partition_report(
    episodes: Sequence[Mapping[str, object]], *, confirmation: bool
) -> dict[str, object]:
    stable = _pooled_rmse(episodes, "stable_normalized_rmse")
    baseline = _pooled_rmse(
        episodes, "current_baseline_stable_normalized_rmse"
    )
    deranged = _pooled_rmse(episodes, "deranged_stable_normalized_rmse")
    latency = int(episodes[0]["action_latency_steps"])
    checks = {
        (
            "eight_episodes"
            if confirmation
            else "at_least_seven_episodes"
        ): len(episodes) == 8 if confirmation else len(episodes) >= 7,
        "stable_normalized_rmse_at_most_0_05": stable <= 0.05,
        "out_of_bounds_rate_zero": max(
            float(value["out_of_bounds_rate"]) for value in episodes
        )
        == 0.0,
        "all_values_finite": all(
            np.isfinite(
                [
                    value["stable_normalized_rmse"],
                    value["current_baseline_stable_normalized_rmse"],
                    value["deranged_stable_normalized_rmse"],
                    value["cold_normalized_rmse"],
                    value["gain_mean"],
                ]
            ).all()
            for value in episodes
        ),
    }
    if confirmation:
        checks["derangement_margin_at_least_0_05"] = deranged - stable >= 0.05
    elif latency == 0:
        checks["latency_zero_degradation_at_most_0_005"] = (
            stable - baseline <= 0.005
        )
    return {
        "passed": all(checks.values()),
        "episode_count": len(episodes),
        "transition_count": sum(
            int(value["transition_count"]) for value in episodes
        ),
        "stable_transition_count": sum(
            int(value["stable_transition_count"]) for value in episodes
        ),
        "stable_normalized_rmse": stable,
        "current_baseline_stable_normalized_rmse": baseline,
        "deranged_stable_normalized_rmse": deranged,
        "derangement_margin": deranged - stable,
        "selected_lag_accuracy": float(
            np.mean([value["selected_lag_accuracy"] for value in episodes])
        ),
        "gain_mean": float(np.mean([value["gain_mean"] for value in episodes])),
        "checks": checks,
    }


def _pooled_rmse(
    episodes: Sequence[Mapping[str, object]], key: str
) -> float:
    weights = np.asarray(
        [value["stable_transition_count"] for value in episodes], np.float64
    )
    squared = np.square([value[key] for value in episodes])
    return float(np.sqrt(np.average(squared, weights=weights)))


def _invalid_experiment_reasons(
    episodes: Sequence[PlantEpisode],
) -> list[str]:
    reasons = []
    if len(episodes) != 144:
        reasons.append("confirmation_episode_count")
    if any(not value.provenance_complete for value in episodes):
        reasons.append("action_latency_provenance")
    if any(value.severe_collisions for value in episodes):
        reasons.append("severe_collision")
    if any(value.terminated_early for value in episodes):
        reasons.append("early_termination")
    return reasons


def _valid_action_latency_provenance(
    value: object, latency: int
) -> bool:
    return isinstance(value, Mapping) and (
        value.get("schema_version") == "hwr.action-latency-only-override/v1"
        and value.get("verified_only_action_latency_changed") is True
        and int(value.get("effective_action_latency_steps", -1)) == latency
    )


def _partition_key(task_id: str, correlation: float, latency: int) -> str:
    return f"{task_id}|rho={correlation:.2f}|lag={latency}"


def _read_json(path: Path) -> Mapping[str, object]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} does not contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
