"""Frozen R0001-P09 observation-to-plant-action alignment diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from hwr.train.foundation_action_probe import (
    ACTION_PROBE_BOOTSTRAP_CONTRACT,
    ACTION_PROBE_HORIZONS,
    _fit_predict_ridge,
    _synchronized_horizon_bootstrap,
)


ALIGNMENT_REPORT_SCHEMA = "hwr.observation-action-alignment/v1"
ALIGNMENT_PROPOSAL_ID = "R0001-P09"
ALIGNMENT_RIDGE = 1.0e-3
ALIGNMENT_BOOTSTRAP_SAMPLES = 200
ALIGNMENT_BOOTSTRAP_SEED = 20260901
ALIGNMENT_TRANSITIONS = 128
ALIGNMENT_TASK_IDS = (
    "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
    "tidy_living_room_3d/v1",
)
ALIGNMENT_TRAINING_SEEDS = (
    20260901,
    20365630,
    20470359,
    20575088,
    20679817,
    20784546,
    20889275,
    20994004,
)
ALIGNMENT_HOLDOUT_SEEDS = (
    520260901,
    520365630,
    520470359,
    520575088,
    520679817,
    520784546,
    520889275,
    520994004,
)
ALIGNMENT_CORRELATIONS = (0.96, 0.50)


@dataclass(frozen=True)
class AlignmentEpisodePlan:
    correlation: float
    task_id: str
    split: str
    seed: int
    observation_latency_steps: int
    episode_id: str


@dataclass(frozen=True)
class AlignmentEpisode:
    plan: AlignmentEpisodePlan
    transition_count: int
    visible_proprioception: np.ndarray
    actor_proposal: np.ndarray
    plant_action: np.ndarray
    safety_intervention: np.ndarray
    physics_contacts: np.ndarray
    severe_collision_count: np.ndarray
    physical_state_count: int
    randomization: Mapping[str, object]
    latency_override: Mapping[str, object]
    artifact_path: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        observations = np.asarray(self.visible_proprioception, np.float64)
        proposals = np.asarray(self.actor_proposal, np.float64)
        plant = np.asarray(self.plant_action, np.float64)
        safety = np.asarray(self.safety_intervention, np.bool_)
        contacts = np.asarray(self.physics_contacts, np.int64)
        severe = np.asarray(self.severe_collision_count, np.int64)
        expected_actions = self.transition_count + 1
        if (
            self.transition_count <= 0
            or observations.ndim != 2
            or observations.shape[0] != expected_actions + 1
            or proposals.shape != (expected_actions, 16)
            or plant.shape != proposals.shape
            or safety.shape != (expected_actions,)
            or contacts.shape != (expected_actions,)
            or severe.shape != (expected_actions,)
            or self.physical_state_count != expected_actions + 1
            or (contacts < 0).any()
            or (severe < 0).any()
            or not all(
                np.isfinite(value).all()
                for value in (observations, proposals, plant)
            )
        ):
            raise ValueError("alignment Episode arrays violate the prefix contract")
        _validate_episode_provenance(self)
        object.__setattr__(self, "visible_proprioception", observations)
        object.__setattr__(self, "actor_proposal", proposals)
        object.__setattr__(self, "plant_action", plant)
        object.__setattr__(self, "safety_intervention", safety)
        object.__setattr__(self, "physics_contacts", contacts)
        object.__setattr__(self, "severe_collision_count", severe)


def build_alignment_episode_plan(
    *,
    task_ids: Sequence[str] = ALIGNMENT_TASK_IDS,
    training_seeds: Sequence[int] = ALIGNMENT_TRAINING_SEEDS,
    holdout_seeds: Sequence[int] = ALIGNMENT_HOLDOUT_SEEDS,
    correlations: Sequence[float] = ALIGNMENT_CORRELATIONS,
) -> tuple[AlignmentEpisodePlan, ...]:
    """Build the pre-registered alternating lag plan without selecting outcomes."""
    tasks = tuple(task_ids)
    training = tuple(int(value) for value in training_seeds)
    holdout = tuple(int(value) for value in holdout_seeds)
    rhos = tuple(float(value) for value in correlations)
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or not training
        or len(training) != len(holdout)
        or len(training) % 2
        or set(training) & set(holdout)
        or len(set(training)) != len(training)
        or len(set(holdout)) != len(holdout)
        or any(seed < 0 for seed in (*training, *holdout))
        or not rhos
        or len(set(rhos)) != len(rhos)
        or any(not 0.0 <= rho < 1.0 for rho in rhos)
    ):
        raise ValueError("alignment seed, task, or correlation plan is invalid")
    plans = []
    for correlation in rhos:
        for task_id in tasks:
            for split, seeds in (("training", training), ("holdout", holdout)):
                for index, seed in enumerate(seeds):
                    lag = index % 2
                    plans.append(
                        AlignmentEpisodePlan(
                            correlation,
                            task_id,
                            split,
                            seed,
                            lag,
                            _episode_id(correlation, task_id, split, seed),
                        )
                    )
    return tuple(plans)


def evaluate_observation_action_alignment(
    episodes: Sequence[AlignmentEpisode],
    expected_plan: Sequence[AlignmentEpisodePlan],
    *,
    transition_count: int = ALIGNMENT_TRANSITIONS,
    ridge: float = ALIGNMENT_RIDGE,
    bootstrap_samples: int = ALIGNMENT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = ALIGNMENT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Evaluate old and lag-aligned action indices on exactly the same Episodes."""
    if (
        transition_count < max(ACTION_PROBE_HORIZONS)
        or ridge <= 0.0
        or bootstrap_samples <= 0
        or bootstrap_seed < 0
    ):
        raise ValueError("alignment probe configuration is invalid")
    plans = tuple(expected_plan)
    by_id = {episode.plan.episode_id: episode for episode in episodes}
    if (
        len(by_id) != len(episodes)
        or {plan.episode_id for plan in plans} != set(by_id)
        or any(by_id[plan.episode_id].plan != plan for plan in plans)
        or any(episode.transition_count != transition_count for episode in episodes)
    ):
        raise ValueError("alignment Episode coverage differs from the frozen plan")
    correlations = tuple(dict.fromkeys(plan.correlation for plan in plans))
    task_ids = tuple(dict.fromkeys(plan.task_id for plan in plans))
    groups = {}
    for correlation_index, correlation in enumerate(correlations):
        task_reports = {}
        for task_index, task_id in enumerate(task_ids):
            selected = [
                by_id[plan.episode_id]
                for plan in plans
                if plan.correlation == correlation and plan.task_id == task_id
            ]
            task_reports[task_id] = _evaluate_task(
                selected,
                transition_count=transition_count,
                ridge=ridge,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=(
                    bootstrap_seed
                    + correlation_index * 1_000_003
                    + task_index * 104_729
                ),
            )
        group = {
            "motion_correlation": correlation,
            "task_reports": task_reports,
        }
        group["checks"] = _group_checks(group)
        group["decision"] = (
            "accepted" if all(group["checks"].values()) else "rejected"
        )
        groups[_correlation_key(correlation)] = group
    report = {
        "schema_version": ALIGNMENT_REPORT_SCHEMA,
        "proposal_id": ALIGNMENT_PROPOSAL_ID,
        "diagnostic_type": "measurement_repair_only",
        "transition_count_per_episode": transition_count,
        "prefix_action_count_per_episode": 1,
        "ridge": ridge,
        "horizons": list(ACTION_PROBE_HORIZONS),
        "bootstrap": {
            "contract": ACTION_PROBE_BOOTSTRAP_CONTRACT,
            "samples": bootstrap_samples,
            "base_seed": bootstrap_seed,
        },
        "episode_count": len(episodes),
        "correlation_groups": groups,
        "episodes": [_episode_report(by_id[plan.episode_id]) for plan in plans],
    }
    report["decision"] = assess_observation_action_alignment(report)
    return report


def assess_observation_action_alignment(report: Mapping[str, object]) -> str:
    """Apply the frozen accepted/rejected/inconclusive decision boundary."""
    try:
        groups = report["correlation_groups"]
        episodes = report["episodes"]
        expected = int(report["episode_count"])
        if (
            report.get("schema_version") != ALIGNMENT_REPORT_SCHEMA
            or not isinstance(groups, Mapping)
            or set(groups) != {"rho_0.96", "rho_0.50"}
            or not isinstance(episodes, Sequence)
            or len(episodes) != expected
            or expected <= 0
        ):
            return "inconclusive"
        if any(
            not isinstance(value, Mapping)
            or not bool(value.get("provenance_complete", False))
            for value in episodes
        ):
            return "inconclusive"
        decisions = [str(value.get("decision", "")) for value in groups.values()]
    except (KeyError, TypeError, ValueError):
        return "inconclusive"
    return "accepted" if decisions == ["accepted", "accepted"] else "rejected"


def _evaluate_task(
    episodes: Sequence[AlignmentEpisode],
    *,
    transition_count: int,
    ridge: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    training = [value for value in episodes if value.plan.split == "training"]
    holdout = [value for value in episodes if value.plan.split == "holdout"]
    if (
        not training
        or len(training) != len(holdout)
        or {value.plan.seed for value in training}
        & {value.plan.seed for value in holdout}
    ):
        raise ValueError("alignment training and holdout Episodes are invalid")
    all_report = _paired_contract_report(
        training,
        holdout,
        transition_count=transition_count,
        ridge=ridge,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    lag_reports = {
        str(lag): _paired_contract_report(
            [value for value in training if value.plan.observation_latency_steps == lag],
            [value for value in holdout if value.plan.observation_latency_steps == lag],
            transition_count=transition_count,
            ridge=ridge,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + (lag + 1) * 10_009,
        )
        for lag in (0, 1)
    }
    lag_one_reductions = {
        str(horizon): 1.0
        - lag_reports["1"]["aligned_index"]["horizons"][str(horizon)][
            "state_action_mse"
        ]
        / max(
            lag_reports["1"]["old_index"]["horizons"][str(horizon)][
                "state_action_mse"
            ],
            1.0e-12,
        )
        for horizon in ACTION_PROBE_HORIZONS
    }
    return {
        **all_report,
        "lag_partitions": lag_reports,
        "lag_one_state_action_mse_relative_reduction_by_horizon": (
            lag_one_reductions
        ),
    }


def _paired_contract_report(
    training: Sequence[AlignmentEpisode],
    holdout: Sequence[AlignmentEpisode],
    *,
    transition_count: int,
    ridge: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    reports = {}
    for aligned, name in ((False, "old_index"), (True, "aligned_index")):
        horizons = {}
        episode_errors = {}
        for horizon in ACTION_PROBE_HORIZONS:
            report, state_errors, action_errors = _evaluate_horizon(
                training, holdout, horizon=horizon, aligned=aligned, ridge=ridge
            )
            _require_sample_count(
                report,
                training_episodes=len(training),
                holdout_episodes=len(holdout),
                transition_count=transition_count,
                horizon=horizon,
            )
            horizons[str(horizon)] = report
            episode_errors[horizon] = (state_errors, action_errors)
        ratios, conservative = _synchronized_horizon_bootstrap(
            episode_errors, samples=bootstrap_samples, seed=bootstrap_seed
        )
        for horizon, values in ratios.items():
            horizons[str(horizon)]["bootstrap"] = _bootstrap_summary(
                values, bootstrap_samples, bootstrap_seed, "none"
            )
        reports[name] = {
            "training_episode_count": len(training),
            "holdout_episode_count": len(holdout),
            "training_episode_ids": [value.plan.episode_id for value in training],
            "holdout_episode_ids": [value.plan.episode_id for value in holdout],
            "minimum_ratio": min(
                value["state_only_to_state_action_ratio"]
                for value in horizons.values()
            ),
            "bootstrap": _bootstrap_summary(
                conservative,
                bootstrap_samples,
                bootstrap_seed,
                "minimum_across_horizons_within_replicate",
            ),
            "horizons": horizons,
        }
    return reports


def _evaluate_horizon(
    training: Sequence[AlignmentEpisode],
    holdout: Sequence[AlignmentEpisode],
    *,
    horizon: int,
    aligned: bool,
    ridge: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    train = [_episode_probe_arrays(value, horizon, aligned=aligned) for value in training]
    test = [_episode_probe_arrays(value, horizon, aligned=aligned) for value in holdout]
    if not train or not test:
        raise ValueError("alignment probe partition is empty")
    train_state = np.concatenate([value[0] for value in train])
    train_action = np.concatenate([value[1] for value in train])
    train_target = np.concatenate([value[2] for value in train])
    test_state = np.concatenate([value[0] for value in test])
    test_action = np.concatenate([value[1] for value in test])
    test_target = np.concatenate([value[2] for value in test])
    state_prediction = _fit_predict_ridge(
        train_state, train_target, test_state, ridge=ridge
    )
    action_prediction = _fit_predict_ridge(
        np.concatenate((train_state, train_action), axis=1),
        train_target,
        np.concatenate((test_state, test_action), axis=1),
        ridge=ridge,
    )
    state_errors = _episode_errors(state_prediction, test_target, test)
    action_errors = _episode_errors(action_prediction, test_target, test)
    state_mse = float(state_errors.mean())
    action_mse = float(action_errors.mean())
    return {
        "training_episode_count": len(training),
        "training_transition_count": len(train_state),
        "holdout_episode_count": len(holdout),
        "holdout_transition_count": len(test_state),
        "holdout_episode_ids": [value.plan.episode_id for value in holdout],
        "episode_weighting": "uniform",
        "state_only_mse": state_mse,
        "state_action_mse": action_mse,
        "state_only_to_state_action_ratio": state_mse / max(action_mse, 1.0e-12),
    }, state_errors, action_errors


def _episode_probe_arrays(
    episode: AlignmentEpisode, horizon: int, *, aligned: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible = episode.visible_proprioception[1:]
    controllable = _controllable_state(visible)
    count = len(visible) - horizon
    lag = episode.plan.observation_latency_steps if aligned else 0
    action_offset = 1 - lag
    actions = np.stack(
        [
            episode.plant_action[
                action_offset + index : action_offset + index + horizon
            ].mean(axis=0)
            for index in range(count)
        ]
    )
    state = visible[:-horizon]
    target = controllable[horizon:] - controllable[:-horizon]
    if len(actions) != count or len(state) != count or len(target) != count:
        raise RuntimeError("alignment action indexing changed the sample count")
    return state, actions, target


def _controllable_state(proprioception: np.ndarray) -> np.ndarray:
    if proprioception.shape[1] < 31:
        return proprioception
    indices = (*range(6, 12), *range(18, 26), *range(29, 31))
    return proprioception[:, indices]


def _episode_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    episodes: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    per_transition = np.square(prediction - target).mean(axis=1)
    lengths = [len(value[0]) for value in episodes]
    offsets = np.cumsum((0, *lengths))
    return np.asarray(
        [
            per_transition[offsets[index] : offsets[index + 1]].mean()
            for index in range(len(lengths))
        ],
        np.float64,
    )


def _require_sample_count(
    report: Mapping[str, object],
    *,
    training_episodes: int,
    holdout_episodes: int,
    transition_count: int,
    horizon: int,
) -> None:
    per_episode = transition_count + 1 - horizon
    expected_training = training_episodes * per_episode
    expected_holdout = holdout_episodes * per_episode
    if (
        int(report["training_episode_count"]) != training_episodes
        or int(report["holdout_episode_count"]) != holdout_episodes
        or int(report["training_transition_count"]) != expected_training
        or int(report["holdout_transition_count"]) != expected_holdout
    ):
        raise RuntimeError("alignment probe sample count violates the prefix contract")


def _bootstrap_summary(
    ratios: np.ndarray, samples: int, seed: int, reduction: str
) -> dict[str, object]:
    return {
        "unit": "episode_cluster",
        "samples": samples,
        "seed": seed,
        "resampling_contract": ACTION_PROBE_BOOTSTRAP_CONTRACT,
        "replicate_reduction": reduction,
        "ratio_p05": float(np.quantile(ratios, 0.05)),
        "ratio_median": float(np.median(ratios)),
        "ratio_p95": float(np.quantile(ratios, 0.95)),
    }


def _group_checks(group: Mapping[str, object]) -> dict[str, bool]:
    tasks = group["task_reports"]
    if not isinstance(tasks, Mapping):
        raise TypeError("alignment task reports must be a mapping")
    aligned_horizons = [
        horizon
        for task in tasks.values()
        for horizon in task["aligned_index"]["horizons"].values()
    ]
    lag_zero_pairs = [
        (
            task["lag_partitions"]["0"]["old_index"]["horizons"][str(horizon)],
            task["lag_partitions"]["0"]["aligned_index"]["horizons"][str(horizon)],
        )
        for task in tasks.values()
        for horizon in ACTION_PROBE_HORIZONS
    ]
    lag_one_reductions = [
        reduction
        for task in tasks.values()
        for reduction in task[
            "lag_one_state_action_mse_relative_reduction_by_horizon"
        ].values()
    ]
    return {
        "all_task_horizon_point_ratios_at_least_1_05": all(
            value["state_only_to_state_action_ratio"] >= 1.05
            for value in aligned_horizons
        ),
        "all_task_synchronized_bootstrap_p05_at_least_1_01": all(
            task["aligned_index"]["bootstrap"]["ratio_p05"] >= 1.01
            for task in tasks.values()
        ),
        "eight_training_and_eight_holdout_episodes_per_task": all(
            task["aligned_index"]["training_episode_count"] == 8
            and task["aligned_index"]["holdout_episode_count"] == 8
            for task in tasks.values()
        ),
        "lag_zero_old_and_aligned_equal_within_1e_10": all(
            max(
                abs(old["state_only_mse"] - new["state_only_mse"]),
                abs(old["state_action_mse"] - new["state_action_mse"]),
                abs(
                    old["state_only_to_state_action_ratio"]
                    - new["state_only_to_state_action_ratio"]
                ),
            )
            <= 1.0e-10
            for old, new in lag_zero_pairs
        ),
        "lag_one_keeps_all_episodes": all(
            task["lag_partitions"]["1"]["aligned_index"][
                "training_episode_count"
            ]
            == 4
            and task["lag_partitions"]["1"]["aligned_index"][
                "holdout_episode_count"
            ]
            == 4
            for task in tasks.values()
        ),
        "lag_one_every_horizon_state_action_mse_reduction_at_least_10_percent": all(
            value >= 0.10 for value in lag_one_reductions
        ),
        "all_metrics_finite": _all_finite(group),
    }


def _all_finite(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _validate_episode_provenance(episode: AlignmentEpisode) -> None:
    randomization = episode.randomization
    override = episode.latency_override
    lag = episode.plan.observation_latency_steps
    if (
        int(randomization.get("observation_latency_steps", -1)) != lag
        or not isinstance(randomization.get("action_latency_steps"), int)
        or not math.isfinite(float(randomization.get("actuator_scale", math.nan)))
        or override.get("schema_version")
        != "hwr.observation-latency-only-override/v1"
        or int(override.get("effective_observation_latency_steps", -1)) != lag
        or override.get("verified_only_observation_latency_changed") is not True
        or len(str(override.get("other_randomization_sha256", ""))) != 64
        or not episode.artifact_path
        or len(episode.artifact_sha256) != 64
    ):
        raise ValueError("alignment Episode provenance is incomplete")


def _episode_report(episode: AlignmentEpisode) -> dict[str, object]:
    plan = episode.plan
    return {
        "episode_id": plan.episode_id,
        "task_id": plan.task_id,
        "split": plan.split,
        "seed": plan.seed,
        "motion_correlation": plan.correlation,
        "observation_latency_steps": plan.observation_latency_steps,
        "action_latency_steps": episode.randomization["action_latency_steps"],
        "actuator_scale": episode.randomization["actuator_scale"],
        "transition_count": episode.transition_count,
        "prefix_action_count": 1,
        "physical_state_count": episode.physical_state_count,
        "safety_intervention_count": int(episode.safety_intervention.sum()),
        "maximum_physics_contacts": int(episode.physics_contacts.max()),
        "final_severe_collision_count": int(episode.severe_collision_count[-1]),
        "artifact": {
            "path": episode.artifact_path,
            "sha256": episode.artifact_sha256,
        },
        "randomization": dict(episode.randomization),
        "latency_override": dict(episode.latency_override),
        "provenance_complete": True,
    }


def _episode_id(
    correlation: float, task_id: str, split: str, seed: int
) -> str:
    task = task_id.replace("/", "-").replace("_", "-")
    rho = f"{correlation:.2f}".replace(".", "p")
    return f"rho-{rho}.{task}.{split}.seed-{seed}"


def _correlation_key(correlation: float) -> str:
    return f"rho_{correlation:.2f}"
