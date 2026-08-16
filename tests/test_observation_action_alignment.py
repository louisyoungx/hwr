from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hwr.apps.evaluate_observation_action_alignment import (
    DEFAULT_OUTPUT_ROOT,
    _resolve_output_root,
)
import hwr.eval.observation_action_alignment as alignment


def _plan():
    return alignment.build_alignment_episode_plan(
        task_ids=("fixture/v1",),
        training_seeds=(10, 11),
        holdout_seeds=(20, 21),
        correlations=(0.96, 0.50),
    )


def _episode(plan, *, transition_count: int = 20):
    rng = np.random.default_rng(plan.seed)
    lag = plan.observation_latency_steps
    plant = rng.normal(size=(transition_count + 1, 16))
    state = np.zeros((transition_count + 2, 16), np.float64)
    for index in range(transition_count + 1):
        action_index = index - lag
        action = plant[action_index] if action_index >= 0 else np.zeros(16)
        state[index + 1] = state[index] + action
    randomization = {
        "observation_latency_steps": lag,
        "action_latency_steps": plan.seed % 2,
        "actuator_scale": 1.0,
    }
    override = {
        "schema_version": "hwr.observation-latency-only-override/v1",
        "effective_observation_latency_steps": lag,
        "other_randomization_sha256": "a" * 64,
        "verified_only_observation_latency_changed": True,
    }
    return alignment.AlignmentEpisode(
        plan=plan,
        transition_count=transition_count,
        visible_proprioception=state,
        actor_proposal=plant.copy(),
        plant_action=plant,
        safety_intervention=np.zeros(transition_count + 1, np.bool_),
        physics_contacts=np.zeros(transition_count + 1, np.int64),
        severe_collision_count=np.zeros(transition_count + 1, np.int64),
        physical_state_count=transition_count + 2,
        randomization=randomization,
        latency_override=override,
        artifact_path=f"episodes/{plan.episode_id}.npz",
        artifact_sha256="b" * 64,
    )


def test_alignment_plan_freezes_seed_banks_and_alternating_lag() -> None:
    plan = alignment.build_alignment_episode_plan()

    assert len(plan) == 96
    assert {value.task_id for value in plan} == set(alignment.ALIGNMENT_TASK_IDS)
    assert {value.correlation for value in plan} == {0.96, 0.50}
    assert {value.seed for value in plan if value.split == "training"} == set(
        alignment.ALIGNMENT_TRAINING_SEEDS
    )
    assert {value.seed for value in plan if value.split == "holdout"} == set(
        alignment.ALIGNMENT_HOLDOUT_SEEDS
    )
    for task_id in alignment.ALIGNMENT_TASK_IDS:
        for correlation in alignment.ALIGNMENT_CORRELATIONS:
            for split in ("training", "holdout"):
                lags = [
                    value.observation_latency_steps
                    for value in plan
                    if value.task_id == task_id
                    and value.correlation == correlation
                    and value.split == split
                ]
                assert lags == [0, 1, 0, 1, 0, 1, 0, 1]


def test_alignment_default_output_root_is_anchored_to_repository() -> None:
    root = Path("/repo")

    assert _resolve_output_root(root, DEFAULT_OUTPUT_ROOT) == (
        root / "runs/research-loop/0001"
    )
    absolute = root / "custom"
    assert _resolve_output_root(root, absolute) == absolute


def test_alignment_prefix_keeps_all_horizon_samples_and_lag_zero_equal() -> None:
    episode = _episode(_plan()[0])

    for horizon in alignment.ACTION_PROBE_HORIZONS:
        old = alignment._episode_probe_arrays(episode, horizon, aligned=False)
        new = alignment._episode_probe_arrays(episode, horizon, aligned=True)
        assert len(old[0]) == episode.transition_count + 1 - horizon
        assert all(np.array_equal(first, second) for first, second in zip(old, new))

    lag_one = _episode(_plan()[1])
    arrays = alignment._episode_probe_arrays(lag_one, 16, aligned=True)
    assert len(arrays[0]) == lag_one.transition_count + 1 - 16
    formal_lag_one = _episode(_plan()[1], transition_count=128)
    formal_arrays = alignment._episode_probe_arrays(
        formal_lag_one, 16, aligned=True
    )
    assert len(formal_arrays[0]) == 113


def test_alignment_report_uses_synchronized_bootstrap_and_expected_counts() -> None:
    plan = _plan()
    episodes = [_episode(value) for value in plan]

    report = alignment.evaluate_observation_action_alignment(
        episodes,
        plan,
        transition_count=20,
        bootstrap_samples=40,
        bootstrap_seed=7,
    )

    task = report["correlation_groups"]["rho_0.96"]["task_reports"]["fixture/v1"]
    assert all(
        value["artifact"]["path"].startswith("episodes/")
        for value in report["episodes"]
    )
    contract = task["aligned_index"]
    assert contract["bootstrap"]["resampling_contract"] == (
        alignment.ACTION_PROBE_BOOTSTRAP_CONTRACT
    )
    assert {value["bootstrap"]["seed"] for value in contract["horizons"].values()} == {
        7
    }
    assert contract["horizons"]["16"]["training_transition_count"] == 10
    assert contract["horizons"]["16"]["holdout_transition_count"] == 10
    lag_zero = task["lag_partitions"]["0"]
    assert lag_zero["old_index"]["horizons"] == lag_zero["aligned_index"]["horizons"]


def test_alignment_rejects_missing_episode_and_wrong_sample_count(monkeypatch) -> None:
    plan = _plan()
    episodes = [_episode(value) for value in plan]
    with pytest.raises(ValueError, match="coverage differs"):
        alignment.evaluate_observation_action_alignment(
            episodes[:-1], plan, transition_count=20
        )

    original = alignment._episode_probe_arrays

    def shortened(episode, horizon, *, aligned):
        state, action, target = original(episode, horizon, aligned=aligned)
        return state[:-1], action[:-1], target[:-1]

    monkeypatch.setattr(alignment, "_episode_probe_arrays", shortened)
    with pytest.raises(RuntimeError, match="sample count"):
        alignment.evaluate_observation_action_alignment(
            episodes,
            plan,
            transition_count=20,
            bootstrap_samples=10,
        )


def test_alignment_acceptance_requires_both_frozen_correlation_groups() -> None:
    passing = {
        "schema_version": alignment.ALIGNMENT_REPORT_SCHEMA,
        "episode_count": 1,
        "episodes": [{"provenance_complete": True}],
        "correlation_groups": {
            "rho_0.96": {"decision": "accepted"},
            "rho_0.50": {"decision": "accepted"},
        },
    }
    assert alignment.assess_observation_action_alignment(passing) == "accepted"
    passing["correlation_groups"]["rho_0.50"]["decision"] = "rejected"
    assert alignment.assess_observation_action_alignment(passing) == "rejected"
    passing["episodes"][0]["provenance_complete"] = False
    assert alignment.assess_observation_action_alignment(passing) == "inconclusive"


def test_alignment_episode_rejects_incomplete_latency_provenance() -> None:
    episode = _episode(_plan()[0])
    with pytest.raises(ValueError, match="provenance"):
        replace(
            episode,
            latency_override={
                **episode.latency_override,
                "verified_only_observation_latency_changed": False,
            },
        )
