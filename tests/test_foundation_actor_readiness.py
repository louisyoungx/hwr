from __future__ import annotations

import json

import numpy as np

import hwr.train.foundation_action_probe as action_probe
from hwr.train.foundation_actor_readiness import (
    FoundationActorReadinessCriteria,
    FoundationActorReadinessTracker,
)


def _diagnostic(passed: bool = True):
    assessment = {
        "passed": passed,
        "components": {
            "visual_latent": {"passed": passed},
            "proprioception": {"passed": passed},
        },
    }
    statistics = {"robust_passed": passed}
    return {
        "assessment": {"passed": passed},
        "shuffle_statistics": {"robust_passed": passed},
        "one_step_action_utilization": {
            "assessment": assessment,
            "shuffle_statistics": statistics,
        },
        "partitions": {
            "partition-a": {
                "assessment": {"passed": passed},
                "shuffle_statistics": {"robust_passed": passed},
                "one_step_action_utilization": {
                    "assessment": assessment,
                    "shuffle_statistics": statistics,
                }
            }
        },
    }


def _probe():
    return {
        "state_only_to_state_action_ratio": 1.2,
        "bootstrap": {"ratio_p05": 1.1},
        "partitions": {
            "partition-a": {
                "state_only_to_state_action_ratio": 1.2,
                "bootstrap": {"ratio_p05": 1.1},
            }
        },
    }


def _coverage():
    return {"active_dimension_fraction": 0.9, "effective_rank": 8.0}


def _interaction():
    return {
        "partitions": {
            "partition-a": {
                "unilateral_contact_episode_count": 1,
                "controlled_motion_episode_count": 1,
                "severe_collision_positive_episode_count": 1,
                "severe_collision_negative_episode_count": 1,
            }
        }
    }


def _collision_validation(passed: bool = True):
    return {
        "passed": passed,
        "criteria": {
            "minimum_positive_episodes_per_task": 8,
            "minimum_negative_episodes_per_task": 8,
            "minimum_recall": 0.8,
            "minimum_pr_auc": 0.5,
            "maximum_brier_score": 0.1,
            "maximum_false_positive_rate": 0.05,
            "minimum_terminal_alignment": 0.8,
            "minimum_action_sensitivity_ratio": 1.02,
        },
        "partitions": {"partition-a": {"passed": passed}},
    }


def _action_execution_validation(passed: bool = True):
    return {"passed": passed, "partitions": {"partition-a": {"passed": passed}}}


def test_actor_readiness_requires_repeated_physical_evidence_and_revokes() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(12, consecutive_passes=2)
    )

    first = tracker.assess(
        _diagnostic(), _probe(), _coverage(), _interaction(),
        _collision_validation(), _action_execution_validation(), replay_episodes=12
    )
    second = tracker.assess(
        _diagnostic(), _probe(), _coverage(), _interaction(),
        _collision_validation(), _action_execution_validation(), replay_episodes=12
    )
    third = tracker.assess(
        _diagnostic(), _probe(), _coverage(), _interaction(),
        _collision_validation(), _action_execution_validation(), replay_episodes=12
    )
    failed = tracker.assess(
        _diagnostic(False), _probe(), _coverage(), _interaction(),
        _collision_validation(), _action_execution_validation(), replay_episodes=12
    )

    assert first["unlocked"] is False
    assert first["exploration_unlocked"] is False
    assert second["unlocked"] is False
    assert second["exploration_unlocked"] is True
    assert third["task_actor_unlocked"] is True
    assert failed["unlocked"] is False
    assert failed["exploration_unlocked"] is False
    assert failed["consecutive_passes"] == 0


def test_data_action_probe_detects_action_identifiability(tmp_path) -> None:
    rng = np.random.default_rng(7)
    manifests = []
    for name, count in (("training", 180), ("holdout", 90)):
        root = tmp_path / name
        root.mkdir()
        action = rng.normal(size=(count, 2)).astype(np.float32)
        state = np.zeros((count + 1, 3), np.float32)
        for index in range(count):
            state[index + 1] = state[index]
            state[index + 1, :2] += action[index]
        np.savez(root / "episode.npz", proprioception=state, executed_action=action)
        manifest = {
            "shards": [{
                "path": "episode.npz",
                "task_id": "fixture/v1",
                "episode_id": f"{name}-episode",
            }]
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        manifests.append((root, manifest))

    report = action_probe.evaluate_foundation_data_action_probe(
        manifests[0][0],
        manifests[0][1],
        manifests[1][0],
        manifests[1][1],
        bootstrap_samples=40,
    )

    assert report["state_only_to_state_action_ratio"] > 100.0
    assert report["bootstrap"]["ratio_p05"] > 10.0
    assert report["bootstrap"]["unit"] == "episode_cluster"
    assert report["schema_version"] == "hwr.foundation-data-action-probe/v4"
    assert report["bootstrap"]["resampling_contract"] == (
        action_probe.ACTION_PROBE_BOOTSTRAP_CONTRACT
    )
    assert report["bootstrap_provenance"] == {
        "contract": action_probe.ACTION_PROBE_BOOTSTRAP_CONTRACT,
        "episode_alignment_key": ["task_id", "holdout_episode_id"],
        "resampling_unit": "holdout_episode",
        "within_task_horizon_coupling": (
            "shared_episode_multiplicity_per_replicate"
        ),
        "within_task_reduction": "minimum_horizon_ratio_per_replicate",
        "task_seed_derivation": "base_seed + sorted_task_index * 104729",
        "across_task_coupling": "independent_episode_resampling",
        "across_task_reduction": "minimum_task_ratio_per_replicate",
    }
    assert set(report["partitions"]) == {"fixture/v1"}
    assert set(report["partitions"]["fixture/v1"]["horizons"]) == {
        "1", "4", "8", "16"
    }
    horizon_bootstraps = report["partitions"]["fixture/v1"]["horizons"]
    assert {value["bootstrap"]["seed"] for value in horizon_bootstraps.values()} == {0}


def test_action_probe_synchronizes_correlated_episode_bootstrap_across_horizons() -> None:
    base = np.asarray([0.8, 1.0, 1.2, 1.4, 1.6, 1.8], np.float64)
    action = np.ones_like(base)
    episode_errors = {
        1: (base + 0.4, action),
        4: (base + 0.3, action),
        8: (base + 0.2, action),
        16: (base, action),
    }

    ratios, conservative = action_probe._synchronized_horizon_bootstrap(
        episode_errors, samples=200, seed=23
    )
    selected = np.random.default_rng(23).integers(
        0, len(base), size=(200, len(base))
    )
    expected = base[selected].mean(axis=1)

    assert np.array_equal(conservative, ratios[16])
    assert np.allclose(conservative, expected)
    assert np.allclose(ratios[1] - conservative, 0.4)
    assert np.allclose(ratios[4] - conservative, 0.3)
    assert np.allclose(ratios[8] - conservative, 0.2)


def test_action_probe_zero_effect_negative_control_never_passes() -> None:
    errors = np.asarray([0.3, 0.8, 1.7, 2.2, 0.5], np.float64)
    episode_errors = {
        horizon: (errors, errors.copy())
        for horizon in action_probe.ACTION_PROBE_HORIZONS
    }

    ratios, conservative = action_probe._synchronized_horizon_bootstrap(
        episode_errors, samples=300, seed=31
    )

    assert all(np.allclose(value, 1.0) for value in ratios.values())
    assert np.quantile(conservative, 0.05) < 1.01


def test_action_probe_shuffled_action_negative_control_never_passes() -> None:
    episode_state_errors = np.asarray([0.4, 1.0, 1.8, 0.7, 1.3], np.float64)
    shuffled_action_errors = episode_state_errors[[2, 4, 0, 3, 1]]
    episode_errors = {
        horizon: (episode_state_errors, shuffled_action_errors)
        for horizon in action_probe.ACTION_PROBE_HORIZONS
    }

    _, conservative = action_probe._synchronized_horizon_bootstrap(
        episode_errors, samples=1_000, seed=37
    )

    assert episode_state_errors.mean() / shuffled_action_errors.mean() == 1.0
    assert np.quantile(conservative, 0.05) < 1.01


def test_action_probe_single_failing_horizon_cannot_pass() -> None:
    action = np.ones(6, np.float64)
    strong = np.asarray([1.3, 1.4, 1.5, 1.6, 1.7, 1.8], np.float64)
    weak = np.asarray([0.75, 0.8, 0.85, 0.9, 0.95, 1.0], np.float64)
    episode_errors = {
        1: (strong, action),
        4: (strong + 0.1, action),
        8: (strong + 0.2, action),
        16: (weak, action),
    }

    ratios, conservative = action_probe._synchronized_horizon_bootstrap(
        episode_errors, samples=300, seed=41
    )

    assert min(values[0].mean() / values[1].mean()
               for values in episode_errors.values()) < 1.05
    assert np.quantile(ratios[1], 0.05) > 1.01
    assert np.quantile(conservative, 0.05) < 1.01


def test_action_probe_synchronized_bootstrap_is_reproducible_and_seed_stable() -> None:
    action = np.ones(8, np.float64)
    base = np.linspace(1.15, 1.85, 8)
    episode_errors = {
        horizon: (base + index * 0.05, action)
        for index, horizon in enumerate(action_probe.ACTION_PROBE_HORIZONS)
    }

    first, first_conservative = action_probe._synchronized_horizon_bootstrap(
        episode_errors, samples=2_000, seed=53
    )
    second, second_conservative = action_probe._synchronized_horizon_bootstrap(
        episode_errors, samples=2_000, seed=53
    )
    p05_by_seed = [
        np.quantile(
            action_probe._synchronized_horizon_bootstrap(
                episode_errors, samples=2_000, seed=seed
            )[1],
            0.05,
        )
        for seed in (53, 59, 61, 67)
    ]

    assert all(np.array_equal(first[horizon], second[horizon])
               for horizon in action_probe.ACTION_PROBE_HORIZONS)
    assert np.array_equal(first_conservative, second_conservative)
    assert max(p05_by_seed) - min(p05_by_seed) < 0.02


def test_physical_gate_can_unlock_explorer_without_task_actor() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(3, consecutive_passes=1)
    )
    diagnostic = _diagnostic()
    diagnostic["assessment"]["passed"] = False
    diagnostic["partitions"]["partition-a"]["assessment"]["passed"] = False

    result = tracker.assess(
        diagnostic, _probe(), _coverage(), _interaction(),
        _collision_validation(), _action_execution_validation(), replay_episodes=3
    )

    assert result["exploration_unlocked"] is True
    assert result["task_actor_unlocked"] is False


def test_explorer_admission_does_not_require_interaction_coverage() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(3, consecutive_passes=1)
    )
    interaction = _interaction()
    partition = interaction["partitions"]["partition-a"]
    for name in tuple(partition):
        partition[name] = 0

    result = tracker.assess(
        _diagnostic(), _probe(), _coverage(), interaction,
        _collision_validation(), _action_execution_validation(), replay_episodes=3
    )

    assert result["exploration_passed_this_cycle"] is True
    assert result["exploration_unlocked"] is True
    assert result["task_interaction_passed_this_cycle"] is False
    assert result["task_actor_unlocked"] is False


def test_task_actor_requires_independent_collision_validation() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(3, consecutive_passes=1)
    )

    result = tracker.assess(
        _diagnostic(),
        _probe(),
        _coverage(),
        _interaction(),
        _collision_validation(False),
        _action_execution_validation(),
        replay_episodes=3,
    )

    assert result["exploration_unlocked"] is True
    assert result["checks"]["collision_model_validation"] is False
    assert result["task_actor_unlocked"] is False


def test_explorer_requires_independent_action_execution_validation() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(3, consecutive_passes=1)
    )

    result = tracker.assess(
        _diagnostic(),
        _probe(),
        _coverage(),
        _interaction(),
        _collision_validation(),
        _action_execution_validation(False),
        replay_episodes=3,
    )

    assert result["checks"]["action_execution_model_validation"] is False
    assert result["exploration_unlocked"] is False
    assert result["task_actor_unlocked"] is False


def test_actor_readiness_rejects_global_probe_pass_when_one_task_fails() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(3, consecutive_passes=1)
    )
    probe = _probe()
    probe["partitions"]["partition-b"] = {
        "state_only_to_state_action_ratio": 0.8,
        "bootstrap": {"ratio_p05": 0.7},
    }

    result = tracker.assess(
        _diagnostic(), probe, _coverage(), _interaction(),
        _collision_validation(), _action_execution_validation(), replay_episodes=3
    )

    assert result["checks"]["data_action_probe_ratio"] is True
    assert result["checks"]["data_action_probe_all_tasks"] is False
    assert result["exploration_unlocked"] is False
