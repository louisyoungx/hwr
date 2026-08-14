from __future__ import annotations

import json

import numpy as np

from hwr.train.foundation_action_probe import evaluate_foundation_data_action_probe
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


def test_actor_readiness_requires_repeated_physical_evidence_and_revokes() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(12, consecutive_passes=2)
    )

    first = tracker.assess(
        _diagnostic(), _probe(), _coverage(), _interaction(),
        _collision_validation(), replay_episodes=12
    )
    second = tracker.assess(
        _diagnostic(), _probe(), _coverage(), _interaction(),
        _collision_validation(), replay_episodes=12
    )
    third = tracker.assess(
        _diagnostic(), _probe(), _coverage(), _interaction(),
        _collision_validation(), replay_episodes=12
    )
    failed = tracker.assess(
        _diagnostic(False), _probe(), _coverage(), _interaction(),
        _collision_validation(), replay_episodes=12
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

    report = evaluate_foundation_data_action_probe(
        manifests[0][0],
        manifests[0][1],
        manifests[1][0],
        manifests[1][1],
        bootstrap_samples=40,
    )

    assert report["state_only_to_state_action_ratio"] > 100.0
    assert report["bootstrap"]["ratio_p05"] > 10.0
    assert report["bootstrap"]["unit"] == "episode_cluster"
    assert set(report["partitions"]) == {"fixture/v1"}
    assert set(report["partitions"]["fixture/v1"]["horizons"]) == {
        "1", "4", "8", "16"
    }


def test_physical_gate_can_unlock_explorer_without_task_actor() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(3, consecutive_passes=1)
    )
    diagnostic = _diagnostic()
    diagnostic["assessment"]["passed"] = False
    diagnostic["partitions"]["partition-a"]["assessment"]["passed"] = False

    result = tracker.assess(
        diagnostic, _probe(), _coverage(), _interaction(),
        _collision_validation(), replay_episodes=3
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
        _collision_validation(), replay_episodes=3
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
        replay_episodes=3,
    )

    assert result["exploration_unlocked"] is True
    assert result["checks"]["collision_model_validation"] is False
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
        _collision_validation(), replay_episodes=3
    )

    assert result["checks"]["data_action_probe_ratio"] is True
    assert result["checks"]["data_action_probe_all_tasks"] is False
    assert result["exploration_unlocked"] is False
