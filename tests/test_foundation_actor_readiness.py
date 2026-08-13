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
        "one_step_action_utilization": {
            "assessment": assessment,
            "shuffle_statistics": statistics,
        },
        "partitions": {
            "partition-a": {
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
    }


def _coverage():
    return {"active_dimension_fraction": 0.9, "effective_rank": 8.0}


def test_actor_readiness_requires_repeated_physical_evidence_and_revokes() -> None:
    tracker = FoundationActorReadinessTracker(
        FoundationActorReadinessCriteria(12, consecutive_passes=2)
    )

    first = tracker.assess(
        _diagnostic(), _probe(), _coverage(), replay_episodes=12
    )
    second = tracker.assess(
        _diagnostic(), _probe(), _coverage(), replay_episodes=12
    )
    failed = tracker.assess(
        _diagnostic(False), _probe(), _coverage(), replay_episodes=12
    )

    assert first["unlocked"] is False
    assert second["unlocked"] is True
    assert failed["unlocked"] is False
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
        manifest = {"shards": [{"path": "episode.npz"}]}
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
