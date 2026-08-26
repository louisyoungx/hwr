from __future__ import annotations

import copy

from hwr.eval.predictive_safety_witness import (
    ANCHOR_ID,
    ANCHOR_IDENTITIES,
    analyze_predictive_witness,
)


def _replay(*, observer: bool) -> dict[str, object]:
    state = {
        "qpos": [1.0],
        "qvel": [2.0],
        "time": 3.0,
    }
    common = {
        "step": 1279,
        "policy_proposal": {"action": [0.12, -0.18]},
        "delayed_scaled_plant_action": [0.10, -0.16],
        "queue_source_step": 1278,
        "queue_source": "scaled_policy_proposal",
        "final_applied_action": {
            "source": "safety",
            "action": [0.0] * 16,
        },
        "events": [{
            "event_type": "action_rejected",
            "source": "safety",
            "details": {"reason": "predicted_severe_collision"},
        }],
        "authoritative_physics_advanced": False,
        "pre_authoritative_state": state,
        "post_authoritative_state": copy.deepcopy(state),
        "boundaries": [
            {
                "boundary_ordinal": 1,
                "cumulative_substep": 25,
                "predictor_input": {"action": [0.10, -0.16]},
                "production_violation": False,
                "predictive_trial_physics_advanced": True,
                "authoritative_physics_advanced": False,
                "witness": None,
            },
            {
                "boundary_ordinal": 2,
                "cumulative_substep": 50,
                "predictor_input": {"action": [0.10, -0.16]},
                "production_violation": True,
                "predictive_trial_physics_advanced": True,
                "authoritative_physics_advanced": False,
                "witness": None,
            },
        ],
    }
    if observer:
        for violation, boundary in zip((False, True), common["boundaries"], strict=True):
            boundary["witness"] = {
                "valid": True,
                "invalid_force_count": 0,
                "witness_violation": violation,
                "maximum_forbidden_contact_point_force": 356.0 if violation else 0.0,
                "display_maximum": (
                    {
                        "contact_index": 3,
                        "normal_force": 356.0,
                        "geom_ids": [1, 2],
                        "geom_names": ["robot", "table"],
                        "body_ids": [1, 2],
                        "body_names": ["robot", "table"],
                        "canonical_geom_pair": ["robot", "table"],
                        "robot_side": {"geom_id": 1},
                        "environment_side": {"geom_id": 2},
                    }
                    if violation else None
                ),
            }
    return {
        "observer_enabled": observer,
        "prefix": {
            **ANCHOR_IDENTITIES,
            "planned_episode_id": ANCHOR_ID,
            "prefix_severe_collision_count": 0,
            "prefix_invalid_force_count": 0,
            "prefix_p40_conservation_maximum_absolute_difference": 0.0,
        },
        "diagnostic": {
            "observer_enabled": observer,
            "steps": [common],
            "final_authoritative_state": state,
        },
    }


def test_analysis_accepts_complete_non_mutating_witness() -> None:
    report = analyze_predictive_witness(
        _replay(observer=False),
        _replay(observer=True),
    )

    assert report["decision"] == "accepted as predictive-safety witness contract"
    assert report["checks"]["passed"] is True
    assert report["rejection_step"] == 1279
    assert report["actual_collision_claim_allowed"] is False


def test_analysis_fails_when_witness_disagrees_with_production() -> None:
    enabled = _replay(observer=True)
    enabled["diagnostic"]["steps"][0]["boundaries"][1]["witness"][
        "witness_violation"
    ] = False

    report = analyze_predictive_witness(
        _replay(observer=False),
        enabled,
    )

    assert report["decision"] == "invalid"
    assert report["checks"]["production_witness_consistent"] is False
