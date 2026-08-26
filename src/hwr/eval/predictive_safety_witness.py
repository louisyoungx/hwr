"""Pure validation for the frozen R0001-P66-E1 safety witness."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

PROPOSAL_ID = "R0001-P66-E1"
REPORT_SCHEMA = "hwr.p66-predictive-safety-witness-report/v1"
ANCHOR_ID = "838dd73530fc555aefa4084574fccb4b5a9ca91496f2cca6791f702d95a27bb8"
ANCHOR_IDENTITIES = {
    "candidate_set_sha256":
        "6d51da1277c6e84ebfe3e6b1b977a30e08ebcf2db106e63e064434fa7c855134",
    "selected_index": 2,
    "runtime_randomization_sha256":
        "cd653560fe05b26894828e2052f2aaf63872754135ce224068da471e03fbd9b6",
    "policy_input_sequence_sha256":
        "e5558b0cf76b58625818da503b3716b5207a3beec8056ad24038c69d34667316",
    "raw_prefix_trace_sha256":
        "09dc4fa6ccf72a77d08c34aba740e5ff2af03fed2c8623041df4a45c2159489a",
    "prefix_step_count": 1280,
    "eligibility_reason": "safety_intervention_during_prefix",
}


def analyze_predictive_witness(
    disabled: Mapping[str, object],
    enabled: Mapping[str, object],
) -> dict[str, object]:
    disabled_prefix = disabled["prefix"]
    enabled_prefix = enabled["prefix"]
    disabled_diagnostic = disabled["diagnostic"]
    enabled_diagnostic = enabled["diagnostic"]
    anchor_checks = {
        name: enabled_prefix.get(name) == expected
        for name, expected in ANCHOR_IDENTITIES.items()
    }
    anchor_checks["planned_episode_id"] = (
        enabled_prefix.get("planned_episode_id") == ANCHOR_ID
    )
    prefix_identity = canonical_bytes(disabled_prefix) == canonical_bytes(
        enabled_prefix
    )
    disabled_projection = diagnostic_projection(disabled_diagnostic)
    enabled_projection = diagnostic_projection(enabled_diagnostic)
    observer_identity = canonical_bytes(disabled_projection) == canonical_bytes(
        enabled_projection
    )
    rejections = [
        step for step in enabled_diagnostic["steps"]
        if _is_predictive_rejection(step)
    ]
    rejection = rejections[0] if len(rejections) == 1 else None
    witness = _crossing_witness(rejection)
    rejected_action_not_committed = _rejected_action_not_committed(rejection)
    witness_complete = _witness_complete(witness)
    production_consistent = _production_consistent(
        enabled_diagnostic["steps"]
    )
    checks = {
        **anchor_checks,
        "observer_disabled_enabled_prefix_bit_identical": prefix_identity,
        "observer_disabled_enabled_authoritative_bit_identical": observer_identity,
        "exactly_one_predictive_rejection": len(rejections) == 1,
        "rejection_step_matches_anchor": (
            rejection is not None and rejection.get("step") == 1279
        ),
        "production_witness_consistent": production_consistent,
        "rejection_witness_complete": witness_complete,
        "rejected_action_not_committed": rejected_action_not_committed,
        "actual_severe_collision_count_zero": (
            enabled_prefix.get("prefix_severe_collision_count") == 0
        ),
        "invalid_force_count_zero": (
            enabled_prefix.get("prefix_invalid_force_count") == 0
        ),
        "p40_conservation_exact": (
            enabled_prefix.get(
                "prefix_p40_conservation_maximum_absolute_difference"
            ) == 0.0
        ),
    }
    anchor_reproduced = all(anchor_checks.values())
    invalid = not all(
        value
        for name, value in checks.items()
        if name not in {"planned_episode_id", *ANCHOR_IDENTITIES}
    )
    if not anchor_reproduced:
        decision = "inconclusive_anchor_not_reproduced"
    elif invalid:
        decision = "invalid"
    else:
        decision = "accepted as predictive-safety witness contract"
    checks["passed"] = decision == "accepted as predictive-safety witness contract"
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "decision": decision,
        "sample_unit": "single deterministic historical Episode anchor",
        "occurrence_rate_claim_allowed": False,
        "checks": checks,
        "rejection_step": None if rejection is None else rejection["step"],
        "witness": witness,
        "plant_action_lineage": None if rejection is None else {
            "policy_proposal": rejection["policy_proposal"],
            "delayed_scaled_plant_action":
                rejection["delayed_scaled_plant_action"],
            "queue_source_step": rejection["queue_source_step"],
            "queue_source": rejection["queue_source"],
            "predictor_input": (
                None if witness is None else witness["predictor_input"]
            ),
            "final_applied_action": rejection["final_applied_action"],
        },
        "training_executed": False,
        "policy_inference_executed": False,
        "capability_claim_allowed": False,
        "task_success_claim_allowed": False,
        "generalization_claim_allowed": False,
        "hardware_safety_claim_allowed": False,
        "actual_collision_claim_allowed": False,
    }


def diagnostic_projection(value: Mapping[str, object]) -> dict[str, object]:
    steps = []
    for step in value["steps"]:
        steps.append(
            {
                key: item
                for key, item in step.items()
                if key != "boundaries"
            }
            | {
                "boundaries": [
                    {
                        key: item
                        for key, item in boundary.items()
                        if key != "witness"
                    }
                    for boundary in step["boundaries"]
                ]
            }
        )
    return {
        "steps": steps,
        "final_authoritative_state": value["final_authoritative_state"],
    }


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_predictive_rejection(step: Mapping[str, object]) -> bool:
    return any(
        event.get("event_type") == "action_rejected"
        and event.get("source") == "safety"
        and event.get("details", {}).get("reason")
        == "predicted_severe_collision"
        for event in step["events"]
    )


def _crossing_witness(
    rejection: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if rejection is None:
        return None
    crossings = [
        boundary
        for boundary in rejection["boundaries"]
        if boundary["production_violation"]
    ]
    return crossings[0] if len(crossings) == 1 else None


def _production_consistent(steps: object) -> bool:
    for step in steps:
        rejected = _is_predictive_rejection(step)
        boundaries = step["boundaries"]
        production = any(item["production_violation"] for item in boundaries)
        if rejected != production:
            return False
        for boundary in boundaries:
            witness = boundary["witness"]
            if (
                witness is None
                or witness["valid"] is not True
                or witness["witness_violation"]
                != boundary["production_violation"]
            ):
                return False
    return True


def _witness_complete(witness: Mapping[str, object] | None) -> bool:
    if witness is None:
        return False
    value = witness.get("witness")
    if not isinstance(value, Mapping):
        return False
    maximum = value.get("display_maximum")
    if not isinstance(maximum, Mapping):
        return False
    required = {
        "contact_index", "normal_force", "geom_ids", "geom_names",
        "body_ids", "body_names", "canonical_geom_pair", "robot_side",
        "environment_side",
    }
    return (
        required <= set(maximum)
        and value.get("invalid_force_count") == 0
        and value.get("maximum_forbidden_contact_point_force", 0.0) >= 220.0
        and witness.get("boundary_ordinal") in (1, 2)
        and witness.get("predictive_trial_physics_advanced") is True
        and witness.get("authoritative_physics_advanced") is False
    )


def _rejected_action_not_committed(
    rejection: Mapping[str, object] | None,
) -> bool:
    if rejection is None:
        return False
    before = rejection["pre_authoritative_state"]
    after = rejection["post_authoritative_state"]
    return (
        rejection["authoritative_physics_advanced"] is False
        and before["qpos"] == after["qpos"]
        and before["qvel"] == after["qvel"]
        and before["time"] == after["time"]
        and rejection["final_applied_action"]["source"] == "safety"
        and all(
            value == 0.0
            for value in rejection["final_applied_action"]["action"][:14]
        )
    )
