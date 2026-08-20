"""Stale-observation action-validity contract diagnostic for R0001-P29."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.core.embodied import DualArmAction
from hwr.safety import DualArmSafetySupervisor, SafetyLimits
from hwr.train.bimanual_runtime import dual_arm_action_frame


STALE_VALIDITY_SCHEMA = "hwr.stale-observation-validity/v1"
CONTROL_HZ = 20
CONTROL_PERIOD_NS = 50_000_000
VALIDITY_DURATION_NS = 100_000_000
TRACE_TRANSITIONS = 64
OBSERVATION_LATENCIES = (0, 1, 2, 3)
EXPECTED_SOURCE_COMMIT = "ef86971ecd9528c00022e3f944e7878f66665f4a"
EXPECTED_REPORT_SHA256 = (
    "79195b94f48e8b59ce04bb7a9a3a680f8717f6484da69b2f28d54b3a9b842cda"
)
EXPECTED_MANIFEST_SHA256 = (
    "509f1b7f11caa2aa0dced3324bcc6ea9af9b045071c85b1f6a2bc66a7160da3a"
)
EXPECTED_EPISODE_COUNTS = {1: 18, 2: 90, 3: 36}
EXPECTED_SAFETY_TOTALS = {1: 0, 2: 0, 3: 2_196}
EXPECTED_SAFETY_PER_EPISODE = {1: 0, 2: 0, 3: 61}
EXPECTED_TASKS = (
    "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
    "tidy_living_room_3d/v1",
)
EXPECTED_CORRELATIONS = (0.50, 0.96)
EXPECTED_ACTION_LATENCIES = (1, 2, 3)
EXPECTED_SEEDS = (
    720_261_101,
    720_365_830,
    720_470_559,
    720_575_288,
    720_680_017,
    720_784_746,
    720_889_475,
    720_994_204,
)


def evaluate_stale_observation_validity(p11_run: Path) -> dict[str, object]:
    timeline = evaluate_synthetic_timeline()
    try:
        evidence = verify_p11_evidence(p11_run)
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {
            "schema_version": STALE_VALIDITY_SCHEMA,
            "proposal_id": "R0001-P29",
            "diagnostic_type": "runtime_contract_only",
            "timeline": timeline,
            "p11_evidence": None,
            "guardrails": _guardrails(),
            "assessment": {
                "decision": "inconclusive",
                "passed": False,
                "checks": {},
                "failure_reason": str(error),
            },
        }
    checks = {
        "synthetic_latency_contract_matches": (
            timeline["rejection_counts"] == {"0": 0, "1": 0, "2": 0, "3": 61}
        ),
        "exact_boundary_is_inclusive": (
            timeline["boundary"]["at_deadline_rejected"] is False
        ),
        "one_nanosecond_after_deadline_is_rejected": (
            timeline["boundary"]["after_deadline_rejected"] is True
        ),
        "p11_artifacts_verified": (
            evidence["verified_artifact_count"] == 145
        ),
        "p11_episode_counts_match": (
            evidence["episode_counts"] == {"1": 18, "2": 90, "3": 36}
        ),
        "p11_safety_totals_match": (
            evidence["safety_totals"] == {"1": 0, "2": 0, "3": 2_196}
        ),
        "p11_per_episode_counts_match": (
            evidence["per_episode_safety_counts"]
            == {"1": [0], "2": [0], "3": [61]}
        ),
        "p11_no_early_termination": evidence["early_termination_count"] == 0,
        "p11_no_severe_collision": evidence["maximum_severe_collision_count"] == 0,
        "no_future_timestamp": timeline["future_timestamp_used"] is False,
        "action_validity_not_extended": (
            timeline["validity_duration_ns"] == VALIDITY_DURATION_NS
        ),
        "latest_bundle_not_substituted": True,
    }
    coverage_complete = evidence["coverage_complete"] is True
    decision = "inconclusive"
    if coverage_complete:
        decision = (
            "contract_incompatible"
            if all(checks.values())
            else "implementation_bug"
        )
    return {
        "schema_version": STALE_VALIDITY_SCHEMA,
        "proposal_id": "R0001-P29",
        "diagnostic_type": "runtime_contract_only",
        "timeline": timeline,
        "p11_evidence": evidence,
        "guardrails": _guardrails(),
        "assessment": {
            "decision": decision,
            "passed": decision == "contract_incompatible",
            "checks": {
                **checks,
                "p11_coverage_complete": coverage_complete,
            },
        },
    }


def evaluate_synthetic_timeline(
    *,
    latencies: Sequence[int] = OBSERVATION_LATENCIES,
    transitions: int = TRACE_TRANSITIONS,
) -> dict[str, object]:
    values = tuple(int(value) for value in latencies)
    if (
        transitions <= 0
        or not values
        or len(set(values)) != len(values)
        or any(value < 0 for value in values)
    ):
        raise ValueError("stale-validity timeline dimensions are invalid")
    supervisor = DualArmSafetySupervisor(SafetyLimits())
    reports = {}
    for latency in values:
        decisions = []
        for step in range(transitions):
            age_steps = min(step, latency)
            now_ns = step * CONTROL_PERIOD_NS
            observation_timestamp_ns = now_ns - age_steps * CONTROL_PERIOD_NS
            frame = dual_arm_action_frame(
                observation_timestamp_ns,
                _zero_action(),
                source="p29_contract_probe",
            )
            _, events = supervisor.filter(
                frame,
                now_ns=now_ns,
                hold_grippers=(0.0, 0.0),
            )
            decisions.append(_decision_report(frame, now_ns, events))
        reports[str(latency)] = {
            "latency_steps": latency,
            "transition_count": transitions,
            "rejection_count": sum(
                decision["rejected"] is True for decision in decisions
            ),
            "maximum_observation_age_ns": max(
                decision["observation_age_ns"] for decision in decisions
            ),
            "first_rejected_step": next(
                (
                    index
                    for index, decision in enumerate(decisions)
                    if decision["rejected"] is True
                ),
                None,
            ),
            "decisions": decisions,
        }
    boundary = _boundary_report(supervisor)
    return {
        "control_hz": CONTROL_HZ,
        "control_period_ns": CONTROL_PERIOD_NS,
        "validity_duration_ns": VALIDITY_DURATION_NS,
        "transition_count": transitions,
        "latencies": reports,
        "rejection_counts": {
            latency: report["rejection_count"]
            for latency, report in reports.items()
        },
        "boundary": boundary,
        "future_timestamp_used": any(
            decision["observation_timestamp_ns"] > decision["runtime_now_ns"]
            for report in reports.values()
            for decision in report["decisions"]
        ),
    }


def verify_p11_evidence(p11_run: Path) -> dict[str, object]:
    report_path = p11_run / "report.json"
    manifest_path = p11_run / "manifest.json"
    if _sha256(report_path) != EXPECTED_REPORT_SHA256:
        raise ValueError("P29 frozen P11 report identity differs")
    if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("P29 frozen P11 manifest identity differs")
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    episodes = report.get("episodes")
    artifacts = manifest.get("artifacts")
    if (
        report.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or report.get("contract_complete") is not True
        or not isinstance(episodes, list)
        or len(episodes) != 144
        or not isinstance(artifacts, Mapping)
        or len(artifacts) != 145
    ):
        raise ValueError("P29 frozen P11 coverage differs")
    _verify_artifacts(p11_run, artifacts)
    groups = {
        latency: [
            episode
            for episode in episodes
            if int(episode["observation_latency_steps"]) == latency
        ]
        for latency in EXPECTED_EPISODE_COUNTS
    }
    episode_counts = {
        str(latency): len(values) for latency, values in groups.items()
    }
    safety_totals = {
        str(latency): sum(
            int(value["safety_intervention_count"]) for value in values
        )
        for latency, values in groups.items()
    }
    per_episode = {
        str(latency): sorted(
            {
                int(value["safety_intervention_count"])
                for value in values
            }
        )
        for latency, values in groups.items()
    }
    _verify_episode_artifacts(p11_run, episodes, artifacts)
    coverage = {
        (
            str(value["task_id"]),
            float(value["motion_correlation"]),
            int(value["action_latency_steps"]),
            int(value["seed"]),
        )
        for value in episodes
    }
    expected_coverage = {
        (task, correlation, latency, seed)
        for task in EXPECTED_TASKS
        for correlation in EXPECTED_CORRELATIONS
        for latency in EXPECTED_ACTION_LATENCIES
        for seed in EXPECTED_SEEDS
    }
    return {
        "run": str(p11_run),
        "source_commit": str(report["source_commit"]),
        "report_sha256": EXPECTED_REPORT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "episode_count": len(episodes),
        "verified_artifact_count": len(artifacts),
        "episode_counts": episode_counts,
        "safety_totals": safety_totals,
        "per_episode_safety_counts": per_episode,
        "coverage_complete": coverage == expected_coverage,
        "early_termination_count": sum(
            value["terminated_early"] is True for value in episodes
        ),
        "maximum_severe_collision_count": max(
            int(value["severe_collision_count"]) for value in episodes
        ),
    }


def _verify_artifacts(
    run: Path, artifacts: Mapping[str, object]
) -> None:
    for relative, identity in artifacts.items():
        if not isinstance(identity, Mapping):
            raise ValueError("P29 P11 artifact identity is invalid")
        path = run / relative
        if (
            _sha256(path) != identity.get("sha256")
            or path.stat().st_size != int(identity.get("bytes", -1))
        ):
            raise ValueError(f"P29 P11 artifact differs: {relative}")


def _verify_episode_artifacts(
    run: Path,
    episodes: Sequence[Mapping[str, object]],
    artifacts: Mapping[str, object],
) -> None:
    for episode in episodes:
        artifact = episode.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("P29 P11 Episode artifact metadata is missing")
        relative = f"episodes/{artifact['path']}"
        identity = artifacts.get(relative)
        if (
            not isinstance(identity, Mapping)
            or identity.get("sha256") != artifact.get("sha256")
            or int(identity.get("bytes", -1)) != int(artifact.get("bytes", -2))
        ):
            raise ValueError("P29 P11 Episode manifest identity differs")
        with np.load(run / relative, allow_pickle=False) as arrays:
            safety = arrays["safety_intervention"].astype(bool)
            applied = arrays["applied_action"]
            if (
                len(safety) != int(episode["transition_count"])
                or int(safety.sum()) != int(episode["safety_intervention_count"])
                or not np.isfinite(applied).all()
                or int(arrays["action_latency_steps"])
                != int(episode["action_latency_steps"])
            ):
                raise ValueError("P29 P11 Episode arrays differ from report")


def _decision_report(frame, now_ns: int, events) -> dict[str, object]:
    reasons = [
        str(event.details.get("reason", ""))
        for event in events
        if event.event_type == "action_rejected"
    ]
    rejected = "outside_validity_window" in reasons
    return {
        "observation_timestamp_ns": frame.created_at_ns,
        "runtime_now_ns": now_ns,
        "observation_age_ns": now_ns - frame.created_at_ns,
        "valid_until_ns": frame.valid_until_ns,
        "rejected": rejected,
        "reason": "outside_validity_window" if rejected else None,
    }


def _boundary_report(
    supervisor: DualArmSafetySupervisor,
) -> dict[str, object]:
    frame = dual_arm_action_frame(1_000_000_000, _zero_action(), source="p29_boundary")
    _, at_deadline = supervisor.filter(
        frame,
        now_ns=frame.valid_until_ns,
        hold_grippers=(0.0, 0.0),
    )
    _, after_deadline = supervisor.filter(
        frame,
        now_ns=frame.valid_until_ns + 1,
        hold_grippers=(0.0, 0.0),
    )
    return {
        "valid_until_ns": frame.valid_until_ns,
        "at_deadline_rejected": bool(at_deadline),
        "after_deadline_rejected": bool(after_deadline),
        "after_deadline_reason": (
            after_deadline[0].details.get("reason")
            if after_deadline
            else None
        ),
    }


def _zero_action() -> DualArmAction:
    return DualArmAction(
        0.0,
        0.0,
        (0.0,) * 6,
        (0.0,) * 6,
        0.0,
        0.0,
    )


def _guardrails() -> dict[str, bool]:
    return {
        "action_validity_extended": False,
        "evaluation_latency_removed": False,
        "latest_bundle_substituted": False,
        "p11_decision_changed": False,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P29 JSON root is invalid: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
