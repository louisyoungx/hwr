from __future__ import annotations

import json
from pathlib import Path

import pytest

from hwr.eval.stale_observation_validity import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_REPORT_SHA256,
    evaluate_stale_observation_validity,
    evaluate_synthetic_timeline,
)


ROOT = Path(__file__).resolve().parents[1]
P11_RUN = (
    ROOT / "runs/research-loop/0003/r0003-p11-causal-plant-s20261101"
)


def test_synthetic_timeline_matches_frozen_stale_validity_contract() -> None:
    report = evaluate_synthetic_timeline()

    assert report["control_hz"] == 20
    assert report["control_period_ns"] == 50_000_000
    assert report["validity_duration_ns"] == 100_000_000
    assert report["rejection_counts"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 61,
    }
    assert report["latencies"]["3"]["first_rejected_step"] == 3
    assert report["future_timestamp_used"] is False


def test_synthetic_timeline_keeps_deadline_inclusive() -> None:
    boundary = evaluate_synthetic_timeline()["boundary"]

    assert boundary["at_deadline_rejected"] is False
    assert boundary["after_deadline_rejected"] is True
    assert boundary["after_deadline_reason"] == "outside_validity_window"


def test_p11_evidence_matches_frozen_contract() -> None:
    if not P11_RUN.exists():
        pytest.skip("frozen P11 run is unavailable")

    report = evaluate_stale_observation_validity(P11_RUN)

    assert report["assessment"]["decision"] == "contract_incompatible"
    assert report["assessment"]["passed"] is True
    assert all(report["assessment"]["checks"].values())
    assert report["p11_evidence"]["episode_counts"] == {
        "1": 18,
        "2": 90,
        "3": 36,
    }
    assert report["p11_evidence"]["safety_totals"] == {
        "1": 0,
        "2": 0,
        "3": 2_196,
    }
    assert report["p11_evidence"]["verified_artifact_count"] == 145
    assert report["guardrails"] == {
        "action_validity_extended": False,
        "evaluation_latency_removed": False,
        "latest_bundle_substituted": False,
        "p11_decision_changed": False,
    }


def test_p11_report_hash_drift_is_rejected(
    tmp_path: Path,
) -> None:
    run = tmp_path / "p11"
    run.mkdir()
    (run / "report.json").write_text("{}\n", encoding="utf-8")
    (run / "manifest.json").write_text("{}\n", encoding="utf-8")

    report = evaluate_stale_observation_validity(run)

    assert report["assessment"]["decision"] == "inconclusive"
    assert report["assessment"]["passed"] is False
    assert "report identity differs" in report["assessment"]["failure_reason"]
    assert EXPECTED_REPORT_SHA256 == (
        "79195b94f48e8b59ce04bb7a9a3a680f8717f6484da69b2f28d54b3a9b842cda"
    )
    assert EXPECTED_MANIFEST_SHA256 == (
        "509f1b7f11caa2aa0dced3324bcc6ea9af9b045071c85b1f6a2bc66a7160da3a"
    )


def test_frozen_p11_report_retains_rejected_decision() -> None:
    if not P11_RUN.exists():
        pytest.skip("frozen P11 run is unavailable")
    report = json.loads((P11_RUN / "report.json").read_text(encoding="utf-8"))

    assert report["decision"] == "rejected"
