from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from hwr.apps import evaluate_support_domain as support_domain_app
from hwr.eval.support_domain import (
    P11_MANIFEST_IDENTITY,
    P11_REPORT_IDENTITY,
    P29_MANIFEST_IDENTITY,
    P29_REPORT_IDENTITY,
    PRIMARY_LEDGER,
    SUPPORT_STATEMENT,
    SupportDomainContractError,
    build_support_domain_report,
    evaluate_support_domain,
)


ROOT = Path(__file__).resolve().parents[1]
P11_RUN = (
    ROOT / "runs/research-loop/0003/r0003-p11-causal-plant-s20261101"
)
P29_RUN = (
    ROOT / "runs/research-loop/0004/r0004-p29-stale-validity-s20262901"
)


@pytest.fixture(scope="module")
def frozen_report() -> dict[str, object]:
    return evaluate_support_domain(P11_RUN, P29_RUN)


@pytest.fixture(scope="module")
def source_reports() -> tuple[dict[str, object], dict[str, object]]:
    return (_read_json(P11_RUN / "report.json"), _read_json(P29_RUN / "report.json"))


def test_frozen_input_identities_and_lineage_are_verified(
    frozen_report: dict[str, object],
) -> None:
    assert _identity(P11_RUN / "report.json") == P11_REPORT_IDENTITY
    assert _identity(P11_RUN / "manifest.json") == P11_MANIFEST_IDENTITY
    assert _identity(P29_RUN / "report.json") == P29_REPORT_IDENTITY
    assert _identity(P29_RUN / "manifest.json") == P29_MANIFEST_IDENTITY
    assert frozen_report["inputs"]["p11"]["verified_manifest_artifacts"] == 145
    assert frozen_report["inputs"]["p29"]["verified_manifest_artifacts"] == 1
    lineage = frozen_report["lineage_and_guardrails"]
    assert lineage["p11_source_commit"] == (
        "ef86971ecd9528c00022e3f944e7878f66665f4a"
    )
    assert lineage["p11_decision"] == "rejected"
    assert lineage["p29_source_commit"] == (
        "b34716d32cccfe5e619b18aaecaaeda8954a765a"
    )
    assert lineage["p29_assessment"]["passed"] is True
    assert lineage["p29_assessment"]["decision"] == "contract_incompatible"
    assert all(lineage["p29_assessment"]["checks"].values())
    assert lineage["p29_guardrails"] == {
        "action_validity_extended": False,
        "evaluation_latency_removed": False,
        "latest_bundle_substituted": False,
        "p11_decision_changed": False,
    }


def test_frozen_latency_mapping_keeps_inclusive_boundary(
    frozen_report: dict[str, object],
) -> None:
    contract = frozen_report["classification_contract"]
    assert contract["control_hz"] == 20
    assert contract["control_period_ns"] == 50_000_000
    assert contract["validity_duration_ns"] == 100_000_000
    assert contract["latencies"] == {
        "0": {"maximum_source_age_ns": 0, "domain": "supported"},
        "1": {"maximum_source_age_ns": 50_000_000, "domain": "supported"},
        "2": {"maximum_source_age_ns": 100_000_000, "domain": "supported"},
        "3": {"maximum_source_age_ns": 150_000_000, "domain": "challenge"},
    }
    assert contract["action_latency_changes_domain"] is False
    assert contract["warmup_splitting_allowed"] is False


def test_every_episode_is_counted_once_in_complete_and_one_domain(
    frozen_report: dict[str, object],
) -> None:
    episodes = frozen_report["episodes"]
    assert len(episodes) == 144
    identities = {
        (
            value["task"],
            value["seed"],
            value["rho"],
            value["action_latency_steps"],
        )
        for value in episodes
    }
    assert len(identities) == 144
    assert all(value["ledger_membership"][0] == PRIMARY_LEDGER for value in episodes)
    assert all(len(value["ledger_membership"]) == 2 for value in episodes)
    supported = [
        value for value in episodes if value["domain"] == "supported"
    ]
    challenge = [
        value for value in episodes if value["domain"] == "challenge"
    ]
    assert len(supported) == 108
    assert len(challenge) == 36
    assert {
        value["observation_latency_steps"] for value in supported
    } == {1, 2}
    assert {
        value["observation_latency_steps"] for value in challenge
    } == {3}
    assert {
        value["maximum_source_age_ns"] for value in supported
    } == {50_000_000, 100_000_000}
    assert {
        value["maximum_source_age_ns"] for value in challenge
    } == {150_000_000}


def test_dual_ledgers_match_frozen_counts_and_safety(
    frozen_report: dict[str, object],
) -> None:
    complete = frozen_report["ledgers"]["complete_challenge"]
    supported = frozen_report["ledgers"]["supported_conditional"]
    challenge = frozen_report["ledgers"]["challenge"]
    assert complete["episode_count"] == 144
    assert supported["episode_count"] == 108
    assert challenge["episode_count"] == 36
    assert supported["safety_intervention_count"] == 0
    assert challenge["safety_intervention_count"] == 2_196
    assert complete["severe_collision_count"] == 0
    assert complete["early_termination_count"] == 0
    assert supported["counts"]["task"] == {
        "clear_dining_table_3d/v1": 36,
        "store_kitchen_items_3d/v1": 36,
        "tidy_living_room_3d/v1": 36,
    }
    assert challenge["counts"]["task"] == {
        "clear_dining_table_3d/v1": 12,
        "store_kitchen_items_3d/v1": 12,
        "tidy_living_room_3d/v1": 12,
    }
    assert supported["counts"]["action_latency"] == {
        "1": 36,
        "2": 36,
        "3": 36,
    }
    assert challenge["counts"]["action_latency"] == {
        "1": 12,
        "2": 12,
        "3": 12,
    }


def test_complete_ledger_reports_all_27_imbalanced_cells(
    frozen_report: dict[str, object],
) -> None:
    counts = frozen_report["ledgers"]["complete_challenge"]["counts"]
    cells = counts["task_observation_action_cells"]
    assert len(cells) == 27
    assert counts["observation_latency"] == {"1": 18, "2": 90, "3": 36}
    assert counts["action_latency"] == {"1": 48, "2": 48, "3": 48}
    assert all(
        cell["count"]
        == {1: 2, 2: 10, 3: 4}[cell["observation_latency_steps"]]
        for cell in cells
    )
    assert frozen_report["coverage"] == {
        "episode_count": 144,
        "cell_count": 27,
        "observation_latency_counts": {"1": 18, "2": 90, "3": 36},
        "observation_latency_imbalanced": True,
        "statement": (
            "P11's observation-latency cells are 2/10/4; "
            "this is not a balanced factorial capability benchmark."
        ),
    }


def test_report_forbids_capability_claims(
    frozen_report: dict[str, object],
) -> None:
    assert frozen_report["support_statement"] == SUPPORT_STATEMENT
    assert frozen_report["capability_claim_allowed"] is False
    assert frozen_report["closed_loop_success_available"] is False
    assert frozen_report["balanced_factorial_benchmark"] is False
    assert frozen_report["primary_ledger"] == "complete_challenge"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_field", "missing fields"),
        ("future_domain", "future domain label"),
        ("unknown_latency", "unknown observation latency"),
        ("duplicate_episode", "duplicate Episode identity"),
        ("missing_episode", "exactly 144 Episodes"),
    ],
)
def test_invalid_episode_inputs_are_rejected(
    source_reports: tuple[dict[str, object], dict[str, object]],
    mutation: str,
    message: str,
) -> None:
    p11_report, p29_report = copy.deepcopy(source_reports)
    episodes = p11_report["episodes"]
    if mutation == "missing_field":
        episodes[0].pop("seed")
    elif mutation == "future_domain":
        episodes[0]["domain"] = "supported"
    elif mutation == "unknown_latency":
        episodes[0]["observation_latency_steps"] = 4
    elif mutation == "duplicate_episode":
        episodes[1] = copy.deepcopy(episodes[0])
    else:
        episodes.pop()

    with pytest.raises(SupportDomainContractError, match=message):
        build_support_domain_report(p11_report, p29_report)


def test_timeline_declared_maximum_must_match_decisions(
    source_reports: tuple[dict[str, object], dict[str, object]],
) -> None:
    p11_report, p29_report = copy.deepcopy(source_reports)
    p29_report["timeline"]["latencies"]["2"]["maximum_observation_age_ns"] = (
        99_999_999
    )

    with pytest.raises(
        SupportDomainContractError, match="latency 2 maximum age differs"
    ):
        build_support_domain_report(p11_report, p29_report)


def test_frozen_file_hash_drift_is_rejected(tmp_path: Path) -> None:
    p11_run = tmp_path / "p11"
    p11_run.mkdir()
    (p11_run / "report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(SupportDomainContractError, match="identity differs"):
        evaluate_support_domain(p11_run, P29_RUN)


def test_cli_atomically_writes_report_manifest_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "support-domain"
    arguments = Namespace(p11_run=P11_RUN, p29_run=P29_RUN, output=output)

    result = support_domain_app.run(arguments)

    report_path = output / "report.json"
    manifest_path = output / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    assert sorted(path.name for path in output.iterdir()) == [
        "manifest.json",
        "report.json",
    ]
    assert manifest["source_commit"] == report["source_commit"]
    assert len(manifest["source_commit"]) == 40
    assert manifest["command"][1:3] == [
        "-m",
        "hwr.apps.evaluate_support_domain",
    ]
    assert manifest["inputs"] == {
        "p11_report": P11_REPORT_IDENTITY,
        "p11_manifest": P11_MANIFEST_IDENTITY,
        "p29_report": P29_REPORT_IDENTITY,
        "p29_manifest": P29_MANIFEST_IDENTITY,
    }
    assert manifest["artifacts"]["report.json"] == _identity(report_path)
    assert result["report_sha256"] == _sha256(report_path)
    assert result["report_bytes"] == report_path.stat().st_size
    assert result["manifest_sha256"] == _sha256(manifest_path)
    assert result["manifest_bytes"] == manifest_path.stat().st_size
    assert not list(output.glob("*.tmp"))

    with pytest.raises(FileExistsError):
        support_domain_app.run(arguments)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _identity(path: Path) -> dict[str, object]:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
