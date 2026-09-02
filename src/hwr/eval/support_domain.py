"""Frozen support-domain evidence aggregation for R0001-P36-E1."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SUPPORT_DOMAIN_SCHEMA = "hwr.support-domain-evaluation/v1"
EXPERIMENT_ID = "R0001-P36-E1"
SUPPORT_STATEMENT = (
    "Supports the evaluation subdomain with visible observation age <=100ms; "
    "the complete evaluation profile is not yet supported."
)
PRIMARY_LEDGER = "complete_challenge"
CONTROL_HZ = 20
CONTROL_PERIOD_NS = 50_000_000
VALIDITY_DURATION_NS = 100_000_000
P11_SOURCE_COMMIT = "ef86971ecd9528c00022e3f944e7878f66665f4a"
P29_SOURCE_COMMIT = "b34716d32cccfe5e619b18aaecaaeda8954a765a"
P11_REPORT_IDENTITY = {
    "sha256": "79195b94f48e8b59ce04bb7a9a3a680f8717f6484da69b2f28d54b3a9b842cda",
    "bytes": 423_970,
}
P11_MANIFEST_IDENTITY = {
    "sha256": "509f1b7f11caa2aa0dced3324bcc6ea9af9b045071c85b1f6a2bc66a7160da3a",
    "bytes": 28_249,
}
P29_REPORT_IDENTITY = {
    "sha256": "7729eed53e034bef5a9d5c50bd1d87d6025370b290226f3055c480f3014274fb",
    "bytes": 71_794,
}
P29_MANIFEST_IDENTITY = {
    "sha256": "41495b1a1efa9f24b65e8c5359bc98cfa28ecee4dbdbc20a5c20d37aa7900cef",
    "bytes": 317,
}
EXPECTED_MAXIMUM_AGE_NS = {
    0: 0,
    1: 50_000_000,
    2: 100_000_000,
    3: 150_000_000,
}
EXPECTED_TASKS = (
    "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
    "tidy_living_room_3d/v1",
)
EXPECTED_OBSERVATION_LATENCIES = (1, 2, 3)
EXPECTED_ACTION_LATENCIES = (1, 2, 3)
EXPECTED_RHOS = (0.50, 0.96)
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
FUTURE_DOMAIN_FIELDS = frozenset(
    {
        "domain",
        "domain_label",
        "observation_age_domain",
        "support_domain",
        "support_domain_label",
    }
)


class SupportDomainContractError(ValueError):
    """Raised when frozen evidence no longer satisfies the E1 contract."""


def evaluate_support_domain(p11_run: Path, p29_run: Path) -> dict[str, object]:
    """Verify frozen runs and produce the P36-E1 dual-ledger report."""
    p11 = _load_frozen_run(
        Path(p11_run),
        report_identity=P11_REPORT_IDENTITY,
        manifest_identity=P11_MANIFEST_IDENTITY,
    )
    p29 = _load_frozen_run(
        Path(p29_run),
        report_identity=P29_REPORT_IDENTITY,
        manifest_identity=P29_MANIFEST_IDENTITY,
    )
    _validate_lineage(
        p11_report=p11["report"],
        p11_manifest=p11["manifest"],
        p29_report=p29["report"],
        p29_manifest=p29["manifest"],
    )
    _verify_episode_manifest_links(
        Path(p11_run), p11["report"], p11["manifest"]
    )
    report = build_support_domain_report(p11["report"], p29["report"])
    report["inputs"] = {
        "p11": {
            "run": str(Path(p11_run).resolve()),
            "source_commit": P11_SOURCE_COMMIT,
            "report": dict(P11_REPORT_IDENTITY),
            "manifest": dict(P11_MANIFEST_IDENTITY),
            "verified_manifest_artifacts": p11["verified_artifacts"],
        },
        "p29": {
            "run": str(Path(p29_run).resolve()),
            "source_commit": P29_SOURCE_COMMIT,
            "report": dict(P29_REPORT_IDENTITY),
            "manifest": dict(P29_MANIFEST_IDENTITY),
            "verified_manifest_artifacts": p29["verified_artifacts"],
        },
    }
    return report


def build_support_domain_report(
    p11_report: Mapping[str, object],
    p29_report: Mapping[str, object],
) -> dict[str, object]:
    """Build ledgers from already identity-verified report documents."""
    latency_domains = _latency_domains(p29_report)
    raw_episodes = p11_report.get("episodes")
    if not isinstance(raw_episodes, list):
        raise SupportDomainContractError("P11 episodes must be a list")
    episodes = [
        _classify_episode(value, latency_domains, index)
        for index, value in enumerate(raw_episodes)
    ]
    _validate_episode_coverage(episodes)
    episodes.sort(
        key=lambda value: (
            value["task"],
            value["observation_latency_steps"],
            value["action_latency_steps"],
            value["rho"],
            value["seed"],
        )
    )
    supported = [value for value in episodes if value["domain"] == "supported"]
    challenge = [value for value in episodes if value["domain"] == "challenge"]
    ledgers = {
        PRIMARY_LEDGER: _aggregate(episodes),
        "supported_conditional": _aggregate(supported),
        "challenge": _aggregate(challenge),
    }
    _validate_aggregates(ledgers)
    return {
        "schema_version": SUPPORT_DOMAIN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "proposal_id": "R0001-P36",
        "support_statement": SUPPORT_STATEMENT,
        "capability_claim_allowed": False,
        "closed_loop_success_available": False,
        "balanced_factorial_benchmark": False,
        "primary_ledger": PRIMARY_LEDGER,
        "classification_contract": {
            "control_hz": CONTROL_HZ,
            "control_period_ns": CONTROL_PERIOD_NS,
            "validity_duration_ns": VALIDITY_DURATION_NS,
            "boundary": "maximum_source_age_ns <= validity_duration_ns",
            "latencies": latency_domains,
            "action_latency_changes_domain": False,
            "warmup_splitting_allowed": False,
        },
        "coverage": {
            "episode_count": len(episodes),
            "cell_count": len(
                ledgers[PRIMARY_LEDGER]["counts"]["task_observation_action_cells"]
            ),
            "observation_latency_counts": ledgers[PRIMARY_LEDGER]["counts"][
                "observation_latency"
            ],
            "observation_latency_imbalanced": True,
            "statement": (
                "P11's observation-latency cells are 2/10/4; "
                "this is not a balanced factorial capability benchmark."
            ),
        },
        "ledgers": ledgers,
        "episodes": episodes,
        "lineage_and_guardrails": {
            "p11_source_commit": P11_SOURCE_COMMIT,
            "p11_decision": p11_report.get("decision"),
            "p29_source_commit": P29_SOURCE_COMMIT,
            "p29_assessment": p29_report.get("assessment"),
            "p29_guardrails": p29_report.get("guardrails"),
        },
    }


def _load_frozen_run(
    run: Path,
    *,
    report_identity: Mapping[str, object],
    manifest_identity: Mapping[str, object],
) -> dict[str, object]:
    report_path = run / "report.json"
    manifest_path = run / "manifest.json"
    _require_identity(report_path, report_identity)
    _require_identity(manifest_path, manifest_identity)
    report = _read_object(report_path)
    manifest = _read_object(manifest_path)
    artifact_count = _verify_manifest_artifacts(run, manifest)
    return {
        "report": report,
        "manifest": manifest,
        "verified_artifacts": artifact_count,
    }


def _validate_lineage(
    *,
    p11_report: Mapping[str, object],
    p11_manifest: Mapping[str, object],
    p29_report: Mapping[str, object],
    p29_manifest: Mapping[str, object],
) -> None:
    if (
        p11_report.get("source_commit") != P11_SOURCE_COMMIT
        or p11_manifest.get("source_commit") != P11_SOURCE_COMMIT
        or p11_report.get("proposal_id") != "R0001-P11"
        or p11_manifest.get("proposal_id") != "R0001-P11"
        or p11_report.get("contract_complete") is not True
        or p11_report.get("decision") != "rejected"
    ):
        raise SupportDomainContractError("P11 lineage or frozen decision differs")
    if (
        p29_report.get("source_commit") != P29_SOURCE_COMMIT
        or p29_manifest.get("source_commit") != P29_SOURCE_COMMIT
        or p29_report.get("proposal_id") != "R0001-P29"
        or p29_manifest.get("proposal_id") != "R0001-P29"
    ):
        raise SupportDomainContractError("P29 lineage differs")
    p29_input = _mapping(p29_report.get("input"), "P29 input")
    p11_evidence = _mapping(
        p29_report.get("p11_evidence"), "P29 P11 evidence"
    )
    expected_hashes = (
        P11_REPORT_IDENTITY["sha256"],
        P11_MANIFEST_IDENTITY["sha256"],
    )
    if (
        (
            p29_input.get("p11_report_sha256"),
            p29_input.get("p11_manifest_sha256"),
        )
        != expected_hashes
        or (
            p11_evidence.get("report_sha256"),
            p11_evidence.get("manifest_sha256"),
        )
        != expected_hashes
        or p11_evidence.get("source_commit") != P11_SOURCE_COMMIT
    ):
        raise SupportDomainContractError("P29 P11 hash lineage differs")
    assessment = _mapping(p29_report.get("assessment"), "P29 assessment")
    checks = _mapping(assessment.get("checks"), "P29 assessment checks")
    if (
        assessment.get("passed") is not True
        or assessment.get("decision") != "contract_incompatible"
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise SupportDomainContractError("P29 assessment differs")
    expected_guardrails = {
        "action_validity_extended": False,
        "evaluation_latency_removed": False,
        "latest_bundle_substituted": False,
        "p11_decision_changed": False,
    }
    if p29_report.get("guardrails") != expected_guardrails:
        raise SupportDomainContractError("P29 guardrails differ")


def _latency_domains(
    p29_report: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    timeline = _mapping(p29_report.get("timeline"), "P29 timeline")
    if (
        timeline.get("control_hz") != CONTROL_HZ
        or timeline.get("control_period_ns") != CONTROL_PERIOD_NS
        or timeline.get("validity_duration_ns") != VALIDITY_DURATION_NS
    ):
        raise SupportDomainContractError("P29 timing contract differs")
    latencies = _mapping(timeline.get("latencies"), "P29 latencies")
    expected_keys = {str(value) for value in EXPECTED_MAXIMUM_AGE_NS}
    if set(latencies) != expected_keys:
        raise SupportDomainContractError("P29 latency strata differ")
    result: dict[str, dict[str, object]] = {}
    for latency, expected_age in EXPECTED_MAXIMUM_AGE_NS.items():
        entry = _mapping(latencies[str(latency)], f"P29 latency {latency}")
        decisions = entry.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            raise SupportDomainContractError(
                f"P29 latency {latency} decisions are missing"
            )
        ages = [
            _strict_int(
                _mapping(value, "P29 timeline decision").get(
                    "observation_age_ns"
                ),
                "P29 observation age",
            )
            for value in decisions
        ]
        declared_age = _strict_int(
            entry.get("maximum_observation_age_ns"),
            "P29 maximum observation age",
        )
        if (
            entry.get("latency_steps") != latency
            or max(ages) != declared_age
            or declared_age != expected_age
        ):
            raise SupportDomainContractError(
                f"P29 latency {latency} maximum age differs"
            )
        result[str(latency)] = {
            "maximum_source_age_ns": declared_age,
            "domain": (
                "supported"
                if declared_age <= VALIDITY_DURATION_NS
                else "challenge"
            ),
        }
    return result


def _classify_episode(
    value: object,
    latency_domains: Mapping[str, Mapping[str, object]],
    index: int,
) -> dict[str, object]:
    episode = _mapping(value, f"P11 Episode {index}")
    future_fields = FUTURE_DOMAIN_FIELDS.intersection(episode)
    if future_fields:
        raise SupportDomainContractError(
            f"P11 Episode {index} contains future domain label"
        )
    required = {
        "task_id",
        "seed",
        "motion_correlation",
        "observation_latency_steps",
        "action_latency_steps",
        "safety_intervention_count",
        "severe_collision_count",
        "terminated_early",
    }
    missing = required.difference(episode)
    if missing:
        raise SupportDomainContractError(
            f"P11 Episode {index} missing fields: {sorted(missing)}"
        )
    task = episode["task_id"]
    if not isinstance(task, str) or not task:
        raise SupportDomainContractError(f"P11 Episode {index} task is invalid")
    seed = _strict_int(episode["seed"], "P11 Episode seed")
    rho = episode["motion_correlation"]
    if (
        isinstance(rho, bool)
        or not isinstance(rho, (int, float))
        or not math.isfinite(float(rho))
    ):
        raise SupportDomainContractError(f"P11 Episode {index} rho is invalid")
    observation_latency = _strict_int(
        episode["observation_latency_steps"], "P11 observation latency"
    )
    action_latency = _strict_int(
        episode["action_latency_steps"], "P11 action latency"
    )
    latency = latency_domains.get(str(observation_latency))
    if latency is None:
        raise SupportDomainContractError(
            f"P11 Episode {index} has unknown observation latency"
        )
    safety = _strict_nonnegative(
        episode["safety_intervention_count"], "P11 safety intervention"
    )
    collision = _strict_nonnegative(
        episode["severe_collision_count"], "P11 severe collision"
    )
    terminated = episode["terminated_early"]
    if type(terminated) is not bool:
        raise SupportDomainContractError(
            f"P11 Episode {index} early termination is invalid"
        )
    domain = latency["domain"]
    return {
        "task": task,
        "seed": seed,
        "rho": float(rho),
        "observation_latency_steps": observation_latency,
        "action_latency_steps": action_latency,
        "maximum_source_age_ns": latency["maximum_source_age_ns"],
        "domain": domain,
        "ledger_membership": [
            PRIMARY_LEDGER,
            "supported_conditional" if domain == "supported" else "challenge",
        ],
        "safety_intervention_count": safety,
        "severe_collision_count": collision,
        "terminated_early": terminated,
    }


def _validate_episode_coverage(
    episodes: Sequence[Mapping[str, object]],
) -> None:
    identities = [
        (
            value["task"],
            value["seed"],
            value["rho"],
            value["action_latency_steps"],
        )
        for value in episodes
    ]
    if len(episodes) != 144:
        raise SupportDomainContractError("P11 must contain exactly 144 Episodes")
    if len(set(identities)) != len(identities):
        raise SupportDomainContractError("P11 contains duplicate Episode identity")
    expected_identities = {
        (task, seed, rho, action)
        for task in EXPECTED_TASKS
        for seed in EXPECTED_SEEDS
        for rho in EXPECTED_RHOS
        for action in EXPECTED_ACTION_LATENCIES
    }
    if set(identities) != expected_identities:
        raise SupportDomainContractError("P11 Episode identity coverage differs")
    if (
        {value["task"] for value in episodes} != set(EXPECTED_TASKS)
        or {value["rho"] for value in episodes} != set(EXPECTED_RHOS)
        or {
            value["observation_latency_steps"] for value in episodes
        }
        != set(EXPECTED_OBSERVATION_LATENCIES)
        or {value["action_latency_steps"] for value in episodes}
        != set(EXPECTED_ACTION_LATENCIES)
    ):
        raise SupportDomainContractError("P11 Episode strata differ")
    cells = Counter(
        (
            value["task"],
            value["observation_latency_steps"],
            value["action_latency_steps"],
        )
        for value in episodes
    )
    expected_cells = {
        (task, observation, action): {1: 2, 2: 10, 3: 4}[observation]
        for task in EXPECTED_TASKS
        for observation in EXPECTED_OBSERVATION_LATENCIES
        for action in EXPECTED_ACTION_LATENCIES
    }
    if dict(cells) != expected_cells:
        raise SupportDomainContractError("P11 27-cell coverage differs")


def _aggregate(
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "episode_count": len(episodes),
        "safety_intervention_count": sum(
            value["safety_intervention_count"] for value in episodes
        ),
        "severe_collision_count": sum(
            value["severe_collision_count"] for value in episodes
        ),
        "early_termination_count": sum(
            value["terminated_early"] is True for value in episodes
        ),
        "counts": {
            "task": _counts(episodes, "task", EXPECTED_TASKS),
            "observation_latency": _counts(
                episodes,
                "observation_latency_steps",
                EXPECTED_OBSERVATION_LATENCIES,
            ),
            "action_latency": _counts(
                episodes,
                "action_latency_steps",
                EXPECTED_ACTION_LATENCIES,
            ),
            "task_observation_action_cells": [
                {
                    "task": task,
                    "observation_latency_steps": observation,
                    "action_latency_steps": action,
                    "count": sum(
                        value["task"] == task
                        and value["observation_latency_steps"] == observation
                        and value["action_latency_steps"] == action
                        for value in episodes
                    ),
                }
                for task in EXPECTED_TASKS
                for observation in EXPECTED_OBSERVATION_LATENCIES
                for action in EXPECTED_ACTION_LATENCIES
            ],
        },
    }


def _validate_aggregates(ledgers: Mapping[str, Mapping[str, object]]) -> None:
    complete = ledgers[PRIMARY_LEDGER]
    supported = ledgers["supported_conditional"]
    challenge = ledgers["challenge"]
    if (
        complete["episode_count"] != 144
        or supported["episode_count"] != 108
        or challenge["episode_count"] != 36
        or supported["safety_intervention_count"] != 0
        or challenge["safety_intervention_count"] != 2_196
        or complete["severe_collision_count"] != 0
        or complete["early_termination_count"] != 0
    ):
        raise SupportDomainContractError("P36 ledger totals differ")
    expected_partition = {
        task: count for task in EXPECTED_TASKS for count in (36,)
    }
    expected_challenge = {
        task: count for task in EXPECTED_TASKS for count in (12,)
    }
    if (
        supported["counts"]["task"] != expected_partition
        or challenge["counts"]["task"] != expected_challenge
        or supported["counts"]["action_latency"]
        != {"1": 36, "2": 36, "3": 36}
        or challenge["counts"]["action_latency"]
        != {"1": 12, "2": 12, "3": 12}
    ):
        raise SupportDomainContractError("P36 ledger strata differ")


def _counts(
    episodes: Sequence[Mapping[str, object]],
    field: str,
    values: Sequence[object],
) -> dict[str, int]:
    counts = Counter(value[field] for value in episodes)
    return {str(value): counts[value] for value in values}


def _verify_episode_manifest_links(
    run: Path,
    report: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    episodes = report.get("episodes")
    artifacts = _mapping(manifest.get("artifacts"), "P11 manifest artifacts")
    if not isinstance(episodes, list):
        raise SupportDomainContractError("P11 episodes must be a list")
    for index, value in enumerate(episodes):
        episode = _mapping(value, f"P11 Episode {index}")
        artifact = _mapping(
            episode.get("artifact"), f"P11 Episode {index} artifact"
        )
        relative = f"episodes/{artifact.get('path')}"
        identity = artifacts.get(relative)
        if (
            not isinstance(identity, Mapping)
            or artifact.get("sha256") != identity.get("sha256")
            or artifact.get("bytes") != identity.get("bytes")
            or not (run / relative).is_file()
        ):
            raise SupportDomainContractError(
                f"P11 Episode {index} artifact lineage differs"
            )


def _verify_manifest_artifacts(
    run: Path, manifest: Mapping[str, object]
) -> int:
    artifacts = _mapping(manifest.get("artifacts"), "manifest artifacts")
    if not artifacts:
        raise SupportDomainContractError("manifest declares no artifacts")
    resolved_run = run.resolve()
    for relative, identity_value in artifacts.items():
        if not isinstance(relative, str):
            raise SupportDomainContractError("manifest artifact path is invalid")
        identity = _mapping(identity_value, f"artifact {relative}")
        path = (run / relative).resolve()
        if (
            Path(relative).is_absolute()
            or path == resolved_run
            or not path.is_relative_to(resolved_run)
        ):
            raise SupportDomainContractError(
                f"manifest artifact escapes run: {relative}"
            )
        _require_identity(path, identity)
    return len(artifacts)


def _require_identity(
    path: Path, expected: Mapping[str, object]
) -> None:
    try:
        size = path.stat().st_size
        digest = _sha256(path)
    except OSError as error:
        raise SupportDomainContractError(f"artifact unavailable: {path}") from error
    if size != expected.get("bytes") or digest != expected.get("sha256"):
        raise SupportDomainContractError(f"artifact identity differs: {path}")


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SupportDomainContractError(f"invalid JSON artifact: {path}") from error
    return _mapping(value, str(path))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SupportDomainContractError(f"{name} must be an object")
    return value


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise SupportDomainContractError(f"{name} must be an integer")
    return value


def _strict_nonnegative(value: object, name: str) -> int:
    result = _strict_int(value, name)
    if result < 0:
        raise SupportDomainContractError(f"{name} must be nonnegative")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
