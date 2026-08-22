"""Command-line assembly points for the HWR platform."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from hwr.eval.candidate_funnel import (
    CandidateFunnelContractError,
    analyze_candidate_funnel,
    candidate_visible_bytes,
)
from hwr.eval.target_selection import deserialize_policy_input


def aggregate_candidate_funnels(
    episodes: Sequence[Mapping[str, object]],
    *,
    expected_episode_count: int = 24,
) -> dict[str, object]:
    identities = [str(value["planned_episode_id"]) for value in episodes]
    cells: dict[str, list[Mapping[str, object]]] = {}
    for episode in episodes:
        cells.setdefault(str(episode["cell_id"]), []).append(episode)
    summaries = [_cell_summary(cell, records) for cell, records in sorted(cells.items())]
    checks = {
        "episode_count": len(episodes) == expected_episode_count,
        "episode_identity_unique": len(identities) == len(set(identities)),
        "twelve_cells": len(cells) == 12,
        "two_episodes_per_cell": all(len(values) == 2 for values in cells.values()),
        "episode_checks": all(
            bool(value["funnel"]["checks"]["passed"]) for value in episodes
        ),
        "capture_identity_guards": all(
            bool(value["capture_enabled_disabled_identity"])
            for value in episodes
        ),
    }
    return {
        "episode_count": len(episodes),
        "cell_count": len(cells),
        "cells": summaries,
        "weakest_task_cell": min(
            summaries,
            key=lambda value: (
                value["final_candidate_count"],
                value["pre_top64_candidate_count"],
                value["raw_candidate_count"],
                value["cell_id"],
            ),
            default=None,
        ),
        "checks": {**checks, "passed": all(checks.values())},
    }


def analyze_candidate_capsule_directory(
    repository: Path,
    capsule_directory: Path,
    current_identities: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = _read_object(capsule_directory / "manifest.json")
    report = _read_object(capsule_directory / "report.json")
    plan = _read_object(capsule_directory / "plan.json")
    index = _read_object(capsule_directory / "capsules.json")
    _verify_artifacts(capsule_directory, manifest)
    source_commit = str(manifest.get("source_commit", ""))
    if not _is_ancestor(repository, source_commit):
        raise CandidateFunnelContractError("E1 source commit is not an ancestor")
    _require_e1_source_identity(manifest, current_identities)
    if (
        manifest.get("status") != "complete"
        or report.get("decision")
        != "accepted as immutable acquisition evidence contract"
        or plan.get("planned_episode_count") != 24
        or index.get("capsule_count") != 24
    ):
        raise CandidateFunnelContractError("E1 evidence is not accepted and complete")
    planned = {str(value["planned_episode_id"]) for value in plan["episodes"]}
    records = index.get("episodes")
    if not isinstance(records, list):
        raise CandidateFunnelContractError("capsule episodes are missing")
    record_ids = [str(value.get("planned_episode_id")) for value in records]
    if len(record_ids) != len(set(record_ids)) or set(record_ids) != planned:
        raise CandidateFunnelContractError("capsule Episode ledger differs")
    episodes = [_analyze_capsule_record(capsule_directory, row) for row in records]
    identity = {
        "path": str(capsule_directory),
        "source_commit": source_commit,
        "report": _relative_identity(capsule_directory, "report.json"),
        "manifest": _relative_identity(capsule_directory, "manifest.json"),
    }
    return {
        "episodes": episodes,
        "aggregate": aggregate_funnel_reports(episodes),
    }, identity


def _analyze_capsule_record(root, record):
    captures, candidate = record.get("captures"), record.get("candidate_set")
    if not isinstance(captures, list) or not isinstance(candidate, Mapping):
        raise CandidateFunnelContractError("capsule record is incomplete")
    ordered = sorted(captures, key=lambda value: int(value["capture_ordinal"]))
    if [int(value["capture_ordinal"]) for value in ordered] != list(
        range(len(ordered))
    ):
        raise CandidateFunnelContractError("capsule capture ordinals differ")
    payloads, visible_by_identity = [], {}
    for capture in ordered:
        payload = read_bound_blob(root, capture["policy_input"])
        visible = read_bound_blob(root, capture["candidate_visible_input"])
        value = deserialize_policy_input(payload)
        if candidate_visible_bytes(value) != visible:
            raise CandidateFunnelContractError("candidate-visible sidecar differs")
        identity = (value.observation_timestamp_ns, value.sequence_id)
        visible_hash = _sha256(visible)
        if identity in visible_by_identity and (
            visible_by_identity[identity] != visible_hash
        ):
            raise CandidateFunnelContractError("observation identity changed payload")
        visible_by_identity.setdefault(identity, visible_hash)
        payloads.append(payload)
    if not ordered or not bool(ordered[-1].get("final_input")):
        raise CandidateFunnelContractError("capsule final input is missing")
    funnel = analyze_candidate_funnel(
        tuple(
            payload
            for payload, row in zip(payloads, ordered, strict=True)
            if not row["final_input"]
        ),
        acquisition_base_pose=record["acquisition_base_pose"],
        final_input=payloads[-1],
        expected_candidate_bytes=read_bound_blob(root, candidate),
        expected_selected_index=int(candidate["selected_index"]),
        expected_score_sha256=str(candidate["score_bytes_sha256"]),
        selection_permitted=record.get("acquisition_failure") is None,
    )
    return {
        "planned_episode_id": record["planned_episode_id"],
        "task_id": record["task_id"],
        "cell_id": record["cell_id"],
        "replicate_ordinal": record["replicate_ordinal"],
        "capture_enabled_disabled_identity": record[
            "capture_enabled_disabled_identity"
        ],
        "funnel": funnel,
    }


def read_bound_blob(root: Path, identity: Mapping[str, object]) -> bytes:
    path = (root / str(identity["path"])).resolve()
    if not path.is_relative_to(root):
        raise CandidateFunnelContractError("artifact path escaped capsule root")
    content = path.read_bytes()
    if len(content) != int(identity["bytes"]) or _sha256(content) != identity["sha256"]:
        raise CandidateFunnelContractError("artifact bytes or hash differ")
    return content


def _verify_artifacts(root, manifest):
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CandidateFunnelContractError("E1 manifest artifacts are missing")
    for name, identity in artifacts.items():
        read_bound_blob(root, {"path": name, **identity})


def _require_e1_source_identity(manifest, current):
    source = manifest.get("source_identities")
    if not isinstance(source, Mapping):
        raise CandidateFunnelContractError("E1 source identities are missing")
    for key in ("binding", "task_config", "recursive_xml"):
        if source.get(key) != current.get(key):
            raise CandidateFunnelContractError(f"E1 {key} identity drifted")
    for name in ("formal_generator", "p41_bridge", "formal_backend"):
        if source["sources"].get(name) != current["sources"].get(name):
            raise CandidateFunnelContractError(f"E1 {name} identity drifted")


def _read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateFunnelContractError(f"{path.name} must contain an object")
    return value


def _is_ancestor(root, commit):
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        return False
    return subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=root, check=False,
    ).returncode == 0


def _relative_identity(root, name):
    content = (root / name).read_bytes()
    return {"path": name, "sha256": _sha256(content), "bytes": len(content)}


def _cell_summary(cell_id, records):
    rates = [_stage_rates(value["funnel"]) for value in records]
    repeatable = sorted(
        stage
        for stage in set.intersection(*(set(value) for value in rates))
        if all(value[stage] >= 0.60 for value in rates)
    ) if rates else []
    return {
        "cell_id": cell_id,
        "task_id": records[0]["task_id"],
        "episode_count": len(records),
        "raw_candidate_count": _sum_metric(
            records, "anchor_ledger", "raw_candidate_count"
        ),
        "component_count": sum(
            int(value["funnel"]["component_ledger"]["ordinal"]["component_count"])
            for value in records
        ),
        "pre_top64_candidate_count": _sum_metric(
            records, "ranking_ledger", "pre_top64_candidate_count"
        ),
        "final_candidate_count": _sum_metric(
            records, "formal_candidate", "candidate_count"
        ),
        "repeatable_descriptive_loss_stage": repeatable,
    }


def _sum_metric(records, section, field):
    return sum(int(value["funnel"][section][field]) for value in records)


def _stage_rates(funnel):
    layers = (
        ("anchor", funnel["anchor_ledger"]["stages"]),
        ("component", funnel["component_ledger"]["ordinal"]["stages"]),
        ("ranking", funnel["ranking_ledger"]["stages"]),
    )
    return {
        f"{layer}.{row['stage']}": (
            int(row["rejection_count"]) / int(row["input_count"])
            if int(row["input_count"]) else 0.0
        )
        for layer, rows in layers
        for row in rows
    }


def _sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()

