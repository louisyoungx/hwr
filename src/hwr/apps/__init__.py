"""Command-line assembly points for the HWR platform."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

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


def persist_candidate_episode(result):
    from hwr.adapters.mujoco.candidate_acquisition import (
        CAPSULE_SCHEMA,
        EPISODE_SCHEMA,
        capture_record,
        replay_candidate_set,
    )
    from hwr.eval.target_selection import ACQUISITION_STEPS, CANDIDATE_SCHEMA

    capsule = result.capsule
    prefix = f"blobs/{capsule.planned_episode_id}"
    blobs: dict[str, bytes] = {}
    captures = []
    for capture in capsule.captures:
        suffix = f"{capture.capture_ordinal:02d}"
        policy_path = f"{prefix}/capture-{suffix}-policy.bin"
        visible_path = f"{prefix}/capture-{suffix}-candidate-visible.bin"
        blobs[policy_path] = capture.policy_input_bytes
        blobs[visible_path] = capture.candidate_visible_bytes
        captures.append(
            capture_record(
                capture,
                policy_blob=policy_path,
                candidate_visible_blob=visible_path,
            )
        )
    replay = replay_candidate_set(capsule) if capsule.candidate_bytes else None
    candidate = {
        "candidate_count": capsule.candidate_count,
        "selected_index": capsule.selected_index,
        "score_bytes_sha256": capsule.candidate_score_sha256,
        "schema_version": CANDIDATE_SCHEMA,
        "generated_online": capsule.acquisition_failure is None,
    }
    if capsule.candidate_bytes:
        candidate_path = f"{prefix}/candidate-set.json"
        blobs[candidate_path] = capsule.candidate_bytes
        candidate.update(
            {
                "path": candidate_path,
                "sha256": capsule.candidate_sha256,
                "bytes": len(capsule.candidate_bytes),
            }
        )
    record = {
        "schema_version": CAPSULE_SCHEMA,
        "planned_episode_id": capsule.planned_episode_id,
        "task_id": capsule.task_id,
        "cell_id": capsule.cell_id,
        "replicate_ordinal": capsule.replicate_ordinal,
        "candidate_ordinal": capsule.candidate_ordinal,
        "environment_seed": capsule.environment_seed,
        "policy_rng_seed": capsule.policy_rng_seed,
        "planned_latency": {
            "observation_steps": capsule.sampled_observation_latency_steps,
            "action_steps": capsule.sampled_action_latency_steps,
        },
        "runtime_latency": {
            "observation_steps": capsule.runtime_observation_latency_steps,
            "action_steps": capsule.runtime_action_latency_steps,
            "override_inactive": capsule.latency_override_inactive,
        },
        "runtime_randomization_sha256": capsule.runtime_randomization_sha256,
        "acquisition_base_pose": list(capsule.acquisition_base_pose),
        "captures": captures,
        "capture_count": len(captures),
        "candidate_set": candidate,
        "acquisition_failure": capsule.acquisition_failure,
        "proposed_action_sha256": capsule.proposed_action_sha256,
        "applied_action_sha256": capsule.applied_action_sha256,
        "observation_identity_trace_sha256": (
            capsule.observation_identity_trace_sha256
        ),
        "primary_run": result.primary_summary,
        "validation_replay": result.validation_summary,
        "replay_comparison": result.replay_comparison,
        "same_seed_validation_replay": capsule.same_seed_validation_replay,
        "capture_enabled_disabled_identity": (
            capsule.capture_enabled_disabled_identity
        ),
        "offline_candidate_replay_bit_identical": (
            None
            if replay is None
            else replay.canonical_bytes == capsule.candidate_bytes
            and replay.candidate_set_sha256 == capsule.candidate_sha256
        ),
        "anchor_blobs_complete": all(
            blobs[row[kind]["path"]]
            and _sha256(blobs[row[kind]["path"]]) == row[kind]["sha256"]
            for row in captures
            for kind in ("policy_input", "candidate_visible_input")
        ),
    }
    terminal = {
        "schema_version": EPISODE_SCHEMA,
        "planned_episode_id": capsule.planned_episode_id,
        "task_id": capsule.task_id,
        "cell_id": capsule.cell_id,
        "replicate_ordinal": capsule.replicate_ordinal,
        "candidate_ordinal": capsule.candidate_ordinal,
        "replacement": False,
        "resolved": True,
        "trace_step_count": result.trace_step_count,
        "planned_step_count": ACQUISITION_STEPS,
        "unexecuted_step_count": ACQUISITION_STEPS - result.trace_step_count,
        "runtime_terminal": result.runtime_terminal,
        "acquisition_failure": capsule.acquisition_failure,
        "candidate_count": capsule.candidate_count,
        "selected_index": capsule.selected_index,
        "action_bounds_valid": result.action_bounds_valid,
        "stale_action_applied_count": result.stale_action_applied_count,
        "severe_collision_count": result.severe_collision_count,
        "invalid_force_count": result.invalid_force_count,
        "p40_conservation_maximum_difference": (
            result.p40_conservation_maximum_difference
        ),
        "safety_intervention_count": result.safety_intervention_count,
        "planned_latency": record["planned_latency"],
        "runtime_latency": record["runtime_latency"],
        "runtime_randomization_sha256": capsule.runtime_randomization_sha256,
        "validation_replay": result.validation_summary,
        "replay_comparison": result.replay_comparison,
    }
    return terminal, record, blobs


def validate_candidate_terminal_ledger(plan, terminals) -> dict[str, object]:
    planned = [str(value["planned_episode_id"]) for value in plan["episodes"]]
    published = [str(value.get("planned_episode_id")) for value in terminals]
    missing = sorted(set(planned) - set(published))
    unplanned = sorted(set(published) - set(planned))
    duplicate_count = len(published) - len(set(published))
    replacement_count = sum(
        bool(value.get("replacement", True)) for value in terminals
    )
    passed = (
        len(published) == len(planned)
        and not missing
        and not unplanned
        and duplicate_count == 0
        and replacement_count == 0
    )
    return {
        "planned_count": len(planned),
        "published_count": len(published),
        "missing": missing,
        "unplanned": unplanned,
        "duplicate_count": duplicate_count,
        "replacement_count": replacement_count,
        "passed": passed,
    }


def candidate_artifact_manifest(
    *,
    schema: str,
    proposal_id: str,
    source_commit: str,
    frozen_document_commit: str,
    command: Sequence[str],
    identities: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    runtime_versions: Mapping[str, str],
    status: str,
    extra: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": schema,
        "proposal_id": proposal_id,
        "status": status,
        "source_commit": source_commit,
        "frozen_document_commit": frozen_document_commit,
        "frozen_document_commit_is_ancestor": status == "complete",
        "command": list(command),
        "source_identities": identities,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            **runtime_versions,
        },
        **extra,
        "artifacts": {
            name: {"sha256": _sha256(content), "bytes": len(content)}
            for name, content in sorted(artifacts.items())
        },
    }


def candidate_source_commit(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P50 runner requires a full Git source commit")
    return commit


def candidate_git_tree(root: Path, path: str) -> str:
    return subprocess.run(
        ("git", "rev-parse", f"HEAD:{path}"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def candidate_file_identity(root: Path, path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(content),
        "bytes": len(content),
    }


def create_candidate_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir()
    try:
        for name, content in sorted(artifacts.items()):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def candidate_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def resolve_candidate_path(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def candidate_sha256(payload: bytes) -> str:
    return _sha256(payload)


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
        "aggregate": aggregate_candidate_funnels(episodes),
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
    if (
        not ordered
        or sum(bool(value.get("final_input")) for value in ordered) != 1
        or not bool(ordered[-1].get("final_input"))
    ):
        raise CandidateFunnelContractError("capsule final input is missing")
    funnel = analyze_candidate_funnel(
        tuple(
            payload
            for payload, row in zip(payloads, ordered, strict=True)
            if not row["final_input"]
        ),
        acquisition_base_pose=record["acquisition_base_pose"],
        final_input=payloads[-1],
        expected_candidate_bytes=(
            read_bound_blob(root, candidate)
            if candidate.get("generated_online") is True
            else b""
        ),
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
    for key in ("binding", "task_config", "recursive_xml", "frozen_document"):
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

