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

from hwr.eval import validate_plan_contract
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
        "cell_ordinal": capsule.cell_ordinal,
        "replicate_ordinal": capsule.replicate_ordinal,
        "candidate_ordinal": capsule.candidate_ordinal,
        "environment_seed": capsule.environment_seed,
        "policy_rng_seed": capsule.policy_rng_seed,
        "replacement": False,
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
        "all_capsule_input_count": len(captures),
        "candidate_keyframe_count": sum(
            not capture.final_input for capture in capsule.captures
        ),
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
        "cell_ordinal": capsule.cell_ordinal,
        "replicate_ordinal": capsule.replicate_ordinal,
        "candidate_ordinal": capsule.candidate_ordinal,
        "environment_seed": capsule.environment_seed,
        "policy_rng_seed": capsule.policy_rng_seed,
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
    result = validate_candidate_record_set(plan, terminals)
    planned = [str(value["planned_episode_id"]) for value in plan["episodes"]]
    published = [str(value.get("planned_episode_id")) for value in terminals]
    replacement_count = sum(
        bool(value.get("replacement", True)) for value in terminals
    )
    return {
        **result,
        "replacement_count": replacement_count,
        "passed": result["passed"] and replacement_count == 0,
    }


def validate_candidate_record_set(plan, records) -> dict[str, object]:
    expected_rows = plan.get("episodes")
    if not isinstance(expected_rows, list):
        raise CandidateFunnelContractError("plan episodes are missing")
    expected_ids = [str(value.get("planned_episode_id")) for value in expected_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise CandidateFunnelContractError("plan Episode identities are duplicated")
    expected = dict(zip(expected_ids, expected_rows, strict=True))
    published = [str(value.get("planned_episode_id")) for value in records]
    duplicate_count = len(published) - len(set(published))
    missing = sorted(set(expected) - set(published))
    unplanned = sorted(set(published) - set(expected))
    mismatches = []
    for record in records:
        identity = str(record.get("planned_episode_id"))
        planned = expected.get(identity)
        if planned is None:
            continue
        for field in (
            "task_id",
            "cell_id",
            "cell_ordinal",
            "replicate_ordinal",
            "candidate_ordinal",
            "environment_seed",
            "policy_rng_seed",
        ):
            if record.get(field) != planned.get(field):
                mismatches.append(
                    {
                        "planned_episode_id": identity,
                        "field": field,
                        "expected": planned.get(field),
                        "actual": record.get(field),
                    }
                )
        planned_latency = {
            "observation_steps": planned.get(
                "sampled_observation_latency_steps"
            ),
            "action_steps": planned.get("sampled_action_latency_steps"),
        }
        if record.get("planned_latency") != planned_latency:
            mismatches.append(
                {
                    "planned_episode_id": identity,
                    "field": "planned_latency",
                    "expected": planned_latency,
                    "actual": record.get("planned_latency"),
                }
            )
        if record.get("replacement") is not False:
            mismatches.append(
                {
                    "planned_episode_id": identity,
                    "field": "replacement",
                    "expected": False,
                    "actual": record.get("replacement"),
                }
            )
    passed = (
        len(records) == len(expected_rows)
        and not missing
        and not unplanned
        and duplicate_count == 0
        and not mismatches
    )
    return {
        "planned_count": len(expected_rows),
        "published_count": len(records),
        "missing": missing,
        "unplanned": unplanned,
        "duplicate_count": duplicate_count,
        "field_mismatches": mismatches,
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
    if "frozen_document_commit_is_ancestor" not in extra:
        raise CandidateFunnelContractError(
            "manifest requires a measured frozen-document ancestry result"
        )
    return {
        "schema_version": schema,
        "proposal_id": proposal_id,
        "status": status,
        "source_commit": source_commit,
        "frozen_document_commit": frozen_document_commit,
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


def require_candidate_disk_capacity(output: Path) -> None:
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 5 * 1024**3:
        raise RuntimeError("P50-E1 requires at least 5GiB free on the data volume")


def candidate_source_identities(
    root, binding_path, task_path, task_ids, source_paths, historical_trees,
    frozen_document_path,
):
    from hwr.adapters.mujoco.training_catalog import load_default_formal_household_catalogs
    from hwr.eval.tool_kinematics import recursive_xml_input_identity

    _, bindings = load_default_formal_household_catalogs(root)
    return {
        "binding": candidate_file_identity(root, root / binding_path),
        "task_config": candidate_file_identity(root, root / task_path),
        "recursive_xml": {
            task_id: recursive_xml_input_identity(root, bindings[task_id].model_path)
            for task_id in task_ids
        },
        "sources": {
            name: candidate_file_identity(root, root / path)
            for name, path in source_paths.items()
        },
        "historical_research_loop_trees": {
            path: candidate_git_tree(root, path) for path in historical_trees
        },
        "frozen_document": candidate_file_identity(root, root / frozen_document_path),
    }


def require_candidate_clean_source(
    root, identities, frozen_document_commit, frozen_document_path,
    protected_paths, historical_trees, runner=subprocess.run,
) -> None:
    status = runner(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P50-E1 runner requires clean committed source")
    ancestor = runner(
        ("git", "merge-base", "--is-ancestor", frozen_document_commit, "HEAD"),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("P50 frozen document commit is not an ancestor")
    protected = runner(
        (
            "git",
            "diff",
            "--quiet",
            frozen_document_commit,
            "HEAD",
            "--",
            *protected_paths,
        ),
        cwd=root,
        check=False,
    )
    if protected.returncode != 0:
        raise RuntimeError("P50 source/config/XML anchors drifted")
    if identities.get("historical_research_loop_trees") != historical_trees:
        raise RuntimeError("P50 historical research-loop documents drifted")
    frozen = runner(
        ("git", "show", f"{frozen_document_commit}:{frozen_document_path}"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if identities["frozen_document"] != {
        "path": frozen_document_path.as_posix(),
        "sha256": _sha256(frozen),
        "bytes": len(frozen),
    }:
        raise RuntimeError("P50 frozen document content drifted")


def analyze_candidate_capsule_directory(
    repository: Path,
    capsule_directory: Path,
    current_identities: Mapping[str, object],
    frozen_document_commit: str,
    latency_sampler,
    required_source_names: Sequence[str] | None = None,
    *,
    expected_plan_schema: str = "hwr.p50-acquisition-plan/v1",
    expected_proposal_id: str = "R0001-P50-E1",
    expected_plan_id: str = "R0001-P50-E1-formal",
    expected_commitment: str = (
        "ed945b2dcfe90c6aab639164da32cc8a1a905df56534c42a443d1bd4753e16a4"
    ),
    expected_cells: Sequence[tuple[str, int, int]] = (),
    acquisition_steps: int = 995,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = _read_object(capsule_directory / "manifest.json")
    report = _read_object(capsule_directory / "report.json")
    plan = _read_object(capsule_directory / "plan.json")
    index = _read_object(capsule_directory / "capsules.json")
    _verify_artifacts(capsule_directory, manifest)
    source_commit = str(manifest.get("source_commit", ""))
    frozen_to_source = candidate_commit_is_ancestor(
        repository, frozen_document_commit, source_commit
    )
    source_to_head = candidate_commit_is_ancestor(
        repository, source_commit, "HEAD"
    )
    if (
        manifest.get("frozen_document_commit") != frozen_document_commit
        or manifest.get("frozen_document_commit_is_ancestor") is not True
        or not frozen_to_source
        or not source_to_head
    ):
        raise CandidateFunnelContractError("E1 frozen/source ancestry differs")
    _require_e1_source_identity(
        repository, source_commit, manifest, current_identities,
        frozen_document_commit, required_source_names,
    )
    validate_plan_contract(
        plan,
        salt=None,
        expected_schema=expected_plan_schema,
        expected_proposal_id=expected_proposal_id,
        expected_plan_id=expected_plan_id,
        expected_commitment=expected_commitment,
        expected_cells=expected_cells,
        acquisition_steps=acquisition_steps,
        latency_sampler=latency_sampler,
    )
    if (
        manifest.get("status") != "complete"
        or report.get("decision")
        != "accepted as immutable acquisition evidence contract"
        or plan.get("planned_episode_count") != 24
        or index.get("capsule_count") != 24
    ):
        raise CandidateFunnelContractError("E1 evidence is not accepted and complete")
    records = index.get("episodes")
    if not isinstance(records, list):
        raise CandidateFunnelContractError("capsule episodes are missing")
    record_ledger = validate_candidate_record_set(plan, records)
    terminal_ledger = validate_candidate_terminal_ledger(
        plan, index.get("terminals", [])
    )
    if not record_ledger["passed"] or not terminal_ledger["passed"]:
        raise CandidateFunnelContractError("capsule Episode ledger differs")
    planned_by_id = {
        str(value["planned_episode_id"]): value for value in plan["episodes"]
    }
    episodes = [
        _analyze_capsule_record(
            capsule_directory,
            row,
            planned_by_id[str(row["planned_episode_id"])],
        )
        for row in records
    ]
    identity = {
        "path": str(capsule_directory),
        "source_commit": source_commit,
        "frozen_document_commit": frozen_document_commit,
        "frozen_to_source_commit_ancestor": frozen_to_source,
        "source_commit_to_current_head_ancestor": source_to_head,
        "report": _relative_identity(capsule_directory, "report.json"),
        "manifest": _relative_identity(capsule_directory, "manifest.json"),
    }
    return {
        "episodes": episodes,
        "aggregate": aggregate_candidate_funnels(episodes),
    }, identity


def _analyze_capsule_record(root, record, planned):
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
        "planned_episode_id": planned["planned_episode_id"],
        "task_id": planned["task_id"],
        "cell_id": planned["cell_id"],
        "replicate_ordinal": planned["replicate_ordinal"],
        "candidate_ordinal": planned["candidate_ordinal"],
        "environment_seed": planned["environment_seed"],
        "policy_rng_seed": planned["policy_rng_seed"],
        "planned_latency": {
            "observation_steps": planned["sampled_observation_latency_steps"],
            "action_steps": planned["sampled_action_latency_steps"],
        },
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


def _require_e1_source_identity(
    repository, source_commit, manifest, current, frozen_document_commit,
    required_source_names,
):
    source = manifest.get("source_identities")
    if not isinstance(source, Mapping):
        raise CandidateFunnelContractError("E1 source identities are missing")
    for key in ("binding", "task_config"):
        _require_committed_identity(repository, source_commit, source.get(key))
    frozen = source.get("frozen_document")
    _require_committed_identity(
        repository, frozen_document_commit, frozen
    )
    recursive = source.get("recursive_xml")
    current_recursive = current.get("recursive_xml")
    if not isinstance(recursive, Mapping) or set(recursive) != set(current_recursive):
        raise CandidateFunnelContractError("E1 recursive XML identity set differs")
    for task_id, identity in recursive.items():
        current_identity = current_recursive[task_id]
        if (
            identity.get("entry_model") != current_identity.get("entry_model")
            or {
                value.get("path") for value in identity.get("dependencies", [])
            } != {
                value.get("path")
                for value in current_identity.get("dependencies", [])
            }
        ):
            raise CandidateFunnelContractError("E1 recursive XML paths differ")
        for dependency in identity.get("dependencies", []):
            _require_committed_identity(repository, source_commit, dependency)
    expected_names = (
        tuple(current.get("sources", ()))
        if required_source_names is None
        else tuple(required_source_names)
    )
    if set(source.get("sources", ())) != set(expected_names):
        raise CandidateFunnelContractError("E1 source identity set differs")
    for name in expected_names:
        identity = source["sources"].get(name)
        if identity.get("path") != current["sources"][name].get("path"):
            raise CandidateFunnelContractError(f"E1 {name} identity drifted")
        _require_committed_identity(repository, source_commit, identity)


def _require_committed_identity(repository, commit, identity) -> None:
    if not isinstance(identity, Mapping):
        raise CandidateFunnelContractError("E1 source identity is missing")
    content = subprocess.run(
        ("git", "show", f"{commit}:{identity.get('path')}"),
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if (
        identity.get("sha256") != _sha256(content)
        or identity.get("bytes") != len(content)
    ):
        raise CandidateFunnelContractError("E1 source identity bytes differ")


def _read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateFunnelContractError(f"{path.name} must contain an object")
    return value


def candidate_commit_is_ancestor(root, ancestor, descendant):
    if any(
        len(commit) != 40
        or any(value not in "0123456789abcdef" for value in commit)
        for commit in (ancestor, descendant)
        if commit != "HEAD"
    ):
        return False
    return subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
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

