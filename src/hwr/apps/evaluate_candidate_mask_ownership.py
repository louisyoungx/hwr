"""Build the frozen R0001-P79-E1 corrected offline candidate bank."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

from hwr.apps import (
    candidate_commit_is_ancestor as _is_ancestor,
    candidate_file_identity as _identity,
    candidate_json_bytes as _json_bytes,
    candidate_sha256 as _sha256,
    candidate_source_commit as _source_commit,
    create_candidate_output as _write_atomic,
    read_bound_blob as _read_bound_blob,
    resolve_candidate_path as _resolve,
)
from hwr.eval import candidate_mask_ownership as ownership
from hwr.eval import target_selection
MODULE_NAME = "hwr.apps.evaluate_candidate_mask_ownership"
PROPOSAL_ID = "R0001-P79-E1"
BANK_SCHEMA = "hwr.p79-candidate-bank/v1"
REGRESSION_SCHEMA = "hwr.p79-candidate-mask-regression/v1"
REPORT_SCHEMA = "hwr.p79-candidate-mask-ownership-report/v1"
MANIFEST_SCHEMA = "hwr.p79-candidate-mask-ownership-artifacts/v1"
FROZEN_DOCUMENT_COMMIT = "61d85cda1b96058831b4f93c7ad21a39f51cc2ab"
FROZEN_DOCUMENT_PATH = Path("docs/research-loop/0014/03-experiment.md")
REVIEW_COMMIT = "41eae6575407263fdcbe1b96667b33bdc2392fd6"
OLD_SOURCE_COMMIT = "d67791a53491ce37cddaef4bd7d6b71ad3e66ac2"
FORMAL_INPUT = Path("runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001")
FORMAL_OUTPUT = Path("runs/research-loop/0014/r0014-p79-candidate-bank-s20267901")
EXPECTED_INPUTS = {
    "capsules.json": "223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf",
    "plan.json": "5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab",
    "report.json": "b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0",
    "manifest.json": "cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86",
}
EXPECTED_CAPTURE_LEDGER_SHA256 = "ff8c5cf53942e89e5ebc04dd8e9020313e5a120dc62ad6ca8764d93a6eda6145"
EXPECTED_MANIFEST_ARTIFACTS = 795
EXPECTED_EPISODES, EXPECTED_CAPTURES, EXPECTED_INPUT_BLOBS = 24, 384, 768
HISTORICAL_ARTIFACT_TREES = {
    "runs/research-loop/0010/r0010-p51-e1-bank-s20265101": "c1990ba894a50fdc6184359983d05f02cffe6a52",
    "runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701": "651371c386eb325c5e96c82e2638f4a3298f1e2d",
    "runs/research-loop/0012/r0012-p60-phase-entry-s20266001": "107e08cb3fde5efca38e2f248bc4d50bc0763e2e",
    "runs/research-loop/0013/r0013-p66-predictive-witness-s20266601": "2595086ad308d06dadeaea0c2caafab862b3638e",
}
MAX_WALL_SECONDS, MAX_RSS_BYTES = 10 * 60, 4 * 1024**3
MAX_ARTIFACT_BYTES, MIN_DISK_FREE_BYTES = 25 * 1024**2, 20 * 1024**3
ALLOWED_CHANGES = frozenset((
    "src/hwr/eval/target_selection.py",
    "src/hwr/eval/candidate_mask_ownership.py",
    "src/hwr/apps/evaluate_candidate_mask_ownership.py",
    "src/hwr/eval/initial_candidate_association.py",
    "tests/test_candidate_mask_ownership.py",
    "tests/test_candidate_mask_ownership_app.py",
    "tests/test_candidate_association.py",
))
SOURCE_PATHS = tuple(Path(value) for value in sorted(ALLOWED_CHANGES)
                     if value.startswith("src/"))
CLAIM_FLAGS = dict.fromkeys((
    "training_executed", "policy_inference_executed",
    "physical_acquisition_executed", "capability_evaluation_executed",
    "capability_claim_allowed", "task_success_claim_allowed",
    "generalization_claim_allowed", "hardware_safety_claim_allowed",
    "candidate_quality_claim_allowed", "selector_improvement_claim_allowed",
), False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser
def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    input_path = _resolve(root, arguments.input)
    output = _resolve(root, arguments.output)
    _require_frozen_paths(root, input_path, output)
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    _require_disk(output)
    started = time.perf_counter()
    source_commit = _source_commit(root)
    provenance = _provenance(root, input_path, source_commit)
    _require_provenance(provenance)
    capsules = _read_json(input_path / "capsules.json")
    first = build_bank(input_path, capsules, started=started)
    second = build_bank(input_path, capsules, started=started)
    replay_identical = _full_bank_replay_bit_identical(first, second)
    history_after = _directory_identities(root, input_path)
    history_unchanged = history_after == provenance["input_files"]
    report = _report(
        first,
        replay_identical=replay_identical,
        history_unchanged=history_unchanged,
        provenance=provenance,
        source_commit=source_commit,
        command=_command(arguments),
        elapsed=time.perf_counter() - started,
    )
    artifacts = {
        **first["artifacts"],
        "report.json": _json_bytes(report),
    }
    manifest = _manifest(
        source_commit,
        _command(arguments),
        provenance,
        artifacts,
        report,
        time.perf_counter() - started,
    )
    artifacts["manifest.json"] = _json_bytes(manifest)
    _require_budget(
        time.perf_counter() - started,
        int(manifest["runtime"]["peak_rss_bytes"]),
        artifacts,
    )
    _write_atomic(output, artifacts)
    return {
        "output": str(output), "decision": report["decision"],
        "episode_count": report["episode_count"],
        "bank_sha256": _sha256(artifacts["bank.json"]),
        "manifest_sha256": _sha256(artifacts["manifest.json"])}


def build_bank(
    input_path: Path,
    capsules: Mapping[str, object],
    *,
    started: float,
) -> dict[str, object]:
    workers = min(EXPECTED_EPISODES, os.cpu_count() or 1)
    return _build_bank(input_path, capsules, started, workers)
def _build_bank(input_path, capsules, started, max_workers):
    deadline = started + MAX_WALL_SECONDS
    episodes = capsules.get("episodes")
    if (
        capsules.get("capsule_count") != EXPECTED_EPISODES
        or not isinstance(episodes, list)
        or len(episodes) != EXPECTED_EPISODES
    ):
        raise RuntimeError("P79 input cohort must contain 24 Episodes")
    identifiers = [str(episode["planned_episode_id"]) for episode in episodes]
    if (
        len(identifiers) != len(set(identifiers))
        or any(episode.get("replacement") is not False for episode in episodes)
    ):
        raise RuntimeError("P79 Episode identity or replacement differs")
    jobs = ((str(input_path), episode) for episode in episodes)
    results = _run_ordered_jobs(
        _build_episode_job, jobs, deadline, max_workers
    )
    bank_records = [value["bank"] for value in results]
    regression_records = [value["regression"] for value in results]
    artifacts = {value["blob_path"]: value["blob"] for value in results}
    bank = {
        "schema_version": BANK_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_acquisition": FORMAL_INPUT.as_posix(),
        "episode_count": len(bank_records),
        "capture_count": sum(int(value["frame_count"]) for value in results),
        "episodes": bank_records,
        **CLAIM_FLAGS,
    }
    regression = {
        "schema_version": REGRESSION_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "sample_unit": "Episode",
        "confirmatory_metrics_exclude_observed_legacy_drift": True,
        "episode_count": len(regression_records),
        "records": regression_records,
        "descriptive": _descriptive(regression_records),
        **CLAIM_FLAGS,
    }
    artifacts.update({
        "bank.json": _json_bytes(bank), "regression.json": _json_bytes(regression),
    })
    if time.perf_counter() > deadline:
        raise RuntimeError("P79 wall-time budget exceeded")
    return {
        "artifacts": artifacts, "bank": bank, "regression": regression,
        "audits": [record["audit"] for record in regression_records]}
def _build_episode_job(job) -> dict[str, object]:
    input_path_value, episode = job
    input_path = Path(input_path_value)
    identity = str(episode["planned_episode_id"])
    pose = episode["acquisition_base_pose"]
    payloads, capture_records = _episode_inputs(input_path, episode)
    candidate_set, audit = ownership.audit_episode(
        payloads[:-1], acquisition_base_pose=pose, final_input=payloads[-1],
    )
    final_value = target_selection.deserialize_policy_input(payloads[-1])
    scores = target_selection.candidate_scores(
        candidate_set, final_value.base_pose, acquisition_base_pose=pose,
    )
    selected_index = target_selection.select_candidate_index(
        candidate_set, final_value.base_pose, acquisition_base_pose=pose,
    )
    blob_path = f"blobs/{identity}/candidate-set.json"
    old = _old_candidate_identity(input_path, episode["candidate_set"])
    new = {
        "schema_version": target_selection.CANDIDATE_SCHEMA,
        "path": blob_path,
        "sha256": candidate_set.candidate_set_sha256,
        "bytes": len(candidate_set.canonical_bytes),
        "candidate_count": len(candidate_set.candidates),
        "score_bytes_sha256": _score_sha256(scores),
        "selected_index": selected_index,
        "selected_canonical_identity": _selected_identity(candidate_set, selected_index),
    }
    bank = {
        name: episode[name] for name in (
            "planned_episode_id", "task_id", "cell_id", "cell_ordinal",
            "replicate_ordinal", "candidate_ordinal", "environment_seed",
            "policy_rng_seed", "replacement", "acquisition_base_pose",
        )
    }
    bank.update({
        "captures": capture_records, "old_candidate_set": old, "candidate_set": new,
    })
    return {
        "bank": bank,
        "regression": _regression_record(
            episode, old, candidate_set, selected_index, audit),
        "blob_path": blob_path, "blob": candidate_set.canonical_bytes,
        "frame_count": len(payloads),
    }
def _run_ordered_jobs(function, jobs, deadline, workers):
    remaining = deadline - time.perf_counter()
    if remaining <= 0.0:
        raise RuntimeError("P79 wall-time budget exceeded")
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        results = tuple(executor.map(
            function, jobs, timeout=remaining, chunksize=1
        ))
    except TimeoutError as error:
        _abort_executor(executor)
        raise RuntimeError("P79 wall-time budget exceeded") from error
    except BaseException:
        _abort_executor(executor)
        raise
    executor.shutdown(wait=True)
    return results
def _abort_executor(executor) -> None:
    processes = tuple((getattr(executor, "_processes", None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + 1.0
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
def _full_bank_replay_bit_identical(first, second) -> bool:
    first_artifacts, second_artifacts = first["artifacts"], second["artifacts"]
    candidate_blobs = {name for name in first_artifacts if name.startswith("blobs/")}
    required = {"bank.json", "regression.json", *candidate_blobs}
    return (
        len(candidate_blobs) == EXPECTED_EPISODES
        and set(first_artifacts) == set(second_artifacts) == required
        and all(first_artifacts[name] == second_artifacts[name] for name in required)
    )
def _episode_inputs(
    input_path: Path,
    episode: Mapping[str, object],
) -> tuple[tuple[bytes, ...], list[dict[str, object]]]:
    captures = episode.get("captures")
    if not isinstance(captures, list):
        raise RuntimeError("P79 Episode captures are missing")
    ordered = sorted(captures, key=lambda value: int(value["capture_ordinal"]))
    if (
        [int(value["capture_ordinal"]) for value in ordered]
        != list(range(len(ordered)))
        or not ordered
        or sum(bool(value.get("final_input")) for value in ordered) != 1
        or ordered[-1].get("final_input") is not True
    ):
        raise RuntimeError("P79 capture order or final input differs")
    payloads = []
    records = []
    identities = {}
    for capture in ordered:
        policy = _read_bound_blob(input_path, capture["policy_input"])
        visible = _read_bound_blob(
            input_path, capture["candidate_visible_input"]
        )
        value = target_selection.deserialize_policy_input(policy)
        if (
            ownership.candidate_visible_bytes(value) != visible
            or [value.observation_timestamp_ns, value.sequence_id]
            != [
                capture["observation_timestamp_ns"],
                capture["sequence_id"],
            ]
        ):
            raise RuntimeError("P79 capture identity or visible bytes differ")
        observation_identity = (
            value.observation_timestamp_ns, value.sequence_id
        )
        visible_sha256 = _sha256(visible)
        if (
            observation_identity in identities
            and identities[observation_identity] != visible_sha256
        ):
            raise RuntimeError("P79 repeated observation identity changed bytes")
        identities.setdefault(observation_identity, visible_sha256)
        payloads.append(policy)
        records.append({
            "capture_ordinal": capture["capture_ordinal"],
            "final_input": capture["final_input"],
            "observation_identity": [
                capture["observation_timestamp_ns"],
                capture["sequence_id"],
            ],
            "policy_input": dict(capture["policy_input"]),
            "candidate_visible_input": dict(
                capture["candidate_visible_input"]
            ),
        })
    return tuple(payloads), records
def _regression_record(episode, old, new, selected_index, audit):
    return {
        "planned_episode_id": episode["planned_episode_id"],
        "task_id": episode["task_id"],
        "old": {
            "schema_version": old["schema_version"],
            "candidate_count": old["candidate_count"],
            "candidate_set_sha256": old["sha256"],
            "selected_index": old["selected_index"],
            "selected_canonical_identity":
                old["selected_canonical_identity"],
        },
        "new": {
            "schema_version": target_selection.CANDIDATE_SCHEMA,
            "candidate_count": len(new.candidates),
            "candidate_set_sha256": new.candidate_set_sha256,
            "selected_index": selected_index,
            "selected_canonical_identity": _selected_identity(
                new, selected_index
            ),
        },
        "audit": audit,
    }
def _old_candidate_identity(
    input_path: Path, value: Mapping[str, object]
) -> dict[str, object]:
    content = _read_bound_blob(input_path, value)
    document = json.loads(content)
    if (
        value.get("schema_version") != target_selection.LEGACY_CANDIDATE_SCHEMA
        or document.get("schema_version")
        != target_selection.LEGACY_CANDIDATE_SCHEMA
    ):
        raise RuntimeError("P79 old candidate schema differs")
    result = dict(value)
    result["selected_canonical_identity"] = (
        None
        if int(value["selected_index"]) < 0
        else _record_identity(
            document["candidates"][int(value["selected_index"])]
        )
    )
    return result
def _descriptive(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    tasks = {}
    for task_id in sorted({str(value["task_id"]) for value in records}):
        tasks[task_id] = _paired_counts(
            [value for value in records if value["task_id"] == task_id]
        )
    return {**_paired_counts(records), "by_task": tasks}
def _paired_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        "episode_count": len(records),
        "candidate_hash_changed_count": sum(
            value["old"]["candidate_set_sha256"]
            != value["new"]["candidate_set_sha256"]
            for value in records
        ),
        "candidate_count_increased_count": sum(
            value["new"]["candidate_count"] > value["old"]["candidate_count"]
            for value in records
        ),
        "candidate_count_decreased_count": sum(
            value["new"]["candidate_count"] < value["old"]["candidate_count"]
            for value in records
        ),
        "candidate_count_unchanged_count": sum(
            value["new"]["candidate_count"] == value["old"]["candidate_count"]
            for value in records
        ),
        "selected_identity_changed_count": sum(
            value["new"]["selected_canonical_identity"]
            != value["old"]["selected_canonical_identity"]
            for value in records
        ),
        "empty_to_nonempty_count": sum(
            value["old"]["candidate_count"] == 0
            and value["new"]["candidate_count"] > 0
            for value in records
        ),
        "nonempty_to_empty_count": sum(
            value["old"]["candidate_count"] > 0
            and value["new"]["candidate_count"] == 0
            for value in records
        ),
        "total_candidate_count_delta": sum(
            value["new"]["candidate_count"] - value["old"]["candidate_count"]
            for value in records
        ),
    }
def _report(first, *, replay_identical, history_unchanged, provenance,
            source_commit, command, elapsed):
    audits = first["audits"]
    frame_count = sum(int(value["frame_count"]) for value in audits)
    frame_checks = [frame for audit in audits for frame in audit["frames"]]
    fixture = ownership.overlap_fixture_audit()
    boundaries = ownership.boundary_fixture_audit()
    checks = {
        "legacy_ast_defect_confirmed":
            provenance["checks"]["legacy_ast_defect_confirmed"] is True,
        "legacy_overlap_fixture": fixture["checks"]["passed"] is True,
        "boundary_controls": boundaries["passed"] is True,
        "corrected_parent_mask_mutation_count_zero": all(
            frame["input_head_depth_valid_byte_identical"]
            and all(
                frame["traversals"][name]["parent_mutation_count"] == 0
                for name in ownership.TRAVERSALS
            )
            for frame in frame_checks
        ),
        "production_oracle_frame_equality": (
            frame_count == EXPECTED_CAPTURES
            and all(
                frame["production_matches_oracle_row_major"]
                for frame in frame_checks
            )
        ),
        "three_traversal_raw_multiset_equality": all(
            frame["oracle_traversal_multisets_equal"]
            for frame in frame_checks
        ),
        "episode_final_bytes_equality": (
            len(audits) == EXPECTED_EPISODES
            and all(
                audit["checks"]["traversal_final_candidate_bytes_equal"]
                for audit in audits
            )
        ),
        "production_final_bytes_match_oracle": all(
            audit["checks"]["production_final_equals_oracle_row_major"]
            for audit in audits
        ),
        "full_bank_replay_bit_identical": replay_identical,
        "twenty_four_episode_bijection": (
            first["bank"]["episode_count"] == EXPECTED_EPISODES
            and len({
                value["planned_episode_id"]
                for value in first["bank"]["episodes"]
            }) == EXPECTED_EPISODES
            and all(
                value["replacement"] is False
                for value in first["bank"]["episodes"]
            )
        ),
        "candidate_schema_v2": all(
            value["candidate_set"]["schema_version"]
            == target_selection.CANDIDATE_SCHEMA
            for value in first["bank"]["episodes"]
        ),
        "historical_artifacts_byte_identical": history_unchanged,
    }
    passed = all(checks.values())
    audit_decisions = {str(value["decision"]) for value in audits}
    if not fixture["checks"]["passed"] or "invalid" in audit_decisions:
        decision = "invalid"
    elif "rejected" in audit_decisions:
        decision = "rejected"
    elif "inconclusive_secondary_order_dependence" in audit_decisions:
        decision = "inconclusive_secondary_order_dependence"
    elif passed:
        decision = "accepted as deterministic candidate-generator correction"
    else:
        decision = "invalid"
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "decision": decision,
        "episode_count": len(audits),
        "capture_frame_count": frame_count,
        "full_bank_replay_bit_identical": replay_identical,
        "overlap_fixture": fixture,
        "boundary_controls": boundaries,
        "checks": {**checks, "passed": passed},
        "descriptive": first["regression"]["descriptive"],
        "wall_seconds": elapsed,
        **CLAIM_FLAGS,
    }
def _provenance(
    root: Path, input_path: Path, source_commit: str
) -> dict[str, object]:
    input_files = _directory_identities(root, input_path)
    manifest = _read_json(input_path / "manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("P79 input manifest artifacts are missing")
    artifact_checks = {
        name: _input_identity(input_path, input_path / name)
        for name in sorted(artifacts)
    }
    input_prefix = FORMAL_INPUT.as_posix()
    expected_input_paths = {
        f"{input_prefix}/{name}" for name in artifacts
    } | {f"{input_prefix}/manifest.json"}
    capture_ledger = _capture_ledger(input_path)
    current_source = (root / "src/hwr/eval/target_selection.py").read_text()
    legacy_source = _git_show(
        root, REVIEW_COMMIT, Path("src/hwr/eval/target_selection.py")
    ).decode()
    single_variable = ownership.audit_single_variable_source(
        current_source, legacy_source
    )
    historical_artifacts = _historical_artifact_identities(root)
    checks = {
        "workspace_clean":
            not _git(root, "status", "--porcelain", "--untracked-files=all"),
        "source_commit_matches_head":
            source_commit == _git(root, "rev-parse", "HEAD"),
        "old_source_commit_is_ancestor":
            _is_ancestor(root, OLD_SOURCE_COMMIT, "HEAD"),
        "frozen_document_commit_is_ancestor":
            _is_ancestor(root, FROZEN_DOCUMENT_COMMIT, "HEAD"),
        "frozen_document_blob_matches":
            _git(root, "rev-parse", f"HEAD:{FROZEN_DOCUMENT_PATH}")
            == _git(
                root,
                "rev-parse",
                f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}",
            ),
        "historical_document_trees_match": all(
            _git(root, "rev-parse", f"HEAD:docs/research-loop/{index:04d}")
            == _git(
                root,
                "rev-parse",
                f"{FROZEN_DOCUMENT_COMMIT}:docs/research-loop/{index:04d}",
            )
            for index in range(1, 14)
        ),
        "historical_artifact_trees_match": all(
            value["frozen_tree"] == value["head_tree"]
            == value["expected_tree"]
            for value in historical_artifacts.values()
        ),
        "implementation_scope_matches": set(
            _git_lines(
                root, "diff", "--name-only", FROZEN_DOCUMENT_COMMIT, "HEAD"
            )
        ) <= ALLOWED_CHANGES,
        "source_files_match_head": all(
            _git(root, "hash-object", "--", path.as_posix())
            == _git(root, "rev-parse", f"HEAD:{path.as_posix()}")
            for path in SOURCE_PATHS
        ),
        "frozen_input_hashes_match": all(
            _sha256((input_path / name).read_bytes()) == expected
            for name, expected in EXPECTED_INPUTS.items()
        ),
        "manifest_artifact_count": len(artifacts)
            == EXPECTED_MANIFEST_ARTIFACTS,
        "input_file_set_matches_manifest": (
            len(input_files) == EXPECTED_MANIFEST_ARTIFACTS + 1
            and {value["path"] for value in input_files}
            == expected_input_paths
        ),
        "manifest_artifacts_match": all(
            identity["sha256"] == artifacts[name]["sha256"]
            and identity["bytes"] == artifacts[name]["bytes"]
            for name, identity in artifact_checks.items()
        ),
        "capture_input_ledger_matches": (
            capture_ledger["entry_count"] == EXPECTED_INPUT_BLOBS
            and capture_ledger["capture_count"] == EXPECTED_CAPTURES
            and capture_ledger["sha256"]
            == EXPECTED_CAPTURE_LEDGER_SHA256
        ),
        "legacy_ast_defect_confirmed":
            ownership.audit_legacy_source(legacy_source)["passed"] is True,
        "single_variable_source_change": single_variable["passed"] is True,
    }
    return {
        "checks": {**checks, "passed": all(checks.values())},
        "input_files": input_files,
        "manifest_artifacts": artifact_checks,
        "capture_input_ledger": capture_ledger,
        "legacy_defect_source": {
            "commit": REVIEW_COMMIT,
            "path": "src/hwr/eval/target_selection.py",
            "audit": ownership.audit_legacy_source(legacy_source),
        },
        "single_variable_source_audit": single_variable,
        "historical_artifact_trees": historical_artifacts,
        "frozen_document": _identity(root, root / FROZEN_DOCUMENT_PATH),
        "sources": {
            path.as_posix(): _identity(root, root / path)
            for path in SOURCE_PATHS
        },
    }
def _require_provenance(value: Mapping[str, object]) -> None:
    checks = value["checks"]
    if checks["passed"] is not True:
        failed = sorted(
            name for name, passed in checks.items()
            if name != "passed" and not passed
        )
        raise RuntimeError(f"P79 provenance gate failed: {', '.join(failed)}")
def _manifest(
    source_commit, command, provenance, artifacts, report, elapsed
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": "complete",
        "source_commit": source_commit,
        "decision": report["decision"],
        "command": list(command),
        "input": {
            "path": FORMAL_INPUT.as_posix(),
            "files": provenance["input_files"],
        },
        "provenance": provenance,
        "runtime": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "wall_seconds": elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
            "disk_free_bytes": shutil.disk_usage(
                Path(__file__).resolve().parents[3]
            ).free,
        },
        "budgets": {
            "maximum_wall_seconds": MAX_WALL_SECONDS,
            "maximum_peak_rss_bytes": MAX_RSS_BYTES,
            "maximum_artifact_bytes": MAX_ARTIFACT_BYTES,
            "minimum_disk_free_bytes": MIN_DISK_FREE_BYTES,
        },
        "artifacts": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in sorted(artifacts.items())
        },
        **CLAIM_FLAGS,
    }
def _capture_ledger(input_path: Path) -> dict[str, object]:
    capsules = _read_json(input_path / "capsules.json")
    rows = []
    capture_count = 0
    for episode in capsules["episodes"]:
        for capture in episode["captures"]:
            capture_count += 1
            for name in ("policy_input", "candidate_visible_input"):
                value = capture[name]
                rows.append([
                    value["path"], value["sha256"], value["bytes"]
                ])
    return {
        "capture_count": capture_count,
        "entry_count": len(rows),
        "sha256": _sha256(json.dumps(
            rows, separators=(",", ":")).encode("ascii")),
    }
def _historical_artifact_identities(root: Path) -> dict[str, object]:
    return {
        path: {
            "expected_tree": expected,
            "frozen_tree": _git(root, "rev-parse",
                                f"{FROZEN_DOCUMENT_COMMIT}:{path}"),
            "head_tree": _git(root, "rev-parse", f"HEAD:{path}"),
        }
        for path, expected in HISTORICAL_ARTIFACT_TREES.items()
    }
def _directory_identities(root: Path, directory: Path) -> list[dict[str, object]]:
    del root
    return [
        _input_identity(directory, path)
        for path in sorted(value for value in directory.rglob("*") if value.is_file())
    ]
def _input_identity(directory: Path, path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": (FORMAL_INPUT / path.relative_to(directory)).as_posix(),
        "bytes": len(content), "sha256": _sha256(content),
    }
def _selected_identity(candidate_set, index: int) -> str | None:
    return (
        None
        if index < 0
        else _record_identity(list(candidate_set.candidates[index].canonical_record()))
    )
def _record_identity(record: object) -> str:
    return _sha256(json.dumps(record, separators=(",", ":")).encode("ascii"))
def _score_sha256(scores: Sequence[float]) -> str:
    return _sha256(np.ascontiguousarray(scores, dtype="<f8").tobytes())
def _require_frozen_paths(root: Path, input_path: Path, output: Path) -> None:
    if input_path != (root / FORMAL_INPUT).resolve():
        raise ValueError("P79 input path differs from frozen path")
    if output != (root / FORMAL_OUTPUT).resolve():
        raise ValueError("P79 output path differs from frozen path")


def _require_budget(
    elapsed: float, peak_rss: int, artifacts: Mapping[str, bytes]
) -> None:
    if elapsed > MAX_WALL_SECONDS:
        raise RuntimeError("P79 wall-time budget exceeded")
    if peak_rss > MAX_RSS_BYTES:
        raise RuntimeError("P79 RSS budget exceeded")
    if sum(len(value) for value in artifacts.values()) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("P79 artifact budget exceeded")


def _require_disk(output: Path) -> None:
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("P79 disk-free guard failed")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _git_lines(root: Path, *arguments: str) -> list[str]:
    output = _git(root, *arguments)
    return [] if not output else output.splitlines()


def _git_show(root: Path, commit: str, path: Path) -> bytes:
    return subprocess.run(
        ("git", "show", f"{commit}:{path.as_posix()}"), cwd=root,
        check=True, capture_output=True,
    ).stdout


def _command(arguments: argparse.Namespace) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        MODULE_NAME,
        "--input",
        arguments.input.as_posix(),
        "--output",
        arguments.output.as_posix(),
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024
def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["decision"] !=
               "accepted as deterministic candidate-generator correction")

if __name__ == "__main__":
    raise SystemExit(main())
