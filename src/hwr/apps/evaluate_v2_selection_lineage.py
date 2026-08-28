"""Run the frozen R0001-P83 blind v2 selection-lineage oracle."""
from __future__ import annotations
import argparse, ast, copy, difflib, hashlib, importlib.metadata, io, json, os, platform, resource, shutil, stat, subprocess, sys, time, tokenize
from pathlib import Path
from typing import Mapping, Sequence
MODULE_NAME = "hwr.apps.evaluate_v2_selection_lineage"; PROPOSAL_ID = "R0001-P83"
PLAN_SCHEMA = "hwr.p83-blind-plan/v1"; RECEIPT_SCHEMA = "hwr.p83-blind-selection-receipt/v1"; BOUNDARY_SCHEMA = "hwr.p83-boundary-controls/v1"
COMPARISON_SCHEMA = "hwr.p83-selection-lineage-comparison/v1"; REPORT_SCHEMA = "hwr.p83-selection-lineage-report/v1"; MANIFEST_SCHEMA = "hwr.p83-selection-lineage-artifacts/v1"
P50_SCHEMA = "hwr.p50-acquisition-capsule-index/v1"; P79_BANK_SCHEMA = "hwr.p79-candidate-bank/v1"; P79_MANIFEST_SCHEMA = "hwr.p79-candidate-mask-ownership-artifacts/v1"; CANDIDATE_SCHEMA = "hwr.p79-target-candidates/v2"
FORMAL_P50 = Path("runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001"); FORMAL_P79 = Path("runs/research-loop/0014/r0014-p79-candidate-bank-s20267901")
FORMAL_OUTPUT = Path("runs/research-loop/0016/r0016-p83-selection-lineage-s20268301"); WORKER_PATH = Path("scripts/evaluate_v2_selection_lineage_oracle.py"); APP_PATH = Path("src/hwr/apps/evaluate_v2_selection_lineage.py")
WORKER_SHA256 = "714e8ffb8eeb1c28dee83dc5c687694e23e50d36c414c2d604f4a9eca41387a6"
ISOLATED_BOOTSTRAP = ("import sys;sys.path[:0]=sys.argv[1:3];"
    "import selection_lineage_worker as worker;raise SystemExit(worker.main(sys.argv[3:]))")
FROZEN_COMMIT = "2e9eb1d0426749b1cd6239a5982ffd19e5c422fd"; FROZEN_DOCUMENT = Path("docs/research-loop/0016/03-experiment.md")
FROZEN_DOCUMENT_BLOB = "d4aa21cd525342da7df76f3e66d9c6ff51f20435"
P79_ARTIFACT_COMMIT = "93ea4e7afad8c52d83abd54f41a2d08d40a3cab4"
P79_TREE = "9a78c75e1f26b2c80399626042252b4e87404169"; P79_PRODUCER_COMMIT = "9eef9953f8a8558228a5e8870d7d2d8f7499ee1e"
P79_PRODUCER_BLOB = "3d3839605eb290f9f2e0b77ec7db22ac7de15a31"; SELECTOR_BLOB = "d7e588ba76ce18882255e3e22b1f86459ab235dd"
P50_FILES = dict(zip(("capsules.json", "plan.json", "report.json", "manifest.json"),
    ("223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf", "5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab",
     "b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0", "cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86")))
P79_FILES = dict(zip(("bank.json", "manifest.json"),
    ("888bf3ee42854a726bd86cc6c703c0c33a74f72d9029dc2cb6a9824ae9becd8e", "162e4bb6d06daf8e53e7192385274f059adde1a254bf81f2f4f8750ccce8d9c9")))
CAPTURE_LEDGER_SHA256 = "ff8c5cf53942e89e5ebc04dd8e9020313e5a120dc62ad6ca8764d93a6eda6145"
EXPECTED_INPUT_FILES, EXPECTED_EPISODES, EXPECTED_CAPTURES = 796, 24, 384
EXPECTED_CANDIDATES, EXPECTED_NONEMPTY, EXPECTED_EMPTY = 36, 22, 2
EXPECTED_SINGLETON, EXPECTED_MULTI = 14, 8
MAX_WALL_SECONDS, MAX_RSS_BYTES = 180.0, 2 * 1024**3
MAX_ARTIFACT_BYTES, MIN_DISK_FREE_BYTES = 16 * 1024**2, 20 * 1024**3
ALLOWED_PATHS = frozenset((
    "scripts/evaluate_v2_selection_lineage_oracle.py",
    "src/hwr/apps/evaluate_v2_selection_lineage.py",
    "tests/test_v2_selection_lineage_oracle.py",
    "tests/test_v2_selection_lineage_app.py"))
CONSUMER_PATHS = (
    Path("src/hwr/apps/evaluate_initial_candidate_association.py"),
    Path("src/hwr/apps/evaluate_phase_entry_geometry.py"))
CONSUMER_BLOBS = {
    "src/hwr/apps/evaluate_initial_candidate_association.py": "f0794de2f46582150b61c862a85c5cb75a594758",
    "src/hwr/apps/evaluate_phase_entry_geometry.py": "028910b9b0dd04182104c3ddd42c27978258de81"}
HISTORY_TREES = dict(zip(
    (f"docs/research-loop/{index:04d}" for index in range(1, 16)),
    ("416912b7dc1c19611bcfc4375028180014a1989b", "6fb603dbd52451fe1749157daf05aa482ca7222f", "f56011eda321ea803bc24051db001e632c1549fb",
     "611c420e539a53a8c7578cd66aa8bdfe46fe82b7", "0352d379d5754adb03e9158c0fa72393ab322d58", "ee3a6f5b25887f67f812750d2a75424df12823d4",
     "0a696caa153abc9c13403fbc9bd3c081ce71c327", "65e626cddbcb0ec9c2e17cca5184b7d40950e1c6", "316db8b9ad9739ef491778f641603dbca25e75c9",
     "8a193a24788027d715750c3cd89c2509e71fdbda", "85bb445726ecb8e35ff4d8e90606874e2ee36fe4", "db73bb9a6c6155d0366d7d92718aec614e044a5f",
     "2d885c0ad96af70b1f8808f0d8a6b700444b4a51", "0f25b24cf5a854af7f9712ed52610cb417395ad7", "e2495c4d70014a231ccbaf3ba5900f0ed57acc88")))
CLAIM_FLAGS = dict.fromkeys((
    "training_executed", "policy_inference_executed", "physical_acquisition_executed",
    "capability_evaluation_executed", "candidate_quality_claim_allowed",
    "selector_improvement_claim_allowed", "association_claim_allowed",
    "reachability_claim_allowed", "generalization_claim_allowed",
    "hardware_safety_claim_allowed", "task_success_claim_allowed",
    "artifact_self_contained_claim_allowed",
    "whole_program_completeness_claim_allowed"), False)
class LineageContractError(ValueError):
    """Raised when frozen selection-lineage evidence is invalid."""
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p50", type=Path, required=True); parser.add_argument("--p79", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); return parser
def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    p50 = _resolve(root, arguments.p50); p79 = _resolve(root, arguments.p79); output = _resolve(root, arguments.output)
    _require_frozen_paths(root, p50, p79, output)
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists(): raise FileExistsError(output)
    _require_disk(output)
    started = time.perf_counter(); parent_rss_start = _peak_rss_bytes()
    provenance = validate_frozen_provenance(root, p50, p79)
    capsules = _read_bound_json(p50 / "capsules.json", provenance["p50_top_files"]["capsules.json"])
    p79_manifest = _read_bound_json(p79 / "manifest.json", provenance["p79_manifest"])
    plan = build_blind_plan(capsules, p79_manifest)
    artifacts: dict[str, bytes] = {"blind-plan.json": _json_bytes(plan)}
    staging.mkdir(parents=True, exist_ok=False)
    try:
        blind_root = staging / "blind-input"; blind_root.mkdir()
        materialized = materialize_blind_inputs(p50, blind_root, plan)
        plan_path = blind_root / "blind-plan.json"
        _atomic_write(plan_path, artifacts["blind-plan.json"])
        _validate_blind_root(blind_root, plan)
        staged_worker, staged_source = materialize_worker(root, staging, provenance)
        isolation = validate_isolated_runtime(root, staged_worker)
        worker_runs = []
        for label in ("a", "b"):
            receipt_name = f"blind-receipt-{label}.json"
            receipt_path = staging / receipt_name; workdir = staging / f"worker-{label}"
            workdir.mkdir()
            worker_runs.append(run_blind_worker(root, blind_root, plan_path, receipt_path,
                workdir, deadline=started + MAX_WALL_SECONDS,
                forbidden_values=(str(p50), FORMAL_P50.as_posix(), str(p79),
                                  FORMAL_P79.as_posix()), isolation=isolation))
            artifacts[receipt_name] = _stable_read(receipt_path, None, root=staging)[0]
            shutil.rmtree(workdir)
        first = json.loads(artifacts["blind-receipt-a.json"]); second = json.loads(artifacts["blind-receipt-b.json"])
        validate_receipt(first, plan, provenance, worker_runs[0]); validate_receipt(second, plan, provenance, worker_runs[1]); _validate_blind_root(blind_root, plan); materialized["closed_world_before_and_after_workers"] = True
        if (_directory_identities(root, p50) != provenance["p50_input_files"]
                or _directory_stats(root, p79) != provenance["p79_pre_reveal_stats"]):
            raise LineageContractError("input_or_artifact_modified")
        bit_identical = artifacts["blind-receipt-a.json"] == artifacts["blind-receipt-b.json"]
        reveal_started = time.perf_counter()
        p79_files = _directory_identities(root, p79)
        if _public_identities(p79_files) != provenance["p79_expected_files"]:
            raise LineageContractError("p79_artifact_drift")
        provenance["p79_artifact_files"] = p79_files
        bank_identity = next(value for value in p79_files if value["path"].endswith("/bank.json"))
        bank = _read_bound_json(p79 / "bank.json", bank_identity)
        comparison = compare_reveal(p79, bank, first)
        boundary = run_boundary_controls(root, blind_root, plan, first, bank, p79,
            comparison, provenance, worker_runs, isolation, started + MAX_WALL_SECONDS)
        provenance["source_end"] = validate_source_stability(
            root, staged_worker, provenance, staged_source)
        comparer_elapsed = time.perf_counter() - reveal_started
        artifacts["boundary-controls.json"] = _json_bytes(boundary)
        artifacts["comparison.json"] = _json_bytes(comparison)
        report, recorded_wall, recorded_peak = _seal_staging(
            root, arguments, staging, artifacts, comparison, boundary, first,
            provenance, worker_runs, bit_identical, started, parent_rss_start,
            comparer_elapsed, materialized)
        os.replace(staging, output); _fsync_directory(output.parent)
        try:
            validate_repo_source_stability(root, provenance)
            finalized = _require_finalized_budget(
                started, recorded_peak, output, recorded_wall, worker_runs)
        except BaseException:
            shutil.rmtree(output, ignore_errors=True)
            _fsync_directory(output.parent)
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output": str(output), "decision": report["decision"], "candidate_exact_match_count": comparison["candidate_exact_match_count"], "score_hash_exact_match_count": comparison["score_hash_exact_match_count"], "selected_index_exact_match_count": comparison["selected_index_exact_match_count"], "finalized_wall_seconds": finalized}
def build_blind_plan(capsules: Mapping[str, object],
                     p79_manifest: Mapping[str, object]) -> dict[str, object]:
    if capsules.get("schema_version") != P50_SCHEMA: raise LineageContractError("p50_schema")
    episodes = capsules.get("episodes")
    if (not isinstance(episodes, list) or len(episodes) != EXPECTED_EPISODES
            or capsules.get("capsule_count") != EXPECTED_EPISODES): raise LineageContractError("episode_count")
    committed = _manifest_input_map(p79_manifest); records = []; seen: set[str] = set(); capture_count = 0
    for episode_ordinal, episode in enumerate(episodes):
        identity = str(episode.get("planned_episode_id", ""))
        if not identity or identity in seen: raise LineageContractError("episode_duplicate")
        seen.add(identity)
        if episode.get("schema_version") != "hwr.p50-acquisition-capsule/v1": raise LineageContractError("episode_schema")
        captures = _sanitized_captures(episode, committed)
        capture_count += len(captures)
        records.append({"episode_ordinal": episode_ordinal, "planned_episode_id": identity,
            "task_id": episode["task_id"], "cell_id": episode["cell_id"],
            "replicate_ordinal": episode["replicate_ordinal"],
            "acquisition_base_pose": episode["acquisition_base_pose"], "captures": captures})
    if capture_count != EXPECTED_CAPTURES: raise LineageContractError("capture_count")
    return {"schema_version": PLAN_SCHEMA, "proposal_id": PROPOSAL_ID,
            "episode_count": len(records), "capture_count": capture_count,
            "input_file_count": 2 * capture_count, "episodes": records}
def _sanitized_captures(episode: Mapping[str, object],
        committed: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    captures = episode.get("captures")
    if not isinstance(captures, list) or not captures: raise LineageContractError("capture_missing")
    records = []
    for ordinal, capture in enumerate(captures):
        if (capture.get("schema_version") != "hwr.p50-acquisition-capture/v1"
                or capture.get("capture_ordinal") != ordinal): raise LineageContractError("capture_order")
        record = {"capture_ordinal": ordinal, "final_input": capture.get("final_input"),
            "observation_identity": [capture.get("observation_timestamp_ns"),
                                     capture.get("sequence_id")]}
        for name in ("policy_input", "candidate_visible_input"):
            record[name] = _sanitize_descriptor(capture.get(name), committed)
        records.append(record)
    final_flags = [value["final_input"] for value in records]
    if (any(type(value) is not bool for value in final_flags)
            or sum(final_flags) != 1 or final_flags[-1] is not True): raise LineageContractError("final_input")
    return records
def _sanitize_descriptor(value: object,
        committed: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if not isinstance(value, Mapping): raise LineageContractError("input_descriptor")
    if set(value) != {"path", "sha256", "bytes"}: raise LineageContractError("input_descriptor")
    relative = _relative_path(value["path"])
    expected = committed.get(relative.as_posix())
    sanitized = {"path": relative.as_posix(), "bytes": value["bytes"],
                 "sha256": value["sha256"]}
    if expected != sanitized: raise LineageContractError("p79_input_commitment")
    return sanitized
def materialize_blind_inputs(source: Path, destination: Path,
                             plan: Mapping[str, object]) -> dict[str, object]:
    descriptors = [capture[name] for episode in plan["episodes"]
                   for capture in episode["captures"]
                   for name in ("policy_input", "candidate_visible_input")]
    if len(descriptors) != 2 * EXPECTED_CAPTURES: raise LineageContractError("blind_input_count")
    ledger = []
    for value in descriptors:
        relative = _relative_path(value["path"])
        content, identity = _stable_read(source / relative, value, root=source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, content); target.chmod(0o400)
        copied, copied_identity = _stable_read(target, value, root=destination)
        if copied != content: raise LineageContractError("blind_input_copy")
        ledger.append({"source": identity, "blind": copied_identity})
    files = [path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()]
    if sorted(files) != sorted(value["path"] for value in descriptors):
        raise LineageContractError("blind_input_extra_or_missing")
    for directory in sorted({path.parent for path in destination.rglob("*") if path.is_file()}, reverse=True): _fsync_directory(directory)
    _fsync_directory(destination)
    return {"file_count": len(files), "entries": ledger}
def materialize_worker(root: Path, staging: Path,
        provenance: Mapping[str, object]) -> tuple[Path, dict[str, object]]:
    expected = provenance["worker_source"]; source = root / WORKER_PATH
    content, source_fd = _stable_read(source, expected, root=root)
    if _sha256(content) != WORKER_SHA256: raise LineageContractError("worker_blob")
    target = staging / "selection_lineage_worker.py"; _atomic_write(target, content); target.chmod(0o400)
    copied, staged_fd = _stable_read(target, expected, root=staging)
    if copied != content: raise LineageContractError("worker_stage_copy")
    return target, {"source_fd_identity": source_fd, "staged_fd_identity": staged_fd, "bytes": len(content), "sha256": _sha256(content)}
def validate_source_stability(root: Path, staged_worker: Path,
        provenance: Mapping[str, object],
        staged_source: Mapping[str, object]) -> dict[str, object]:
    worker = _file_identity(root, root / WORKER_PATH); app = _file_identity(root, root / APP_PATH); staged = _file_identity(staged_worker.parent, staged_worker)
    checks = {"worker_matches_start": _public_identity(worker) == _public_identity(provenance["worker_source"]),
        "worker_blob_bound": worker["sha256"] == WORKER_SHA256,
        "app_matches_start": _public_identity(app) == _public_identity(provenance["app_source"]),
        "staged_worker_matches": staged["sha256"] == staged_source["sha256"]
        == WORKER_SHA256 and staged["bytes"] == staged_source["bytes"]}
    checks["passed"] = all(checks.values())
    if not checks["passed"]: raise LineageContractError("source_changed_during_run")
    return {"worker": worker, "app": app, "staged_worker": staged, "checks": checks}
def validate_repo_source_stability(root: Path,
                                   provenance: Mapping[str, object]) -> None:
    for name, path in (("worker", WORKER_PATH), ("app", APP_PATH)):
        current = _file_identity(root, root / path)
        if _public_identity(current) != _public_identity(provenance["source_end"][name]):
            raise LineageContractError("source_changed_after_seal")
def validate_isolated_runtime(root: Path, worker: Path | None = None) -> dict[str, object]:
    worker = root / WORKER_PATH if worker is None else worker
    numpy_site = Path(importlib.metadata.distribution("numpy").locate_file("")).resolve()
    probe = ("import importlib.util,sys;sys.path.insert(0,sys.argv[1]);"
             "import numpy;print(numpy.__version__);print(importlib.util.find_spec('hwr'))")
    result = subprocess.run((sys.executable, "-I", "-S", "-c", probe, str(numpy_site)), cwd=worker.parent, capture_output=True)
    lines = result.stdout.decode().splitlines()
    if result.returncode or len(lines) != 2 or lines[1] != "None":
        raise LineageContractError("isolated_runtime")
    return {"numpy_site": str(numpy_site), "numpy_version": lines[0], "hwr_find_spec": None, "worker_path": str(worker), "worker_sha256": _file_identity(worker.parent, worker)["sha256"]}
def run_blind_worker(root: Path, blind_root: Path, plan_path: Path,
        receipt_path: Path, workdir: Path, *, deadline: float,
        forbidden_values: Sequence[str],
        isolation: Mapping[str, object]) -> dict[str, object]:
    del root
    worker = Path(str(isolation["worker_path"])).absolute()
    plan_identity = _file_identity(blind_root, plan_path)
    plan = _read_bound_json(plan_path, plan_identity)
    audit_paths = ["blind-plan.json", *(str(capture[kind]["path"])
        for episode in plan["episodes"] for capture in episode["captures"]
        for kind in ("policy_input", "candidate_visible_input"))]
    audit_sha256 = _sha256(json.dumps(sorted(audit_paths), separators=(",", ":")).encode("utf-8"))
    worker_before = _file_identity(worker.parent, worker)
    if (worker_before["sha256"] != isolation["worker_sha256"] or
            worker_before["sha256"] != WORKER_SHA256): raise LineageContractError("worker_stage_hash")
    worker_command = (sys.executable, "-I", "-S", "-c", ISOLATED_BOOTSTRAP,
        str(worker.parent), str(isolation["numpy_site"]),
        "--plan", str(plan_path), "--plan-bytes", str(plan_identity["bytes"]),
        "--plan-sha256", str(plan_identity["sha256"]),
        "--input-root", str(blind_root), "--output", str(receipt_path),
        "--worker-sha256", str(isolation["worker_sha256"]))
    time_output = workdir / "resource.txt"
    command = ("/usr/bin/time", "-l", "-o", str(time_output), *worker_command)
    environment = _blind_environment(forbidden_values); environment["PWD"] = str(workdir); environment["HWR_P83_ISOLATED"] = "1"
    audit_worker_invocation(worker_command, environment, workdir, forbidden_values)
    remaining = deadline - time.perf_counter()
    if remaining <= 0.0: raise RuntimeError("P83 wall-time budget exceeded before worker")
    child_before = _children_peak_rss_bytes(); child_started = time.perf_counter()
    result = subprocess.run(command, cwd=workdir, env=environment, capture_output=True, timeout=remaining, check=False)
    external_wall = time.perf_counter() - child_started; child_after = _children_peak_rss_bytes()
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")
        raise RuntimeError("blind worker failed: " + detail)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    if not receipt_path.is_file() or temporary.exists(): raise LineageContractError("receipt_atomic_complete")
    try:
        worker_summary = json.loads(result.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LineageContractError("worker_summary") from error
    receipt_bytes, _ = _stable_read(receipt_path, None, root=receipt_path.parent)
    if worker_summary.get("output_sha256") != _sha256(receipt_bytes): raise LineageContractError("worker_receipt_hash")
    worker_after = _file_identity(worker.parent, worker)
    if _public_identity(worker_after) != _public_identity(worker_before): raise LineageContractError("worker_changed_during_run")
    time_bytes, _ = _stable_read(time_output, None, root=workdir)
    external_peak = _parse_time_peak_rss(time_bytes.decode("utf-8"))
    return {"returncode": result.returncode, "stdout_sha256": _sha256(result.stdout),
        "stderr_sha256": _sha256(result.stderr), "argv": [str(value) for value in worker_command],
        "cwd": str(workdir), "worker_path": str(worker), "pythonpath": environment.get("PYTHONPATH", ""),
        "isolated_flags": ["-I", "-S"], "wall_seconds": worker_summary["wall_seconds"],
        "external_wall_seconds": external_wall, "outer_observed_child_peak_rss_bytes":
        max(child_before, child_after, external_peak,
            int(worker_summary["process_tree_peak_rss_upper_bound_bytes"])),
        "external_time_peak_rss_bytes": external_peak, "expected_read_audit_sha256": audit_sha256,
        "worker_source_before": worker_before, "worker_source_after": worker_after,
        "process_tree_peak_rss_upper_bound_bytes": worker_summary["process_tree_peak_rss_upper_bound_bytes"],
        "receipt_present_after_exit": True, "receipt_temporary_absent_after_exit": True}
def audit_worker_invocation(command: Sequence[str], environment: Mapping[str, str],
                            workdir: Path, forbidden_values: Sequence[str]) -> None:
    if (not workdir.is_dir() or environment.get("PYTHONPATH") != "" or
            environment.get("HWR_P83_ISOLATED") != "1" or
            list(command[1:3]) != ["-I", "-S"]):
        raise LineageContractError("worker_isolation")
    haystacks = [*map(str, command), *environment.keys(), *environment.values()]
    forbidden = [value for value in forbidden_values if value]
    if any(value in item for value in forbidden for item in haystacks):
        raise LineageContractError("worker_p79_exposure")
    forbidden_names = ("expected_score", "selected_index", "selected_identity", "candidate_set_sha256")
    if any(marker in item.lower().replace("-", "_")
           for marker in forbidden_names for item in haystacks):
        raise LineageContractError("worker_metadata_exposure")
def validate_receipt(receipt: Mapping[str, object], plan: Mapping[str, object],
        provenance: Mapping[str, object],
        worker_run: Mapping[str, object]) -> None:
    required = {"schema_version", "proposal_id", "status", "plan_sha256",
        "worker_source_sha256", "candidate_schema_version", "score_weights",
        "tie_break_id", "episode_count", "capture_count", "candidate_count",
        "input_file_match_count", "execution", "read_ledger", "read_audit",
        "episodes", "mutation_evidence"}
    if set(receipt) != required: raise LineageContractError("receipt_fields")
    if (receipt["schema_version"] != RECEIPT_SCHEMA or receipt["proposal_id"] != PROPOSAL_ID or
            receipt["status"] != "complete" or receipt["candidate_schema_version"] != CANDIDATE_SCHEMA):
        raise LineageContractError("receipt_schema")
    if receipt["plan_sha256"] != _sha256(_json_bytes(plan)): raise LineageContractError("receipt_plan")
    if receipt["worker_source_sha256"] != provenance["worker_source"]["sha256"]: raise LineageContractError("receipt_source")
    if (receipt["episode_count"] != EXPECTED_EPISODES or receipt["capture_count"] != EXPECTED_CAPTURES or receipt["candidate_count"] != EXPECTED_CANDIDATES or receipt["input_file_match_count"] != 2 * EXPECTED_CAPTURES):
        raise LineageContractError("receipt_counts")
    execution = receipt["execution"]
    if (execution.get("job_count") != EXPECTED_EPISODES or execution.get("worker_count") != min(EXPECTED_EPISODES, os.cpu_count() or 1) or execution.get("parallel_path_used") is not (EXPECTED_EPISODES > 1) or (EXPECTED_EPISODES > 1 and not 1 < execution.get("worker_process_count", 0) <= execution["worker_count"])):
        raise LineageContractError("worker_execution")
    if receipt["score_weights"] != [0.30, 0.25, 0.20, 0.15, 0.10] or receipt["tie_break_id"] != "maximum-score-then-lowest-index/v1":
        raise LineageContractError("receipt_scoring_contract")
    _validate_read_ledger(receipt["read_ledger"], plan)
    audit = receipt["read_audit"]
    if (audit.get("trust_role") != "auxiliary" or audit.get("audited_open_count") != 1 + 2 * EXPECTED_CAPTURES or audit.get("expected_open_count") != 1 + 2 * EXPECTED_CAPTURES or audit.get("path_sequence_sha256") != worker_run["expected_read_audit_sha256"]):
        raise LineageContractError("read_audit_count")
    plan_identity = audit.get("plan_fd_identity", {})
    if (plan_identity.get("bytes") != len(_json_bytes(plan)) or plan_identity.get("sha256") != _sha256(_json_bytes(plan)) or plan_identity.get("fd_identity", {}).get("size") != plan_identity.get("bytes")):
        raise LineageContractError("read_audit_plan")
    if worker_run["returncode"] != 0 or worker_run["receipt_present_after_exit"] is not True or worker_run["receipt_temporary_absent_after_exit"] is not True:
        raise LineageContractError("receipt_atomic_complete")
    if any(value["sha256"] != receipt["worker_source_sha256"] for value in (worker_run["worker_source_before"], worker_run["worker_source_after"])): raise LineageContractError("receipt_source_parent_observation")
def _validate_read_ledger(ledger: object, plan: Mapping[str, object]) -> None:
    if not isinstance(ledger, list): raise LineageContractError("read_ledger")
    expected = [{"kind": kind, **capture[kind]}
        for episode in plan["episodes"]
        for capture in episode["captures"]
        for kind in ("policy_input", "candidate_visible_input")]
    projected = [{key: value[key] for key in ("kind", "path", "bytes", "sha256")}
                 for value in ledger]
    if projected != expected: raise LineageContractError("read_ledger")
    if any(set(value["fd_identity"]) != {"device", "inode", "size"}
           or value["fd_identity"]["size"] != value["bytes"] for value in ledger):
        raise LineageContractError("read_ledger_fd_identity")
    forbidden = ("candidate-set", "bank.json", "manifest.json", "score",
                 "selected", "segmentation", "contact", "force")
    if any(marker in str(value["path"]).lower() for value in ledger for marker in forbidden):
        raise LineageContractError("read_ledger_forbidden")
def compare_reveal(p79: Path, bank: Mapping[str, object],
                   receipt: Mapping[str, object]) -> dict[str, object]:
    if (bank.get("schema_version") != P79_BANK_SCHEMA
            or bank.get("proposal_id") != "R0001-P79-E1"):
        raise LineageContractError("p79_schema")
    bank_episodes = bank.get("episodes"); receipt_episodes = receipt.get("episodes")
    if not isinstance(bank_episodes, list) or not isinstance(receipt_episodes, list):
        raise LineageContractError("reveal_episodes")
    expected_ids = [str(value["planned_episode_id"]) for value in bank_episodes]
    actual_ids = [str(value["planned_episode_id"]) for value in receipt_episodes]
    if (len(expected_ids) != EXPECTED_EPISODES
            or len(expected_ids) != len(set(expected_ids))
            or actual_ids != expected_ids):
        raise LineageContractError("reveal_episode_identity")
    records = []
    for expected, actual in zip(bank_episodes, receipt_episodes):
        committed = expected["candidate_set"]
        candidate_bytes = _read_p79_candidate(p79, committed)
        candidate_document = json.loads(candidate_bytes)
        if candidate_document.get("schema_version") != CANDIDATE_SCHEMA:
            raise LineageContractError("candidate_schema")
        candidate_match = (actual["candidate_canonical_ascii"].encode("ascii") == candidate_bytes and actual["candidate_sha256"] == committed["sha256"] and actual["candidate_bytes"] == committed["bytes"] and actual["candidate_count"] == committed["candidate_count"])
        score_match = actual["score_bytes_sha256"] == committed["score_bytes_sha256"]; count = committed["candidate_count"]
        index_match = actual["selected_index"] == committed["selected_index"]; identity_match = actual["selected_canonical_identity"] == committed["selected_canonical_identity"]
        kind = "empty" if count == 0 else "singleton" if count == 1 else "multi"
        records.append({"planned_episode_id": expected["planned_episode_id"], "task_id": expected["task_id"], "kind": kind, "candidate_count": count, "candidate_exact_match": candidate_match, "score_hash_exact_match": score_match, "selected_index_exact_match": index_match, "selected_identity_exact_match": identity_match, "canonical_only_score_hash_matches": actual["canonical_only_score_bytes_sha256"] == committed["score_bytes_sha256"], "top_two_score_margin": actual["top_two_score_margin"]})
    counts = _comparison_counts(records)
    tasks = {}
    for value in records:
        row = tasks.setdefault(value["task_id"], {"episodes": 0, "candidates": 0})
        row["episodes"] += 1; row["candidates"] += int(value["candidate_count"])
    return {"schema_version": COMPARISON_SCHEMA, "proposal_id": PROPOSAL_ID, "sample_unit": "Episode", **counts, "task_breakdown": tasks, "episodes": records}
def _comparison_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {"episode_count": len(records), "candidate_exact_match_count": sum(bool(value["candidate_exact_match"]) for value in records), "score_hash_exact_match_count": sum(bool(value["score_hash_exact_match"]) for value in records), "selected_index_exact_match_count": sum(bool(value["selected_index_exact_match"]) for value in records), "selected_identity_exact_match_count": sum(value["kind"] != "empty" and bool(value["selected_identity_exact_match"]) for value in records), "empty_selection_exact_match_count": sum(value["kind"] == "empty" and bool(value["selected_index_exact_match"]) and bool(value["selected_identity_exact_match"]) for value in records), "candidate_count": sum(int(value["candidate_count"]) for value in records), "empty_episode_count": sum(value["kind"] == "empty" for value in records), "singleton_episode_count": sum(value["kind"] == "singleton" for value in records), "multi_episode_count": sum(value["kind"] == "multi" for value in records), "canonical_only_score_hash_mismatch_count": sum(not bool(value["canonical_only_score_hash_matches"]) for value in records)}
def run_boundary_controls(root: Path, blind_root: Path, plan: Mapping[str, object], receipt: Mapping[str, object], bank: Mapping[str, object], p79: Path, comparison: Mapping[str, object], provenance: Mapping[str, object], worker_runs: Sequence[Mapping[str, object]], isolation: Mapping[str, object], deadline: float) -> dict[str, object]:
    started = time.perf_counter()
    cases = (('schema_interchange', 'plan_schema'), ('episode_duplicate', 'episode_duplicate'), ('episode_missing', 'episode_count'), ('episode_order', 'episode_order'), ('capture_ordinal', 'capture_order'), ('final_missing', 'final_input'), ('final_multiple', 'final_input'), ('observation_identity', 'observation_identity'), ('path_absolute', 'path_escape'), ('path_traversal', 'path_escape'), ('input_size', 'policy_input_size_or_hash'), ('input_hash', 'policy_input_size_or_hash'), ('input_missing', 'policy_input_missing'), ('symlink', 'policy_input_symlink'), ('input_duplicate', 'input_path_duplicate'))
    controls = [{'name': name, 'expected_category': category, 'observed_category': (observed := _run_plan_mutation(root, blind_root, plan, name, isolation, deadline)), 'passed': observed == category} for name, category in cases]
    evidence = receipt['mutation_evidence']
    algorithm = {'candidate_order': evidence['candidate_order_changed_count'] == evidence['candidate_order_denominator'] == EXPECTED_MULTI, **{name: evidence[key] == evidence['nonempty_denominator'] == EXPECTED_NONEMPTY for name, key in (('final_base_pose', 'final_base_score_changed_count'), ('score_weights', 'weight_score_changed_count'), ('tie_break', 'tie_break_flip_count'), ('canonical_only', 'canonical_only_score_mismatch_count'))}}
    mutated_bank = copy.deepcopy(bank)
    selected = next((value for value in mutated_bank['episodes'] if value['candidate_set']['candidate_count'] > 0))
    original_index = selected['candidate_set']['selected_index']
    selected['candidate_set']['selected_index'] = -1 if original_index != -1 else 0
    selected['candidate_set']['selected_canonical_identity'] = None
    mutated = compare_reveal(p79, mutated_bank, receipt)
    selected_control = mutated['selected_index_exact_match_count'] < comparison['selected_index_exact_match_count'] and mutated['selected_identity_exact_match_count'] < comparison['selected_identity_exact_match_count']
    observed_reads = [value['path'] for value in receipt['read_ledger']]
    isolation_checks = {'worker_isolated': all((value['isolated_flags'] == ['-I', '-S'] and value['pythonpath'] == '' for value in worker_runs)), 'worker_input_is_blind_root': all((value['argv'][value['argv'].index('--input-root') + 1] == str(blind_root) for value in worker_runs)), 'worker_executes_staged_copy': all(value['worker_path'] == isolation['worker_path'] for value in worker_runs), 'worker_blob_bound': isolation['worker_sha256'] == provenance['worker_source']['sha256'] == WORKER_SHA256, 'hwr_not_importable': isolation['hwr_find_spec'] is None, 'read_audit_auxiliary_complete': receipt['read_audit']['audited_open_count'] == receipt['read_audit']['expected_open_count'], 'read_paths_are_plan_only': observed_reads == [capture[kind]['path'] for episode in plan['episodes'] for capture in episode['captures'] for kind in ('policy_input', 'candidate_visible_input')], 'source_independent': provenance['worker_source_audit']['passed'] and provenance['worker_similarity_audit']['passed'], 'selected_metadata_mutation': selected_control}
    passed = all((value['passed'] for value in controls)) and all(algorithm.values()) and all(isolation_checks.values())
    elapsed = time.perf_counter() - started
    if time.perf_counter() > deadline: raise RuntimeError("P83 wall-time budget exceeded during controls")
    return {'schema_version': BOUNDARY_SCHEMA, 'proposal_id': PROPOSAL_ID, 'plan_mutations': controls, 'algorithm_mutations': algorithm, 'isolation_controls': isolation_checks, 'runtime': {'wall_seconds': elapsed, 'blind_blob_copy_count': 0, 'maximum_mutation_episode_count': min(3, len(plan['episodes']) + 1)}, 'control_count': len(controls) + len(algorithm) + len(isolation_checks), 'pass_count': sum((value['passed'] for value in controls)) + sum(algorithm.values()) + sum(isolation_checks.values()), 'passed': passed}
def _run_plan_mutation(
    root: Path, blind_root: Path, plan: Mapping[str, object], name: str,
    isolation: Mapping[str, object], deadline: float,
) -> str | None:
    case_root = blind_root.parent / f"mutation-{name}"; case_root.mkdir()
    mutated = copy.deepcopy(plan)
    episodes = mutated["episodes"][:2]
    for episode_ordinal, episode in enumerate(episodes):
        captures = [episode["captures"][0], episode["captures"][-1]]
        captures[0]["capture_ordinal"] = 0; captures[0]["final_input"] = False
        captures[1]["capture_ordinal"] = 1; captures[1]["final_input"] = True
        episode["captures"] = captures; episode["episode_ordinal"] = 0
        episode["episode_ordinal"] = episode_ordinal
    mutated["episodes"] = episodes; mutated["episode_count"] = len(episodes)
    mutated["capture_count"] = 2 * len(episodes); mutated["input_file_count"] = 4 * len(episodes)
    descriptor = mutated["episodes"][0]["captures"][0]["policy_input"]
    if name == "schema_interchange": mutated["schema_version"] = P79_BANK_SCHEMA
    elif name == "episode_duplicate":
        duplicate = copy.deepcopy(mutated["episodes"][0])
        duplicate["episode_ordinal"] = len(mutated["episodes"])
        mutated["episodes"].append(duplicate)
        mutated["episode_count"] += 1
        mutated["capture_count"] += len(duplicate["captures"])
        mutated["input_file_count"] += 2 * len(duplicate["captures"])
    elif name == "episode_missing": mutated["episodes"].pop()
    elif name == "episode_order": mutated["episodes"][0]["episode_ordinal"] = 1
    elif name == "capture_ordinal": mutated["episodes"][0]["captures"][1]["capture_ordinal"] = 0
    elif name == "final_missing": mutated["episodes"][0]["captures"][-1]["final_input"] = False
    elif name == "final_multiple": mutated["episodes"][0]["captures"][0]["final_input"] = True
    elif name == "observation_identity": mutated["episodes"][0]["captures"][0]["observation_identity"][0] += 1
    elif name == "path_absolute": descriptor["path"] = str((blind_root / descriptor["path"]).resolve())
    elif name == "path_traversal": descriptor["path"] = "../escape.bin"
    elif name == "input_size": descriptor["bytes"] += 1
    elif name == "input_hash": descriptor["sha256"] = "0" * 64
    elif name == "input_missing": descriptor["path"] = "missing.bin"
    elif name == "input_duplicate":
        descriptor.update(mutated["episodes"][0]["captures"][1]["policy_input"])
    elif name == "symlink":
        link = blind_root / "mutation-symlink.bin"
        link.symlink_to(blind_root / descriptor["path"])
        descriptor["path"] = link.relative_to(blind_root).as_posix()
    else: raise AssertionError(name)
    plan_bytes = _json_bytes(mutated); plan_path = case_root / "plan.json"
    _atomic_write(plan_path, plan_bytes)
    output = case_root / "receipt.json"; workdir = case_root / "cwd"; workdir.mkdir()
    try:
        run_blind_worker(root, blind_root, plan_path, output, workdir,
                         deadline=deadline, forbidden_values=(),
                         isolation=isolation)
    except (LineageContractError, RuntimeError) as error:
        detail = str(error).split("blind worker failed: ", 1)[-1]
        return detail.strip().splitlines()[-1].rsplit(":", 1)[-1].strip()
    finally:
        if name == "symlink": (blind_root / "mutation-symlink.bin").unlink(missing_ok=True)
        shutil.rmtree(case_root, ignore_errors=True)
    return None
def build_report(comparison: Mapping[str, object], boundary: Mapping[str, object],
        *, bit_identical: bool, provenance: Mapping[str, object],
        worker_runs: Sequence[Mapping[str, object]], elapsed: float,
        peak_rss: int, parent_rss_start: int, comparer_elapsed: float,
        materialized: Mapping[str, object],
        receipt: Mapping[str, object]) -> dict[str, object]:
    lineage_exact = comparison["candidate_exact_match_count"] == EXPECTED_EPISODES and comparison["selected_index_exact_match_count"] == EXPECTED_EPISODES and comparison["selected_identity_exact_match_count"] == EXPECTED_NONEMPTY and comparison["empty_selection_exact_match_count"] == EXPECTED_EMPTY
    score_exact = comparison["score_hash_exact_match_count"] == EXPECTED_EPISODES
    guards = provenance["checks"]["passed"] and boundary["passed"] and bit_identical and comparison["episode_count"] == EXPECTED_EPISODES and comparison["candidate_count"] == EXPECTED_CANDIDATES and comparison["empty_episode_count"] == EXPECTED_EMPTY and comparison["singleton_episode_count"] == EXPECTED_SINGLETON and comparison["multi_episode_count"] == EXPECTED_MULTI
    decision = "invalid" if not guards else "rejected" if not lineage_exact else "inconclusive_score_bytes" if not score_exact else "accepted as consumer-local v2 selection-lineage evidence"
    multi_margins = [value["top_two_score_margin"] for value in comparison["episodes"] if value["kind"] == "multi"]
    metric_names = ("episode_count", "candidate_count", "candidate_exact_match_count",
        "score_hash_exact_match_count", "selected_index_exact_match_count",
        "selected_identity_exact_match_count", "empty_selection_exact_match_count",
        "empty_episode_count", "singleton_episode_count", "multi_episode_count",
        "canonical_only_score_hash_mismatch_count")
    p79_reads = sum(any(marker in value["path"].lower() for marker in ("p79", "candidate-set", "score", "selected")) for value in receipt["read_ledger"])
    truth_reads = sum(any(marker in value["path"].lower() for marker in ("segmentation", "geom", "body", "contact", "force")) for value in receipt["read_ledger"])
    cohort_exact = comparison["episode_count"] == EXPECTED_EPISODES and comparison["candidate_count"] == EXPECTED_CANDIDATES and comparison["empty_episode_count"] == EXPECTED_EMPTY and comparison["singleton_episode_count"] == EXPECTED_SINGLETON and comparison["multi_episode_count"] == EXPECTED_MULTI
    runtime = {"wall_seconds": elapsed, "wall_seconds_is_upper_bound": True, "process_tree_peak_rss_bytes": peak_rss, "process_tree_peak_rss_is_upper_bound": True, "parent_peak_rss_at_start_bytes": parent_rss_start, "worker_wall_seconds": [value["wall_seconds"] for value in worker_runs], "comparer_wall_seconds": comparer_elapsed, "worker_runs": list(worker_runs)}
    checks = {"input_files_exact": provenance["input_file_match_count"] == EXPECTED_INPUT_FILES, "cohort_counts_exact": cohort_exact, "blind_rebuild_bit_identical": bit_identical, "boundary_controls_passed": boundary["passed"], "provenance_passed": provenance["checks"]["passed"]}
    return {"schema_version": REPORT_SCHEMA, "proposal_id": PROPOSAL_ID, "decision": decision, "sample_unit": "Episode", "metrics": {key: comparison[key] for key in metric_names}, "task_breakdown": comparison["task_breakdown"], "top_two_score_margins": multi_margins, "blind_rebuild_bit_identical": bit_identical, "blind_p79_path_or_metadata_read_count": p79_reads, "private_truth_read_count": truth_reads, "legacy_v1_generator_call_count": len(provenance["worker_source_audit"]["forbidden_helper_calls"]), "input_file_match_count": provenance["input_file_match_count"], "blind_input_file_count": materialized["file_count"], "mutation_control_pass_count": boundary["pass_count"], "mutation_control_count": boundary["control_count"], "phase_a_trust_basis": {"blind_root_closed_world": materialized["closed_world_before_and_after_workers"] is True and materialized["file_count"] == 2 * EXPECTED_CAPTURES, "worker_blob_parent_verified": boundary["isolation_controls"]["worker_blob_bound"], "staging_copy_executed": boundary["isolation_controls"]["worker_executes_staged_copy"], "isolated_hwr_unavailable": boundary["isolation_controls"]["hwr_not_importable"], "source_audit_passed": provenance["worker_source_audit"]["passed"], "source_similarity_passed": provenance["worker_similarity_audit"]["passed"], "sources_stable_after_workers": provenance["source_end"]["checks"]["passed"], "read_ledger_role": "auxiliary"}, "runtime": runtime, "checks": checks, **CLAIM_FLAGS}
def build_manifest(root: Path, arguments: argparse.Namespace,
        provenance: Mapping[str, object], report: Mapping[str, object],
        artifacts: Mapping[str, bytes], elapsed: float,
        peak_rss: int) -> dict[str, object]:
    runtime = {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"), "platform": platform.platform(), "cpu_count": os.cpu_count(), "wall_seconds": elapsed, "process_tree_peak_rss_bytes": peak_rss, "disk_free_bytes": shutil.disk_usage(root).free}
    budgets = {"maximum_wall_seconds": MAX_WALL_SECONDS, "maximum_peak_rss_bytes": MAX_RSS_BYTES, "maximum_artifact_bytes": MAX_ARTIFACT_BYTES, "minimum_disk_free_bytes": MIN_DISK_FREE_BYTES}
    artifact_index = {name: {"bytes": len(content), "sha256": _sha256(content)} for name, content in sorted(artifacts.items())}
    return {"schema_version": MANIFEST_SCHEMA, "proposal_id": PROPOSAL_ID, "status": "complete", "source_commit": provenance["source_commit"], "decision": report["decision"], "command": _command(arguments), "provenance": provenance, "runtime": runtime, "budgets": budgets, "artifacts": artifact_index, **CLAIM_FLAGS}
def _seal_staging(root: Path, arguments: argparse.Namespace, staging: Path,
        artifacts: dict[str, bytes], comparison: Mapping[str, object],
        boundary: Mapping[str, object], receipt: Mapping[str, object],
        provenance: Mapping[str, object], worker_runs: Sequence[Mapping[str, object]],
        bit_identical: bool, started: float, parent_rss_start: int,
        comparer_elapsed: float, materialized: Mapping[str, object]
        ) -> tuple[dict[str, object], float, int]:
    for _ in range(4):
        measured = time.perf_counter() - started; wall_upper = measured + 1.0
        peak = _process_tree_peak(worker_runs) + 16 * 1024**2
        report = build_report(comparison, boundary, bit_identical=bit_identical,
            provenance=provenance, worker_runs=worker_runs, elapsed=wall_upper,
            peak_rss=peak, parent_rss_start=parent_rss_start,
            comparer_elapsed=comparer_elapsed, materialized=materialized,
            receipt=receipt)
        artifacts["report.json"] = _json_bytes(report); artifacts.pop("manifest.json", None)
        artifacts["manifest.json"] = _json_bytes(build_manifest(
            root, arguments, provenance, report, artifacts, wall_upper, peak))
        _require_budget(wall_upper, peak, artifacts)
        _write_staging(staging, artifacts)
        if time.perf_counter() - started <= wall_upper and _process_tree_peak(worker_runs) <= peak:
            return report, wall_upper, peak
    raise RuntimeError("P83 resource measurement did not stabilize")
def validate_frozen_provenance(root: Path, p50: Path, p79: Path) -> dict[str, object]:
    source_commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    changed = set(_git_lines(root, "diff", "--name-only", f"{FROZEN_COMMIT}..HEAD"))
    p50_top = {name: _file_identity(p50, p50 / name) for name in sorted(P50_FILES)}
    manifest_content, manifest_fd = _stable_read(p79 / "manifest.json", None, root=p79)
    p79_manifest = json.loads(manifest_content)
    expected_inputs = p79_manifest.get("provenance", {}).get("input_files")
    if not isinstance(expected_inputs, list): raise LineageContractError("p79_input_manifest")
    actual_inputs = _directory_identities(root, p50)
    p79_manifest_identity = {"path": _path_name(root, p79 / "manifest.json"), "bytes": len(manifest_content), "sha256": _sha256(manifest_content), "fd_identity": manifest_fd}
    p79_expected = _p79_expected_files(root, p79, p79_manifest, p79_manifest_identity)
    p79_stats = _directory_stats(root, p79)
    worker_path = root / WORKER_PATH; worker_source = _file_identity(root, worker_path)
    app_source = _file_identity(root, root / APP_PATH)
    worker_bytes, _ = _stable_read(worker_path, None, root=root)
    worker_text = worker_bytes.decode("utf-8"); worker_audit = audit_worker_source(worker_text)
    references = tuple(_git_show(root, P79_PRODUCER_COMMIT, Path(path)).decode() for path in ("src/hwr/eval/candidate_mask_ownership.py", "src/hwr/eval/target_selection.py"))
    similarity = source_similarity_audit(worker_text, references)
    consumer_sources = {path.as_posix(): {"expected_blob": CONSUMER_BLOBS[path.as_posix()], "head_blob": _git(root, "rev-parse", f"HEAD:{path.as_posix()}"), "imports": _source_imports(_stable_read(root / path, None, root=root)[0].decode("utf-8"))} for path in CONSUMER_PATHS}
    history = {path: {"expected_tree": expected, "head_tree": _git(root, "rev-parse", f"HEAD:{path}")} for path, expected in HISTORY_TREES.items()}
    p79_sizes = [{"path": value["path"], "bytes": value["bytes"]} for value in p79_stats]
    expected_sizes = [{"path": value["path"], "bytes": value["bytes"]} for value in p79_expected]
    capture_ledger = _capture_ledger(_read_bound_json(p50 / "capsules.json", p50_top["capsules.json"]))
    checks = {"workspace_clean": status == "", "implementation_scope_matches": changed == ALLOWED_PATHS, "frozen_document_commit_is_ancestor": _is_ancestor(root, FROZEN_COMMIT, "HEAD"), "frozen_document_blob_matches": _git(root, "rev-parse", f"HEAD:{FROZEN_DOCUMENT.as_posix()}") == FROZEN_DOCUMENT_BLOB, "history_trees_match": all(value["head_tree"] == value["expected_tree"] for value in history.values()), "p50_top_files_match": all(p50_top[name]["sha256"] == digest for name, digest in P50_FILES.items()), "p79_manifest_matches": p79_manifest_identity["sha256"] == P79_FILES["manifest.json"], "p79_bank_commitment_matches": p79_manifest["artifacts"]["bank.json"].get("sha256") == P79_FILES["bank.json"], "p79_file_set_and_sizes_match": p79_sizes == expected_sizes, "p79_tree_matches": _git(root, "rev-parse", f"{P79_ARTIFACT_COMMIT}:{FORMAL_P79.as_posix()}") == P79_TREE and _git(root, "rev-parse", f"HEAD:{FORMAL_P79.as_posix()}") == P79_TREE, "producer_commit_is_ancestor": _is_ancestor(root, P79_PRODUCER_COMMIT, "HEAD"), "producer_blob_matches": _git(root, "rev-parse", f"{P79_PRODUCER_COMMIT}:src/hwr/eval/candidate_mask_ownership.py") == P79_PRODUCER_BLOB, "selector_blob_matches": _git(root, "rev-parse", f"{P79_PRODUCER_COMMIT}:src/hwr/eval/target_selection.py") == SELECTOR_BLOB, "consumer_blobs_match": all(value["head_blob"] == value["expected_blob"] for value in consumer_sources.values()), "worker_blob_matches": worker_source["sha256"] == WORKER_SHA256, "worker_source_boundary": worker_audit["passed"], "worker_similarity_boundary": similarity["passed"], "input_file_set_matches_manifest": _public_identities(actual_inputs) == expected_inputs, "input_file_count": len(actual_inputs) == EXPECTED_INPUT_FILES, "capture_ledger_matches": capture_ledger == {"capture_count": EXPECTED_CAPTURES, "entry_count": 2 * EXPECTED_CAPTURES, "sha256": CAPTURE_LEDGER_SHA256}, "p79_manifest_schema": p79_manifest.get("schema_version") == P79_MANIFEST_SCHEMA}
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise LineageContractError("provenance:" + ",".join(failed))
    return {"source_commit": source_commit, "frozen_document": {"commit": FROZEN_COMMIT, "path": FROZEN_DOCUMENT.as_posix(), "blob": FROZEN_DOCUMENT_BLOB}, "p50_top_files": p50_top, "p79_manifest": p79_manifest_identity, "p79_pre_reveal_stats": p79_stats, "p79_expected_files": p79_expected, "p79_tree": P79_TREE, "producer_blob": P79_PRODUCER_BLOB, "selector_blob": SELECTOR_BLOB, "consumer_sources": consumer_sources, "history_trees": history, "worker_source": worker_source, "app_source": app_source, "worker_source_audit": worker_audit, "worker_similarity_audit": similarity, "p50_input_files": actual_inputs, "input_file_match_count": len(actual_inputs), "checks": checks}
def audit_worker_source(source: str) -> dict[str, object]:
    tree = ast.parse(source); imports = _source_imports(source)
    imported_bindings = {alias.asname or alias.name.split(".", 1)[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    import_aliases = [alias.asname for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
        if alias.asname and (alias.name, alias.asname) != ("numpy", "np")]
    nonstandard_imports = [name for name in imports if name.split(".", 1)[0] not in sys.stdlib_module_names and name.split(".", 1)[0] != "numpy" and name != "__future__"]
    forbidden_imports = [name for name in imports if name == "hwr" or name.startswith("hwr.") or name == "torch" or name.startswith("torch.") or name == "mujoco" or name.startswith("mujoco.")]
    forbidden_calls = {"generate_candidate_set_v2", "generate_candidate_set", "candidate_scores", "select_candidate_index", "_merge_candidates"}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    dynamic = {"__import__", "exec", "eval", "compile", "open", "import_module"}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    attribute_calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    dynamic_attributes = {"import_module", "__import__", "exec", "eval", "compile", "run_path", "run_module", "exec_module", "load_module", "module_from_spec", "spec_from_file_location"}
    dynamic_aliases = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names if alias.name in dynamic | dynamic_attributes}
    file_aliases = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module in {"os", "io"} for alias in node.names if alias.name in {"open", "read", "pread", "preadv", "fdopen"}}
    dangerous_attributes = {"open", "read_bytes", "read_text", "pread", "preadv", "fdopen", *dynamic_attributes, "__import__", "__getattribute__", "__subclasses__"}
    reflection = {"getattr", "setattr", "delattr", "globals", "locals", "vars", "attrgetter"}
    reflection_aliases = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names if alias.name in reflection}
    alias_assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.NamedExpr, ast.AnnAssign)) and ((isinstance(node.value, ast.Name) and node.value.id in dynamic | reflection | imported_bindings) or (isinstance(node.value, ast.Attribute) and node.value.attr in dangerous_attributes))]
    file_api_calls = [(function.name, node.func.attr) for function in (value for value in ast.walk(tree) if isinstance(value, ast.FunctionDef)) for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in dangerous_attributes]
    unsafe_attribute_reads = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"open", "read_bytes", "read_text", "pread", "preadv", "fdopen"} and not (node.func.attr == "open" and isinstance(node.func.value, ast.Name) and node.func.value.id == "os")]
    forbidden_file_recovery = any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(tree))
    io_calls = [(function.name, node.func.attr) for function in (value for value in ast.walk(tree) if isinstance(value, ast.FunctionDef)) for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr in {"open", "read", "close", "lstat", "fstat"}]
    stable_callers = {function.name for function in (value for value in ast.walk(tree) if isinstance(value, ast.FunctionDef)) for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "stable_file_read"}
    checks = {"imports_only_standard_library_and_numpy": not forbidden_imports and not nonstandard_imports, "does_not_call_production_helpers": not (calls & forbidden_calls), "does_not_use_dynamic_code_or_import": not (calls & (dynamic | dynamic_aliases)) and not (attribute_calls & dynamic_attributes), "does_not_use_alias_or_reflection": not import_aliases and not (calls & (reflection | reflection_aliases)) and not (attribute_calls & reflection) and not (attributes & {"__dict__", "__getattribute__", "__subclasses__"}) and not any(isinstance(node, ast.Name) and node.id == "__builtins__" for node in ast.walk(tree)) and not alias_assignments, "file_reads_use_stable_reader": not (calls & ({"open"} | file_aliases)) and not unsafe_attribute_reads and all(name in {"stable_file_read", "_atomic_write"} for name, operation in io_calls if operation in {"open", "read"}) and all(name in {"stable_file_read", "_atomic_write"} for name, operation in file_api_calls), "stable_reader_call_sites_closed": stable_callers <= {"run", "_read_bound_blob"}, "does_not_recover_root_from_file": not forbidden_file_recovery, "does_not_modify_sys_path": not any(isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr == "path" for node in ast.walk(tree))}
    return {"imports": imports, "forbidden_imports": forbidden_imports, "nonstandard_imports": nonstandard_imports, "forbidden_helper_calls": sorted(calls & forbidden_calls), "alias_assignment_count": len(alias_assignments), "stable_reader_callers": sorted(stable_callers), "os_file_io_calls": [list(value) for value in io_calls], "checks": checks, "passed": all(checks.values())}
def source_similarity_audit(worker: str, references: Sequence[str]) -> dict[str, object]:
    trusted = {"acquisition_from_robot", "_rotation", "_pose", "_quantize"}
    worker_tokens = _normalized_tokens(worker); parsed_worker = ast.parse(worker)
    worker_ast = [type(node).__name__ for node in ast.walk(parsed_worker)]
    whole_token = max(_ratio(worker_tokens, _normalized_tokens(value)) for value in references)
    whole_ast = max(_ratio(worker_ast, [type(node).__name__ for node in ast.walk(ast.parse(value))]) for value in references)
    worker_functions = _function_sequences(worker, trusted)
    reference_functions = [item for value in references for item in _function_sequences(value, trusted)]
    pairs = [(_ratio(left[1], right[1]), left[0], right[0]) for left in worker_functions for right in reference_functions if len(left[1]) >= 120 and len(right[1]) >= 120]
    maximum = max(pairs, default=(0.0, None, None))
    checks = {"whole_token_ratio_le_0_45": whole_token <= 0.45, "whole_ast_ratio_le_0_45": whole_ast <= 0.45, "major_function_token_ratio_le_0_82": maximum[0] <= 0.82}
    return {"whole_token_ratio": whole_token, "whole_ast_ratio": whole_ast, "maximum_major_function_ratio": maximum[0], "maximum_major_function_pair": list(maximum[1:]), "shared_trusted_math_functions": sorted(trusted), "checks": checks, "passed": all(checks.values())}
def _function_sequences(source: str, excluded: set[str]) -> list[tuple[str, list[str]]]:
    return [(node.name, _normalized_tokens(ast.unparse(node))) for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in excluded]
def _normalized_tokens(source: str) -> list[str]:
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT}
    replace = {tokenize.NAME, tokenize.NUMBER, tokenize.STRING}
    return [tokenize.tok_name[token.type] if token.type in replace else token.string for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type not in ignored]
def _ratio(left: Sequence[str], right: Sequence[str]) -> float:
    return difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()
def _write_staging(staging: Path, artifacts: Mapping[str, bytes]) -> None:
    allowed = set(artifacts)
    for path in staging.iterdir():
        if path.name not in allowed:
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
    for name, content in artifacts.items():
        path = staging / name
        if not path.exists(): _atomic_write(path, content)
        elif _stable_read(path, None, root=staging)[0] != content: _atomic_write(path, content)
    _fsync_directory(staging)
def _validate_blind_root(root: Path, plan: Mapping[str, object]) -> None:
    expected = {"blind-plan.json", *(capture[kind]["path"] for episode in plan["episodes"] for capture in episode["captures"] for kind in ("policy_input", "candidate_visible_input"))}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected or any(path.is_symlink() for path in root.rglob("*")):
        raise LineageContractError("blind_input_extra_or_missing")
def _manifest_input_map(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    if manifest.get("schema_version") != P79_MANIFEST_SCHEMA: raise LineageContractError("p79_manifest_schema")
    values = manifest.get("provenance", {}).get("input_files")
    if not isinstance(values, list) or len(values) != EXPECTED_INPUT_FILES: raise LineageContractError("p79_input_manifest")
    result = {}; prefix = FORMAL_P50.as_posix() + "/"
    for value in values:
        path = str(value.get("path", ""))
        if not path.startswith(prefix): raise LineageContractError("p79_input_root")
        relative = path.removeprefix(prefix)
        if relative in result: raise LineageContractError("p79_input_duplicate")
        result[relative] = {"path": relative, "bytes": value.get("bytes"),
                            "sha256": value.get("sha256")}
    return result
def _p79_expected_files(root: Path, p79: Path, manifest: Mapping[str, object],
        manifest_identity: Mapping[str, object]) -> list[dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping): raise LineageContractError("p79_artifacts")
    expected = [{
        "path": (p79 / _relative_path(name)).relative_to(root).as_posix(),
        "bytes": value.get("bytes"), "sha256": value.get("sha256")}
        for name, value in artifacts.items() if isinstance(value, Mapping)]
    if len(expected) != len(artifacts): raise LineageContractError("p79_artifacts")
    return sorted([*expected, _public_identity(manifest_identity)],
                  key=lambda value: value["path"])
def _read_p79_candidate(p79: Path, descriptor: Mapping[str, object]) -> bytes:
    if descriptor.get("schema_version") != CANDIDATE_SCHEMA: raise LineageContractError("candidate_schema")
    relative = _relative_path(descriptor.get("path")); path = p79 / relative
    content, _ = _stable_read(path, descriptor, root=p79)
    return content
def _capture_ledger(capsules: Mapping[str, object]) -> dict[str, object]:
    rows = []; capture_count = 0
    for episode in capsules["episodes"]:
        for capture in episode["captures"]:
            capture_count += 1
            for name in ("policy_input", "candidate_visible_input"):
                value = capture[name]
                rows.append([value["path"], value["sha256"], value["bytes"]])
    return {"capture_count": capture_count, "entry_count": len(rows), "sha256": _sha256(json.dumps(rows, separators=(",", ":")).encode("ascii"))}
def _source_imports(source: str) -> list[str]:
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import): imports.extend(value.name for value in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: imports.append(node.module)
    return sorted(set(imports))
def _blind_environment(forbidden_values: Sequence[str]) -> dict[str, str]:
    excluded = ("PYTHONPATH", "PYTHONHOME", "P79", "EXPECTED", "SCORE_HASH",
                "SELECTED_INDEX", "SELECTED_IDENTITY")
    environment = {key: value for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in excluded)
        and not any(item and item in value for item in forbidden_values)}
    environment["PYTHONPATH"] = ""; environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment
def _require_frozen_paths(root: Path, p50: Path, p79: Path, output: Path) -> None:
    expected = ((root / FORMAL_P50).resolve(), (root / FORMAL_P79).resolve(), (root / FORMAL_OUTPUT).resolve())
    if (p50, p79, output) != expected:
        raise ValueError("P83 path differs from frozen path")
def _require_budget(elapsed: float, peak_rss: int,
                    artifacts: Mapping[str, bytes]) -> None:
    if elapsed > MAX_WALL_SECONDS: raise RuntimeError("P83 wall-time budget exceeded")
    if peak_rss > MAX_RSS_BYTES: raise RuntimeError("P83 RSS budget exceeded")
    if sum(len(value) for value in artifacts.values()) > MAX_ARTIFACT_BYTES: raise RuntimeError("P83 artifact budget exceeded")
def _require_disk(output: Path) -> None:
    parent = output.parent
    while not parent.exists(): parent = parent.parent
    if shutil.disk_usage(parent).free < MIN_DISK_FREE_BYTES: raise RuntimeError("P83 disk-free guard failed")
def _file_identity(root: Path, path: Path) -> dict[str, object]:
    boundary = root if path.is_relative_to(root) else path.parent
    content, identity = _stable_read(path, None, root=boundary)
    return {"path": _path_name(root, path), "bytes": len(content), "sha256": _sha256(content), "fd_identity": identity}
def _directory_identities(root: Path, directory: Path) -> list[dict[str, object]]:
    return [_file_identity(root, path) for path in sorted(value for value in directory.rglob("*") if value.is_file())]
def _directory_stats(root: Path, directory: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        identity = _stat_identity(os.lstat(path))
        result.append({"path": _path_name(root, path), "bytes": identity["size"], "device": identity["device"], "inode": identity["inode"]})
    return result
def _public_identity(value: Mapping[str, object]) -> dict[str, object]: return {key: value[key] for key in ("path", "bytes", "sha256")}
def _public_identities(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]: return [_public_identity(value) for value in values]
def _stable_read(path: Path, expected: Mapping[str, object] | None, *,
                 root: Path | None = None) -> tuple[bytes, dict[str, int]]:
    root = path.parent if root is None else root
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise LineageContractError("path_escape") from error
    if not relative.parts or ".." in relative.parts: raise LineageContractError("path_escape")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    directories: list[int] = []; descriptor: int | None = None
    try:
        root_before = os.lstat(root)
        if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode): raise LineageContractError("path_symlink")
        root_fd = os.open(root, directory_flags); directories.append(root_fd)
        root_open = os.fstat(root_fd)
        if (root_before.st_dev, root_before.st_ino) != (root_open.st_dev, root_open.st_ino): raise LineageContractError("path_changed_during_read")
        parent_fd = root_fd
        for part in relative.parts[:-1]:
            before_directory = os.lstat(part, dir_fd=parent_fd)
            if stat.S_ISLNK(before_directory.st_mode) or not stat.S_ISDIR(before_directory.st_mode): raise LineageContractError("path_symlink")
            next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            opened_directory = os.fstat(next_fd); directories.append(next_fd)
            if (before_directory.st_dev, before_directory.st_ino) != (opened_directory.st_dev, opened_directory.st_ino): raise LineageContractError("path_changed_during_read")
            parent_fd = next_fd
        leaf = relative.parts[-1]; before = os.lstat(leaf, dir_fd=parent_fd)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): raise LineageContractError("path_symlink")
        descriptor = os.open(leaf, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
        opened = os.fstat(descriptor); chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk)
        after_read = os.fstat(descriptor); after_path = os.lstat(leaf, dir_fd=parent_fd)
    except OSError as error:
        raise LineageContractError("path_open") from error
    finally:
        if descriptor is not None: os.close(descriptor)
        for directory in reversed(directories): os.close(directory)
    identities = [_stat_identity(value) for value in (before, opened, after_read, after_path)]
    if len({tuple(value.values()) for value in identities}) != 1:
        raise LineageContractError("path_changed_during_read")
    content = b"".join(chunks)
    if expected is not None and (len(content) != expected.get("bytes") or _sha256(content) != expected.get("sha256")):
        raise LineageContractError("bound_file_size_or_hash")
    return content, identities[1]
def _read_bound_json(path: Path, expected: Mapping[str, object]) -> object:
    content, _ = _stable_read(path, expected, root=path.parent); return json.loads(content)
def _stat_identity(value: os.stat_result) -> dict[str, int]: return {"device": int(value.st_dev), "inode": int(value.st_ino), "size": int(value.st_size)}
def _path_name(root: Path, path: Path) -> str:
    try: return path.relative_to(root).as_posix()
    except ValueError: return path.as_posix()
def _relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value: raise LineageContractError("path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts: raise LineageContractError("path_escape")
    return path
def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", ancestor, descendant), cwd=root, check=False).returncode == 0
def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, text=True).stdout.strip()
def _git_lines(root: Path, *arguments: str) -> list[str]: return _git(root, *arguments).splitlines()
def _git_show(root: Path, commit: str, path: Path) -> bytes: return subprocess.run(("git", "show", f"{commit}:{path.as_posix()}"), cwd=root, check=True, capture_output=True).stdout
def _json_bytes(value: object) -> bytes: return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(content)
        while view: view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally: os.close(descriptor)
    os.replace(temporary, path)
def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
def _parse_time_peak_rss(output: str) -> int:
    for line in output.splitlines():
        if "maximum resident set size" in line: return int(line.split()[0])
    raise LineageContractError("external_rss_missing")
def _process_tree_peak(worker_runs: Sequence[Mapping[str, object]]) -> int:
    external = max([_children_peak_rss_bytes(), *(int(value["outer_observed_child_peak_rss_bytes"]) for value in worker_runs)]); return _peak_rss_bytes() + external
def _require_finalized_budget(started: float, peak_rss: int, output: Path,
        recorded_wall: float, worker_runs: Sequence[Mapping[str, object]]) -> float:
    artifacts = {path.relative_to(output).as_posix(): path.stat().st_size for path in output.rglob("*") if path.is_file()}
    report = _read_bound_json(output / "report.json", _file_identity(output, output / "report.json"))
    manifest = _read_bound_json(output / "manifest.json", _file_identity(output, output / "manifest.json"))
    observed_peak = _process_tree_peak(worker_runs); elapsed = time.perf_counter() - started
    if (report["runtime"]["wall_seconds"] != recorded_wall or manifest["runtime"]["wall_seconds"] != recorded_wall or report["runtime"]["process_tree_peak_rss_bytes"] != peak_rss
            or manifest["runtime"]["process_tree_peak_rss_bytes"] != peak_rss):
        raise LineageContractError("final_resource_record")
    if elapsed > recorded_wall: raise RuntimeError("P83 recorded wall-time bound exceeded")
    if observed_peak > peak_rss: raise RuntimeError("P83 recorded RSS bound exceeded")
    if elapsed > MAX_WALL_SECONDS: raise RuntimeError("P83 wall-time budget exceeded after rename")
    if observed_peak > MAX_RSS_BYTES: raise RuntimeError("P83 RSS budget exceeded after rename")
    if sum(artifacts.values()) > MAX_ARTIFACT_BYTES: raise RuntimeError("P83 artifact budget exceeded after rename")
    return elapsed
def _sha256(content: bytes) -> str: return hashlib.sha256(content).hexdigest()
def _resolve(root: Path, path: Path) -> Path: return path.resolve() if path.is_absolute() else (root / path).resolve()
def _command(arguments: argparse.Namespace) -> list[str]:
    return [sys.executable, "-m", MODULE_NAME, "--p50", arguments.p50.as_posix(), "--p79", arguments.p79.as_posix(), "--output", arguments.output.as_posix()]
def _rss_bytes(who: int) -> int:
    value = int(resource.getrusage(who).ru_maxrss); return value if sys.platform == "darwin" else value * 1024
def _peak_rss_bytes() -> int: return _rss_bytes(resource.RUSAGE_SELF)
def _children_peak_rss_bytes() -> int: return _rss_bytes(resource.RUSAGE_CHILDREN)
def main(argv: Sequence[str] | None = None) -> int: result = run(build_parser().parse_args(argv)); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["decision"].startswith("accepted") else 2
if __name__ == "__main__": raise SystemExit(main())
