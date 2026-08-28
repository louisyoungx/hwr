"""Run the frozen R0001-P87 experiment-contract reachability oracle."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from hwr.eval import experiment_contract_oracle as oracle
PROPOSAL_ID = "R0001-P87"
REGISTRY_PATH = Path("configs/eval/r0017_experiment_contracts.json")
REGISTRY_SHA256 = "c13ce840c4342772f46e77d87d0d856595f761bf02880266bad98ba6ffee6e8b"
REGISTRY_BLOB = "270ad643a2aa5b90a16151696c2122d61e5f179d"
FROZEN_COMMIT = "9c07cf6e44cb050b3977364ca7862938a7948904"
FROZEN_DOCUMENT = Path("docs/research-loop/0017/03-experiment.md")
FROZEN_DOCUMENT_BLOB = "18cbf3c67b4ff464b2a77d50c5e2c449d80f499e"
FORMAL_OUTPUT = Path(
    "runs/research-loop/0017/r0017-p87-contract-oracle-s20268701"
)
APP_PATH = Path("src/hwr/apps/evaluate_experiment_contracts.py")
ORACLE_PATH = Path("src/hwr/eval/experiment_contract_oracle.py")
ALLOWED_PATHS = frozenset({
    APP_PATH.as_posix(), ORACLE_PATH.as_posix(),
    "tests/test_experiment_contract_oracle.py",
    "tests/test_experiment_contract_oracle_app.py"})
HISTORY_PATHS = tuple(f"docs/research-loop/{index:04d}" for index in range(1, 17))
MAX_WALL_SECONDS = 30.0
MAX_RSS_BYTES = 512 * 1024**2
MAX_ARTIFACT_BYTES = 8 * 1024**2
MIN_DISK_FREE_BYTES = 20 * 1024**3
CLAIM_FLAGS = dict.fromkeys((
    "training_executed", "policy_inference_executed",
    "physical_acquisition_executed", "capability_evaluation_executed",
    "candidate_quality_claim_allowed", "selector_improvement_claim_allowed",
    "association_claim_allowed", "reachability_claim_allowed",
    "generalization_claim_allowed", "hardware_safety_claim_allowed",
    "task_success_claim_allowed"), False)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=FORMAL_OUTPUT)
    return parser
def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    registry_path = _resolve_inside(root, arguments.registry)
    output, staging = _require_output_available(root, arguments.output, formal=True)
    if registry_path != (root / REGISTRY_PATH).resolve():
        raise oracle.ContractOracleError("registry_path")
    _require_disk(output.parent)
    started = time.perf_counter()
    rss_start = _peak_rss_bytes()
    provenance = validate_provenance(root, registry_path)
    registry = _read_bound_json(registry_path, REGISTRY_SHA256, root)
    sources = _load_sources(root, registry)
    validate_p83_identity(registry, sources["p83_report"], sources["p83_manifest"])
    cohort = oracle.build_cohort(sources["p50_capsules"], sources["p79_bank"], registry)
    first = oracle.analyze_registry(registry, cohort)
    second = oracle.analyze_registry(registry, cohort)
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise oracle.ContractOracleError("analysis_bit_identity")
    boundary = run_boundary_controls(root, provenance)
    controls = run_controls(registry, cohort, first, boundary)
    if not controls["passed"]:
        raise oracle.ContractOracleError("mutation_control")
    elapsed = time.perf_counter() - started
    peak_rss = max(rss_start, _peak_rss_bytes())
    report = build_report(first, controls, elapsed, peak_rss)
    contracts = {
        "schema_version": "hwr.r0017-experiment-contract-copy/v1",
        "registry_identity": provenance["registry"],
        "registry": registry,
    }
    artifacts = {
        "cohort.json": canonical_json_bytes(cohort),
        "contracts.json": canonical_json_bytes(contracts),
        "analysis.json": canonical_json_bytes(first),
        "controls.json": canonical_json_bytes(controls),
        "report.json": canonical_json_bytes(report),
    }
    manifest = build_manifest(
        root, arguments, provenance, artifacts, elapsed, peak_rss
    )
    artifacts["manifest.json"] = canonical_json_bytes(manifest)
    total_bytes = sum(map(len, artifacts.values()))
    _check_resources(elapsed, peak_rss, total_bytes)
    finalized = _publish_artifacts(
        staging, output, artifacts,
        lambda: _finalize_run(root, provenance, started, output))
    return {
        "output": str(output),
        "decision": report["decision"],
        "contract_count": first["metrics"]["contract_count"],
        "mutation_control_pass_count": controls["pass_count"],
        "artifact_bytes": total_bytes,
        "finalized_wall_seconds": finalized["wall_seconds"],
        "finalized_process_tree_peak_rss_bytes": finalized["peak_rss_bytes"],
    }
def validate_provenance(root: Path, registry_path: Path) -> dict[str, object]:
    _require_clean_committed_source(root)
    if _git(root, "cat-file", "-t", FROZEN_COMMIT) != "commit":
        raise oracle.ContractOracleError("frozen_commit")
    if _git(root, "rev-parse", f"{FROZEN_COMMIT}:{FROZEN_DOCUMENT}") != (
        FROZEN_DOCUMENT_BLOB
    ):
        raise oracle.ContractOracleError("frozen_design_blob")
    if _git(root, "rev-parse", f"{FROZEN_COMMIT}:{REGISTRY_PATH}") != REGISTRY_BLOB:
        raise oracle.ContractOracleError("registry_blob")
    frozen_history = {
        path: _git(root, "rev-parse", f"{FROZEN_COMMIT}:{path}")
        for path in HISTORY_PATHS}
    history = {path: _git(root, "rev-parse", f"HEAD:{path}")
               for path in HISTORY_PATHS}
    if history != frozen_history: raise oracle.ContractOracleError("history_tree")
    changed = set(
        filter(None, _git(root, "diff", "--name-only", f"{FROZEN_COMMIT}...HEAD").splitlines())
    )
    if not changed <= ALLOWED_PATHS:
        raise oracle.ContractOracleError("allowed_scope", sorted(changed - ALLOWED_PATHS))
    registry_identity = _file_identity(root, registry_path)
    if registry_identity["sha256"] != REGISTRY_SHA256:
        raise oracle.ContractOracleError("registry_hash")
    source = {
        path.as_posix(): _file_identity(root, root / path)
        for path in (APP_PATH, ORACLE_PATH)
    }
    result = {
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "registry": registry_identity,
        "frozen_commit": FROZEN_COMMIT,
        "frozen_document_blob": FROZEN_DOCUMENT_BLOB,
        "registry_blob": REGISTRY_BLOB,
        "frozen_history_trees": frozen_history,
        "history_trees": history,
        "allowed_changed_paths": sorted(changed),
        "source": source,
    }
    _validate_provenance_contract(result)
    return result
def _require_clean_committed_source(root: Path) -> None:
    if _git(root, "status", "--porcelain=v1"):
        raise oracle.ContractOracleError("source_dirty")
    for path in (APP_PATH, ORACLE_PATH):
        tracked = _git_bytes(root, "show", f"HEAD:{path}")
        if tracked != (root / path).read_bytes():
            raise oracle.ContractOracleError("source_uncommitted", path)
def validate_source_stability(
    root: Path, provenance: Mapping[str, object]
) -> None:
    if _git(root, "rev-parse", "HEAD") != provenance["source_commit"]:
        raise oracle.ContractOracleError("source_changed")
    for name, identity in provenance["source"].items():
        if _file_identity(root, root / name) != identity:
            raise oracle.ContractOracleError("source_changed", name)
def _load_sources(
    root: Path, registry: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    loaded: dict[str, Mapping[str, object]] = {}
    for source, names in (
        ("p50", ("capsules.json", "plan.json", "report.json", "manifest.json")),
        ("p79", ("bank.json", "manifest.json")),
        ("p83", ("report.json", "manifest.json")),
    ):
        definition = registry["sources"][source]
        base = _resolve_inside(root, Path(definition["path"]))
        if base.is_symlink() or not base.is_dir():
            raise oracle.ContractOracleError("source_scope", source)
        for name in names:
            path = base / name; key = f"{source}_{name[:-5]}"
            if key in {"p50_capsules", "p79_bank", "p83_report", "p83_manifest"}:
                loaded[key] = _read_bound_json(path, definition["files"][name], root)
            elif _sha256(_stable_read(path, root)) != definition["files"][name]: raise oracle.ContractOracleError("input_hash", name)
    return loaded
def validate_p83_identity(
    registry: Mapping[str, object],
    report: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    report_identity = (
        report.get("schema_version"), report.get("proposal_id"),
        report.get("sample_unit"))
    if report_identity != (
        "hwr.p83-selection-lineage-report/v1", "R0001-P83", "Episode"):
        raise oracle.ContractOracleError("p83_report_identity")
    manifest_identity = (
        manifest.get("schema_version"), manifest.get("proposal_id"),
        manifest.get("status"), manifest.get("decision"))
    if manifest_identity != (
        "hwr.p83-selection-lineage-artifacts/v1", "R0001-P83", "complete",
        report.get("decision")):
        raise oracle.ContractOracleError("p83_manifest_identity")
    artifacts = manifest.get("artifacts")
    expected_report = registry["sources"]["p83"]["files"]["report.json"]
    if not isinstance(artifacts, Mapping) or (
        artifacts.get("report.json", {}).get("sha256") != expected_report):
        raise oracle.ContractOracleError("p83_report_manifest_binding")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise oracle.ContractOracleError("p83_provenance")
    p50 = provenance.get("p50_top_files")
    if not isinstance(p50, Mapping) or {
        name: value.get("sha256") for name, value in p50.items()
    } != registry["sources"]["p50"]["files"]:
        raise oracle.ContractOracleError("p83_p50_binding")
    p79_manifest = provenance.get("p79_manifest")
    expected_manifest = registry["sources"]["p79"]["files"]["manifest.json"]
    if not isinstance(p79_manifest, Mapping) or (
        p79_manifest.get("sha256") != expected_manifest):
        raise oracle.ContractOracleError("p83_p79_binding")
    bank_hashes = {
        value.get("sha256")
        for value in provenance.get("p79_artifact_files", [])
        if isinstance(value, Mapping) and str(value.get("path", "")).endswith("/bank.json")
    }
    if bank_hashes != {registry["sources"]["p79"]["files"]["bank.json"]}:
        raise oracle.ContractOracleError("p83_p79_binding")
def run_controls(
    registry: Mapping[str, object],
    cohort: Mapping[str, object],
    analysis: Mapping[str, object],
    boundary_controls: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Exercise every preregistered semantic and boundary control."""
    contracts = {row["contract_id"]: row for row in registry["contracts"]}
    controls: list[dict[str, object]] = []
    def record(control_id: str, passed: bool, observed: object) -> None:
        controls.append(
            {"control_id": control_id, "passed": bool(passed), "observed": observed}
        )

    selector = contracts["r0016-p68-e3-selector-negative"]
    base = oracle.solve_analytic(selector, registry, cohort)
    mutated = copy.deepcopy(selector)
    mutated["target_minimum"] = 8
    relaxed = copy.deepcopy(mutated)
    relaxed["claim_scope"] = ["overall"]
    relaxed["stratum_minimums"] = {}
    reachable = oracle.solve_enumeration(relaxed, registry, cohort)
    record(
        "C01-selector-target-boundary",
        any(row["category"] == "required_gt_eligible" for row in base["contradictions"])
        and not any(
            row["category"] == "required_gt_eligible"
            for row in oracle.solve_analytic(mutated, registry, cohort)["contradictions"]
        )
        and reachable["reachable"],
        {"target_mutation": "18_to_8",
         "independent_task_floors_relaxed_for_assignment_probe": True},
    )
    pooled = contracts["r0016-p76-e3-pooled-prefix"]
    denominator = [row for row in cohort["episodes"]
                   if "nonempty" in row["denominators"]]
    negatives = {}
    for row in denominator:
        if (row["observation_latency_steps"], row["action_latency_steps"]) == (
                2, 2) and row["task_id"] not in negatives:
            negatives[row["task_id"]] = row["episode_id"]
    positive = sorted(row["episode_id"] for row in denominator
                      if row["episode_id"] not in set(negatives.values()))
    witness = {"positive_episode_ids": positive}
    old_contract = copy.deepcopy(pooled)
    old_contract["claim_scope"] = ["overall", "task"]
    old_check = oracle.verify_assignment(old_contract, registry, cohort, witness)
    with_latency = copy.deepcopy(pooled)
    with_latency["stratum_minimums"]["latency_pair"] = {
        "o1-a1": 0, "o1-a2": 0, "o2-a1": 0, "o2-a2": 4}
    new_check = oracle.verify_assignment(with_latency, registry, cohort, witness)
    record(
        "C02-pooled-hidden-latency-collapse",
        old_check["passed"] and not new_check["passed"] and len(positive) == 19,
        {"old": old_check, "with_latency_floor": new_check},
    )
    _error_control(
        controls, "C03-empty-nonempty-accounting", "denominator_partition",
        lambda value: value["denominators"].pop("empty"),
        registry,
    )
    _error_control(
        controls, "C04-denominator-name-swap", "denominator_semantics",
        lambda value: value["denominators"].update({
            "empty": value["denominators"]["nonempty"],
            "nonempty": value["denominators"]["empty"]}),
        registry,
    )
    _error_control(
        controls, "C05-choice-as-nonempty", "denominator_semantics",
        lambda value: value["denominators"]["choice_opportunity"].update(
            candidate_count_minimum=1),
        registry,
    )
    for unit in ("Candidate", "Frame", "Pixel", "Arm", "ControlStep"):
        _error_control(
            controls, f"C06-sample-unit-{unit.lower()}", "sample_unit",
            lambda value, unit=unit: value.update(sample_unit=unit),
            registry,
        )
    _cohort_error_controls(controls, registry, cohort)
    expected_mutation = copy.deepcopy(registry)
    expected_mutation["expected_cohort"]["episode_count"] += 1
    _capture_control(
        controls, "C11-expected-count-self-proof", "expected_cohort",
        lambda: oracle.build_cohort(
            _capsules_from_cohort(cohort), _bank_from_cohort(cohort), expected_mutation
        ),
    )
    include = contracts["r0017-p76-e5-include-exposed-draft"]
    include_result = oracle.solve_analytic(include, registry, cohort)
    record("C12-confirmatory-include-exposed", any(
        row["category"] == "confirmatory_include_exposed"
        for row in include_result["contradictions"]),
        include_result["contradictions"])
    _exposure_controls(controls, registry)
    excluded = contracts["r0017-p76-e5-exclude-exposed-draft"]
    denominator_result = oracle.solve_analytic(excluded, registry, cohort)["denominator"]
    fake = dict(denominator_result)
    fake["effective_count"] += fake["excluded_count"]
    _capture_control(
        controls, "C14-threshold-only-exclusion", "denominator_conservation",
        lambda: oracle.validate_denominator_accounting(
            excluded, registry, cohort, fake
        ),
    )
    missing_floor = copy.deepcopy(excluded)
    missing_floor["stratum_minimums"].pop("cell")
    record("C15-claim-without-floor", any(
        row["category"] == "claim_without_minimum"
        for row in oracle.solve_analytic(
            missing_floor, registry, cohort)["contradictions"]),
        "claim_without_minimum")
    reachable_result = next(row for row in analysis["contracts"] if row["reachable"])
    reachable_contract = contracts[reachable_result["contract_id"]]
    broken_witness = copy.deepcopy(reachable_result["solver_b"]["accepted_witness"])
    broken_witness["positive_episode_ids"] = broken_witness["positive_episode_ids"][:1]
    record(
        "C16-invalid-reachable-witness",
        not oracle.verify_assignment(
            reachable_contract, registry, cohort, broken_witness
        )["passed"],
        "witness_rejected",
    )
    unreachable_result = next(
        row for row in analysis["contracts"] if not row["reachable"])
    unreachable_contract = contracts[unreachable_result["contract_id"]]
    forged = {"positive_episode_ids": [
        row["episode_id"] for row in cohort["episodes"]]}
    record(
        "C17-forged-unreachable-witness",
        not oracle.verify_assignment(
            unreachable_contract, registry, cohort, forged
        )["passed"],
        "witness_rejected",
    )
    controls.extend(dict(row) for row in boundary_controls)
    _capture_control(
        controls, "C20-solver-disagreement", "invalid_solver_disagreement",
        lambda: oracle.combine_solver_results(
            {"reachable": True}, {"reachable": False}
        ),
    )
    return {
        "schema_version": "hwr.r0017-experiment-contract-controls/v1",
        "controls": controls,
        "control_count": len(controls),
        "pass_count": sum(row["passed"] for row in controls),
        "passed": all(row["passed"] for row in controls),
    }
def run_boundary_controls(
    root: Path, provenance: Mapping[str, object]
) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    def guard(identity: str, category: str, action) -> None:
        _capture_control(controls, identity, category, action)
    drift = copy.deepcopy(provenance); drift["frozen_document_blob"] = "0" * 40
    guard("C18-frozen-design", "frozen_design_blob",
          lambda: _validate_provenance_contract(drift))
    drift = copy.deepcopy(provenance); drift["registry_blob"] = "0" * 40
    guard("C18-registry-blob", "registry_blob",
          lambda: _validate_provenance_contract(drift))
    drift = copy.deepcopy(provenance); drift["registry"]["sha256"] = "0" * 64
    guard("C18-registry-hash", "registry_hash",
          lambda: _validate_provenance_contract(drift))
    drift = copy.deepcopy(provenance); drift["allowed_changed_paths"] = ["forbidden.py"]
    guard("C18-source-scope", "allowed_scope",
          lambda: _validate_provenance_contract(drift))
    drift = copy.deepcopy(provenance); drift["history_trees"][HISTORY_PATHS[0]] = "0" * 40
    guard("C18-history-tree", "history_tree",
          lambda: _validate_provenance_contract(drift))
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        sandbox = Path(temporary); payload = sandbox / "input.json"
        payload.write_bytes(b"{}")
        guard("C18-input-hash", "input_hash",
              lambda: _read_bound_json(payload, "0" * 64, sandbox))
        output = sandbox / "output"; output.mkdir()
        guard("C19-output-preexists", "output_or_staging_preexists",
              lambda: _require_output_available(sandbox, Path("output")))
        output.rmdir(); (sandbox / "output.staging").mkdir()
        guard("C19-staging-preexists", "output_or_staging_preexists",
              lambda: _require_output_available(sandbox, Path("output")))
        shutil.rmtree(sandbox / "output.staging")
        outside = sandbox.parent / f"{sandbox.name}-outside"; outside.write_bytes(b"x")
        link = sandbox / "link"; link.symlink_to(outside)
        guard("C19-symlink", "path_symlink",
              lambda: _require_output_available(sandbox, Path("link")))
        guard("C19-path-escape", "path_escape",
              lambda: _resolve_inside(sandbox, Path("../escape")))
        final = sandbox / "final"; stage = sandbox / "stage"
        calls = 0
        def fail_write(path: Path, content: bytes) -> None:
            nonlocal calls
            calls += 1
            if calls == 2: raise oracle.ContractOracleError("partial_write")
            _atomic_write(path, content)
        def fail_finalize() -> None:
            _publish_artifacts(stage, final, {"a.json": b"{}", "b.json": b"{}"},
                               lambda: {}, writer=fail_write)
        guard("C19-partial-write", "partial_write", fail_finalize)
        if final.exists() or stage.exists(): raise oracle.ContractOracleError("partial_cleanup")
        outside.unlink()
    return controls
def _validate_provenance_contract(provenance: Mapping[str, object]) -> None:
    if provenance.get("frozen_document_blob") != FROZEN_DOCUMENT_BLOB:
        raise oracle.ContractOracleError("frozen_design_blob")
    if provenance.get("registry_blob") != REGISTRY_BLOB:
        raise oracle.ContractOracleError("registry_blob")
    registry = provenance.get("registry")
    if not isinstance(registry, Mapping) or registry.get("sha256") != REGISTRY_SHA256:
        raise oracle.ContractOracleError("registry_hash")
    changed = set(provenance.get("allowed_changed_paths", []))
    if not changed <= ALLOWED_PATHS:
        raise oracle.ContractOracleError("allowed_scope")
    history, expected = (provenance.get(key) for key in (
        "history_trees", "frozen_history_trees"))
    if not isinstance(history, Mapping) or history != expected:
        raise oracle.ContractOracleError("history_tree")
def _cohort_error_controls(
    controls: list[dict[str, object]],
    registry: Mapping[str, object],
    cohort: Mapping[str, object],
) -> None:
    cases = (
        ("C07-task-key-missing", "task_identity", "task_missing"),
        ("C07-latency-key-unknown", "latency_identity", "observation"),
        ("C07-cell-key-unknown", "cell_identity", "cell"),
        ("C08-duplicate-episode", "episode_duplicate", "duplicate"),
        ("C08-missing-episode", "episode_identity_mismatch", "missing"),
        ("C09-p50-p79-identity", "episode_identity_mismatch", "identity"),
        ("C10-candidate-count", "candidate_count", "candidate"),
        ("C10-task", "task_identity", "task"),
        ("C10-latency", "latency_identity", "action"),
    )
    for identity, category, mutation in cases:
        capsules, bank = _capsules_from_cohort(cohort), _bank_from_cohort(cohort)
        _mutate_cohort_inputs(capsules, bank, mutation)
        _capture_control(
            controls, identity, category,
            lambda capsules=capsules, bank=bank: oracle.build_cohort(
                capsules, bank, registry))
def _mutate_cohort_inputs(
    capsules: dict[str, object], bank: dict[str, object], mutation: str
) -> None:
    left, right = capsules["episodes"][0], bank["episodes"][0]
    if mutation == "task_missing": left.pop("task_id")
    elif mutation == "observation": left["planned_latency"]["observation_steps"] = 99
    elif mutation == "cell": left["cell_id"] = "bad-cell"
    elif mutation == "duplicate":
        capsules["episodes"].append(copy.deepcopy(left)); capsules["capsule_count"] += 1
    elif mutation == "missing":
        capsules["episodes"].pop(); capsules["capsule_count"] -= 1
    elif mutation == "identity": right["planned_episode_id"] = "f" * 64
    elif mutation == "candidate": right["candidate_set"]["candidate_count"] = 99
    elif mutation == "task": right["task_id"] = "unknown"
    elif mutation == "action": left["planned_latency"]["action_steps"] = 99
def _exposure_controls(
    controls: list[dict[str, object]], registry: Mapping[str, object]
) -> None:
    cases = (
        ("C13-exposure-episode-missing", "exposure_episode", "episode"),
        ("C13-exposure-field-missing", "exposure_fields", "field"),
        ("C13-exposure-duplicate", "exposure_duplicate", "duplicate"),
        ("C13-exposure-unknown", "exposure_fields", "unknown"),
    )
    for identity, category, mutation in cases:
        value = copy.deepcopy(registry); ledger = value["result_exposure_ledger"]
        if mutation == "episode": ledger[0].pop("episode_id")
        elif mutation == "field": ledger[0]["fields"] = []
        elif mutation == "duplicate": ledger.append(copy.deepcopy(ledger[0]))
        else: ledger[0]["fields"].append("private_truth")
        _capture_control(
            controls, identity, category, lambda value=value: oracle.validate_registry(value))
def _error_control(
    controls: list[dict[str, object]],
    identity: str,
    category: str,
    mutation,
    registry: Mapping[str, object],
) -> None:
    value = copy.deepcopy(registry)
    mutation(value)
    _capture_control(
        controls, identity, category, lambda: oracle.validate_registry(value)
    )
def _capture_control(
    controls: list[dict[str, object]],
    identity: str,
    category: str,
    action,
) -> None:
    observed = "no_error"
    try:
        action()
    except oracle.ContractOracleError as error:
        observed = error.category
    controls.append(
        {
            "control_id": identity,
            "expected_category": category,
            "observed": observed,
            "passed": observed == category,
        }
    )
def _capsules_from_cohort(
    cohort: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": oracle.P50_SCHEMA,
        "capsule_count": len(cohort["episodes"]),
        "episodes": [
            {
                "planned_episode_id": row["episode_id"],
                "task_id": row["task_id"],
                "cell_id": row["cell_id"],
                "planned_latency": {
                    "observation_steps": row["observation_latency_steps"],
                    "action_steps": row["action_latency_steps"],
                },
                "candidate_set": {"candidate_count": row["candidate_count"]},
            }
            for row in cohort["episodes"]
        ],
    }
def _bank_from_cohort(
    cohort: Mapping[str, object],
    source_acquisition: str = "runs/research-loop/0010/"
    "r0010-p50-e1-acquisition-s20265001",
) -> dict[str, object]:
    return {
        "schema_version": oracle.P79_SCHEMA,
        "episode_count": len(cohort["episodes"]),
        "source_acquisition": source_acquisition,
        "episodes": [
            {
                "planned_episode_id": row["episode_id"],
                "task_id": row["task_id"],
                "cell_id": row["cell_id"],
                "candidate_set": {"candidate_count": row["candidate_count"]},
            }
            for row in cohort["episodes"]
        ],
    }
def build_report(
    analysis: Mapping[str, object],
    controls: Mapping[str, object],
    wall_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, object]:
    metrics = analysis["metrics"]
    contract_count = metrics["contract_count"]
    checks = {
        "five_contracts_analyzed": contract_count == 5,
        "solver_agreement_complete": metrics["solver_agreement_count"]
        == contract_count,
        "accepted_witnesses_valid": metrics["valid_accepted_witness_count"]
        == metrics["reachable_contract_count"],
        "contradictions_valid": metrics["valid_contradiction_count"]
        == metrics["rejected_contract_count"],
        "denominator_conservation_complete": metrics[
            "denominator_conservation_count"
        ]
        == contract_count,
        "exposure_policy_complete": metrics["exposure_policy_valid_count"]
        == contract_count,
        "private_outcome_unread": metrics["private_outcome_read_count"] == 0,
        "episode_sample_unit": metrics["sample_unit_violation_count"] == 0,
        "mutation_controls_complete": controls["passed"],
        "wall_within_budget": wall_seconds < MAX_WALL_SECONDS,
        "rss_within_budget": peak_rss_bytes < MAX_RSS_BYTES,
    }
    return {
        "schema_version": "hwr.r0017-experiment-contract-report/v1",
        "proposal_id": PROPOSAL_ID,
        "decision": (
            "accepted as frozen experiment-contract oracle"
            if all(checks.values())
            else "invalid"
        ),
        "checks": {**checks, "passed": all(checks.values())},
        "metrics": {
            **metrics,
            "mutation_control_count": controls["control_count"],
            "mutation_control_pass_count": controls["pass_count"],
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "process_tree_peak_rss_bytes": peak_rss_bytes,
            "cpu_only": True,
        },
        **CLAIM_FLAGS,
    }
def build_manifest(
    root: Path,
    arguments: argparse.Namespace,
    provenance: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    wall_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": "hwr.r0017-experiment-contract-artifacts/v1",
        "proposal_id": PROPOSAL_ID,
        "status": "complete",
        "source_commit": provenance["source_commit"],
        "command": [
            sys.executable,
            "-m",
            "hwr.apps.evaluate_experiment_contracts",
            "--registry",
            str(arguments.registry),
            "--output",
            str(arguments.output),
        ],
        "provenance": provenance,
        "runtime": {
            "wall_seconds_before_publish": wall_seconds,
            "process_tree_peak_rss_bytes": peak_rss_bytes,
            "cpu_only": True,
        },
        "budgets": {
            "wall_seconds_exclusive": MAX_WALL_SECONDS,
            "process_tree_peak_rss_bytes_exclusive": MAX_RSS_BYTES,
            "artifact_bytes_exclusive": MAX_ARTIFACT_BYTES,
            "minimum_disk_free_bytes": MIN_DISK_FREE_BYTES,
        },
        "artifacts": {
            name: {"sha256": _sha256(content), "bytes": len(content)}
            for name, content in sorted(artifacts.items())
        },
        **CLAIM_FLAGS,
    }
def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
def _read_bound_json(
    path: Path, expected_sha256: str, root: Path
) -> Mapping[str, object]:
    content = _stable_read(path, root)
    if _sha256(content) != expected_sha256:
        raise oracle.ContractOracleError("input_hash", path.name)
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise oracle.ContractOracleError("input_json", path.name) from error
    if not isinstance(value, Mapping):
        raise oracle.ContractOracleError("input_json", path.name)
    return value
def _stable_read(path: Path, root: Path) -> bytes:
    if path.is_symlink(): raise oracle.ContractOracleError("input_file_type", path)
    resolved = _resolve_inside(root, path)
    before = resolved.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise oracle.ContractOracleError("input_file_type", path)
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        content = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            content.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (opened.st_dev, opened.st_ino, opened.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or len(content) != after.st_size
    ):
        raise oracle.ContractOracleError("input_changed", path)
    return bytes(content)
def _file_identity(root: Path, path: Path) -> dict[str, object]:
    content = _stable_read(path, root)
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return {"path": relative, "sha256": _sha256(content), "bytes": len(content)}
def _resolve_inside(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise oracle.ContractOracleError("path_escape", path)
    return resolved
def _require_output_available(
    root: Path, value: Path, *, formal: bool = False
) -> tuple[Path, Path]:
    candidate = value if value.is_absolute() else root / value
    sibling = candidate.with_name(candidate.name + ".staging")
    if candidate.is_symlink() or sibling.is_symlink():
        raise oracle.ContractOracleError("path_symlink")
    output, staging = _resolve_inside(root, candidate), _resolve_inside(root, sibling)
    if formal and output != (root / FORMAL_OUTPUT).resolve():
        raise oracle.ContractOracleError("output_path")
    if output.exists() or staging.exists():
        raise oracle.ContractOracleError("output_or_staging_preexists")
    return output, staging
def _require_disk(parent: Path) -> None:
    existing = parent
    while not existing.exists(): existing = existing.parent
    if shutil.disk_usage(existing).free < MIN_DISK_FREE_BYTES:
        raise oracle.ContractOracleError("disk_budget")
def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("artifact_preexists")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)
def _publish_artifacts(
    staging: Path, output: Path, artifacts: Mapping[str, bytes], finalize,
    *, writer=None,
) -> Mapping[str, object]:
    writer = _atomic_write if writer is None else writer
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for name, content in artifacts.items(): writer(staging / name, content)
        if sum(path.stat().st_size for path in staging.iterdir()) != sum(
            map(len, artifacts.values())):
            raise oracle.ContractOracleError("partial_write")
        os.replace(staging, output); _fsync_directory(output.parent)
        return finalize()
    except BaseException:
        shutil.rmtree(output if output.exists() else staging, ignore_errors=True)
        try: _fsync_directory(output.parent)
        except OSError: pass
        raise
def _check_resources(wall: float, rss: int, artifact_bytes: int) -> None:
    if wall >= MAX_WALL_SECONDS: raise oracle.ContractOracleError("wall_budget")
    if rss >= MAX_RSS_BYTES: raise oracle.ContractOracleError("rss_budget")
    if artifact_bytes >= MAX_ARTIFACT_BYTES:
        raise oracle.ContractOracleError("artifact_budget")
def _finalize_run(
    root: Path, provenance: Mapping[str, object], started: float, output: Path
) -> dict[str, object]:
    validate_source_stability(root, provenance)
    finalized = {"wall_seconds": time.perf_counter() - started,
                 "peak_rss_bytes": _peak_rss_bytes(),
                 "artifact_bytes": sum(path.stat().st_size for path in output.iterdir())}
    _check_resources(finalized["wall_seconds"], finalized["peak_rss_bytes"],
                     finalized["artifact_bytes"])
    return finalized
def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
def _peak_rss_bytes() -> int:
    rss = sum(resource.getrusage(who).ru_maxrss for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)); return int(
        rss if sys.platform == "darwin" else rss * 1024)
def _sha256(content: bytes) -> str: return hashlib.sha256(content).hexdigest()
def _git(root: Path, *arguments: str) -> str: return _git_bytes(root, *arguments).decode("utf-8").strip()
def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", *arguments), cwd=root, check=False, capture_output=True
    )
    if result.returncode:
        raise oracle.ContractOracleError(
            "git_provenance", result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout
def main(arguments: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(arguments))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
