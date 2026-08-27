"""Evaluate the frozen R0001-P80 version-sealed artifact consumer contract."""
from __future__ import annotations
import argparse, ast, hashlib, json, os
from dataclasses import asdict
import platform, re, resource, shutil, subprocess, sys, tempfile, time
from typing import Mapping, Sequence
from hwr.apps import create_candidate_output
from hwr.eval.candidate_artifact_contract import (
    ACCEPTED_DECISION, FORMAL_TRUST_ANCHOR, FORMAL_TRUST_ANCHOR_FINGERPRINT,
    LEGACY_CANDIDATE_SCHEMA, PREREGISTERED_NEGATIVE_CONTROLS, git_bytes,
    git_is_ancestor, git_text, repository_path, repository_root, resolve_candidate_artifact)
MODULE_NAME = "hwr.apps.evaluate_candidate_artifact_contract"; PROPOSAL_ID = "R0001-P80"
REPORT_SCHEMA = "hwr.p80-candidate-artifact-contract-report/v1"; MANIFEST_SCHEMA = "hwr.p80-candidate-artifact-contract-artifacts/v1"
INCONCLUSIVE_DECISION = "inconclusive_artifact_contract_insufficient"
FROZEN_DOCUMENT_COMMIT = "f224149e0a5ab0ae3cea981e669dd661d7d64ffe"; FROZEN_DOCUMENT_PATH = "docs/research-loop/0015/03-experiment.md"
FROZEN_DOCUMENT_BLOB = "768c1fb309ea662f72319cc9688b1f50ce7eeada"
FROZEN_DOCUMENT_SHA256 = "1672848c3c0907eef56af4ca56d98adcc540ce242b2b7b55ea4eda2d41b0a153"
FORMAL_BANK = FORMAL_TRUST_ANCHOR.artifact_root; FORMAL_LEGACY_SOURCE = FORMAL_TRUST_ANCHOR.legacy_root
FORMAL_OUTPUT = "runs/research-loop/0015/r0015-p80-artifact-contract-s20268001"
SOURCE_PATHS = ("src/hwr/eval/candidate_artifact_contract.py", "src/hwr/apps/evaluate_candidate_artifact_contract.py")
ALLOWED_CHANGES = frozenset((*SOURCE_PATHS,
    "tests/test_candidate_artifact_contract.py", "tests/test_candidate_artifact_contract_app.py"))
FULL_PYTEST_COMMAND = (".venv/bin/python", "-m", "pytest", "-vv")
ALLOWED_FULL_PYTEST_FAILURES = frozenset((
    "tests/test_entity_candidate_mapping.py::test_frozen_document_history_and_input_provenance_are_complete",))
REPOSITORY_GATE_COMMANDS = (
    (".venv/bin/python", "scripts/check_python_size.py"),
    (".venv/bin/python", "scripts/check_architecture.py"),
    (".venv/bin/python", "-m", "compileall", "-q", "src", "tests"),
    ("git", "diff", "--check"),
)
MAX_WALL_SECONDS = 5 * 60; MAX_RSS_BYTES = 2 * 1024**3
MAX_ARTIFACT_BYTES = 5 * 1024**2; MIN_DISK_FREE_BYTES = 20 * 1024**3
CLAIM_FLAGS = dict.fromkeys("""
training_executed policy_inference_executed physical_acquisition_executed
capability_evaluation_executed capability_claim_allowed task_success_claim_allowed
generalization_claim_allowed hardware_safety_claim_allowed
candidate_quality_claim_allowed selector_improvement_claim_allowed
association_claim_allowed feasibility_claim_allowed default_migration_allowed
""".split(), False)
EVALUATOR_IMPORTS = frozenset("""
from:__future__:annotations import:hashlib import:json import:subprocess
from:dataclasses:asdict from:dataclasses:dataclass from:enum:Enum
from:pathlib:Path from:pathlib:PurePosixPath from:typing:Mapping
from:typing:Sequence from:hwr.apps:read_bound_blob
""".split())
APP_IMPORTS = frozenset("""
from:__future__:annotations import:argparse import:ast import:hashlib import:json
import:os from:dataclasses:asdict import:platform import:re
import:resource import:shutil import:subprocess import:sys import:tempfile
import:time from:typing:Mapping from:typing:Sequence
from:hwr.apps:create_candidate_output
from:hwr.eval.candidate_artifact_contract:ACCEPTED_DECISION
from:hwr.eval.candidate_artifact_contract:FORMAL_TRUST_ANCHOR
from:hwr.eval.candidate_artifact_contract:FORMAL_TRUST_ANCHOR_FINGERPRINT
from:hwr.eval.candidate_artifact_contract:LEGACY_CANDIDATE_SCHEMA
from:hwr.eval.candidate_artifact_contract:PREREGISTERED_NEGATIVE_CONTROLS
from:hwr.eval.candidate_artifact_contract:git_bytes
from:hwr.eval.candidate_artifact_contract:git_is_ancestor
from:hwr.eval.candidate_artifact_contract:git_text
from:hwr.eval.candidate_artifact_contract:repository_path
from:hwr.eval.candidate_artifact_contract:repository_root
from:hwr.eval.candidate_artifact_contract:resolve_candidate_artifact
""".split())
SUBPROCESS_CALLS = frozenset((
    ("_pytest_receipt", "subprocess.run(FULL_PYTEST_COMMAND, cwd=root, env=environment, capture_output=True, timeout=remaining)"),
    ("_repository_gate_receipt", "subprocess.run(command, cwd=root, env=environment, capture_output=True, timeout=remaining)"),
    ("_baseline_failure_receipt", "subprocess.run(('git', 'worktree', 'add', '--detach', worktree.as_posix(), FROZEN_DOCUMENT_COMMIT), cwd=root, capture_output=True)"),
    ("_baseline_failure_receipt", "subprocess.run((root / '.venv/bin/python', '-m', 'pytest', '-q', path), cwd=worktree, env=_validation_environment(), capture_output=True, timeout=remaining)"),
    ("_baseline_failure_receipt", "subprocess.run(('git', 'worktree', 'remove', '--force', worktree.as_posix()), cwd=root, check=False, capture_output=True)"),
))
APP_CALL_FINGERPRINT = "f4af636426e69cbeff12459fb3ac35a67a678df06a261090619597ef2471bf30"
APP_AST_FINGERPRINT = "61f107386a79b4aafcbff8d6aa54dc3a0a582ddb7af1fcb6f905e3833180f02b"
EVALUATOR_AST_FINGERPRINT = "95310b7fc8bdec385041c7e066cfc7b545d390b1a24ca3fddc741c6b05799c84"
def audit_consumer_architecture(evaluator_source: str,
                                app_source: str) -> dict[str, object]:
    evaluator = ast.parse(evaluator_source); application = ast.parse(app_source)
    evaluator_imports, evaluator_alias = _import_contract(evaluator); app_imports, app_alias = _import_contract(application)
    calls = _subprocess_contract(application)
    call_fingerprint = _call_fingerprint(application)
    ast_fingerprint = _ast_fingerprint(application)
    checks = {
        "scope_is_frozen_consumer_source_ast": True,
        "resolver_imports_exact": evaluator_imports == EVALUATOR_IMPORTS and not evaluator_alias,
        "complete_resolver_ast_matches_frozen_allowlist": _ast_fingerprint(evaluator) == EVALUATOR_AST_FINGERPRINT,
        "app_imports_exact": app_imports == APP_IMPORTS and not app_alias,
        "every_app_call_matches_frozen_receiver_and_shape": call_fingerprint == APP_CALL_FINGERPRINT,
        "complete_app_ast_matches_frozen_allowlist": ast_fingerprint == APP_AST_FINGERPRINT,
        "command_constants_exact": _command_constants_exact(application),
        "app_subprocess_calls_exact": calls == SUBPROCESS_CALLS,
    }
    return {"category": "architecture",
            "scope": "frozen evaluator/app source AST allowlist",
            "app_call_fingerprint": call_fingerprint,
            "app_ast_fingerprint": ast_fingerprint,
            "checks": checks, "passed": all(checks.values())}
def audit_default_generator_schema(source: str) -> dict[str, object]:
    tree = ast.parse(source); schema = None; generator = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and any(isinstance(target, ast.Name)
                        and target.id == "CANDIDATE_SCHEMA"
                        for target in node.targets)):
            schema = node.value.value
        if isinstance(node, ast.FunctionDef) and node.name == "generate_candidate_set":
            generator = node
    assignments = [] if generator is None else [
        node for node in ast.walk(generator) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "schema_version"
        and isinstance(value, ast.Name) and value.id == "CANDIDATE_SCHEMA"
    ]
    forbidden = () if generator is None else tuple(node
        for node in ast.walk(generator)
        if ((isinstance(node, (ast.Name, ast.Attribute))
             and any(value in getattr(node, "id", getattr(node, "attr", "")).lower()
                     for value in ("v2", "ownership")))
            or (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and ("p79-target-candidates/v2" in node.value
                     or "ownership" in node.value.lower()))))
    checks = {
        "candidate_schema_constant_is_frozen_v1": schema == LEGACY_CANDIDATE_SCHEMA,
        "canonical_document_schema_uses_candidate_schema": len(assignments) == 1,
        "generator_has_no_v2_or_ownership_reference": not forbidden,
    }
    return {
        "candidate_schema": schema, "checks": checks,
        "default_generator_uses_candidate_schema": len(assignments) == 1,
        "default_generator_has_no_v2_or_ownership_reference": not forbidden,
        "passed": all(checks.values()),
    }
def _import_contract(tree: ast.Module) -> tuple[frozenset[str], bool]:
    imports = set(); aliased = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(f"import:{alias.name}"); aliased |= alias.asname is not None
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.add(f"from:{module}:{alias.name}")
                aliased |= alias.asname is not None
    return frozenset(imports), aliased
def _subprocess_contract(tree: ast.Module) -> frozenset[tuple[str, str]]:
    calls = set()
    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        for node in ast.walk(function):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"):
                calls.add((function.name, ast.unparse(node)))
    return frozenset(calls)
def _call_fingerprint(tree: ast.Module) -> str:
    calls = sorted((function.name, ast.dump(node, include_attributes=False))
        for function in tree.body if isinstance(function, ast.FunctionDef)
        for node in ast.walk(function) if isinstance(node, ast.Call))
    return hashlib.sha256(json.dumps(calls, separators=(",", ":")).encode()).hexdigest()
def _ast_fingerprint(tree: ast.Module) -> str:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {
                "APP_CALL_FINGERPRINT", "APP_AST_FINGERPRINT",
                "EVALUATOR_AST_FINGERPRINT"}
            for target in node.targets
        ):
            node.value = ast.Constant(value="<sealed>")
    return hashlib.sha256(ast.dump(
        tree, include_attributes=False).encode()).hexdigest()
def _command_constants_exact(tree: ast.Module) -> bool:
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
        and target.id in {"FULL_PYTEST_COMMAND", "REPOSITORY_GATE_COMMANDS"}
    }
    return assignments == {
        "FULL_PYTEST_COMMAND": FULL_PYTEST_COMMAND,
        "REPOSITORY_GATE_COMMANDS": REPOSITORY_GATE_COMMANDS,
    }
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument(
        "--bank", required=True); parser.add_argument("--output", required=True)
    return parser
def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = repository_root(__file__)
    bank = _resolve_argument(root, arguments.bank); output = _resolve_argument(
        root, arguments.output)
    _require_frozen_paths(root, bank, output)
    if os.path.lexists(output) or os.path.lexists(
        output.with_name(output.name + ".tmp")
    ):
        raise FileExistsError(output)
    _require_disk(output)
    started = time.perf_counter(); deadline = started + MAX_WALL_SECONDS
    source_commit = _source_commit(root)
    provenance = _provenance(root, source_commit)
    _require_provenance(provenance)
    receipt, pytest_output = _pytest_receipt(root, deadline)
    baseline_failure_receipt, baseline_failure_output = (
        _baseline_failure_receipt(root, deadline)
    )
    gates, gate_output = _repository_gate_receipt(root, deadline)
    first = resolve_candidate_artifact(
        root,
        p79_v2_bank=FORMAL_BANK,
        p50_legacy_source=FORMAL_LEGACY_SOURCE,
        trust_anchor=FORMAL_TRUST_ANCHOR,
    )
    second = resolve_candidate_artifact(
        root,
        p79_v2_bank=FORMAL_BANK,
        p50_legacy_source=FORMAL_LEGACY_SOURCE,
        trust_anchor=FORMAL_TRUST_ANCHOR,
    )
    first_bytes = first.canonical_receipt_bytes(); second_bytes = second.canonical_receipt_bytes()
    final_provenance = _provenance(root, source_commit)
    _require_provenance(final_provenance)
    report, artifacts = _write_validated_output(
        output=output, started=started, source_commit=source_commit,
        command=_command(arguments), counts=first.validation_counts.as_dict(),
        first_receipt=first_bytes, repeated=first_bytes == second_bytes,
        pytest_receipt=receipt, pytest_output=pytest_output,
        repository_gates=gates, gate_output=gate_output,
        baseline_failure_receipt=baseline_failure_receipt,
        baseline_failure_output=baseline_failure_output,
        provenance={
            **final_provenance,
            "preflight_checks": provenance["checks"],
        },
    )
    return {
        "output": str(output),
        "decision": report["decision"],
        "receipt_sha256": report["receipt"]["sha256"],
        "manifest_sha256": _sha256(artifacts["manifest.json"]),
    }
def _report(
    *, source_commit: str, command: Sequence[str], counts: Mapping[str, int],
    first_receipt: bytes, repeated: bool, pytest_receipt: Mapping[str, object],
    repository_gates: Mapping[str, object], provenance: Mapping[str, object],
    elapsed: float, peak_rss: Mapping[str, int], pytest_output: bytes,
    gate_outputs: Mapping[str, bytes],
    baseline_failure_receipt: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "episode_count": 24, "capture_count": 384,
        "v2_candidate_count": 24, "legacy_candidate_count": 24,
        "capture_blob_count": 768, "p79_artifact_count": 28,
        "p50_artifact_count": 795,
        "p50_input_file_count": 796,
    }
    pytest_valid = _validate_pytest_receipt(pytest_receipt, pytest_output)
    gates_valid = _validate_repository_gate_receipt(
        repository_gates, gate_outputs
    )
    baseline_valid = _validate_baseline_failure_receipt(
        baseline_failure_receipt, baseline_failure_receipt["_output"]
    )
    checks = {
        "complete_validation_counts": dict(counts) == expected,
        "repeated_receipt_bit_identical": repeated,
        "all_preregistered_negative_controls_passed": pytest_valid,
        "full_pytest_matches_frozen_allowed_failure": pytest_valid,
        "repository_gates_passed": gates_valid,
        "allowed_failure_reproduced_at_frozen_commit": baseline_valid,
        "provenance_passed": provenance["checks"]["passed"] is True,
        "consumer_architecture_guard": (
            provenance["architecture"]["passed"] is True
        ),
        "default_generator_remains_v1": (
            provenance["default_generator"]["passed"] is True
        ),
    }
    technical_passed = all(checks.values())
    checks["independent_v2_score_selection_evidence"] = False
    passed = all(checks.values())
    return {
        "schema_version": REPORT_SCHEMA, "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "decision": INCONCLUSIVE_DECISION if technical_passed else "invalid",
        "validation_counts": dict(counts),
        "evidence_scope": (
            "current source consistency with committed frozen document and "
            "artifact Git receipt; not proof against later code, document, "
            "or history rewriting"
        ),
        "path_safety_scope": "symlink rejection plus pre/post lstat snapshots; not complete TOCTOU immunity",
        "selection_metadata_scope": (
            "bank/capsule bind legacy selected index, score hash, and identity; "
            "bank/regression bind v2 and legacy selected index and identity; "
            "regression has no v2 score hash, so the v2 score/selection relation "
            "is bound metadata and is not independently validated"
        ),
        "formal_prerequisites": {
            "full_pytest_executed_in_runner": True,
            "repository_gates_executed_in_runner": True,
            "no_external_validation_receipt_required": True},
        "execution_boundaries": {
            "validation_child_cuda_visible_devices": "",
            "validation_child_pytorch_mps_fallback": "0",
            "validation_may_exercise_configured_accelerators": True,
            "p80_resolver_ast_excludes_mujoco_and_torch_imports": True,
            "p80_physics_or_scientific_compute_executed": False,
            "runner_uses_foreground_subprocesses_only": True,
            "tmux_or_background_command_used": False,
        },
        "receipt": {
            "schema_version": "hwr.p80-candidate-artifact-receipt/v1",
            "path": "receipt.json", "bytes": len(first_receipt),
            "sha256": _sha256(first_receipt),
            "repeated_bit_identical": repeated,
        },
        "pytest_receipt": _public_receipt(pytest_receipt),
        "repository_gate_receipt": _public_receipt(repository_gates),
        "baseline_failure_receipt": _public_receipt(baseline_failure_receipt),
        "provenance_checks": dict(provenance["checks"]),
        "resource_usage": {
            "wall_seconds_through_staging_probe": elapsed, **peak_rss},
        "final_budget_check": {"performed_after_staging_write": True,
                               "performed_before_atomic_rename": True},
        "checks": {**checks, "technical_validation_passed": technical_passed,
                   "passed": passed},
        **CLAIM_FLAGS,
    }
def _manifest(
    *, source_commit: str, command: Sequence[str],
    artifacts: Mapping[str, bytes], report: Mapping[str, object],
    provenance: Mapping[str, object], pytest_receipt: Mapping[str, object],
    repository_gates: Mapping[str, object], elapsed: float,
    peak_rss: Mapping[str, int],
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA, "proposal_id": PROPOSAL_ID,
        "status": "complete", "source_commit": source_commit,
        "decision": report["decision"],
        "command": list(command),
        "trust_anchor": asdict(FORMAL_TRUST_ANCHOR),
        "trust_anchor_fingerprint": FORMAL_TRUST_ANCHOR_FINGERPRINT,
        "provenance": dict(provenance),
        "pytest_receipt": dict(pytest_receipt),
        "repository_gate_receipt": dict(repository_gates),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "wall_seconds_through_staging_probe": elapsed,
            **peak_rss,
            "disk_free_bytes": shutil.disk_usage(repository_root(__file__)).free,
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
def _provenance(root, source_commit: str) -> dict[str, object]:
    evaluator_source = git_bytes(root, "show", f"{source_commit}:{SOURCE_PATHS[0]}").decode()
    app_source = git_bytes(root, "show", f"{source_commit}:{SOURCE_PATHS[1]}").decode()
    default_source = git_bytes(root, "show",
        f"{source_commit}:src/hwr/eval/target_selection.py").decode()
    architecture = audit_consumer_architecture(evaluator_source, app_source)
    default_generator = audit_default_generator_schema(default_source)
    frozen = git_bytes(root, "show",
        f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}")
    changed = set(git_text(root, "diff", "--name-only",
                           FROZEN_DOCUMENT_COMMIT, source_commit).splitlines())
    commit_count = int(git_text(
        root, "rev-list", "--count", f"{FROZEN_DOCUMENT_COMMIT}..{source_commit}"
    ))
    checks = {
        "workspace_clean": not git_text(
            root, "status", "--porcelain", "--untracked-files=all"
        ),
        "source_commit_matches_head":
            source_commit == git_text(root, "rev-parse", "HEAD"),
        "source_files_match_head": all(
            git_text(root, "hash-object", path)
            == git_text(root, "rev-parse", f"HEAD:{path}")
            for path in SOURCE_PATHS
        ),
        "implementation_scope_matches": changed == ALLOWED_CHANGES,
        "implementation_commit_count": commit_count == 1,
        "frozen_document_commit_is_ancestor":
            git_is_ancestor(root, FROZEN_DOCUMENT_COMMIT, source_commit),
        "frozen_document_blob_matches": (
            git_text(root, "rev-parse",
                 f"{FROZEN_DOCUMENT_COMMIT}:{FROZEN_DOCUMENT_PATH}")
            == FROZEN_DOCUMENT_BLOB
            and _sha256(frozen) == FROZEN_DOCUMENT_SHA256
        ),
        "historical_document_trees_match": all(
            git_text(root, "rev-parse",
                 f"{source_commit}:docs/research-loop/{index:04d}")
            == git_text(root, "rev-parse",
                f"{FROZEN_DOCUMENT_COMMIT}:docs/research-loop/{index:04d}")
            for index in range(1, 15)
        ),
        "consumer_architecture_guard": architecture["passed"] is True,
        "default_generator_remains_v1": default_generator["passed"] is True,
    }
    return {
        "source_commit": source_commit,
        "frozen_document": {
            "commit": FROZEN_DOCUMENT_COMMIT,
            "path": FROZEN_DOCUMENT_PATH,
            "git_blob": FROZEN_DOCUMENT_BLOB,
            "sha256": FROZEN_DOCUMENT_SHA256,
        },
        "architecture": architecture,
        "default_generator": default_generator,
        "implementation": {
            "commit_count": commit_count,
            "allowed_files": sorted(ALLOWED_CHANGES),
            "changed_files": sorted(changed),
        },
        "checks": {**checks, "passed": all(checks.values())},
    }
def _pytest_receipt(
    root, deadline: float
) -> tuple[dict[str, object], bytes]:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise RuntimeError("P80 wall-time budget exceeded before full pytest")
    environment = _validation_environment()
    try:
        result = subprocess.run(
            FULL_PYTEST_COMMAND,
            cwd=root,
            env=environment,
            capture_output=True,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("P80 wall-time budget exceeded during full pytest") from error
    output = result.stdout + result.stderr
    text = output.decode(errors="replace")
    summary = _pytest_summary(text)
    failed_ids = sorted(set(re.findall(
        r"^FAILED\s+(\S+)(?:\s+-\s+.*)?$", text, re.MULTILINE)))
    controls = []
    for name in PREREGISTERED_NEGATIVE_CONTROLS:
        node_id = (
            "tests/test_candidate_artifact_contract.py::"
            "test_preregistered_negative_control_fails_closed"
            f"[{name}]"
        )
        controls.append({
            "name": name,
            "status": "passed" if f"{node_id} PASSED" in text else "missing",
            "test_node_id": node_id,
        })
    passed = (
        result.returncode in (0, 1)
        and set(failed_ids) == ALLOWED_FULL_PYTEST_FAILURES
        and summary["passed"] > 0
        and all(
        value["status"] == "passed" for value in controls
        )
    )
    if not passed:
        raise RuntimeError("P80 full pytest or negative-control receipt differs")
    return ({
        "command": list(FULL_PYTEST_COMMAND),
        "returncode": result.returncode,
        "allowed_failure_ids": sorted(ALLOWED_FULL_PYTEST_FAILURES),
        "failed_ids": failed_ids,
        "negative_controls": controls,
        "negative_control_count": len(controls),
        "all_negative_controls_passed": True,
        "summary": summary,
        "output_path": "pytest-output.txt",
        "output_bytes": len(output),
        "output_sha256": _sha256(output),
        "execution": {
            "foreground_subprocess": True,
            "accelerator_environment": {
                "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
                "PYTORCH_ENABLE_MPS_FALLBACK":
                    environment["PYTORCH_ENABLE_MPS_FALLBACK"],
            },
            "validation_may_exercise_configured_accelerators": True,
        },
    }, output)
def _repository_gate_receipt(
    root, deadline: float
) -> tuple[dict[str, object], dict[str, bytes]]:
    environment = _validation_environment()
    records = []
    outputs = {}
    for index, command in enumerate(REPOSITORY_GATE_COMMANDS):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise RuntimeError("P80 wall-time budget exceeded during repository gates")
        result = subprocess.run(
            command, cwd=root, env=environment, capture_output=True,
            timeout=remaining,
        )
        output = result.stdout + result.stderr
        name = ("python-size", "architecture", "compileall", "diff-check")[index]
        output_path = f"gates/{name}.txt"
        outputs[output_path] = output
        records.append({
            "name": name, "command": list(command),
            "returncode": result.returncode, "output_path": output_path,
            "output_bytes": len(output), "output_sha256": _sha256(output),
        })
        if result.returncode != 0:
            raise RuntimeError(f"P80 repository gate failed: {name}")
    receipt = {
        "commands": records,
        "gate_count": len(records),
        "execution": {
            "foreground_subprocesses": True,
            "tmux_or_background_command_used": False,
            "accelerator_environment": {
                "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
                "PYTORCH_ENABLE_MPS_FALLBACK":
                    environment["PYTORCH_ENABLE_MPS_FALLBACK"],
            },
            "validation_may_exercise_configured_accelerators": True,
        },
    }
    return receipt, outputs
def _baseline_failure_receipt(
    root, deadline: float
) -> tuple[dict[str, object], bytes]:
    path = next(iter(ALLOWED_FULL_PYTEST_FAILURES))
    with tempfile.TemporaryDirectory(prefix="p80-frozen-") as temporary:
        worktree = repository_path(root, os.path.join(temporary, "repo"))
        result = subprocess.run(
            ("git", "worktree", "add", "--detach", worktree.as_posix(),
             FROZEN_DOCUMENT_COMMIT),
            cwd=root, capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError("P80 could not create frozen validation worktree")
        try:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise RuntimeError("P80 wall-time budget exceeded before frozen test")
            baseline = subprocess.run(
                (root / ".venv/bin/python", "-m", "pytest", "-q", path),
                cwd=worktree, env=_validation_environment(), capture_output=True,
                timeout=remaining,
            )
            output = baseline.stdout + baseline.stderr
            passed = baseline.returncode == 1 and (
                f"FAILED {path}".encode() in output
            )
            if not passed:
                raise RuntimeError("P80 allowed failure is not frozen-baseline")
            return ({
                "commit": FROZEN_DOCUMENT_COMMIT, "test_node_id": path,
                "returncode": baseline.returncode, "output_bytes": len(output),
                "output_path": "frozen-baseline-output.txt",
                "output_sha256": _sha256(output), "passed": True,
                "_output": output,
            }, output)
        finally:
            subprocess.run(
                ("git", "worktree", "remove", "--force", worktree.as_posix()),
                cwd=root, check=False, capture_output=True,
            )
def _validate_pytest_receipt(
    value: Mapping[str, object], output: bytes
) -> bool:
    controls = value.get("negative_controls")
    if not isinstance(controls, list):
        return False
    expected_nodes = {
        name: (
            "tests/test_candidate_artifact_contract.py::"
            "test_preregistered_negative_control_fails_closed"
            f"[{name}]"
        )
        for name in PREREGISTERED_NEGATIVE_CONTROLS
    }
    observed = {
        item.get("name"): item for item in controls
        if isinstance(item, Mapping)
    }
    try:
        parsed_summary = _pytest_summary(output.decode(errors="replace"))
    except RuntimeError:
        return False
    return (
        value.get("command") == list(FULL_PYTEST_COMMAND)
        and value.get("returncode") in (0, 1)
        and set(value.get("allowed_failure_ids", ()))
            == ALLOWED_FULL_PYTEST_FAILURES
        and set(value.get("failed_ids", ())) == ALLOWED_FULL_PYTEST_FAILURES
        and value.get("negative_control_count") == 25
        and len(controls) == 25
        and value.get("all_negative_controls_passed") is True
        and isinstance(value.get("summary"), Mapping)
        and value["summary"] == parsed_summary
        and parsed_summary.get("passed", 0) > 0
        and parsed_summary.get("failed", 0) == len(ALLOWED_FULL_PYTEST_FAILURES)
        and not any(parsed_summary.get(name, 0)
                    for name in ("error", "errors", "xpassed"))
        and set(observed) == set(expected_nodes)
        and all(
            observed[name].get("status") == "passed"
            and observed[name].get("test_node_id") == node
            and f"{node} PASSED".encode() in output
            for name, node in expected_nodes.items()
        )
        and value.get("output_path") == "pytest-output.txt"
        and value.get("output_bytes") == len(output)
        and value.get("output_sha256") == _sha256(output)
    )
def _validate_repository_gate_receipt(
    value: Mapping[str, object], outputs: Mapping[str, bytes]
) -> bool:
    records = value.get("commands")
    if not isinstance(records, list) or len(records) != len(REPOSITORY_GATE_COMMANDS):
        return False
    names = ("python-size", "architecture", "compileall", "diff-check")
    for record, command, name in zip(
        records, REPOSITORY_GATE_COMMANDS, names, strict=True
    ):
        if not isinstance(record, Mapping):
            return False
        path = record.get("output_path")
        output = outputs.get(path) if isinstance(path, str) else None
        if (
            record.get("name") != name
            or record.get("command") != list(command)
            or record.get("returncode") != 0
            or output is None
            or record.get("output_bytes") != len(output)
            or record.get("output_sha256") != _sha256(output)
        ):
            return False
    return (
        value.get("gate_count") == len(REPOSITORY_GATE_COMMANDS)
        and set(outputs) == {f"gates/{name}.txt" for name in names}
    )
def _validate_baseline_failure_receipt(
    value: Mapping[str, object], output: bytes) -> bool:
    path = next(iter(ALLOWED_FULL_PYTEST_FAILURES))
    return (
        value.get("commit") == FROZEN_DOCUMENT_COMMIT
        and value.get("test_node_id") == path
        and value.get("returncode") == 1
        and value.get("output_path") == "frozen-baseline-output.txt"
        and value.get("output_bytes") == len(output)
        and value.get("output_sha256") == _sha256(output)
        and f"FAILED {path}".encode() in output
        and value.get("passed") is True
    )
def _public_receipt(value: Mapping[str, object]) -> dict[str, object]:
    return {name: item for name, item in value.items() if not name.startswith("_")}
def _write_validated_output(
    *, output, started: float, source_commit: str,
    command: Sequence[str], counts: Mapping[str, int],
    first_receipt: bytes, repeated: bool,
    pytest_receipt: Mapping[str, object], pytest_output: bytes,
    repository_gates: Mapping[str, object], gate_output: Mapping[str, bytes],
    baseline_failure_receipt: Mapping[str, object],
    baseline_failure_output: bytes,
    provenance: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, bytes]]:
    public_pytest = _public_receipt(pytest_receipt)
    public_gates = _public_receipt(repository_gates)
    staging = output.with_name(output.name + ".tmp")
    try:
        memory = _memory_summary()
        report, artifacts = _output_documents(
            source_commit, command, counts, first_receipt, repeated,
            public_pytest, pytest_output, public_gates, gate_output,
            provenance, baseline_failure_receipt, baseline_failure_output,
            time.perf_counter() - started, memory,
        )
        _require_budget(
            time.perf_counter() - started,
            memory["peak_rss_upper_bound_bytes"], artifacts,
        )
        create_candidate_output(staging, artifacts)
        measured_elapsed = time.perf_counter() - started
        measured_memory = _memory_summary()
        shutil.rmtree(staging)
        report, artifacts = _output_documents(
            source_commit, command, counts, first_receipt, repeated,
            public_pytest, pytest_output, public_gates, gate_output,
            provenance, baseline_failure_receipt, baseline_failure_output,
            measured_elapsed, measured_memory,
        )
        create_candidate_output(staging, artifacts)
        _require_budget(
            time.perf_counter() - started,
            _memory_summary()["peak_rss_upper_bound_bytes"], artifacts,
        )
        if os.path.lexists(output): raise FileExistsError(output)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(staging.with_name(staging.name + ".tmp"), ignore_errors=True)
        raise
    return report, artifacts
def _output_documents(
    source_commit, command, counts, first_receipt, repeated,
    pytest_receipt, pytest_output, repository_gates, gate_output,
    provenance, baseline_failure_receipt, baseline_failure_output,
    elapsed, memory,
):
    report = _report(
        source_commit=source_commit, command=command, counts=counts,
        first_receipt=first_receipt, repeated=repeated,
        pytest_receipt=pytest_receipt, repository_gates=repository_gates,
        provenance=provenance, elapsed=elapsed, peak_rss=memory,
        pytest_output=pytest_output, gate_outputs=gate_output,
        baseline_failure_receipt={
            **baseline_failure_receipt,
            "_output": baseline_failure_output,
        },
    )
    artifacts = {
        **gate_output, "pytest-output.txt": pytest_output,
        "frozen-baseline-output.txt": baseline_failure_output,
        "receipt.json": first_receipt, "report.json": _json_bytes(report),
    }
    artifacts["manifest.json"] = _json_bytes(_manifest(
        source_commit=source_commit, command=command, artifacts=artifacts,
        report=report, provenance=provenance, pytest_receipt=pytest_receipt,
        repository_gates=repository_gates, elapsed=elapsed, peak_rss=memory,
    ))
    return report, artifacts
def _validation_environment() -> dict[str, str]:
    return {**os.environ, "PYTEST_ADDOPTS": "", "CUDA_VISIBLE_DEVICES": "",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
def _pytest_summary(output: str) -> dict[str, object]:
    match = re.search(r"^=+ (?P<body>.+) in (?P<seconds>[0-9.]+)s"
                      r"(?: \([^)]*\))? =+$",
                      output, re.MULTILINE)
    if match is None:
        raise RuntimeError("P80 full pytest summary is missing")
    counts = {name: int(count) for count, name in re.findall(
        r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|warnings?)",
        match.group("body"),
    )}
    return {**counts, "seconds": float(match.group("seconds"))}
def _require_provenance(value: Mapping[str, object]) -> None:
    checks = value["checks"]
    if checks["passed"] is not True:
        failed = sorted(
            name for name, passed in checks.items()
            if name != "passed" and not passed
        )
        raise RuntimeError(f"P80 provenance gate failed: {', '.join(failed)}")
def _require_frozen_paths(root, bank, output) -> None:
    if bank != repository_path(root, FORMAL_BANK): raise ValueError(
        "bank path differs from frozen path")
    if output != repository_path(root, FORMAL_OUTPUT): raise ValueError(
        "output path differs from frozen path")
def _require_budget(
    elapsed: float, peak_rss: int, artifacts: Mapping[str, bytes]
) -> None:
    if elapsed > MAX_WALL_SECONDS: raise RuntimeError("P80 wall-time budget exceeded")
    if peak_rss > MAX_RSS_BYTES: raise RuntimeError("P80 RSS budget exceeded")
    if sum(map(len, artifacts.values())) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("P80 artifact budget exceeded")
def _require_disk(output) -> None:
    parent = output.parent
    while not os.path.lexists(parent): parent = parent.parent
    if shutil.disk_usage(parent).free < MIN_DISK_FREE_BYTES: raise RuntimeError(
        "P80 disk-free guard failed")
def _resolve_argument(root, value):
    return repository_path(root, value)
def _source_commit(root) -> str:
    commit = git_text(root, "rev-parse", "HEAD")
    if len(commit) != 40: raise RuntimeError(
        "P80 requires a full Git source commit")
    return commit
def _command(arguments: argparse.Namespace) -> tuple[str, ...]:
    return (sys.executable, "-m", MODULE_NAME, "--bank",
            str(arguments.bank), "--output", str(arguments.output))
def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n").encode()
def _sha256(content: bytes) -> str: return hashlib.sha256(content).hexdigest()
def _rss_bytes(who: int) -> int:
    value = int(resource.getrusage(who).ru_maxrss); return value if sys.platform == "darwin" else value * 1024
def _memory_summary() -> dict[str, int]:
    self_rss = _rss_bytes(resource.RUSAGE_SELF); child_rss = _rss_bytes(resource.RUSAGE_CHILDREN)
    return {
        "self_peak_rss_bytes": self_rss, "children_peak_rss_bytes": child_rss,
        "peak_rss_upper_bound_bytes": self_rss + child_rss,
    }
def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv)); print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["decision"] != ACCEPTED_DECISION)
if __name__ == "__main__":
    raise SystemExit(main())
