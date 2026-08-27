from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from hwr.apps import evaluate_candidate_artifact_contract as app
from hwr.eval import candidate_artifact_contract as contract
import test_candidate_artifact_contract as contract_test

ROOT = Path(__file__).resolve().parents[1]


def _accepted_provenance() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "frozen_document": {
            "commit": app.FROZEN_DOCUMENT_COMMIT,
            "path": app.FROZEN_DOCUMENT_PATH,
            "git_blob": app.FROZEN_DOCUMENT_BLOB,
            "sha256": app.FROZEN_DOCUMENT_SHA256,
        },
        "architecture": {
            "checks": {"consumer_only": True},
            "passed": True,
        },
        "default_generator": {
            "candidate_schema": contract.LEGACY_CANDIDATE_SCHEMA,
            "default_generator_uses_candidate_schema": True,
            "passed": True,
        },
        "implementation": {"commit_count": 1},
        "checks": {
            "workspace_clean": True,
            "source_commit_matches_head": True,
            "source_files_match_head": True,
            "implementation_scope_matches": True,
            "implementation_commit_count": True,
            "frozen_document_commit_is_ancestor": True,
            "frozen_document_blob_matches": True,
            "historical_document_trees_match": True,
            "consumer_architecture_guard": True,
            "default_generator_remains_v1": True,
            "passed": True,
        },
    }


def _pytest_receipt() -> tuple[dict[str, object], bytes]:
    failure = next(iter(app.ALLOWED_FULL_PYTEST_FAILURES))
    lines = [
        (
            "tests/test_candidate_artifact_contract.py::"
            "test_preregistered_negative_control_fails_closed"
            f"[{name}] PASSED"
        )
        for name in contract.PREREGISTERED_NEGATIVE_CONTROLS
    ]
    content = (
        "\n".join((
            *lines,
            f"FAILED {failure} - AssertionError",
            "================ 1 failed, 45 passed in 1.00s ================",
        )) + "\n"
    ).encode()
    return ({
        "command": list(app.FULL_PYTEST_COMMAND),
        "returncode": 1,
        "allowed_failure_ids": [failure],
        "failed_ids": [failure],
        "negative_controls": [
            {
                "name": name,
                "status": "passed",
                "test_node_id": (
                    "tests/test_candidate_artifact_contract.py::"
                    "test_preregistered_negative_control_fails_closed"
                    f"[{name}]"
                ),
            }
            for name in contract.PREREGISTERED_NEGATIVE_CONTROLS
        ],
        "negative_control_count": 25,
        "all_negative_controls_passed": True,
        "summary": {"passed": 45, "failed": 1, "seconds": 1.0},
        "output_path": "pytest-output.txt",
        "output_bytes": len(content),
        "output_sha256": hashlib.sha256(content).hexdigest(),
        "execution": {
            "foreground_subprocess": True,
            "accelerator_environment": {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            },
        },
    }, content)


def _gate_receipt() -> tuple[dict[str, object], dict[str, bytes]]:
    outputs = {
        f"gates/{name}.txt": f"{name} passed\n".encode()
        for name in ("python-size", "architecture", "compileall", "diff-check")
    }
    records = [
        {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "output_path": path,
            "output_bytes": len(outputs[path]),
            "output_sha256": hashlib.sha256(outputs[path]).hexdigest(),
        }
        for name, command, path in zip(
            ("python-size", "architecture", "compileall", "diff-check"),
            app.REPOSITORY_GATE_COMMANDS,
            outputs,
            strict=True,
        )
    ]
    return ({
        "commands": records,
        "gate_count": 4,
        "execution": {
            "foreground_subprocesses": True,
            "tmux_or_background_command_used": False,
            "accelerator_environment": {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            },
        },
    }, outputs)


def _baseline_failure_receipt() -> tuple[dict[str, object], bytes]:
    failure = next(iter(app.ALLOWED_FULL_PYTEST_FAILURES))
    output = f"FAILED {failure} - AssertionError\n".encode()
    return ({
        "commit": app.FROZEN_DOCUMENT_COMMIT,
        "test_node_id": failure,
        "returncode": 1,
        "output_path": "frozen-baseline-output.txt",
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "passed": True,
    }, output)


def _counts() -> contract.ValidationCounts:
    return contract.ValidationCounts(
        episode_count=24,
        capture_count=384,
        v2_candidate_count=24,
        legacy_candidate_count=24,
        capture_blob_count=768,
        p79_artifact_count=28,
        p50_artifact_count=795,
        p50_input_file_count=796,
    )


def test_formal_bank_validates_complete_frozen_ledger_twice() -> None:
    arguments = {
        "p79_v2_bank": Path(contract.FORMAL_TRUST_ANCHOR.artifact_root),
        "p50_legacy_source": Path(contract.FORMAL_TRUST_ANCHOR.legacy_root),
        "trust_anchor": contract.FORMAL_TRUST_ANCHOR,
    }
    first = contract.resolve_candidate_artifact(ROOT, **arguments)
    assert first.validation_counts == _counts()
    captures = tuple(item for episode in first.episodes
                     for item in episode.captures)
    assert len({item.composite_identity for item in captures}) == 384
    assert {item.schema for item in captures} == {contract.P50_CAPTURE_SCHEMA}
    assert all(not episode.candidate.selection_relation_validated
               for episode in first.episodes)
    receipt = first.canonical_receipt_bytes()
    assert contract.resolve_candidate_artifact(
        ROOT, **arguments
    ).canonical_receipt_bytes() == receipt


@pytest.mark.parametrize("legacy_inside_bank", (True, False))
def test_root_ancestor_overlap_is_rejected(
    tmp_path: Path, legacy_inside_bank: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank"; legacy = bank / "nested"
    if not legacy_inside_bank:
        legacy, bank = tmp_path / "legacy", tmp_path / "legacy" / "nested"
    bank.mkdir(parents=True); legacy.mkdir(parents=True, exist_ok=True)
    root = tmp_path
    anchor = replace(
        contract.FORMAL_TRUST_ANCHOR,
        artifact_root=bank.relative_to(root).as_posix(),
        legacy_root=legacy.relative_to(root).as_posix(),
    )
    monkeypatch.setattr(
        contract, "FORMAL_TRUST_ANCHOR_FINGERPRINT",
        contract.candidate_artifact_anchor_fingerprint(anchor),
    )
    with pytest.raises(contract.CandidateArtifactContractError) as raised:
        contract.resolve_candidate_artifact(
            root, p79_v2_bank=Path(anchor.artifact_root),
            p50_legacy_source=Path(anchor.legacy_root), trust_anchor=anchor,
        )
    assert raised.value.category == "root_overlap"


@pytest.mark.parametrize("kind", ("capture", "legacy_candidate"))
def test_v2_blob_role_cannot_satisfy_legacy_consumers(kind: str) -> None:
    content = b"same-bytes"
    identity = {
        "path": "same.bin", "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    blob = contract.BoundBlob(
        contract.RootRole.P79_V2_BANK, "same.bin", len(content),
        identity["sha256"], content,
    )
    if kind == "capture":
        with pytest.raises(contract.CandidateArtifactContractError) as raised:
            contract._input_blob(
                {f"{contract.FORMAL_TRUST_ANCHOR.legacy_root}/same.bin": blob},
                identity, contract.FORMAL_TRUST_ANCHOR,
            )
        assert raised.value.category == "capture_root"
    else:
        with pytest.raises(contract.CandidateArtifactContractError) as raised:
            contract._resolve_candidate(
                {**identity, "schema_version": contract.LEGACY_CANDIDATE_SCHEMA},
                {"same.bin": blob}, contract.RootRole.P50_LEGACY_SOURCE,
                contract.LEGACY_CANDIDATE_SCHEMA, contract.P50_SOURCE_COMMIT, (),
            )
        assert raised.value.category == "candidate_root"


@pytest.mark.parametrize(
    "sources",
    (
        (),
        contract.FORMAL_TRUST_ANCHOR.producer_sources[:2],
        (*contract.FORMAL_TRUST_ANCHOR.producer_sources,
         contract.GitBlobAnchor("extra.py", "0" * 40, "0" * 64, 0)),
    ),
)
def test_producer_source_anchor_set_must_equal_frozen_three(
    sources: tuple[contract.GitBlobAnchor, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = replace(contract.FORMAL_TRUST_ANCHOR, producer_sources=sources)
    monkeypatch.setattr(
        contract, "FORMAL_TRUST_ANCHOR_FINGERPRINT",
        contract.candidate_artifact_anchor_fingerprint(anchor),
    )
    with pytest.raises(contract.CandidateArtifactContractError) as raised:
        contract.resolve_candidate_artifact(
            ROOT, p79_v2_bank=Path(anchor.artifact_root),
            p50_legacy_source=Path(anchor.legacy_root), trust_anchor=anchor,
        )
    assert raised.value.category == "producer_source_set"


@pytest.mark.parametrize("field", ("git_blob", "sha256", "byte_count"))
def test_producer_source_identity_fields_are_all_bound(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = contract.FORMAL_TRUST_ANCHOR.producer_sources[0]
    changed = 0 if field == "byte_count" else "0" * (40 if field == "git_blob" else 64)
    anchor = replace(
        contract.FORMAL_TRUST_ANCHOR,
        producer_sources=(
            replace(source, **{field: changed}),
            *contract.FORMAL_TRUST_ANCHOR.producer_sources[1:],
        ),
    )
    monkeypatch.setattr(
        contract, "FORMAL_TRUST_ANCHOR_FINGERPRINT",
        contract.candidate_artifact_anchor_fingerprint(anchor),
    )
    with pytest.raises(contract.CandidateArtifactContractError) as raised:
        contract.resolve_candidate_artifact(
            ROOT, p79_v2_bank=Path(anchor.artifact_root),
            p50_legacy_source=Path(anchor.legacy_root), trust_anchor=anchor,
        )
    assert raised.value.category == "producer_source"


def test_joint_anchor_rewrite_fails_immutable_fingerprint() -> None:
    anchor = replace(contract.FORMAL_TRUST_ANCHOR, expected_captures=385)
    with pytest.raises(contract.CandidateArtifactContractError) as raised:
        contract.resolve_candidate_artifact(
            ROOT, p79_v2_bank=Path(anchor.artifact_root),
            p50_legacy_source=Path(anchor.legacy_root), trust_anchor=anchor,
        )
    assert raised.value.category == "anchor_fingerprint"


@pytest.mark.parametrize(
    ("field", "expected_category"),
    (
        ("acquisition_base_pose", "episode_identity"),
        ("capture_schema", "capture_contract"),
        ("capture_phase", "capture_contract"),
        ("legacy_generated_online", "selection_metadata"),
        ("regression_task_id", "regression_audit"),
        ("regression_audit", "regression_audit"),
    ),
)
def test_complete_field_guards_use_semantic_validators(
    tmp_path: Path, field: str, expected_category: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = contract_test._build_fixture(tmp_path / "repository")
    bank_path = fixture.repository / fixture.bank_root / "bank.json"
    capsules_path = fixture.repository / fixture.legacy_root / "capsules.json"
    regression_path = fixture.repository / fixture.bank_root / "regression.json"
    bank = contract_test._load(bank_path)
    capsules = contract_test._load(capsules_path)
    regression = contract_test._load(regression_path)
    if field == "acquisition_base_pose":
        capsules["episodes"][0]["acquisition_base_pose"][0] = 1.0
    elif field == "capture_schema":
        capsules["episodes"][0]["captures"][0]["schema_version"] = "unknown"
    elif field == "capture_phase":
        capsules["episodes"][0]["captures"][0]["acquisition_phase"] = "A1_panorama"
    elif field == "legacy_generated_online":
        capsules["episodes"][0]["candidate_set"]["generated_online"] = False
        bank["episodes"][0]["old_candidate_set"]["generated_online"] = False
    elif field == "regression_task_id":
        regression["records"][0]["task_id"] = "other/v1"
    else:
        regression["records"][0]["audit"]["checks"]["passed"] = False
    contract_test._store(capsules_path, capsules)
    contract_test._store(regression_path, regression)
    fixture = contract_test._refreshed_anchor(fixture)
    monkeypatch.setattr(
        contract, "FORMAL_TRUST_ANCHOR_FINGERPRINT", fixture.anchor_fingerprint
    )
    with pytest.raises(contract.CandidateArtifactContractError) as raised:
        contract.resolve_candidate_artifact(
            fixture.repository, p79_v2_bank=fixture.bank_root,
            p50_legacy_source=fixture.legacy_root, trust_anchor=fixture.anchor,
        )
    assert raised.value.category == expected_category


def test_architecture_guard_accepts_frozen_consumer_and_default_v1() -> None:
    evaluator = (ROOT / "src/hwr/eval/candidate_artifact_contract.py").read_text()
    application = (
        ROOT / "src/hwr/apps/evaluate_candidate_artifact_contract.py"
    ).read_text()
    assert app.audit_consumer_architecture(evaluator, application)["passed"] is True
    audit = app.audit_default_generator_schema(
        (ROOT / "src/hwr/eval/target_selection.py").read_text()
    )
    assert audit["candidate_schema"] == contract.LEGACY_CANDIDATE_SCHEMA
    assert audit["default_generator_uses_candidate_schema"] is True
    assert audit["default_generator_has_no_v2_or_ownership_reference"] is True
    assert audit["passed"] is True


def test_capture_cannot_resolve_from_v2_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = contract_test._build_fixture(tmp_path / "repository")
    anchor = replace(fixture.anchor, legacy_root=fixture.bank_root.as_posix())
    monkeypatch.setattr(
        contract, "FORMAL_TRUST_ANCHOR_FINGERPRINT",
        contract.candidate_artifact_anchor_fingerprint(anchor),
    )
    with pytest.raises(contract.CandidateArtifactContractError) as raised:
        contract.resolve_candidate_artifact(
            fixture.repository, p79_v2_bank=fixture.bank_root,
            p50_legacy_source=fixture.bank_root, trust_anchor=anchor,
        )
    assert raised.value.category == "root_overlap"


@pytest.mark.parametrize("mutation", (
    "\nimport subprocess as sp\n",
    "\nfrom subprocess import run as sr\n",
    "\ndef bypass():\n    g = getattr\n    return g(Path, 'read_bytes')\n",
    "\ndef bypass():\n    reader = Path('x').read_bytes\n    return reader()\n",
    "\nfrom hwr.apps import read_bound_blob as rb\n",
    "\ndef bypass():\n    return os.popen('cat x')\n",
    "\nimport importlib\npathlib = importlib.import_module('pathlib')\n",
    "\ndef bypass():\n    return subprocess.run(('cat', 'x'))\n",
    "\ndef bypass():\n    return subprocess.run(('sed', '-n', '1p', 'x'))\n",
    "\ndef bypass():\n    return subprocess.run(('awk', '{print}', 'x'))\n",
    "\ndef bypass():\n    return subprocess.run(('python', '-c', \"open('x').read()\"))\n",
    "\ndef bypass():\n    if False:\n        return 'hwr.p79-target-candidates/v2'\n",
))
def test_architecture_guard_rejects_indirect_readers(mutation: str) -> None:
    source = Path(app.__file__).read_text(encoding="utf-8") + mutation
    audit = app.audit_consumer_architecture(
        (ROOT / "src/hwr/eval/candidate_artifact_contract.py").read_text(),
        source,
    )
    assert audit["category"] == "architecture"
    assert audit["scope"] == "frozen evaluator/app source AST allowlist"
    assert audit["passed"] is False


def test_default_generator_rejects_dead_reference_and_actual_v2() -> None:
    source = (ROOT / "src/hwr/eval/target_selection.py").read_text()
    source = source.replace(
        "    document = {\n        \"schema_version\": CANDIDATE_SCHEMA,",
        "    if False:\n        unused = CANDIDATE_SCHEMA\n"
        "    document = {\n        \"schema_version\": CANDIDATE_SCHEMA_V2,",
        1,
    ).replace(
        'CANDIDATE_SCHEMA = "hwr.p41-target-candidates/v1"',
        'CANDIDATE_SCHEMA = "hwr.p41-target-candidates/v1"\n'
        'CANDIDATE_SCHEMA_V2 = "hwr.p79-target-candidates/v2"',
        1,
    )
    audit = app.audit_default_generator_schema(source)
    assert audit["checks"]["canonical_document_schema_uses_candidate_schema"] is False
    assert audit["checks"]["generator_has_no_v2_or_ownership_reference"] is False
    assert audit["passed"] is False
    original = (ROOT / "src/hwr/eval/target_selection.py").read_text()
    dead_literal = original.replace(
        "    origin = _pose(acquisition_base_pose)",
        "    dead = 'hwr.p79-target-candidates/v2'\n"
        "    origin = _pose(acquisition_base_pose)",
        1,
    )
    assert app.audit_default_generator_schema(dead_literal)["passed"] is False


def test_run_writes_atomic_auditable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank"
    legacy = tmp_path / "legacy"
    output = tmp_path / "output"
    bank.mkdir()
    legacy.mkdir()
    monkeypatch.setattr(app, "FORMAL_BANK", bank)
    monkeypatch.setattr(app, "FORMAL_LEGACY_SOURCE", legacy)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(app, "_require_disk", lambda path: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        app,
        "_provenance",
        lambda root, source_commit: _accepted_provenance(),
    )
    monkeypatch.setattr(app, "_pytest_receipt", lambda root, deadline: _pytest_receipt())
    monkeypatch.setattr(
        app, "_repository_gate_receipt", lambda root, deadline: _gate_receipt()
    )
    monkeypatch.setattr(
        app,
        "_baseline_failure_receipt",
        lambda root, deadline: _baseline_failure_receipt(),
    )
    monkeypatch.setattr(
        app,
        "_memory_summary",
        lambda: {
            "self_peak_rss_bytes": 100,
            "children_peak_rss_bytes": 200,
            "peak_rss_upper_bound_bytes": 300,
        },
    )
    calls = []

    class Envelope:
        validation_counts = _counts()

        @staticmethod
        def canonical_receipt_bytes() -> bytes:
            return b"sealed receipt"

    def resolve(
        repository,
        *,
        p79_v2_bank,
        p50_legacy_source,
        trust_anchor,
    ):
        calls.append((
            repository,
            p79_v2_bank,
            p50_legacy_source,
            trust_anchor,
        ))
        return Envelope()

    monkeypatch.setattr(app, "resolve_candidate_artifact", resolve)

    result = app.run(SimpleNamespace(bank=bank, output=output))

    assert len(calls) == 2
    assert result["decision"] == app.INCONCLUSIVE_DECISION
    assert not output.with_name(output.name + ".tmp").exists()
    assert sorted(path.name for path in output.iterdir()) == [
        "frozen-baseline-output.txt",
        "gates",
        "manifest.json",
        "pytest-output.txt",
        "receipt.json",
        "report.json",
    ]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == app.REPORT_SCHEMA
    assert report["decision"] == app.INCONCLUSIVE_DECISION
    assert report["validation_counts"] == _counts().as_dict()
    assert report["receipt"]["repeated_bit_identical"] is True
    assert report["receipt"]["sha256"] == hashlib.sha256(
        b"sealed receipt"
    ).hexdigest()
    assert (output / "receipt.json").read_bytes() == b"sealed receipt"
    assert report["pytest_receipt"]["negative_control_count"] == 25
    assert report["repository_gate_receipt"]["gate_count"] == 4
    assert report["resource_usage"]["peak_rss_upper_bound_bytes"] == 300
    assert report["selection_metadata_scope"].endswith(
        "is bound metadata and is not independently validated"
    )
    assert report["checks"]["technical_validation_passed"] is True
    assert report["checks"]["independent_v2_score_selection_evidence"] is False
    assert report["checks"]["passed"] is False
    assert report["execution_boundaries"]["validation_may_exercise_configured_accelerators"] is True
    assert all(report[name] is False for name in app.CLAIM_FLAGS)
    assert manifest["status"] == "complete"
    assert manifest["source_commit"] == "a" * 40
    assert manifest["provenance"]["checks"]["passed"] is True
    assert manifest["trust_anchor_fingerprint"] == (
        contract.FORMAL_TRUST_ANCHOR_FINGERPRINT
    )
    assert manifest["budgets"] == {
        "maximum_artifact_bytes": 5 * 1024**2,
        "maximum_peak_rss_bytes": 2 * 1024**3,
        "maximum_wall_seconds": 5 * 60,
        "minimum_disk_free_bytes": 20 * 1024**3,
    }
    assert manifest["artifacts"]["pytest-output.txt"]["sha256"] == (
        hashlib.sha256((output / "pytest-output.txt").read_bytes()).hexdigest()
    )


def test_runner_rejects_wrong_paths_and_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = tmp_path / "bank"
    legacy = tmp_path / "legacy"
    output = tmp_path / "output"
    bank.mkdir()
    legacy.mkdir()
    monkeypatch.setattr(app, "FORMAL_BANK", bank)
    monkeypatch.setattr(app, "FORMAL_LEGACY_SOURCE", legacy)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)

    with pytest.raises(ValueError, match="bank path"):
        app.run(SimpleNamespace(bank=tmp_path / "wrong", output=output))
    output.mkdir()
    with pytest.raises(FileExistsError):
        app.run(SimpleNamespace(bank=bank, output=output))


def test_provenance_and_resource_guards_fail_closed() -> None:
    for failed in (
        "workspace_clean",
        "source_files_match_head",
        "implementation_scope_matches",
        "implementation_commit_count",
        "frozen_document_blob_matches",
        "historical_document_trees_match",
        "consumer_architecture_guard",
        "default_generator_remains_v1",
    ):
        provenance = _accepted_provenance()
        provenance["checks"][failed] = False
        provenance["checks"]["passed"] = False
        with pytest.raises(RuntimeError, match=failed):
            app._require_provenance(provenance)
    with pytest.raises(RuntimeError, match="wall-time"):
        app._require_budget(app.MAX_WALL_SECONDS + 1, 1, {})
    with pytest.raises(RuntimeError, match="RSS"):
        app._require_budget(1, app.MAX_RSS_BYTES + 1, {})
    with pytest.raises(RuntimeError, match="artifact"):
        app._require_budget(
            1,
            1,
            {"report.json": b"x" * (app.MAX_ARTIFACT_BYTES + 1)},
        )


def test_pytest_receipt_requires_all_named_negative_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lines = [
        (
            "tests/test_candidate_artifact_contract.py::"
            "test_preregistered_negative_control_fails_closed"
            f"[{name}] PASSED"
        )
        for name in contract.PREREGISTERED_NEGATIVE_CONTROLS
    ]
    failure = next(iter(app.ALLOWED_FULL_PYTEST_FAILURES))
    output = ("\n".join((
        *lines, f"FAILED {failure} - AssertionError",
        "================ 1 failed, 45 passed in 1.00s ================",
    )) + "\n").encode()
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=output,
            stderr=b"",
            returncode=1,
        ),
    )

    receipt, captured = app._pytest_receipt(
        tmp_path, time.perf_counter() + 1000.0
    )

    assert captured == output
    assert receipt["negative_control_count"] == 25
    assert receipt["all_negative_controls_passed"] is True
    assert receipt["summary"]["passed"] == 45
    assert app._validate_pytest_receipt(receipt, captured) is True

    missing = output.replace(
        (
            "tests/test_candidate_artifact_contract.py::"
            "test_preregistered_negative_control_fails_closed"
            "[trust_anchor_drift] PASSED\n"
        ).encode(),
        b"",
    )
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=missing,
            stderr=b"",
            returncode=1,
        ),
    )
    with pytest.raises(RuntimeError, match="negative-control"):
        app._pytest_receipt(tmp_path, time.perf_counter() + 1000.0)


def test_pytest_receipt_accepts_real_vv_failure_summary_without_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt, old_output = _pytest_receipt()
    failure = next(iter(app.ALLOWED_FULL_PYTEST_FAILURES))
    output = old_output.replace(
        f"FAILED {failure} - AssertionError".encode(),
        f"FAILED {failure}".encode(),
    ).replace(
        b"1 failed, 45 passed in 1.00s",
        b"1 failed, 1177 passed, 11 skipped in 254.45s",
    )
    monkeypatch.setattr(
        app.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
            stdout=output, stderr=b"", returncode=1)
    )

    parsed, captured = app._pytest_receipt(
        tmp_path, time.perf_counter() + 1000.0
    )

    assert parsed["failed_ids"] == [failure]
    assert parsed["summary"] == {
        "failed": 1, "passed": 1177, "skipped": 11, "seconds": 254.45
    }
    assert app._validate_pytest_receipt(parsed, captured) is True
    assert app._validate_pytest_receipt(receipt, old_output) is True
    wrong = output.replace(f"FAILED {failure}".encode(), b"FAILED tests/test_other.py::test_new")
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=wrong, stderr=b"", returncode=1))
    with pytest.raises(RuntimeError, match="differs"):
        app._pytest_receipt(tmp_path, time.perf_counter() + 1000.0)


def test_receipt_validators_reject_boolean_only_or_hash_drift() -> None:
    receipt, output = _pytest_receipt()
    status_drift = {**receipt, "negative_controls": [
        *receipt["negative_controls"][:-1],
        {**receipt["negative_controls"][-1], "status": "missing"},
    ]}
    node_drift = {**receipt, "negative_controls": [
        *receipt["negative_controls"][:-1],
        {**receipt["negative_controls"][-1], "test_node_id": "wrong"},
    ]}
    for mutation in (
        {**receipt, "negative_controls": []},
        {**receipt, "negative_control_count": 24},
        status_drift,
        node_drift,
        {**receipt, "output_bytes": len(output) + 1},
        {**receipt, "output_sha256": "0" * 64},
    ):
        assert app._validate_pytest_receipt(mutation, output) is False
    gates, outputs = _gate_receipt()
    assert app._validate_repository_gate_receipt(gates, outputs) is True
    changed = dict(gates)
    changed["commands"] = [dict(value) for value in gates["commands"]]
    changed["commands"][0]["output_sha256"] = "0" * 64
    assert app._validate_repository_gate_receipt(changed, outputs) is False


def test_frozen_baseline_failure_is_reproduced_and_hash_bound() -> None:
    receipt, output = app._baseline_failure_receipt(
        ROOT, time.perf_counter() + 30.0
    )

    assert app._validate_baseline_failure_receipt(receipt, output) is True
    assert app._validate_baseline_failure_receipt(
        {**receipt, "output_sha256": "0" * 64}, output
    ) is False


def test_staging_is_removed_when_final_budget_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    pytest_receipt, pytest_output = _pytest_receipt()
    gates, outputs = _gate_receipt()
    baseline, baseline_output = _baseline_failure_receipt()
    calls = iter((
        {
            "self_peak_rss_bytes": 1,
            "children_peak_rss_bytes": 1,
            "peak_rss_upper_bound_bytes": 2,
        },
        {
            "self_peak_rss_bytes": 1,
            "children_peak_rss_bytes": 1,
            "peak_rss_upper_bound_bytes": 2,
        },
        {
            "self_peak_rss_bytes": 1,
            "children_peak_rss_bytes": app.MAX_RSS_BYTES,
            "peak_rss_upper_bound_bytes": app.MAX_RSS_BYTES + 1,
        },
    ))
    monkeypatch.setattr(app, "_memory_summary", lambda: next(calls))

    with pytest.raises(RuntimeError, match="RSS"):
        app._write_validated_output(
            output=output,
            started=time.perf_counter(),
            source_commit="a" * 40,
            command=("p80",),
            counts=_counts().as_dict(),
            first_receipt=b"receipt",
            repeated=True,
            pytest_receipt=pytest_receipt,
            pytest_output=pytest_output,
            repository_gates=gates,
            gate_output=outputs,
            baseline_failure_receipt=baseline,
            baseline_failure_output=baseline_output,
            provenance=_accepted_provenance(),
        )

    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()


@pytest.mark.parametrize(
    ("decision", "expected"),
    (
        (contract.ACCEPTED_DECISION, 0),
        ("rejected", 1),
        ("inconclusive_artifact_contract_insufficient", 1),
        ("invalid", 1),
    ),
)
def test_main_exit_code_reflects_decision(
    decision: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        app,
        "run",
        lambda arguments: {"decision": decision, "output": "unused"},
    )

    assert app.main(("--bank", "x", "--output", "y")) == expected
    assert json.loads(capsys.readouterr().out)["decision"] == decision
