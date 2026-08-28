from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hwr.apps import evaluate_experiment_contracts as app
from hwr.eval import experiment_contract_oracle as oracle


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / app.REGISTRY_PATH).read_text())
P79 = (
    ROOT
    / "runs/research-loop/0014/r0014-p79-candidate-bank-s20267901"
)
P83 = (
    ROOT
    / "runs/research-loop/0016/r0016-p83-selection-lineage-s20268301"
)


def _cohort() -> dict[str, object]:
    bank = json.loads((P79 / "bank.json").read_text())
    capsules = app._capsules_from_cohort(
        {
            "episodes": [
                {
                    "episode_id": row["planned_episode_id"],
                    "task_id": row["task_id"],
                    "cell_id": row["cell_id"],
                    "observation_latency_steps": int(
                        row["cell_id"].split("-obs-")[1].split("-action-")[0]
                    ),
                    "action_latency_steps": int(
                        row["cell_id"].rsplit("-action-", 1)[1]
                    ),
                    "candidate_count": row["candidate_set"]["candidate_count"],
                }
                for row in bank["episodes"]
            ]
        }
    )
    return oracle.build_cohort(capsules, bank, REGISTRY)


def test_frozen_registry_and_available_top_files_match_hashes() -> None:
    assert hashlib.sha256((ROOT / app.REGISTRY_PATH).read_bytes()).hexdigest() == (
        app.REGISTRY_SHA256
    )
    for source, directory in (("p79", P79), ("p83", P83)):
        for name, expected in REGISTRY["sources"][source]["files"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == (
                expected
            )


def test_p83_report_manifest_and_upstream_bindings_are_verified() -> None:
    report = json.loads((P83 / "report.json").read_text())
    manifest = json.loads((P83 / "manifest.json").read_text())

    app.validate_p83_identity(REGISTRY, report, manifest)
    broken = json.loads(json.dumps(manifest))
    broken["provenance"]["p50_top_files"]["capsules.json"]["sha256"] = "0" * 64
    with pytest.raises(oracle.ContractOracleError, match="p83_p50_binding"):
        app.validate_p83_identity(REGISTRY, report, broken)


def test_all_preregistered_controls_hit_their_semantic_guards() -> None:
    cohort = _cohort()
    analysis = oracle.analyze_registry(REGISTRY, cohort)
    history = {path: "a" * 40 for path in app.HISTORY_PATHS}
    provenance = {
        "frozen_document_blob": app.FROZEN_DOCUMENT_BLOB,
        "registry_blob": app.REGISTRY_BLOB,
        "registry": {"sha256": app.REGISTRY_SHA256},
        "allowed_changed_paths": sorted(app.ALLOWED_PATHS),
        "frozen_history_trees": history,
        "history_trees": dict(history),
    }

    boundary = app.run_boundary_controls(ROOT, provenance)
    controls = app.run_controls(REGISTRY, cohort, analysis, boundary)

    assert controls["control_count"] >= 20
    assert controls["pass_count"] == controls["control_count"]
    assert controls["passed"] is True
    observed = {row["control_id"]: row["observed"] for row in controls["controls"]}
    assert observed["C03-empty-nonempty-accounting"] == "denominator_partition"
    assert observed["C10-candidate-count"] == "candidate_count"
    assert observed["C13-exposure-unknown"] == "exposure_fields"
    assert observed["C18-frozen-design"] == "frozen_design_blob"
    assert observed["C18-registry-hash"] == "registry_hash"
    assert observed["C19-output-preexists"] == "output_or_staging_preexists"
    assert observed["C19-partial-write"] == "partial_write"
    assert observed["C20-solver-disagreement"] == "invalid_solver_disagreement"


def test_canonical_json_and_atomic_write_fail_closed(tmp_path: Path) -> None:
    first = app.canonical_json_bytes({"b": 2, "a": 1})
    second = app.canonical_json_bytes({"a": 1, "b": 2})
    target = tmp_path / "artifact.json"

    assert first == second == b'{"a":1,"b":2}'
    app._atomic_write(target, first)
    assert target.read_bytes() == first
    assert not target.with_name(target.name + ".tmp").exists()
    with pytest.raises(FileExistsError, match="artifact_preexists"):
        app._atomic_write(target, first)


def test_stable_reader_rejects_hash_drift_symlink_and_escape(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.json"
    payload.write_bytes(b'{"value":1}')
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    assert app._read_bound_json(payload, digest, tmp_path) == {"value": 1}
    with pytest.raises(oracle.ContractOracleError, match="input_hash"):
        app._read_bound_json(payload, "0" * 64, tmp_path)
    outside = tmp_path.parent / "outside-p87.json"
    outside.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(outside)
    try:
        with pytest.raises(oracle.ContractOracleError, match="input_file_type"):
            app._stable_read(link, tmp_path)
    finally:
        outside.unlink()


def test_formal_path_guard_rejects_nonformal_output(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(oracle.ContractOracleError, match="output_path"):
        app._require_output_available(ROOT, Path("runs/not-formal"), formal=True)


def test_report_records_resources_metrics_and_false_claim_flags() -> None:
    cohort = _cohort()
    analysis = oracle.analyze_registry(REGISTRY, cohort)
    controls = app.run_controls(REGISTRY, cohort, analysis, ())

    report = app.build_report(analysis, controls, 0.25, 1024)

    assert report["decision"] == "accepted as frozen experiment-contract oracle"
    assert report["checks"]["passed"] is True
    assert report["runtime"]["cpu_only"] is True
    assert all(report[name] is False for name in app.CLAIM_FLAGS)


def test_publisher_removes_final_after_post_rename_gate_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.staging"

    with pytest.raises(oracle.ContractOracleError, match="rss_budget"):
        app._publish_artifacts(
            staging,
            output,
            {"report.json": b"{}"},
            lambda: (_ for _ in ()).throw(
                oracle.ContractOracleError("rss_budget")
            ),
        )

    assert not output.exists()
    assert not staging.exists()


def test_publisher_removes_final_when_parent_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.staging"
    calls = 0
    original = app._fsync_directory

    def fail_parent(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("parent fsync failed")
        if calls > 2:
            return
        original(path)

    monkeypatch.setattr(app, "_fsync_directory", fail_parent)
    with pytest.raises(OSError, match="parent fsync failed"):
        app._publish_artifacts(
            staging, output, {"report.json": b"{}"}, lambda: {}
        )

    assert not output.exists()
    assert not staging.exists()


def test_finalize_samples_resources_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "report.json").write_bytes(b"{}")
    calls = []
    monkeypatch.setattr(
        app, "validate_source_stability", lambda root, provenance: calls.append("source")
    )
    monkeypatch.setattr(app.time, "perf_counter", lambda: 12.5)
    monkeypatch.setattr(app, "_peak_rss_bytes", lambda: 4096)

    result = app._finalize_run(tmp_path, {}, 10.0, output)

    assert calls == ["source"]
    assert result == {
        "wall_seconds": 2.5,
        "peak_rss_bytes": 4096,
        "artifact_bytes": 2,
    }


def test_provenance_rejects_history_and_allowed_scope_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "_require_clean_committed_source", lambda root: None)
    monkeypatch.setattr(
        app,
        "_file_identity",
        lambda root, path: {
            "path": path.name,
            "sha256": app.REGISTRY_SHA256,
            "bytes": 1,
        },
    )

    def fake_git(root, *arguments):
        if arguments[:2] == ("cat-file", "-t"):
            return "commit"
        if arguments[:2] == ("rev-parse", f"{app.FROZEN_COMMIT}:{app.FROZEN_DOCUMENT}"):
            return app.FROZEN_DOCUMENT_BLOB
        if arguments[:2] == ("rev-parse", f"{app.FROZEN_COMMIT}:{app.REGISTRY_PATH}"):
            return app.REGISTRY_BLOB
        if arguments[:2] == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments[:2] == ("diff", "--name-only"):
            return "src/hwr/policy/forbidden.py"
        if arguments[:1] == ("rev-parse",):
            return "b" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(app, "_git", fake_git)
    with pytest.raises(oracle.ContractOracleError, match="allowed_scope"):
        app.validate_provenance(ROOT, ROOT / app.REGISTRY_PATH)
