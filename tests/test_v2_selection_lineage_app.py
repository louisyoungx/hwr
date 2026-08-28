from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
from hwr.apps import evaluate_v2_selection_lineage as app
from hwr.eval import candidate_mask_ownership as production_v2
from hwr.eval import target_selection
from hwr.eval.target_selection import PolicyVisibleInput, serialize_policy_input

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/evaluate_v2_selection_lineage_oracle.py"
def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p83_app_worker", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
oracle = _load_worker()
def _input(timestamp: int, sequence: int) -> PolicyVisibleInput:
    depth = np.ones((192, 256), dtype="<f4")
    valid = np.zeros((192, 256), dtype=np.bool_)
    for row, column, height in ((96, 60, 0.75), (96, 200, 0.8)):
        valid[row - 10 : row + 11, column - 10 : column + 11] = True
        depth[row - 2 : row + 3, column - 2 : column + 3] = height
    proprioception = np.zeros(37, dtype="<f8")
    proprioception[24:26] = 0.25
    return PolicyVisibleInput(
        observation_timestamp_ns=timestamp,
        sequence_id=sequence,
        phase_index=1,
        phase_step=0,
        policy_rng_seed=17,
        safety_state="ok",
        head_rgb_uint8=np.zeros((192, 256, 3), dtype=np.uint8),
        head_depth_m=depth,
        head_depth_valid=valid,
        head_camera_intrinsics=np.asarray(
            (80.0, 80.0, 127.5, 95.5), dtype="<f8"
        ),
        robot_from_head_camera=np.eye(4, dtype="<f8"),
        proprioception=proprioception,
        executed_action_history=np.zeros((4, 16), dtype="<f8"),
        history_available=np.asarray(
            (False, False, False, True), dtype=np.bool_
        ),
    )
def _descriptor(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
def _fixture_roots(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    p50 = tmp_path / "p50"
    p79 = tmp_path / "p79"
    identity = "fixture-episode"
    values = (_input(1, 1), _input(2, 2), _input(3, 3))
    captures = []
    manifest_inputs = []
    payloads = []
    for ordinal, value in enumerate(values):
        payload = serialize_policy_input(value)
        visible = production_v2.candidate_visible_bytes(value)
        policy_path = f"blobs/{identity}/capture-{ordinal:02d}-policy.bin"
        visible_path = (
            f"blobs/{identity}/capture-{ordinal:02d}-candidate-visible.bin"
        )
        (p50 / policy_path).parent.mkdir(parents=True, exist_ok=True)
        (p50 / policy_path).write_bytes(payload)
        (p50 / visible_path).write_bytes(visible)
        policy = _descriptor(policy_path, payload)
        candidate_visible = _descriptor(visible_path, visible)
        captures.append(
            {
                "schema_version": "hwr.p50-acquisition-capture/v1",
                "capture_ordinal": ordinal,
                "final_input": ordinal == len(values) - 1,
                "observation_timestamp_ns": value.observation_timestamp_ns,
                "sequence_id": value.sequence_id,
                "policy_input": policy,
                "candidate_visible_input": candidate_visible,
            }
        )
        for descriptor in (policy, candidate_visible):
            manifest_inputs.append(
                {
                    **descriptor,
                    "path": (app.FORMAL_P50 / descriptor["path"]).as_posix(),
                }
            )
        payloads.append(payload)
    episode = {
        "schema_version": "hwr.p50-acquisition-capsule/v1",
        "planned_episode_id": identity,
        "task_id": "fixture-task",
        "cell_id": "fixture-cell",
        "replicate_ordinal": 0,
        "acquisition_base_pose": [0.0, 0.0, 0.0],
        "captures": captures,
    }
    capsules = {
        "schema_version": app.P50_SCHEMA,
        "capsule_count": 1,
        "episodes": [episode],
    }
    (p50 / "capsules.json").write_text(json.dumps(capsules), encoding="utf-8")
    candidate_set = production_v2.generate_candidate_set_v2(
        payloads[:-1],
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=payloads[-1],
    )
    scores = target_selection.candidate_scores(
        candidate_set,
        values[-1].base_pose,
        acquisition_base_pose=(0.0, 0.0, 0.0),
    )
    selected = target_selection.select_candidate_index(
        candidate_set,
        values[-1].base_pose,
        acquisition_base_pose=(0.0, 0.0, 0.0),
    )
    candidate_path = f"blobs/{identity}/candidate-set.json"
    (p79 / candidate_path).parent.mkdir(parents=True, exist_ok=True)
    (p79 / candidate_path).write_bytes(candidate_set.canonical_bytes)
    committed = {
        "schema_version": app.CANDIDATE_SCHEMA,
        "path": candidate_path,
        "sha256": candidate_set.candidate_set_sha256,
        "bytes": len(candidate_set.canonical_bytes),
        "candidate_count": len(candidate_set.candidates),
        "score_bytes_sha256": oracle.score_hash(scores),
        "selected_index": selected,
        "selected_canonical_identity": (
            None if selected < 0 else hashlib.sha256(json.dumps(
                list(candidate_set.candidates[selected].canonical_record()),
                separators=(",", ":")).encode("ascii")).hexdigest()),
    }
    bank = {
        "schema_version": app.P79_BANK_SCHEMA,
        "proposal_id": "R0001-P79-E1",
        "episodes": [
            {
                "planned_episode_id": identity,
                "task_id": "fixture-task",
                "candidate_set": committed,
            }
        ],
    }
    (p79 / "bank.json").write_text(json.dumps(bank), encoding="utf-8")
    artifacts = {}
    for path in (p79 / "bank.json", p79 / candidate_path):
        content = path.read_bytes()
        artifacts[path.relative_to(p79).as_posix()] = {
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()
        }
    manifest = {
        "schema_version": app.P79_MANIFEST_SCHEMA,
        "provenance": {"input_files": manifest_inputs},
        "artifacts": artifacts,
    }
    (p79 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return p50, p79, capsules, manifest
def _patch_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "EXPECTED_INPUT_FILES", 6)
    monkeypatch.setattr(app, "EXPECTED_EPISODES", 1)
    monkeypatch.setattr(app, "EXPECTED_CAPTURES", 3)
    monkeypatch.setattr(app, "EXPECTED_CANDIDATES", 2)
    monkeypatch.setattr(app, "EXPECTED_NONEMPTY", 1)
    monkeypatch.setattr(app, "EXPECTED_EMPTY", 0)
    monkeypatch.setattr(app, "EXPECTED_SINGLETON", 0)
    monkeypatch.setattr(app, "EXPECTED_MULTI", 1)
def _provenance(
    p50: Path | None = None, p79: Path | None = None
) -> dict[str, object]:
    content = WORKER.read_bytes()
    app_path = ROOT / app.APP_PATH
    result = {
        "source_commit": "a" * 40,
        "worker_source": {
            "path": app.WORKER_PATH.as_posix(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        "app_source": app._file_identity(ROOT, app_path),
        "worker_source_audit": app.audit_worker_source(
            content.decode("utf-8")
        ),
        "worker_similarity_audit": {"passed": True},
        "input_file_match_count": 6,
        "p50_top_files": {},
        "p79_manifest": {},
        "checks": {"passed": True},
    }
    if p50 is not None and p79 is not None:
        result["p50_input_files"] = app._directory_identities(ROOT, p50)
        result["p79_pre_reveal_stats"] = app._directory_stats(ROOT, p79)
        result["p79_expected_files"] = app._public_identities(
            app._directory_identities(ROOT, p79))
        result["p50_top_files"] = {
            "capsules.json": app._file_identity(ROOT, p50 / "capsules.json")}
        result["p79_manifest"] = app._file_identity(
            ROOT, p79 / "manifest.json")
    return result
def test_blind_plan_is_sanitized_and_uses_manifest_bound_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p50, _, capsules, manifest = _fixture_roots(tmp_path)
    del p50
    _patch_counts(monkeypatch)

    plan = app.build_blind_plan(capsules, manifest)
    serialized = json.dumps(plan, sort_keys=True)

    assert plan["input_file_count"] == 6
    assert "candidate_set" not in serialized
    assert "score_bytes_sha256" not in serialized
    assert "selected_index" not in serialized
    assert "selected_canonical_identity" not in serialized
    assert "environment_seed" not in serialized
    assert plan["episodes"][0]["captures"][-1]["final_input"] is True
@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        (
            lambda value: value.update(
                schema_version="hwr.p79-candidate-bank/v1"
            ),
            "p50_schema",
        ),
        (
            lambda value: value["episodes"].append(
                copy.deepcopy(value["episodes"][0])
            ),
            "episode_count",
        ),
        (
            lambda value: value["episodes"][0]["captures"][1].update(
                capture_ordinal=0
            ),
            "capture_order",
        ),
        (
            lambda value: value["episodes"][0]["captures"][-1].update(
                final_input=False
            ),
            "final_input",
        ),
        (
            lambda value: value["episodes"][0]["captures"][0][
                "policy_input"
            ].update(path="../escape.bin"),
            "path_escape",
        ),
        (
            lambda value: value["episodes"][0]["captures"][0][
                "policy_input"
            ].update(sha256="0" * 64),
            "p79_input_commitment",
        ),
    ),
)
def test_plan_builder_mutations_fail_in_specific_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    category: str,
) -> None:
    _, _, capsules, manifest = _fixture_roots(tmp_path)
    _patch_counts(monkeypatch)
    mutation(capsules)

    with pytest.raises(app.LineageContractError, match=category):
        app.build_blind_plan(capsules, manifest)
def test_worker_source_and_invocation_guards_reject_production_access(
    tmp_path: Path,
) -> None:
    current = app.audit_worker_source(WORKER.read_text(encoding="utf-8"))
    imported = app.audit_worker_source(
        "from hwr.eval.target_selection import candidate_scores\n"
    )
    called = app.audit_worker_source(
        "def candidate_scores(): pass\ncandidate_scores()\n"
    )
    dynamic = app.audit_worker_source(
        "import importlib\nimportlib.import_module('hwr.eval.target_selection')\n"
    )
    direct_read = app.audit_worker_source(
        "from pathlib import Path\nPath('secret').read_bytes()\n"
    )
    builtin_read = app.audit_worker_source(
        "open('secret', 'rb').read()\n"
    )
    recovered = app.audit_worker_source(
        "from pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n"
    )

    assert current["passed"] is True
    assert imported["passed"] is False
    assert called["passed"] is False
    assert dynamic["passed"] is False
    assert direct_read["passed"] is False
    assert builtin_read["passed"] is False
    assert recovered["passed"] is False
    staging = tmp_path / "stage"; staging.mkdir()
    provenance = _provenance()
    staged_worker, staged_source = app.materialize_worker(ROOT, staging, provenance)
    assert staged_worker != WORKER
    assert staged_worker.stat().st_mode & 0o777 == 0o400
    assert app.validate_source_stability(
        ROOT, staged_worker, provenance, staged_source)["checks"]["passed"] is True
    staged_worker.chmod(0o600); staged_worker.write_bytes(b"tampered")
    with pytest.raises(app.LineageContractError, match="source_changed_during_run"):
        app.validate_source_stability(ROOT, staged_worker, provenance, staged_source)
    with pytest.raises(app.LineageContractError, match="worker_p79_exposure"):
        app.audit_worker_invocation(
            ("python", "-I", "-S", "worker.py", "--input", "/secret/p79"),
            {"PYTHONPATH": "", "HWR_P83_ISOLATED": "1"},
            tmp_path,
            ("/secret/p79",),
        )
    with pytest.raises(app.LineageContractError, match="worker_isolation"):
        app.audit_worker_invocation(
            ("python", "worker.py"),
            {"PYTHONPATH": "src"},
            tmp_path,
            (),
        )
def test_receipt_requires_actual_audit_and_complete_atomic_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p50, _, capsules, manifest = _fixture_roots(tmp_path)
    _patch_counts(monkeypatch)
    plan = app.build_blind_plan(capsules, manifest)
    receipt = oracle.rebuild_plan(
        plan,
        input_root=p50,
        plan_sha256=hashlib.sha256(app._json_bytes(plan)).hexdigest(),
        worker_source_sha256=_provenance()["worker_source"]["sha256"],
    )
    receipt["read_audit"] = {
        "trust_role": "auxiliary",
        "audited_open_count": 7,
        "expected_open_count": 7,
        "path_sequence_sha256": "a" * 64,
        "plan_fd_identity": {
            "bytes": len(app._json_bytes(plan)),
            "sha256": hashlib.sha256(app._json_bytes(plan)).hexdigest(),
            "fd_identity": {
                "device": 1, "inode": 1, "size": len(app._json_bytes(plan)),
            },
        },
    }
    worker_identity = _provenance()["worker_source"]
    worker_run = {
        "returncode": 0,
        "receipt_present_after_exit": True,
        "receipt_temporary_absent_after_exit": True,
        "expected_read_audit_sha256": "a" * 64,
        "worker_source_before": worker_identity,
        "worker_source_after": worker_identity,
    }
    app.validate_receipt(receipt, plan, _provenance(), worker_run)

    early = copy.deepcopy(receipt)
    early["read_audit"]["audited_open_count"] += 1
    with pytest.raises(app.LineageContractError, match="read_audit_count"):
        app.validate_receipt(early, plan, _provenance(), worker_run)
    incomplete = {**worker_run, "receipt_temporary_absent_after_exit": False}
    with pytest.raises(app.LineageContractError, match="atomic_complete"):
        app.validate_receipt(receipt, plan, _provenance(), incomplete)
    changed_ledger = copy.deepcopy(receipt)
    changed_ledger["read_ledger"][0]["sha256"] = "0" * 64
    with pytest.raises(app.LineageContractError, match="read_ledger"):
        app.validate_receipt(changed_ledger, plan, _provenance(), worker_run)
def test_materialized_blind_root_contains_only_plan_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p50, _, capsules, manifest = _fixture_roots(tmp_path)
    _patch_counts(monkeypatch)
    plan = app.build_blind_plan(capsules, manifest)
    blind = tmp_path / "blind"
    blind.mkdir()
    receipt = app.materialize_blind_inputs(p50, blind, plan)
    app._atomic_write(blind / "blind-plan.json", app._json_bytes(plan))
    app._validate_blind_root(blind, plan)

    assert receipt["file_count"] == 6
    assert len(list(blind.rglob("*.bin"))) == 6
    assert not list(blind.rglob("candidate-set.json"))
    assert not (blind / "capsules.json").exists()
    assert not (blind / "manifest.json").exists()
def test_formal_scale_blind_root_has_24_episodes_384_captures_768_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    p50 = tmp_path / "p50"; p50.mkdir()
    episodes, inputs = [], []
    for episode_ordinal in range(24):
        captures = []
        for capture_ordinal in range(16):
            prefix = Path("blobs") / f"e-{episode_ordinal:02d}"
            names = (prefix / f"{capture_ordinal:02d}-policy.bin",
                     prefix / f"{capture_ordinal:02d}-visible.bin")
            values = (f"policy-{episode_ordinal}-{capture_ordinal}".encode(),
                      f"visible-{episode_ordinal}-{capture_ordinal}".encode())
            descriptors = []
            for name, content in zip(names, values):
                path = p50 / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content); descriptor = _descriptor(name.as_posix(), content)
                descriptors.append(descriptor)
                inputs.append({**descriptor, "path": (app.FORMAL_P50 / name).as_posix()})
            captures.append({"schema_version": "hwr.p50-acquisition-capture/v1",
                "capture_ordinal": capture_ordinal, "final_input": capture_ordinal == 15,
                "observation_timestamp_ns": episode_ordinal * 100 + capture_ordinal,
                "sequence_id": capture_ordinal, "policy_input": descriptors[0],
                "candidate_visible_input": descriptors[1]})
        episodes.append({"schema_version": "hwr.p50-acquisition-capsule/v1",
            "planned_episode_id": f"episode-{episode_ordinal:02d}", "task_id": "task",
            "cell_id": "cell", "replicate_ordinal": episode_ordinal,
            "acquisition_base_pose": [0.0, 0.0, 0.0], "captures": captures})
    capsules = {"schema_version": app.P50_SCHEMA, "capsule_count": 24,
                "episodes": episodes}
    manifest = {"schema_version": app.P79_MANIFEST_SCHEMA,
                "provenance": {"input_files": inputs}}
    for ordinal in range(28):
        content = f"unused-{ordinal}".encode()
        name = Path("unused") / f"{ordinal:02d}.bin"
        inputs.append({**_descriptor(name.as_posix(), content),
                       "path": (app.FORMAL_P50 / name).as_posix()})
    plan = app.build_blind_plan(capsules, manifest)
    blind = tmp_path / "blind"; blind.mkdir()
    result = app.materialize_blind_inputs(p50, blind, plan)
    app._atomic_write(blind / "blind-plan.json", app._json_bytes(plan))
    app._validate_blind_root(blind, plan)
    assert (plan["episode_count"], plan["capture_count"],
            plan["input_file_count"], result["file_count"]) == (24, 384, 768, 768)
    assert len([path for path in blind.rglob("*") if path.is_file()]) == 769
    seen = []
    def reject(*args, **kwargs):
        mutated = json.loads(args[2].read_text()); seen.append(
            (mutated["episode_count"], mutated["capture_count"]))
        raise RuntimeError("observation_identity")
    monkeypatch.setattr(app, "run_blind_worker", reject)
    assert app._run_plan_mutation(
        ROOT, blind, plan, "observation_identity", {}, app.time.perf_counter() + 1
    ) == "observation_identity"
    assert seen == [(2, 4)] and len(list(blind.rglob("*.bin"))) == 768


def test_worker_rejects_observation_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p50, _, capsules, manifest = _fixture_roots(tmp_path)
    _patch_counts(monkeypatch)
    plan = app.build_blind_plan(capsules, manifest)
    plan["episodes"][0]["captures"][0]["observation_identity"] = [2, 1]

    with pytest.raises(oracle.OracleContractError, match="observation_identity"):
        oracle.rebuild_plan(
            plan, input_root=p50, plan_sha256="0" * 64,
            worker_source_sha256="1" * 64,
        )


def test_reveal_detects_selected_metadata_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p50, p79, capsules, manifest = _fixture_roots(tmp_path)
    _patch_counts(monkeypatch)
    plan = app.build_blind_plan(capsules, manifest)
    receipt = oracle.rebuild_plan(
        plan,
        input_root=p50,
        plan_sha256=hashlib.sha256(app._json_bytes(plan)).hexdigest(),
        worker_source_sha256=_provenance()["worker_source"]["sha256"],
    )
    bank = json.loads((p79 / "bank.json").read_text(encoding="utf-8"))

    comparison = app.compare_reveal(p79, bank, receipt)
    assert comparison["selected_index_exact_match_count"] == 1
    mutated = copy.deepcopy(bank)
    mutated["episodes"][0]["candidate_set"]["selected_index"] = -1
    comparison = app.compare_reveal(p79, mutated, receipt)
    assert comparison["selected_index_exact_match_count"] == 0
    wrong_schema = copy.deepcopy(bank)
    wrong_schema["schema_version"] = app.P50_SCHEMA
    with pytest.raises(app.LineageContractError, match="p79_schema"):
        app.compare_reveal(p79, wrong_schema, receipt)


def test_actual_boundary_controls_run_worker_and_comparer_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p50, p79, capsules, manifest = _fixture_roots(tmp_path)
    _patch_counts(monkeypatch)
    plan = app.build_blind_plan(capsules, manifest)
    blind = tmp_path / "blind"; blind.mkdir()
    app.materialize_blind_inputs(p50, blind, plan)
    app._atomic_write(blind / "blind-plan.json", app._json_bytes(plan))
    staging = tmp_path / "stage"; staging.mkdir()
    staged_worker, _ = app.materialize_worker(ROOT, staging, _provenance())
    isolation = app.validate_isolated_runtime(ROOT, staged_worker)
    work = tmp_path / "worker"; work.mkdir()
    receipt_path = tmp_path / "receipt.json"
    worker_run = app.run_blind_worker(
        ROOT, blind, blind / "blind-plan.json", receipt_path, work,
        deadline=app.time.perf_counter() + 60, forbidden_values=(str(p79),),
        isolation=isolation)
    receipt = json.loads(receipt_path.read_text())
    bank = json.loads((p79 / "bank.json").read_text())
    comparison = app.compare_reveal(p79, bank, receipt)
    boundary = app.run_boundary_controls(
        ROOT, blind, plan, receipt, bank, p79, comparison, _provenance(),
        (worker_run,),
        isolation, app.time.perf_counter() + 60)

    assert boundary["passed"] is True
    assert boundary["pass_count"] == boundary["control_count"]
    assert boundary["runtime"]["blind_blob_copy_count"] == 0
    assert boundary["runtime"]["maximum_mutation_episode_count"] <= 3
    assert boundary["runtime"]["wall_seconds"] < 60
    assert {row["observed_category"] for row in boundary["plan_mutations"]} >= {
        "plan_schema", "episode_count", "episode_duplicate", "episode_order",
        "capture_order", "final_input", "observation_identity", "path_escape",
        "policy_input_size_or_hash", "policy_input_missing",
        "policy_input_symlink", "input_path_duplicate"}
    assert all(not path.is_file() or path.is_symlink()
               for path in tmp_path.glob("mutation-*/*/*.bin"))


def test_external_rss_and_post_rename_budget_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert app._parse_time_peak_rss(
        "  123456 maximum resident set size\n") == 123456
    output = tmp_path / "output"
    output.mkdir()
    (output / "artifact").write_bytes(b"x" * 8)
    started = app.time.perf_counter()
    worker_runs = ({"outer_observed_child_peak_rss_bytes": 1},)
    peak = app._process_tree_peak(worker_runs) + 1024
    runtime = {"wall_seconds": 1.0, "process_tree_peak_rss_bytes": peak}
    (output / "report.json").write_text(json.dumps({"runtime": runtime}))
    (output / "manifest.json").write_text(json.dumps({"runtime": runtime}))
    assert app._require_finalized_budget(
        started, peak, output, 1.0, worker_runs) >= 0.0
    with pytest.raises(RuntimeError, match="RSS"):
        monkeypatch.setattr(app, "_process_tree_peak",
                            lambda runs: app.MAX_RSS_BYTES + 1)
        runtime["process_tree_peak_rss_bytes"] = app.MAX_RSS_BYTES + 1
        (output / "report.json").write_text(json.dumps({"runtime": runtime}))
        (output / "manifest.json").write_text(json.dumps({"runtime": runtime}))
        app._require_finalized_budget(
            started, app.MAX_RSS_BYTES + 1, output, 1.0, worker_runs)
    monkeypatch.undo()
    peak = app._process_tree_peak(worker_runs) + 1024
    runtime["process_tree_peak_rss_bytes"] = peak
    (output / "report.json").write_text(json.dumps({"runtime": runtime}))
    (output / "manifest.json").write_text(json.dumps({"runtime": runtime}))
    with pytest.raises(RuntimeError, match="artifact"):
        original = app.MAX_ARTIFACT_BYTES
        app.MAX_ARTIFACT_BYTES = 1
        try:
            app._require_finalized_budget(
                started, peak, output, 1.0, worker_runs)
        finally:
            app.MAX_ARTIFACT_BYTES = original


def test_run_uses_two_blind_workers_before_bank_reveal_and_writes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p50, p79 = tmp_path / "p50", tmp_path / "p79"
    output = tmp_path / "output"
    monkeypatch.setattr(app, "FORMAL_P50", p50)
    monkeypatch.setattr(app, "FORMAL_P79", p79)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    _fixture_roots(tmp_path)
    _patch_counts(monkeypatch)
    monkeypatch.setattr(app, "MIN_DISK_FREE_BYTES", 0)
    monkeypatch.setattr(app, "MAX_WALL_SECONDS", 120.0)
    provenance = _provenance(p50, p79)
    monkeypatch.setattr(
        app, "validate_frozen_provenance", lambda *args: provenance
    )
    events = []
    original_worker = app.run_blind_worker
    original_identities = app._directory_identities

    def tracked_worker(*args, **kwargs):
        events.append("worker")
        return original_worker(*args, **kwargs)

    def tracked_identities(root: Path, directory: Path):
        if directory == p79:
            events.append("p79-hash")
        return original_identities(root, directory)

    monkeypatch.setattr(app, "run_blind_worker", tracked_worker)
    monkeypatch.setattr(app, "_directory_identities", tracked_identities)

    result = app.run(SimpleNamespace(p50=p50, p79=p79, output=output))

    assert result["decision"] == (
        "accepted as consumer-local v2 selection-lineage evidence"
    )
    assert events[:2] == ["worker", "worker"]
    assert events[2] == "p79-hash"
    assert events[3:] == ["worker"] * 15
    assert not output.with_name(output.name + ".tmp").exists()
    assert sorted(path.name for path in output.iterdir()) == [
        "blind-plan.json",
        "blind-receipt-a.json",
        "blind-receipt-b.json",
        "boundary-controls.json",
        "comparison.json",
        "manifest.json",
        "report.json",
    ]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    manifest_result = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert report["blind_rebuild_bit_identical"] is True
    assert report["blind_p79_path_or_metadata_read_count"] == 0
    assert report["private_truth_read_count"] == 0
    assert report["blind_input_file_count"] == 6
    assert report["mutation_control_pass_count"] == report["mutation_control_count"]
    assert report["phase_a_trust_basis"]["read_ledger_role"] == "auxiliary"
    assert all(value is True for key, value in report["phase_a_trust_basis"].items()
               if key != "read_ledger_role")
    assert report["runtime"]["wall_seconds"] == manifest_result["runtime"]["wall_seconds"]
    assert report["runtime"]["wall_seconds_is_upper_bound"] is True
    assert report["runtime"]["process_tree_peak_rss_is_upper_bound"] is True
    assert report["runtime"]["process_tree_peak_rss_bytes"] == manifest_result[
        "runtime"]["process_tree_peak_rss_bytes"]
    assert result["finalized_wall_seconds"] <= report["runtime"]["wall_seconds"]
    assert manifest_result["provenance"]["source_end"]["checks"]["passed"] is True
    assert all(run["worker_path"].endswith("/selection_lineage_worker.py")
               for run in report["runtime"]["worker_runs"])
    assert manifest_result["training_executed"] is False
    assert manifest_result["physical_acquisition_executed"] is False


def test_failure_removes_staging_and_never_creates_final_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p50, p79 = tmp_path / "p50", tmp_path / "p79"
    output = tmp_path / "output"
    monkeypatch.setattr(app, "FORMAL_P50", p50)
    monkeypatch.setattr(app, "FORMAL_P79", p79)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    _fixture_roots(tmp_path)
    _patch_counts(monkeypatch)
    monkeypatch.setattr(app, "MIN_DISK_FREE_BYTES", 0)
    provenance = _provenance(p50, p79)
    monkeypatch.setattr(
        app, "validate_frozen_provenance", lambda *args: provenance
    )
    monkeypatch.setattr(
        app,
        "run_blind_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("worker")),
    )

    with pytest.raises(RuntimeError, match="worker"):
        app.run(SimpleNamespace(p50=p50, p79=p79, output=output))
    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()


@pytest.mark.parametrize(
    ("target", "category"),
    (
        (f"HEAD:{app.FROZEN_DOCUMENT}", "frozen_document_blob_matches"),
        ("HEAD:docs/research-loop/0005", "history_trees_match"),
        (f"HEAD:{app.FORMAL_P79}", "p79_tree_matches"),
        (
            f"{app.P79_PRODUCER_COMMIT}:src/hwr/eval/target_selection.py",
            "selector_blob_matches",
        ),
    ),
)
def test_provenance_drift_categories_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    category: str,
) -> None:
    p50, p79 = tmp_path / "p50", tmp_path / "p79"
    p50.mkdir(); p79.mkdir()
    manifest = {
        "schema_version": app.P79_MANIFEST_SCHEMA,
        "provenance": {"input_files": []},
        "artifacts": {
            "bank.json": {
                "bytes": 328423,
                "sha256": app.P79_FILES["bank.json"],
            }
        },
    }
    (p50 / "capsules.json").write_text("{}", encoding="utf-8")
    (p79 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(app, "EXPECTED_INPUT_FILES", 0)
    monkeypatch.setattr(app, "EXPECTED_CAPTURES", 0)
    monkeypatch.setattr(app, "_directory_identities", lambda *args: [])
    monkeypatch.setattr(app, "_directory_stats", lambda *args: [])
    monkeypatch.setattr(app, "_p79_expected_files", lambda *args: [])
    monkeypatch.setattr(
        app, "_stable_read",
        lambda path, expected, root=None: (
            path.read_bytes(), {"device": 1, "inode": 1, "size": path.stat().st_size}
        ),
    )
    monkeypatch.setattr(
        app, "source_similarity_audit", lambda *args: {"passed": True}
    )
    monkeypatch.setattr(
        app,
        "_capture_ledger",
        lambda value: {
            "capture_count": 0,
            "entry_count": 0,
            "sha256": app.CAPTURE_LEDGER_SHA256,
        },
    )
    monkeypatch.setattr(app, "_is_ancestor", lambda *args: True)
    monkeypatch.setattr(app, "_git_lines", lambda *args: sorted(app.ALLOWED_PATHS))

    def git(root: Path, *arguments: str) -> str:
        del root
        requested = arguments[-1]
        values = {
            "HEAD": "a" * 40,
            f"HEAD:{app.FROZEN_DOCUMENT}": app.FROZEN_DOCUMENT_BLOB,
            f"{app.P79_ARTIFACT_COMMIT}:{app.FORMAL_P79}": app.P79_TREE,
            f"HEAD:{app.FORMAL_P79}": app.P79_TREE,
            (
                f"{app.P79_PRODUCER_COMMIT}:"
                "src/hwr/eval/candidate_mask_ownership.py"
            ): app.P79_PRODUCER_BLOB,
            (
                f"{app.P79_PRODUCER_COMMIT}:"
                "src/hwr/eval/target_selection.py"
            ): app.SELECTOR_BLOB,
            **{f"HEAD:{path}": value for path, value in app.HISTORY_TREES.items()},
            **{
                f"HEAD:{path}": app.CONSUMER_BLOBS[path]
                for path in app.CONSUMER_BLOBS
            },
        }
        return "0" * 40 if requested == target else values.get(requested, "")

    monkeypatch.setattr(app, "_git", git)
    original_identity = app._file_identity

    def identity(root: Path, path: Path) -> dict[str, object]:
        if path == p79 / "manifest.json":
            return {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": app.P79_FILES["manifest.json"],
                    "fd_identity": {"device": 1, "inode": 1, "size": path.stat().st_size},
            }
        if path.parent == p50 and path.name in app.P50_FILES:
            return {
                "path": path.as_posix(),
                "bytes": 1,
                "sha256": app.P50_FILES[path.name],
                    "fd_identity": {"device": 1, "inode": 1, "size": 1},
            }
        return original_identity(root, path)

    monkeypatch.setattr(app, "_file_identity", identity)
    with pytest.raises(app.LineageContractError, match=category):
        app.validate_frozen_provenance(ROOT, p50, p79)


def test_main_exit_code_tracks_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        app,
        "run",
        lambda arguments: {"decision": "rejected", "output": "fixture"},
    )

    assert app.main(("--p50", "a", "--p79", "b", "--output", "c")) == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "rejected"
