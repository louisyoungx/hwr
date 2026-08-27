from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hwr.apps import evaluate_candidate_mask_ownership as app
from hwr.eval import candidate_mask_ownership as ownership
from hwr.eval import target_selection
from hwr.eval.target_selection import PolicyVisibleInput


def _policy_input() -> PolicyVisibleInput:
    proprioception = np.zeros(37, dtype="<f8")
    proprioception[24:26] = 0.25
    return PolicyVisibleInput(
        observation_timestamp_ns=50_000_000,
        sequence_id=3,
        phase_index=5,
        phase_step=0,
        policy_rng_seed=17,
        safety_state="ok",
        head_rgb_uint8=np.zeros((192, 256, 3), dtype=np.uint8),
        head_depth_m=np.ones((192, 256), dtype="<f4"),
        head_depth_valid=np.ones((192, 256), dtype=np.bool_),
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


def _identity(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _input_directory(path: Path) -> dict[str, object]:
    payload = target_selection.serialize_policy_input(_policy_input())
    visible = ownership.candidate_visible_bytes(_policy_input())
    candidate_document = {
        "schema_version": target_selection.LEGACY_CANDIDATE_SCHEMA,
        "acquisition_input_sha256": [hashlib.sha256(payload).hexdigest()],
        "candidate_count": 0,
        "candidates": [],
    }
    candidate = json.dumps(
        candidate_document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    policy_path = "blobs/episode/capture-00-policy.bin"
    visible_path = "blobs/episode/capture-00-candidate-visible.bin"
    candidate_path = "blobs/episode/candidate-set.json"
    for name, content in (
        (policy_path, payload),
        (visible_path, visible),
        (candidate_path, candidate),
    ):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    capsule = {
        "schema_version": "hwr.p50-acquisition-capsule-index/v1",
        "capsule_count": 1,
        "episodes": [{
            "planned_episode_id": "episode",
            "task_id": "tidy_living_room_3d/v1",
            "cell_id": "cell",
            "cell_ordinal": 0,
            "replicate_ordinal": 0,
            "candidate_ordinal": 0,
            "environment_seed": 1,
            "policy_rng_seed": 2,
            "replacement": False,
            "acquisition_base_pose": [0.0, 0.0, 0.0],
            "captures": [{
                "capture_ordinal": 0,
                "final_input": True,
                "observation_timestamp_ns": 50_000_000,
                "sequence_id": 3,
                "policy_input": _identity(policy_path, payload),
                "candidate_visible_input": _identity(visible_path, visible),
            }],
            "candidate_set": {
                **_identity(candidate_path, candidate),
                "schema_version": target_selection.LEGACY_CANDIDATE_SCHEMA,
                "candidate_count": 0,
                "selected_index": -1,
                "score_bytes_sha256": hashlib.sha256(b"").hexdigest(),
                "generated_online": True,
            },
        }],
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "capsules.json").write_text(json.dumps(capsule), encoding="utf-8")
    return capsule


def test_run_writes_v2_bank_atomically_without_copying_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input"
    output = tmp_path / "output"
    _input_directory(input_path)
    identity_snapshot = [{"path": "frozen", "bytes": 1, "sha256": "a" * 64}]
    monkeypatch.setattr(app, "FORMAL_INPUT", input_path)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(app, "EXPECTED_EPISODES", 1)
    monkeypatch.setattr(app, "EXPECTED_CAPTURES", 1)
    monkeypatch.setattr(app, "_require_disk", lambda output: None)
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        app,
        "_directory_identities",
        lambda root, directory: identity_snapshot,
    )
    monkeypatch.setattr(
        app,
        "_provenance",
        lambda *args: {
            "checks": {
                "legacy_ast_defect_confirmed": True,
                "passed": True,
            },
            "input_files": identity_snapshot,
        },
    )
    monkeypatch.setattr(app, "_peak_rss_bytes", lambda: 1)
    calls = 0
    original = app.build_bank

    def counted_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(app, "build_bank", counted_build)

    result = app.run(SimpleNamespace(input=input_path, output=output))

    assert calls == 2
    assert result["decision"] == (
        "accepted as deterministic candidate-generator correction"
    )
    assert not output.with_name(output.name + ".tmp").exists()
    assert sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    ) == [
        "bank.json",
        "blobs/episode/candidate-set.json",
        "manifest.json",
        "regression.json",
        "report.json",
    ]
    bank = json.loads((output / "bank.json").read_text())
    candidate = json.loads(
        (output / "blobs/episode/candidate-set.json").read_text()
    )
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert bank["schema_version"] == app.BANK_SCHEMA
    assert candidate["schema_version"] == target_selection.CANDIDATE_SCHEMA
    assert bank["episodes"][0]["old_candidate_set"]["schema_version"] == (
        target_selection.LEGACY_CANDIDATE_SCHEMA
    )
    assert not any(path.suffix == ".bin" for path in output.rglob("*"))
    assert manifest["training_executed"] is False
    assert manifest["physical_acquisition_executed"] is False
    assert manifest["policy_inference_executed"] is False
    assert manifest["capability_evaluation_executed"] is False
    assert report["full_bank_replay_bit_identical"] is True
    assert report["checks"]["full_bank_replay_bit_identical"] is True


def test_runner_rejects_non_frozen_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input path"):
        app.run(
            SimpleNamespace(
                input=tmp_path / "wrong-input",
                output=tmp_path / "wrong-output",
            )
        )


def test_capture_ledger_preserves_manifest_order(tmp_path: Path) -> None:
    path = tmp_path / "input"
    capsule = _input_directory(path)
    expected = []
    for episode in capsule["episodes"]:
        for capture in episode["captures"]:
            for name in ("policy_input", "candidate_visible_input"):
                value = capture[name]
                expected.append(
                    [value["path"], value["sha256"], value["bytes"]]
                )

    ledger = app._capture_ledger(path)

    assert ledger == {
        "capture_count": 1,
        "entry_count": 2,
        "sha256": hashlib.sha256(
            json.dumps(expected, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


def test_episode_loader_rejects_duplicate_identity_with_changed_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input"
    capsule = _input_directory(path)
    first = capsule["episodes"][0]["captures"][0]
    first["final_input"] = False
    changed = _policy_input()
    changed.head_depth_m[0, 0] = 0.5
    payload = target_selection.serialize_policy_input(changed)
    visible = ownership.candidate_visible_bytes(changed)
    policy_path = "blobs/episode/capture-01-policy.bin"
    visible_path = "blobs/episode/capture-01-candidate-visible.bin"
    (path / policy_path).write_bytes(payload)
    (path / visible_path).write_bytes(visible)
    capsule["episodes"][0]["captures"].append({
        **first,
        "capture_ordinal": 1,
        "final_input": True,
        "policy_input": _identity(policy_path, payload),
        "candidate_visible_input": _identity(visible_path, visible),
    })

    with pytest.raises(RuntimeError, match="repeated observation identity"):
        app._episode_inputs(path, capsule["episodes"][0])


def test_provenance_and_budget_guards_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="workspace_clean"):
        app._require_provenance(
            {"checks": {"workspace_clean": False, "passed": False}}
        )
    with pytest.raises(RuntimeError, match="historical_artifact_trees_match"):
        app._require_provenance({
            "checks": {
                "historical_artifact_trees_match": False,
                "passed": False,
            }
        })
    with pytest.raises(RuntimeError, match="wall-time"):
        app._require_budget(
            app.MAX_WALL_SECONDS + 1,
            1,
            {"report.json": b"{}"},
        )
    with pytest.raises(RuntimeError, match="RSS"):
        app._require_budget(
            1,
            app.MAX_RSS_BYTES + 1,
            {"report.json": b"{}"},
        )
    with pytest.raises(RuntimeError, match="artifact"):
        app._require_budget(
            1,
            1,
            {"report.json": b"x" * (app.MAX_ARTIFACT_BYTES + 1)},
        )


def test_full_bank_replay_compares_indexes_regression_and_every_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "EXPECTED_EPISODES", 1)
    first = {
        "artifacts": {
            "bank.json": b"bank",
            "regression.json": b"regression",
            "blobs/episode/candidate-set.json": b"candidate",
        }
    }
    second = {"artifacts": dict(first["artifacts"])}

    assert app._full_bank_replay_bit_identical(first, second) is True
    for name in second["artifacts"]:
        changed = {"artifacts": dict(second["artifacts"])}
        changed["artifacts"][name] += b"x"
        assert app._full_bank_replay_bit_identical(first, changed) is False


def test_historical_artifact_trees_are_bound_to_frozen_and_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def git(root, *arguments):
        del root
        calls.append(arguments)
        path = arguments[-1].split(":", 1)[1]
        return app.HISTORICAL_ARTIFACT_TREES[path]

    monkeypatch.setattr(app, "_git", git)

    identities = app._historical_artifact_identities(Path("."))

    assert set(identities) == set(app.HISTORICAL_ARTIFACT_TREES)
    assert all(
        value["expected_tree"] == value["frozen_tree"] == value["head_tree"]
        for value in identities.values()
    )
    assert len(calls) == 2 * len(app.HISTORICAL_ARTIFACT_TREES)


@pytest.mark.parametrize(
    ("decision", "expected"),
    (
        ("accepted as deterministic candidate-generator correction", 0),
        ("rejected", 1),
        ("inconclusive_secondary_order_dependence", 1),
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

    assert app.main(("--input", "x", "--output", "y")) == expected
    assert json.loads(capsys.readouterr().out)["decision"] == decision


def test_bank_builder_rejects_replacement_or_wrong_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input"
    capsule = _input_directory(input_path)
    monkeypatch.setattr(app, "EXPECTED_EPISODES", 1)
    capsule["episodes"][0]["replacement"] = True

    with pytest.raises(RuntimeError, match="replacement"):
        app.build_bank(input_path, capsule, started=time.perf_counter())
    capsule["episodes"] = []
    with pytest.raises(RuntimeError, match="24 Episodes"):
        app.build_bank(input_path, capsule, started=time.perf_counter())
