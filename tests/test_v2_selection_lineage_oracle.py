from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from hwr.apps import evaluate_v2_selection_lineage as app

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/evaluate_v2_selection_lineage_oracle.py"


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p83_oracle_test", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = _load_worker()


def binary_input(timestamp: int, sequence: int, *, final_x: float = 0.0) -> bytes:
    depth = np.ones((192, 256), dtype="<f4")
    valid = np.zeros((192, 256), dtype=np.uint8)
    valid[86:107, 190:211] = 1
    depth[94:99, 198:203] = 0.8
    proprioception = np.zeros(37, dtype="<f8")
    proprioception[24:26] = 0.25
    proprioception[26] = final_x
    return b"".join((
        oracle.INPUT_SCHEMA.encode("ascii"), b"\0",
        struct.pack("<qqiiQ", timestamp, sequence, 1, 0, 17), b"\0",
        np.zeros((192, 256, 3), np.uint8).tobytes(),
        depth.tobytes(), valid.tobytes(),
        np.asarray((80.0, 80.0, 127.5, 95.5), dtype="<f8").tobytes(),
        np.eye(4, dtype="<f8").tobytes(), proprioception.tobytes(),
        np.zeros((4, 16), dtype="<f8").tobytes(),
        np.asarray((0, 0, 0, 1), dtype=np.uint8).tobytes(),
    ))


def visible_bytes(payload: bytes) -> bytes:
    frame = oracle.parse_policy_input(payload)
    proprioception = frame["proprioception"]
    selected = np.concatenate((
        proprioception[:6], proprioception[12:18], proprioception[26:29]))
    return b"".join((
        oracle.CANDIDATE_VISIBLE_SCHEMA.encode("ascii"), b"\0",
        np.asarray(frame["rgb"], dtype=np.uint8).tobytes(),
        np.asarray(frame["depth"], dtype="<f4").tobytes(),
        np.asarray(frame["depth_valid"], dtype=np.uint8).tobytes(),
        np.asarray(frame["intrinsics"], dtype="<f8").tobytes(),
        np.asarray(frame["robot_from_camera"], dtype="<f8").tobytes(),
        np.asarray(selected, dtype="<f8").tobytes(),
    ))


def descriptor(path: str, content: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def fixture_plan(root: Path, count: int = 1,
                 captures_per_episode: int = 3) -> dict[str, object]:
    episodes = []
    for episode in range(count):
        captures = []
        for ordinal in range(captures_per_episode):
            payload = binary_input(
                episode * captures_per_episode + ordinal + 1,
                episode * captures_per_episode + ordinal + 1,
                final_x=0.2 if ordinal == captures_per_episode - 1 else 0.0)
            visible = visible_bytes(payload)
            prefix = Path("blobs") / f"episode-{episode:02d}"
            policy = prefix / f"capture-{ordinal:02d}-policy.bin"
            candidate = prefix / f"capture-{ordinal:02d}-candidate-visible.bin"
            (root / policy).parent.mkdir(parents=True, exist_ok=True)
            (root / policy).write_bytes(payload)
            (root / candidate).write_bytes(visible)
            captures.append({
                "capture_ordinal": ordinal,
                "final_input": ordinal == captures_per_episode - 1,
                "observation_identity": [
                    episode * captures_per_episode + ordinal + 1] * 2,
                "policy_input": descriptor(policy.as_posix(), payload),
                "candidate_visible_input": descriptor(candidate.as_posix(), visible),
            })
        episodes.append({
            "episode_ordinal": episode,
            "planned_episode_id": f"episode-{episode:02d}",
            "task_id": "fixture-task", "cell_id": "fixture-cell",
            "replicate_ordinal": episode,
            "acquisition_base_pose": [0.0, 0.0, 0.0],
            "captures": captures,
        })
    return {
        "schema_version": oracle.PLAN_SCHEMA, "proposal_id": oracle.PROPOSAL_ID,
        "episode_count": count, "capture_count": captures_per_episode * count,
        "input_file_count": 2 * captures_per_episode * count, "episodes": episodes,
    }


def test_manual_binary_parser_geometry_merge_and_score_known_values(
    tmp_path: Path,
) -> None:
    plan = fixture_plan(tmp_path)
    payloads = [
        (tmp_path / capture["policy_input"]["path"]).read_bytes()
        for capture in plan["episodes"][0]["captures"]
    ]
    frames = [oracle.parse_policy_input(payload) for payload in payloads]
    candidates = oracle.generate_candidates(
        frames[:-1], acquisition_base_pose=(0.0, 0.0, 0.0))
    assert frames[0]["identity"] == (1, 1)
    assert frames[-1]["base_pose"] == (0.2, 0.0, 0.0)
    assert oracle.candidate_visible_bytes(frames[0]) == visible_bytes(payloads[0])
    assert [oracle._candidate_record(row) for row in candidates] == [
        (725, 5, 800, 0, 0, -10000, 40, 0, 96, 200, 200, 50, 2)]
    assert oracle.compute_scores(
        candidates, (0.2, 0.0, 0.0), (0.0, 0.0, 0.0)
    ) == (0.7644583415088799,)
    raw = np.asarray((
        (1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.10, 0.05, 10, 0, 12, 16),
        (1.01, 0.0, 0.5, 1.0, 0.0, 0.0, 0.12, 0.06, 20, 1, 16, 20),
        (3.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.10, 0.05, 10, 2, 12, 16),
    ), np.float64)
    assert np.allclose(oracle.merge_components(raw)[0], (
        1.005, 0.0, 0.5, 1.0, 0.0, 0.0, 0.11, 0.06,
        30.0, 2.0, 0.0, 12.0, 16.0))
    assert np.array_equal(
        oracle.camera_points(
            np.asarray((1, 5)), np.asarray((3, 1)),
            np.asarray((2.0, 4.0)), np.asarray((2.0, 4.0, 1.0, 1.0))),
        np.asarray(((2.0, 0.0, 2.0), (0.0, 4.0, 4.0))))
    transform = oracle.acquisition_from_robot(
        (1.0, 2.0, np.pi / 2), (1.0, 3.0, np.pi))
    assert np.allclose(transform[:2, :], ((0.0, -1.0, 0.0, 1.0),
                                          (1.0, 0.0, 0.0, 0.0)))
    assert np.allclose(oracle.point_segment_distances(
        np.asarray(((0.5, 1.0, 0.0),)),
        np.asarray(((0.0, 0.0, 0.0),)),
        np.asarray(((1.0, 0.0, 0.0),))), ((1.0,),))


@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        (lambda value: value.update(schema_version="wrong"), "plan_schema"),
        (lambda value: value["episodes"].append(
            copy.deepcopy(value["episodes"][0])), "episode_count"),
        (lambda value: value["episodes"][0].update(episode_ordinal=1),
         "episode_order"),
        (lambda value: value["episodes"][0]["captures"][1].update(
            capture_ordinal=0), "capture_order"),
        (lambda value: value["episodes"][0]["captures"][-1].update(
            final_input=False), "final_input"),
        (lambda value: value["episodes"][0]["captures"][0].update(
            observation_identity=[-1, 0]), "observation_identity"),
        (lambda value: value["episodes"][0]["captures"][0][
            "policy_input"].update(path="../escape"), "path_escape"),
    ),
)
def test_plan_mutations_fail_by_category(
    tmp_path: Path, mutation, category: str,
) -> None:
    plan = fixture_plan(tmp_path)
    mutation(plan)
    with pytest.raises(oracle.OracleContractError, match=category):
        oracle._validate_plan(plan)


def test_fd_reader_rejects_hash_symlink_and_mid_read_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"bound")
    with pytest.raises(oracle.OracleContractError, match="size_or_hash"):
        oracle.stable_file_read(
            path, {**descriptor("input.bin", b"bound"), "sha256": "0" * 64},
            kind="policy_input", logical_path="input.bin", root=tmp_path)
    link = tmp_path / "link.bin"
    link.symlink_to(path)
    with pytest.raises(oracle.OracleContractError, match="symlink"):
        oracle.stable_file_read(
            link, descriptor("link.bin", b"bound"), kind="policy_input",
            logical_path="link.bin", root=tmp_path)
    original = oracle.os.fstat
    calls = 0

    def changing(descriptor_fd: int):
        nonlocal calls
        calls += 1
        value = original(descriptor_fd)
        if calls == 2:
            return os.stat_result((*value[:6], value.st_size + 1, *value[7:]))
        return value

    monkeypatch.setattr(oracle.os, "fstat", changing)
    with pytest.raises(oracle.OracleContractError, match="changed_during_read"):
        oracle.stable_file_read(
            path, descriptor("input.bin", b"bound"), kind="policy_input",
            logical_path="input.bin", root=tmp_path)


def test_source_similarity_and_dynamic_read_guards() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode()).hexdigest() == app.WORKER_SHA256
    references = tuple(
        subprocess.run(
            ("git", "show", f"9eef9953:{path}"),
            cwd=ROOT, check=True, capture_output=True, text=True).stdout
        for path in (
            "src/hwr/eval/candidate_mask_ownership.py",
            "src/hwr/eval/target_selection.py"))
    similarity = app.source_similarity_audit(source, references)
    assert similarity["passed"] is True
    assert similarity["whole_token_ratio"] <= 0.45
    assert similarity["whole_ast_ratio"] <= 0.45
    assert similarity["maximum_major_function_ratio"] <= 0.82
    for bad in (
        "import hwr\n",
        "import importlib\nimportlib.import_module('hwr')\n",
        "from importlib import import_module as load\nload('hwr')\n",
        "exec('x=1')\n",
        "open('secret').read()\n",
        "from pathlib import Path\nPath('secret').read_bytes()\n",
        "from pathlib import Path\nPath('secret').open('rb')\n",
        "from pathlib import Path\nROOT=Path(__file__).parent\n",
        "from os import read as steal\nsteal(3, 1)\n",
        "import builtins\nreader=builtins.open\nreader('secret')\n",
        "import os\nhidden=os\nhidden.open('secret')\n",
        "import runpy\nrunpy.run_path('secret')\n",
        "from runpy import run_path as load\nload('secret')\n",
        "import builtins\nbuiltins.eval('1')\n",
        "from pathlib import Path\ngetattr(Path('secret'), 'read_bytes')()\n",
        "from builtins import getattr as reflect\nreflect(object(), 'x')\n",
        "import operator\noperator.attrgetter('read_bytes')(object())\n",
        "import os as hidden\ndef stable_file_read(): hidden.open('secret')\n",
    ):
        assert app.audit_worker_source(bad)["passed"] is False


def test_read_audit_normalizes_pathlike_and_treats_ledger_as_auxiliary() -> None:
    audit = oracle.ReadAudit()
    for path in ("capture.bin", b"capture.bin", Path("capture.bin")):
        audit.authorize(path, "logical.bin")
        audit.hook("open", (path, "rb", 0))
        audit.clear()
    audit.hook("open", ("ignored.bin", "wb", os.O_WRONLY))
    audit.hook("open", ("runtime-internal.bin", None, os.O_RDONLY))
    assert audit.events == ["logical.bin"] * 3


def test_isolated_worker_and_twenty_four_job_parallel_path(tmp_path: Path) -> None:
    plan = fixture_plan(tmp_path, count=24, captures_per_episode=16)
    plan_path = tmp_path / "blind-plan.json"
    output = tmp_path / "receipt.json"
    plan_bytes = oracle.canonical_json_bytes(plan)
    plan_path.write_bytes(plan_bytes)
    numpy_site = Path(importlib.metadata.distribution("numpy").locate_file(""))
    staged_worker = tmp_path / "selection_lineage_worker.py"
    staged_worker.write_bytes(WORKER.read_bytes())
    command = (
        sys.executable, "-I", "-S", "-c", app.ISOLATED_BOOTSTRAP,
        str(tmp_path), str(numpy_site), "--plan", str(plan_path),
        "--plan-bytes", str(len(plan_bytes)),
        "--plan-sha256", hashlib.sha256(plan_bytes).hexdigest(),
        "--input-root", str(tmp_path), "--output", str(output),
        "--worker-sha256", hashlib.sha256(staged_worker.read_bytes()).hexdigest(),
    )
    result = subprocess.run(
        command, cwd=tmp_path,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": "",
             "HWR_P83_ISOLATED": "1"}, check=True, capture_output=True)
    receipt = json.loads(output.read_text())
    assert receipt["execution"]["job_count"] == 24
    assert receipt["execution"]["parallel_path_used"] is True
    assert receipt["execution"]["worker_count"] == min(24, os.cpu_count() or 1)
    assert receipt["execution"]["worker_process_count"] > 1
    assert receipt["read_audit"]["trust_role"] == "auxiliary"
    assert receipt["capture_count"] == 384
    assert receipt["input_file_match_count"] == 768
    assert receipt["read_audit"]["audited_open_count"] == 769
    assert all(set(value["fd_identity"]) == {"device", "inode", "size"}
               for value in receipt["read_ledger"])
    assert json.loads(result.stdout)["output_sha256"] == hashlib.sha256(
        output.read_bytes()).hexdigest()
