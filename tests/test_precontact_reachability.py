from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hwr.apps import evaluate_precontact_reachability as app
from hwr.eval import precontact_reachability as contract

ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = (
    ROOT
    / "runs/research-loop/0010/r0010-p51-e1-bank-s20265101/bank.json"
)
TERMINALS_PATH = (
    ROOT
    / "runs/research-loop/0010/"
    "r0010-p51-e1-convergence-s20265101/terminals.json"
)


@pytest.fixture(scope="module")
def frozen_inputs() -> tuple[dict[str, object], dict[str, object]]:
    return _read(BANK_PATH), _read(TERMINALS_PATH)


@pytest.fixture(scope="module")
def analysis(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
) -> dict[str, object]:
    bank, terminals = frozen_inputs
    return contract.analyze_precontact_reachability(bank, terminals)


def test_frozen_p51_inputs_recompute_complete_p57_contract(
    analysis: dict[str, object],
) -> None:
    assert analysis["decision"] == (
        "accepted as bilateral pre-contact reachability measurement evidence"
    )
    assert analysis["diagnostic"] in {
        "precontact_support_deficit_supported",
        "precontact_support_deficit_rejected",
        "diagnostic_inconclusive",
    }
    assert analysis["checks"]["passed"] is True
    assert analysis["summary"]["pair_count"] == 36
    assert analysis["summary"]["arm_count"] == 72
    assert len(analysis["pairs"]) == 36
    assert all(
        len(row["bilateral_ready_by_step"]) == 101
        and len(row["frame_fixed_applied_actions"]) == 100
        and len(row["arms"]["left"]["distances_m"]) == 101
        and len(row["arms"]["right"]["distances_m"]) == 101
        for row in analysis["pairs"]
    )
    assert {
        value["pair_count"] for value in analysis["summary"]["by_task"].values()
    } == {12}
    assert {
        value["pair_count"] for value in analysis["summary"]["by_cell"].values()
    } == {3}
    assert {
        value["pair_count"]
        for value in analysis["summary"]["by_observation_latency"].values()
    } == {18}
    assert {
        value["pair_count"]
        for value in analysis["summary"]["by_action_latency"].values()
    } == {18}
    assert {
        value["pair_count"]
        for value in analysis["summary"]["by_latency_combination"].values()
    } == {9}


def test_same_frozen_input_is_canonical_bit_identical(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    analysis: dict[str, object],
) -> None:
    bank, terminals = frozen_inputs
    replay = contract.analyze_precontact_reachability(bank, terminals)
    assert _canonical(analysis["pairs"]) == _canonical(replay["pairs"])
    assert _canonical(analysis["summary"]) == _canonical(replay["summary"])
    assert _canonical(analysis) == _canonical(replay)


def test_budget_uses_only_raw_applied_arm_linear_commands(
    analysis: dict[str, object],
) -> None:
    pair = analysis["pairs"][0]
    actions = np.asarray(pair["frame_fixed_applied_actions"], np.float64)
    for arm, (start, stop) in contract.ARM_FIELDS.items():
        expected_norms = np.linalg.norm(actions[:, start:stop], axis=1)
        expected_budget = float(
            np.sum(expected_norms)
            * contract.ACTION_SCALE_M_PER_S
            / contract.CONTROL_HZ
        )
        assert pair["arms"][arm]["applied_command_norms"] == pytest.approx(
            expected_norms
        )
        assert pair["arms"][arm]["actual_applied_command_budget_m"] == pytest.approx(
            expected_budget
        )
        assert pair["arms"][arm]["initial_command_margin_m"] == pytest.approx(
            expected_budget - pair["arms"][arm]["d_0_m"]
        )


def test_bilateral_readiness_requires_same_step(
    analysis: dict[str, object],
) -> None:
    pair = analysis["pairs"][0]
    left = pair["arms"]["left"]["distances_m"]
    right = pair["arms"]["right"]["distances_m"]
    expected = [
        left[index] <= contract.READY_DISTANCE_M
        and right[index] <= contract.READY_DISTANCE_M
        for index in range(101)
    ]
    assert pair["bilateral_ready_by_step"] == expected
    assert pair["ever_bilateral_ready"] is any(expected)
    assert pair["endpoint_bilateral_ready"] is expected[-1]


def test_contact_target_and_nominal_margin_reconstruct_from_bank(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    analysis: dict[str, object],
) -> None:
    bank, _ = frozen_inputs
    rebuilt = contract.contact_targets(bank["pairs"][0])
    pair = analysis["pairs"][0]
    assert rebuilt == pair["target_reconstruction"]
    assert rebuilt["preposition"] == bank["pairs"][0]["preposition_targets"]
    assert rebuilt["b3_nominal_maximum_m"] == pytest.approx(0.075)
    assert rebuilt["b4_nominal_maximum_m"] == pytest.approx(0.020)
    assert rebuilt["total_nominal_maximum_m"] == pytest.approx(0.095)
    for arm in ("left", "right"):
        preposition = np.asarray(rebuilt["preposition"][arm])
        contact = np.asarray(rebuilt["contact"][arm])
        distance = float(np.linalg.norm(contact - preposition))
        assert pair["arms"][arm][
            "contact_to_preposition_distance_m"
        ] == pytest.approx(distance)
        assert pair["arms"][arm]["contact_transition_margin_m"] == pytest.approx(
            0.095 - distance
        )


@pytest.mark.parametrize(
    ("ready", "negative", "tasks", "expected"),
    (
        (6, 30, (2, 2, 2), "precontact_support_deficit_supported"),
        (24, 0, (8, 8, 8), "precontact_support_deficit_rejected"),
        (7, 30, (2, 2, 3), "diagnostic_inconclusive"),
    ),
)
def test_diagnostic_boundaries_are_frozen(
    ready: int,
    negative: int,
    tasks: tuple[int, int, int],
    expected: str,
) -> None:
    summary = {
        "overall": {
            "ever_bilateral_ready_count": ready,
            "both_initial_command_margins_negative_count": negative,
        },
        "by_task": {
            task: {"ever_bilateral_ready_count": count}
            for task, count in zip(contract.TASK_IDS, tasks, strict=True)
        },
    }
    assert contract.diagnostic_decision(summary) == expected


def test_applied_action_tampering_is_invalid(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
) -> None:
    bank, terminals = frozen_inputs
    tampered = copy.deepcopy(terminals)
    tampered["records"][0]["arms"]["frame_fixed"]["applied_actions"][0][2] += 0.01
    result = contract.analyze_precontact_reachability(bank, tampered)
    assert result["decision"] == "invalid"
    assert result["diagnostic"] is None
    assert result["pairs"] == []


def test_tool_distance_tampering_is_invalid(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
) -> None:
    bank, terminals = frozen_inputs
    tampered = copy.deepcopy(terminals)
    tampered["records"][0]["arms"]["frame_fixed"]["tool_distances"][0][
        "left_m"
    ] += 0.01
    result = contract.analyze_precontact_reachability(bank, tampered)
    assert result["decision"] == "invalid"
    assert result["diagnostic"] is None


def test_input_specs_bind_all_frozen_files() -> None:
    for spec in app.INPUT_SPECS.values():
        payload = (ROOT / spec["path"]).read_bytes()
        assert len(payload) == spec["bytes"]
        assert hashlib.sha256(payload).hexdigest() == spec["sha256"]


def test_input_provenance_only_requires_tracked_bank_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = app.subprocess.run
    ls_files_paths = []

    def recording_run(arguments, *args, **kwargs):
        if tuple(arguments[:3]) == ("git", "ls-files", "--error-unmatch"):
            ls_files_paths.append(arguments[3])
        return original_run(arguments, *args, **kwargs)

    monkeypatch.setattr(app.subprocess, "run", recording_run)
    inputs = app._input_identities(
        ROOT,
        argparse.Namespace(bank=BANK_PATH, terminals=TERMINALS_PATH),
    )
    assert set(ls_files_paths) == {
        app.INPUT_SPECS["bank"]["path"].as_posix(),
        app.INPUT_SPECS["bank_manifest"]["path"].as_posix(),
    }
    producer = _read(
        ROOT / app.INPUT_SPECS["terminal_manifest"]["path"]
    )["source_commit"]
    for name in app.TRACKED_INPUTS:
        assert inputs[name]["provenance_kind"] == "tracked_committed_artifact"
        assert inputs[name]["commit"] != ""
    for name in app.MANIFEST_BOUND_INPUTS:
        assert (
            inputs[name]["provenance_kind"]
            == "manifest_bound_ignored_artifact"
        )
        assert inputs[name]["commit"] == producer


def test_manifest_bound_inputs_require_producer_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = app._input_identities(
        ROOT,
        argparse.Namespace(bank=BANK_PATH, terminals=TERMINALS_PATH),
    )
    bank, terminals = _read(BANK_PATH), _read(TERMINALS_PATH)
    inputs["terminals"]["commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="producer commit"):
        app._validate_input_provenance(ROOT, bank, terminals, inputs)


def test_nonformal_output_is_rejected_before_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = argparse.Namespace(
        bank=BANK_PATH,
        terminals=TERMINALS_PATH,
        output=tmp_path / "not-formal",
    )
    monkeypatch.setattr(
        app,
        "_source_commit",
        lambda root: pytest.fail("analysis started before output rejection"),
    )
    with pytest.raises(ValueError, match="frozen formal output"):
        app.run(arguments)


def test_atomic_output_rejects_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        app._create_output(output, {"report.json": b"{}\n"})


def test_atomic_output_rejects_existing_staging_directory(tmp_path: Path) -> None:
    output = tmp_path / "result"
    staging = tmp_path / "result.tmp"
    staging.mkdir()
    with pytest.raises(FileExistsError):
        app._create_output(output, {"report.json": b"{}\n"})


def test_pairs_document_and_report_keep_pair_as_sample_unit(
    analysis: dict[str, object],
) -> None:
    pairs = app._pairs_document(analysis["pairs"])
    report = app._report(
        "a" * 40,
        analysis,
        {
            name: {
                "path": str(spec["path"]),
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
                "commit": "b" * 40,
                "provenance_kind": (
                    "tracked_committed_artifact"
                    if name in app.TRACKED_INPUTS
                    else "manifest_bound_ignored_artifact"
                ),
            }
            for name, spec in app.INPUT_SPECS.items()
        },
        pairs_equal=True,
        report_equal=True,
    )
    assert pairs["pair_count"] == 36
    assert pairs["arm_count"] == 72
    assert pairs["sample_unit"] == "pair"
    assert report["sample_unit"] == "pair"
    assert report["checks"]["passed"] is True
    assert report["mujoco_executed"] is False
    assert report["training_executed"] is False
    assert report["contact_or_grasp_claim_allowed"] is False


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
