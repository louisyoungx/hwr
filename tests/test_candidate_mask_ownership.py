from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from hwr.eval import candidate_mask_ownership as ownership
from hwr.eval import target_selection
from hwr.eval.target_selection import PolicyVisibleInput, serialize_policy_input

ROOT = Path(__file__).resolve().parents[1]


def _input(
    *,
    timestamp: int = 1,
    sequence: int = 1,
    depth: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> PolicyVisibleInput:
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
        head_depth_m=(
            np.ones((192, 256), dtype="<f4") if depth is None else depth
        ),
        head_depth_valid=(
            np.ones((192, 256), dtype=np.bool_) if valid is None else valid
        ),
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


def _single_anchor_input() -> PolicyVisibleInput:
    depth = np.ones((192, 256), dtype="<f4")
    valid = np.zeros((192, 256), dtype=np.bool_)
    row, column = 96, 200
    valid[row - 10 : row + 11, column - 10 : column + 11] = True
    depth[row - 2 : row + 3, column - 2 : column + 3] = 0.8
    return _input(depth=depth, valid=valid)


def test_overlap_fixture_proves_legacy_alias_and_corrected_ownership() -> None:
    report = ownership.overlap_fixture_audit()

    assert report["checks"]["passed"] is True
    assert all(value["mutation_count"] > 0 for value in report["legacy"])
    assert all(value["mutation_count"] == 0 for value in report["corrected"])
    assert len({
        json.dumps(value["canonical_support"], sort_keys=True)
        for value in report["legacy"]
    }) > 1
    assert len({
        json.dumps(value["canonical_support"], sort_keys=True)
        for value in report["corrected"]
    }) == 1


def test_legacy_defect_is_derived_from_review_source_ast() -> None:
    legacy = subprocess.run(
        (
            "git",
            "show",
            "61d85cda1b96058831b4f93c7ad21a39f51cc2ab:"
            "src/hwr/eval/target_selection.py",
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert ownership.audit_legacy_source(legacy) == {
        "slice_assignment_count": 1,
        "bitand_augmented_assignment_count": 1,
        "assignment_precedes_mutation": True,
        "passed": True,
    }
    current = Path(ownership.__file__).read_text()
    assert ownership.audit_single_variable_source(current, legacy) == {
        "one_ownership_copy_gate": True,
        "frame_generator_otherwise_unchanged": True,
        "one_candidate_schema_change": True,
        "candidate_reduction_otherwise_unchanged": True,
        "geometry_merge_ranking_selector_unchanged": True,
        "passed": True,
    }
    target_blob = subprocess.run(
        ("git", "hash-object", "src/hwr/eval/target_selection.py"),
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    frozen_blob = subprocess.run(
        ("git", "rev-parse", "61d85cd:src/hwr/eval/target_selection.py"),
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert target_blob == frozen_blob == (
        "d7e588ba76ce18882255e3e22b1f86459ab235dd"
    )


def test_oracle_three_traversals_match_production_without_mutation() -> None:
    keyframe = serialize_policy_input(_single_anchor_input())
    final = serialize_policy_input(_input(timestamp=2, sequence=2))
    before = hashlib.sha256(
        target_selection.deserialize_policy_input(
            keyframe
        ).head_depth_valid.tobytes()
    ).hexdigest()

    candidate_set, audit = ownership.audit_episode(
        (keyframe,),
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=final,
    )

    assert json.loads(candidate_set.canonical_bytes)["schema_version"] == (
        ownership.CANDIDATE_SCHEMA_V2
    )
    assert audit["decision"] == (
        "accepted as deterministic candidate-generator correction"
    )
    assert audit["frame_count"] == 2
    assert audit["checks"]["passed"] is True
    assert all(
        value["raw_multiset_sha256"]
        == audit["traversals"]["row_major"]["raw_multiset_sha256"]
        for value in audit["traversals"].values()
    )
    assert all(
        frame["input_head_depth_valid_sha256_before"]
        == frame["input_head_depth_valid_sha256_after"]
        == before
        for frame in audit["frames"][:1]
    )
    assert all(
        traversal["probe_count"]
        == len(ownership.ROWS) * len(ownership.COLUMNS)
        for frame in audit["frames"]
        for traversal in frame["traversals"].values()
    )


def test_oracle_rehashes_parent_mask_before_and_after_every_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ownership._sha256
    measured = []

    def measured_sha256(content: bytes) -> str:
        measured.append(content)
        return original(content)

    monkeypatch.setattr(ownership, "_sha256", measured_sha256)
    result = ownership.oracle_frame_candidates(
        _input(valid=np.zeros((192, 256), dtype=np.bool_)),
        (0.0, 0.0, 0.0),
        0,
        "row_major",
    )

    expected = len(ownership.ROWS) * len(ownership.COLUMNS)
    parent_bytes = 192 * 256
    assert result.probe_count == expected
    assert sum(len(value) == parent_bytes for value in measured) == 1 + 2 * expected
    assert result.probe_sha256_pair_counts == (
        (result.parent_sha256, result.parent_sha256, expected),
    )


def test_boundary_controls_cover_invalid_flat_single_duplicate_and_nonfinite() -> None:
    invalid = _input(valid=np.zeros((192, 256), dtype=np.bool_))
    flat = _input(timestamp=2, sequence=2)
    single = _single_anchor_input()
    nonfinite_depth = np.ones((192, 256), dtype="<f4")
    nonfinite_depth[96, 200] = np.nan
    nonfinite = _input(
        timestamp=3,
        sequence=3,
        depth=nonfinite_depth,
    )

    assert all(
        len(
            ownership.oracle_frame_candidates(
                frame, (0.0, 0.0, 0.0), 0, traversal
            ).raw
        ) == expected
        for frame, expected in (
            (invalid, 0),
            (flat, 0),
            (single, 1),
            (nonfinite, 0),
        )
        for traversal in ownership.TRAVERSALS
    )
    duplicate = serialize_policy_input(single)
    candidate_set, audit = ownership.audit_episode(
        (duplicate, duplicate),
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=duplicate,
    )
    assert audit["checks"]["passed"] is True
    assert len(candidate_set.candidates) == 1
    assert candidate_set.candidates[0].view_count == 2
    assert ownership.boundary_fixture_audit()["passed"] is True


def test_v2_generator_is_explicit_and_legacy_generator_stays_frozen() -> None:
    payload = serialize_policy_input(_single_anchor_input())
    arguments = {
        "acquisition_base_pose": (0.0, 0.0, 0.0),
        "final_input": payload,
    }

    v2 = ownership.generate_candidate_set_v2((payload, payload), **arguments)
    legacy = target_selection.generate_candidate_set((payload, payload), **arguments)
    repeated = ownership.generate_candidate_set_v2((payload, payload), **arguments)

    assert json.loads(v2.canonical_bytes)["schema_version"] == (
        "hwr.p79-target-candidates/v2"
    )
    assert json.loads(legacy.canonical_bytes)["schema_version"] == (
        "hwr.p41-target-candidates/v1"
    )
    assert repeated.canonical_bytes == v2.canonical_bytes
