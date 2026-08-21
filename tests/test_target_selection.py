from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hwr.eval.target_selection import (
    Candidate,
    CandidateSet,
    PolicyVisibleInput,
    TargetSelectionContractError,
    _candidate_from_points,
    candidate_scores,
    deserialize_policy_input,
    primitive_action,
    select_candidate_index,
    select_control_index,
    serialize_policy_input,
)
from hwr.eval.target_selection_safety import paired_primary_statistics


def _input(*, phase_step: int = 0, safety_state: str = "ok") -> PolicyVisibleInput:
    proprioception = np.zeros(37, dtype="<f8")
    proprioception[24:26] = 0.25
    return PolicyVisibleInput(
        observation_timestamp_ns=50_000_000,
        sequence_id=3,
        phase_index=5,
        phase_step=phase_step,
        policy_rng_seed=17,
        safety_state=safety_state,
        head_rgb_uint8=np.zeros((192, 256, 3), dtype=np.uint8),
        head_depth_m=np.ones((192, 256), dtype="<f4"),
        head_depth_valid=np.ones((192, 256), dtype=np.bool_),
        head_camera_intrinsics=np.asarray((200.0, 200.0, 127.5, 95.5), dtype="<f8"),
        robot_from_head_camera=np.eye(4, dtype="<f8"),
        proprioception=proprioception,
        executed_action_history=np.zeros((4, 16), dtype="<f8"),
        history_available=np.asarray((False, False, False, True), dtype=np.bool_),
    )


def _candidate_set() -> CandidateSet:
    candidates = (
        Candidate((1.0, 0.0, 0.7), (-1.0, 0.0, 0.0), 0.14, 0.15, 80, 4, 0, 20, 20),
        Candidate((2.5, 0.5, 0.7), (-1.0, 0.0, 0.0), 0.35, 0.03, 50, 2, 1, 24, 24),
    )
    canonical = b"frozen-candidate-bytes"
    return CandidateSet((), candidates, canonical, hashlib.sha256(canonical).hexdigest())


def test_policy_input_round_trip_is_canonical_and_fail_closed() -> None:
    payload = serialize_policy_input(_input())
    restored = deserialize_policy_input(payload)

    assert serialize_policy_input(restored) == payload
    assert restored.proprioception.shape == (37,)
    assert restored.head_depth_valid.dtype == np.bool_
    assert restored.head_rgb_uint8.flags.c_contiguous
    with pytest.raises(TargetSelectionContractError):
        deserialize_policy_input(payload + b"x")
    invalid = _input()
    invalid.head_depth_m[0, 0] = np.nan
    with pytest.raises(TargetSelectionContractError, match="nonfinite"):
        serialize_policy_input(invalid)


def test_selectors_only_change_index_and_are_replayable() -> None:
    candidates = _candidate_set()

    candidate_index = select_candidate_index(candidates, (0.0, 0.0, 0.0))
    first = select_control_index(candidates, 123)
    second = select_control_index(candidates, 123)

    assert candidate_index == 0
    assert first == second
    assert first in (0, 1)
    assert select_control_index(CandidateSet((), (), b"x", hashlib.sha256(b"x").hexdigest()), 3) == -1


def test_candidate_score_transforms_final_base_into_acquisition_frame() -> None:
    candidates = CandidateSet(
        (),
        (
            Candidate((0.0, 0.0, 0.7), (-1.0, 0.0, 0.0), 0.14, 0.15, 80, 4, 0, 20, 20),
            Candidate((2.0, 0.0, 0.7), (-1.0, 0.0, 0.0), 0.14, 0.15, 80, 4, 1, 20, 20),
        ),
        b"frozen",
        hashlib.sha256(b"frozen").hexdigest(),
    )

    scores = candidate_scores(
        candidates,
        (11.0, 20.0, 0.0),
        acquisition_base_pose=(10.0, 20.0, 0.0),
    )

    assert scores[0] == pytest.approx(scores[1])


def test_candidate_range_is_measured_from_sampling_base_not_camera() -> None:
    rows, columns = np.indices((6, 6), dtype=np.float64)
    points = np.column_stack(
        (
            np.full(rows.size, 0.50),
            (columns.ravel() - 2.5) * 0.01,
            0.70 + (rows.ravel() - 2.5) * 0.01,
        )
    )

    candidate = _candidate_from_points(
        points,
        np.asarray((0.45, 0.0, 0.70)),
        np.asarray((0.0, 0.0, 0.0)),
        0.10,
        0,
        20,
        20,
    )

    assert candidate is not None


def test_primitive_is_same_index_bit_identical_and_uses_frozen_bounds() -> None:
    payload = serialize_policy_input(_input())
    candidate = _candidate_set().candidates[0]

    candidate_action = primitive_action(payload, candidate, (0.0, 0.0, 0.0), 550)
    control_action = primitive_action(payload, candidate, (0.0, 0.0, 0.0), 550)

    assert np.asarray(candidate_action, dtype="<f8").tobytes() == np.asarray(
        control_action, dtype="<f8"
    ).tobytes()
    assert len(candidate_action) == 16
    assert all(-0.35 <= value <= 0.35 for value in candidate_action[2:14])
    assert candidate_action[14:] == pytest.approx((0.0375, 0.0375))
    stopped = primitive_action(
        serialize_policy_input(_input(safety_state="stopped")),
        candidate,
        (0.0, 0.0, 0.0),
        550,
    )
    assert stopped == (0.0, 0.0, *(0.0,) * 12, 0.25, 0.25)


def test_primary_statistics_keep_all_supported_itt_pairs() -> None:
    records = []
    outcomes = [(1, 0)] * 12 + [(0, 1)] * 2 + [(1, 1)] * 2 + [(0, 0)] * 2
    for index, (candidate, control) in enumerate(outcomes):
        records.append(
            {
                "domain": "supported",
                "resolved": True,
                "task_id": ("a", "b", "c")[index % 3],
                "observation_latency_steps": 1 + index % 2,
                "candidate_event": candidate,
                "control_event": control,
            }
        )
    records.append(
        {
            "domain": "challenge",
            "resolved": True,
            "task_id": "a",
            "observation_latency_steps": 3,
            "candidate_event": 0,
            "control_event": 1,
        }
    )

    report = paired_primary_statistics(records)

    assert report["planned_pair_count"] == 18
    assert report["candidate_only"] == 12
    assert report["control_only"] == 2
    assert report["delta_itt"] == pytest.approx(10 / 18)
    assert set(report["by_observation_latency"]) == {"1", "2"}
