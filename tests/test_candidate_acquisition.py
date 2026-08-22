from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from hwr.adapters.mujoco.candidate_acquisition import (
    AcquisitionCapsule,
    AcquisitionContractError,
    AcquisitionEpisodeResult,
    capture_policy_input,
    compare_episode_replays,
    replay_candidate_set,
    validate_capture_identities,
)
from hwr.eval.target_selection import (
    PolicyVisibleInput,
    generate_candidate_set,
    serialize_policy_input,
)


def _input(**changes: object) -> PolicyVisibleInput:
    proprioception = np.zeros(37, dtype="<f8")
    proprioception[:6] = np.linspace(0.01, 0.06, 6)
    proprioception[12:18] = np.linspace(-0.01, -0.06, 6)
    proprioception[26:29] = (0.2, -0.1, 0.3)
    values = {
        "observation_timestamp_ns": 50_000_000,
        "sequence_id": 3,
        "phase_index": 1,
        "phase_step": 4,
        "policy_rng_seed": 17,
        "safety_state": "ok",
        "head_rgb_uint8": np.zeros((192, 256, 3), dtype=np.uint8),
        "head_depth_m": np.full((192, 256), 2.0, dtype="<f4"),
        "head_depth_valid": np.ones((192, 256), dtype=np.bool_),
        "head_camera_intrinsics": np.asarray(
            (200.0, 200.0, 127.5, 95.5), dtype="<f8"
        ),
        "robot_from_head_camera": np.eye(4, dtype="<f8"),
        "proprioception": proprioception,
        "executed_action_history": np.zeros((4, 16), dtype="<f8"),
        "history_available": np.asarray(
            (False, False, False, True), dtype=np.bool_
        ),
    }
    values.update(changes)
    return PolicyVisibleInput(**values)


def _capture(
    value: PolicyVisibleInput,
    *,
    ordinal: int = 0,
    phase: str = "A1_panorama",
    final: bool = False,
):
    return capture_policy_input(
        serialize_policy_input(value),
        capture_ordinal=ordinal,
        acquisition_phase=phase,
        final_input=final,
    )


def _capsule() -> AcquisitionCapsule:
    first = _capture(_input())
    second_value = _input(
        observation_timestamp_ns=100_000_000,
        sequence_id=4,
        phase_index=4,
        phase_step=5,
    )
    final = _capture(
        second_value, ordinal=1, phase="A4_seal", final=True
    )
    candidate_set = generate_candidate_set(
        (first.policy_input_bytes,),
        acquisition_base_pose=(0.2, -0.1, 0.3),
        final_input=final.policy_input_bytes,
    )
    return AcquisitionCapsule(
        planned_episode_id="a" * 64,
        task_id="tidy_living_room_3d/v1",
        cell_id="obs-1-action-1",
        replicate_ordinal=0,
        candidate_ordinal=2,
        environment_seed=11,
        policy_rng_seed=17,
        sampled_observation_latency_steps=1,
        sampled_action_latency_steps=1,
        acquisition_base_pose=(0.2, -0.1, 0.3),
        captures=(first, final),
        candidate_bytes=candidate_set.canonical_bytes,
        candidate_sha256=candidate_set.candidate_set_sha256,
        candidate_count=len(candidate_set.candidates),
        selected_index=-1,
        candidate_score_sha256=hashlib.sha256(b"").hexdigest(),
        acquisition_failure=None,
        proposed_action_sha256="b" * 64,
        applied_action_sha256="c" * 64,
        observation_identity_trace_sha256="d" * 64,
        same_seed_lockstep_replay=True,
        capture_enabled_disabled_identity=True,
    )


def test_capture_round_trip_is_bit_identical_and_immutable() -> None:
    payload = serialize_policy_input(_input())
    capture = capture_policy_input(
        payload,
        capture_ordinal=0,
        acquisition_phase="A1_panorama",
        final_input=False,
    )

    assert capture.policy_input_bytes == payload
    assert capture.policy_input_byte_count == len(payload)
    assert capture.observation_identity == (50_000_000, 3)
    assert capture.policy_input_sha256 != capture.candidate_visible_sha256
    with pytest.raises(FrozenInstanceError):
        capture.capture_ordinal = 2  # type: ignore[misc]


def test_candidate_visible_identity_ignores_non_candidate_policy_fields() -> None:
    baseline = _capture(_input())
    changed = _input(
        phase_index=8,
        phase_step=99,
        policy_rng_seed=999,
        safety_state="degraded",
        executed_action_history=np.ones((4, 16), dtype="<f8"),
        history_available=np.ones(4, dtype=np.bool_),
    )

    other = _capture(changed)

    assert baseline.policy_input_sha256 != other.policy_input_sha256
    assert baseline.candidate_visible_bytes == other.candidate_visible_bytes
    assert baseline.candidate_visible_sha256 == other.candidate_visible_sha256


@pytest.mark.parametrize(
    "field",
    (
        "head_rgb_uint8",
        "head_depth_m",
        "head_depth_valid",
        "head_camera_intrinsics",
        "robot_from_head_camera",
        "base_pose",
        "left_joint_position",
        "right_joint_position",
    ),
)
def test_every_candidate_visible_field_changes_identity(field: str) -> None:
    baseline_value = _input()
    baseline = _capture(baseline_value)
    values = {
        name: np.array(getattr(baseline_value, name), copy=True)
        for name in (
            "head_rgb_uint8",
            "head_depth_m",
            "head_depth_valid",
            "head_camera_intrinsics",
            "robot_from_head_camera",
            "proprioception",
        )
    }
    if field in values:
        values[field].flat[0] = (
            not bool(values[field].flat[0])
            if field == "head_depth_valid"
            else values[field].flat[0] + 1
        )
    else:
        slices = {
            "left_joint_position": slice(0, 6),
            "right_joint_position": slice(12, 18),
            "base_pose": slice(26, 29),
        }
        values["proprioception"][slices[field].start] += 1.0
    changed = _capture(
        replace(
            baseline_value,
            head_rgb_uint8=values["head_rgb_uint8"],
            head_depth_m=values["head_depth_m"],
            head_depth_valid=values["head_depth_valid"],
            head_camera_intrinsics=values["head_camera_intrinsics"],
            robot_from_head_camera=values["robot_from_head_camera"],
            proprioception=values["proprioception"],
        )
    )

    assert changed.candidate_visible_sha256 != baseline.candidate_visible_sha256


def test_same_observation_identity_with_different_visible_bytes_fails_closed() -> None:
    first = _capture(_input())
    depth = np.array(_input().head_depth_m, copy=True)
    depth[0, 0] += 0.5
    conflicting = _capture(
        _input(head_depth_m=depth),
        ordinal=1,
        phase="A3_panorama",
    )

    with pytest.raises(
        AcquisitionContractError, match="observation identity changed bytes"
    ):
        validate_capture_identities((first, conflicting))


def test_offline_candidate_replay_is_bit_identical_to_capsule() -> None:
    capsule = _capsule()

    replay = replay_candidate_set(capsule)

    assert replay.canonical_bytes == capsule.candidate_bytes
    assert replay.candidate_set_sha256 == capsule.candidate_sha256
    assert replay_candidate_set(capsule).canonical_bytes == replay.canonical_bytes


def test_capsule_rejects_noncanonical_payload_and_missing_final_input() -> None:
    capsule = _capsule()

    with pytest.raises(AcquisitionContractError, match="round-trip"):
        replace(
            capsule.captures[0],
            policy_input_bytes=capsule.captures[0].policy_input_bytes + b"x",
        )
    with pytest.raises(AcquisitionContractError, match="one final input"):
        replace(capsule, captures=capsule.captures[:1])


def test_replay_comparison_covers_actions_observations_candidates_and_capture() -> None:
    capsule = _capsule()
    first = AcquisitionEpisodeResult(
        capsule=capsule,
        trace_step_count=995,
        action_bounds_valid=True,
        stale_action_applied_count=0,
        severe_collision_count=0,
        invalid_force_count=0,
        p40_conservation_maximum_difference=0.0,
        safety_intervention_count=0,
    )

    assert compare_episode_replays(first, first)["passed"] is True
    changed = replace(
        first,
        capsule=replace(capsule, applied_action_sha256="e" * 64),
    )
    report = compare_episode_replays(first, changed)
    assert report["passed"] is False
    assert report["checks"]["applied_action_trace"] is False
