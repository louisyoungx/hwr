from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from hwr.adapters.mujoco import candidate_acquisition as acquisition
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
        runtime_observation_latency_steps=1,
        runtime_action_latency_steps=1,
        latency_override_inactive=True,
        runtime_randomization_sha256="e" * 64,
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
        same_seed_validation_replay=True,
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
    first = _replay_record(0, True)
    second = _replay_record(1, False)

    assert compare_episode_replays(first, second)["passed"] is True
    changed = {**second, "applied_action_sha256": "e" * 64}
    report = compare_episode_replays(first, changed)
    assert report["passed"] is False
    assert report["checks"]["applied_action_trace"] is False


def _replay_record(run_ordinal: int, capture_enabled: bool) -> dict[str, object]:
    return {
        "backend_run_ordinal": run_ordinal,
        "_backend_object": object(),
        "reset_count": 1,
        "capture_persistence_enabled": capture_enabled,
        "captures": [object()] if capture_enabled else [],
        "environment_seed": 11,
        "policy_rng_seed": 17,
        "acquisition_pose": (0.0, 0.0, 0.0),
        "runtime_randomization_sha256": "a" * 64,
        "runtime_randomization": {
            "observation_latency_steps": 1,
            "action_latency_steps": 1,
        },
        "runtime_observation_latency_steps": 1,
        "runtime_action_latency_steps": 1,
        "latency_override_inactive": True,
        "physical_trace_sha256": "b" * 64,
        "policy_input_trace_sha256": "c" * 64,
        "observation_identity_trace_sha256": "d" * 64,
        "capture_identity_sequence": ((1, 1),),
        "capture_payload_sha256": "e" * 64,
        "proposed_action_sha256": "f" * 64,
        "applied_action_sha256": "0" * 64,
        "candidate_bytes": b"candidate",
        "candidate_sha256": "1" * 64,
        "candidate_count": 0,
        "candidate_score_sha256": "2" * 64,
        "selected_index": -1,
        "failure": None,
        "runtime_terminal": False,
        "trace_step_count": 995,
    }


class _Ledger:
    def set_enabled(self, enabled: bool) -> None:
        assert enabled is True

    def report(self) -> dict[str, object]:
        return {}


class _Backend:
    def __init__(self, observation, latency=(1, 1), *, terminal=False) -> None:
        self.observation = observation
        self.latency = latency
        self.terminal = terminal
        self.contact_ledger = _Ledger()
        self.reset_calls = 0
        self.closed = False
        self._after_physics_substep = lambda: None

    def reset(self, *, seed: int, task_id: str):
        assert seed == 11
        assert task_id == "tidy_living_room_3d/v1"
        self.reset_calls += 1
        return self.observation

    def task_audit(self) -> dict[str, object]:
        observation, action = self.latency
        return {
            "randomization": {
                "observation_latency_steps": observation,
                "action_latency_steps": action,
                "other": 4,
            },
            "action_latency_diagnostic": None,
            "observation_latency_diagnostic": None,
            "latency_pair_diagnostic": None,
            "severe_collision_count": 0,
        }

    def close(self) -> None:
        self.closed = True


class _Graph:
    def reset(self) -> None:
        pass

    def sample_mujoco_substep(self, model, data) -> None:
        del model, data

    def report(self) -> dict[str, object]:
        return {name: 0 for name in acquisition.INVALID_GRAPH_FIELDS}


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    latency: tuple[int, int] = (1, 1),
    failure: str | None = None,
    terminal: bool = False,
    safety_intervened: bool = False,
):
    observation = SimpleNamespace(
        timestamp_ns=1,
        sequence_id=1,
        proprioception=SimpleNamespace(base_pose=(0.0, 0.0, 0.0)),
    )
    backends = [_Backend(observation, latency, terminal=terminal) for _ in range(2)]
    monkeypatch.setattr(acquisition, "ACQUISITION_STEPS", 2)
    monkeypatch.setattr(acquisition, "_graph_from_backend", lambda backend: _Graph())
    monkeypatch.setattr(
        acquisition,
        "_acquisition_phase",
        lambda step: (1, "A1_panorama", step),
    )
    monkeypatch.setattr(
        acquisition,
        "policy_input_bytes",
        lambda observation, history, available, seed, **kwargs: (
            serialize_policy_input(_input(
                observation_timestamp_ns=observation.timestamp_ns,
                sequence_id=observation.sequence_id,
                phase_index=kwargs["phase_index"],
                phase_step=kwargs["phase_step"],
                policy_rng_seed=seed,
            ))
        ),
    )
    monkeypatch.setattr(
        acquisition,
        "_input_failure",
        lambda *args, **kwargs: failure,
    )
    monkeypatch.setattr(
        acquisition,
        "_step",
        lambda backend, graph, observation, action, step: (
            SimpleNamespace(
                timestamp_ns=observation.timestamp_ns + 1,
                sequence_id=observation.sequence_id + 1,
                proprioception=observation.proprioception,
            ),
            {"reset_settling_excluded": False, "substeps": []},
            {
                "proposed_action": list(action),
                "applied_action": list(action),
                "hold_action": list(action),
                "proprioception": [float(step)],
                "events": [],
                "safety_intervened": safety_intervened,
                "outside_validity_window": False,
                "action_bounds_valid": True,
                "terminated": terminal,
                "truncated": False,
                "terminal": terminal,
                "_motion_start": {},
                "_motion_end": {},
            },
        ),
    )
    monkeypatch.setattr(
        acquisition,
        "p40_conservation_differences",
        lambda graph, ledger: {"maximum_absolute_difference": 0.0},
    )
    diagnostic = acquisition.CandidateAcquisitionDiagnostic(
        SimpleNamespace(task_id="tidy_living_room_3d/v1"), object()
    )
    monkeypatch.setattr(diagnostic, "_backend", lambda: backends.pop(0))
    return diagnostic, backends


def _plan() -> dict[str, object]:
    return {
        "planned_episode_id": "a" * 64,
        "cell_id": "cell-00-obs-1-action-1",
        "replicate_ordinal": 0,
        "candidate_ordinal": 0,
        "environment_seed": 11,
        "policy_rng_seed": 17,
        "sampled_observation_latency_steps": 1,
        "sampled_action_latency_steps": 1,
    }


def test_production_episode_uses_two_fresh_backends_and_disables_replay_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hwr.apps import persist_candidate_episode

    diagnostic, backends = _install_fake_runtime(monkeypatch)
    created = tuple(backends)

    result = diagnostic.run_episode(_plan())

    assert backends == []
    assert [backend.reset_calls for backend in created] == [1, 1]
    assert all(backend.closed for backend in created)
    assert result.replay_comparison["passed"] is True
    assert result.primary_summary["backend_run_ordinal"] == 0
    assert result.validation_summary["backend_run_ordinal"] == 1
    assert result.primary_summary["capture_persistence_enabled"] is True
    assert result.validation_summary["capture_persistence_enabled"] is False
    assert result.validation_summary["capture_payload_sha256"] == (
        result.primary_summary["capture_payload_sha256"]
    )
    terminal, capsule, _ = persist_candidate_episode(result)
    assert terminal["planned_latency"] == {
        "observation_steps": 1,
        "action_steps": 1,
    }
    assert terminal["runtime_latency"] == {
        "observation_steps": 1,
        "action_steps": 1,
        "override_inactive": True,
    }
    assert capsule["primary_run"]["runtime_randomization"] == (
        capsule["validation_replay"]["runtime_randomization"]
    )


def test_runtime_latency_drift_fails_before_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic, _ = _install_fake_runtime(monkeypatch, latency=(2, 1))

    with pytest.raises(AcquisitionContractError, match="runtime latency") as error:
        diagnostic.run_episode(_plan())

    assert error.value.details["planned_observation_latency_steps"] == 1
    assert error.value.details["runtime_observation_latency_steps"] == 2


def test_acquisition_failure_does_not_call_formal_candidate_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic, _ = _install_fake_runtime(
        monkeypatch, failure="forced_failure", terminal=True
    )
    monkeypatch.setattr(
        acquisition,
        "generate_candidate_set",
        lambda *args, **kwargs: pytest.fail("formal candidate generated after failure"),
    )

    result = diagnostic.run_episode(_plan())
    from hwr.apps import persist_candidate_episode

    terminal, capsule, blobs = persist_candidate_episode(result)

    assert result.capsule.acquisition_failure == "forced_failure"
    assert result.capsule.candidate_bytes == b""
    assert result.capsule.candidate_count == 0
    assert result.capsule.selected_index == -1
    assert terminal["runtime_terminal"] is True
    assert terminal["trace_step_count"] == 1
    assert capsule["candidate_set"]["generated_online"] is False
    assert not any(name.endswith("candidate-set.json") for name in blobs)


def test_production_safety_and_terminal_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic, _ = _install_fake_runtime(
        monkeypatch,
        failure="safety_intervention_during_acquisition",
        terminal=True,
        safety_intervened=True,
    )
    result = diagnostic.run_episode(_plan())

    assert result.runtime_terminal is True
    assert result.trace_step_count == 1
    assert result.safety_intervention_count == 1
    assert result.validation_summary["runtime_terminal"] is True
    assert result.validation_summary["safety_intervention_count"] == 1
