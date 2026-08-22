from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hwr.adapters.mujoco import cartesian_convergence as bridge
from hwr.adapters.mujoco import cartesian_convergence_provenance as provenance
from hwr.adapters.mujoco import dual_arm_backend
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.core.embodied import DualArmAction
from hwr.eval import target_selection
from hwr.eval.target_selection import Candidate, PolicyVisibleInput
from hwr.safety import SafetyLimits


def test_treatment_injection_calls_same_primitive_and_restores_helper(
    monkeypatch,
) -> None:
    calls = []
    helper = target_selection.acquisition_error_to_base_velocity

    def primitive(payload, candidate, acquisition_pose, post_step):
        calls.append(
            (
                payload,
                candidate,
                acquisition_pose,
                post_step,
                target_selection.acquisition_error_to_base_velocity,
            )
        )
        return (0.0,) * 16

    monkeypatch.setattr(target_selection, "primitive_action", primitive)
    candidate = _candidate()

    fixed = bridge._primitive_action(b"input", candidate, (0, 0, 0), 400, "frame_fixed")
    legacy = bridge._primitive_action(
        b"input", candidate, (0, 0, 0), 400, "frame_legacy"
    )

    assert fixed == legacy == (0.0,) * 16
    assert calls[0][-1] is helper
    assert calls[1][-1] is bridge.legacy_transform
    assert calls[0][:4] == calls[1][:4]
    assert target_selection.acquisition_error_to_base_velocity is helper


def test_treatment_injection_changes_only_arm_linear_xy() -> None:
    payload = target_selection.serialize_policy_input(_policy_input())
    candidate = _candidate()
    legacy = bridge._primitive_action(
        payload, candidate, (0.0, 0.0, 0.0), 400, "frame_legacy"
    )
    fixed = bridge._primitive_action(
        payload, candidate, (0.0, 0.0, 0.0), 400, "frame_fixed"
    )
    guard = bridge.first_treatment_guard(legacy, fixed)

    assert guard["different_bytes"] is True
    assert guard["only_arm_linear_xy_differs"] is True
    assert guard["arm_action_noncollapsed"] is True


def test_noncollinear_b1_pose_targets_match_primitive_actual_errors() -> None:
    candidate = _candidate()
    value = _policy_input()
    proprioception = value.proprioception.copy()
    proprioception[26:29] = (0.17, -0.11, np.pi / 2.0)
    value = replace(value, proprioception=proprioception)
    payload = target_selection.serialize_policy_input(value)
    targets = bridge.preposition_targets(
        candidate, (0.0, 0.0, 0.0), value.base_pose
    )

    _, crosscheck = bridge._fixed_action_with_target_crosscheck(
        payload, candidate, (0.0, 0.0, 0.0), targets
    )

    assert crosscheck["passed"] is True
    assert len(crosscheck["actual_error_calls"]) == 2
    assert crosscheck["reconstructed_targets"]["left"] == pytest.approx(
        targets["left"]
    )
    assert crosscheck["reconstructed_targets"]["right"] == pytest.approx(
        targets["right"]
    )


def test_continuation_identity_covers_queues_history_and_counters(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance.mujoco,
        "mj_stateSize",
        lambda model, specification: 3,
    )
    monkeypatch.setattr(
        provenance.mujoco,
        "mj_getState",
        lambda model, data, target, specification: target.__setitem__(
            slice(None), (1.0, 2.0, 3.0)
        ),
    )
    backend = _fake_backend()
    observation = _observation()
    graph = _FakeGraph()

    first = provenance.continuation_identity(
        backend, observation, [(0.0,) * 16], [True], graph
    )
    assert set(first["components"]) == {
        "mujoco_model_state",
        "mujoco_data_state",
        "actuator_servo_targets",
        "action_latency_queue",
        "observation_latency_queue",
        "policy_history_availability",
        "current_observation",
        "timestamp_sequence_runtime_safety_counters",
    }

    backend._action_queue.append(
        DualArmAction(0.0, 0.0, (0.1,) * 6, (0.0,) * 6, 0.0, 0.0)
    )
    second = provenance.continuation_identity(
        backend, observation, [(0.0,) * 16], [True], graph
    )
    assert first["identity"] != second["identity"]
    assert first["components"]["action_latency_queue"] != second["components"][
        "action_latency_queue"
    ]

    backend._observation_queue.append(observation)
    queued = provenance.continuation_identity(
        backend, observation, [(0.0,) * 16], [True], graph
    )
    assert second["identity"] != queued["identity"]
    assert second["components"]["observation_latency_queue"] != queued[
        "components"
    ]["observation_latency_queue"]

    third = provenance.continuation_identity(
        backend, observation, [(0.1,) * 16], [True], graph
    )
    assert queued["identity"] != third["identity"]
    backend._steps += 1
    fourth = provenance.continuation_identity(
        backend, observation, [(0.1,) * 16], [True], graph
    )
    assert third["identity"] != fourth["identity"]


def test_distance_uses_live_mujoco_grasp_center_sites() -> None:
    run = SimpleNamespace(
        acquisition_world_origin=(1.0, 2.0, 0.0),
        acquisition_pose=(0.0, 0.0, np.pi / 2.0),
        backend=SimpleNamespace(
            data=SimpleNamespace(
                site_xpos=np.asarray(
                    ((1.0, 3.0, 0.0), (2.0, 2.0, 0.0)),
                    np.float64,
                )
            ),
            _left_tool_site=0,
            _right_tool_site=1,
        ),
    )
    targets = {"left": (1.0, 0.0, 0.0), "right": (0.0, -1.0, 0.0)}

    distance = bridge._distance_record(run, targets)

    assert distance["left_m"] == pytest.approx(0.0)
    assert distance["right_m"] == pytest.approx(0.0)
    assert distance["mean_m"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "task_id",
    (
        "tidy_living_room_3d/v1",
        "clear_dining_table_3d/v1",
        "store_kitchen_items_3d/v1",
    ),
)
def test_natural_latency_sampler_matches_backend(
    task_id: str, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    tasks, bindings = load_default_formal_household_catalogs(root)
    monkeypatch.setattr(
        dual_arm_backend, "MujocoCameraRenderer", _HeadlessRenderer
    )
    evaluator = bridge.CartesianConvergenceMujoco(
        tasks[task_id], bindings[task_id]
    )
    expected = evaluator.sample_latencies(20_265_101)
    backend = evaluator._diagnostic._backend()
    try:
        backend.reset(seed=20_265_101, task_id=task_id)
        randomization = backend.task_audit()["randomization"]
    finally:
        backend.close()

    assert expected == (
        int(randomization["observation_latency_steps"]),
        int(randomization["action_latency_steps"]),
    )


def _candidate() -> Candidate:
    return Candidate(
        center=(1.0, 0.2, 0.8),
        normal=(-1.0, 0.0, 0.0),
        width=0.12,
        prominence=0.1,
        support_count=30,
        view_count=2,
        first_frame=0,
        first_row=20,
        first_column=30,
    )


def _policy_input() -> PolicyVisibleInput:
    proprioception = np.zeros(37, dtype="<f8")
    proprioception[26:29] = (0.0, 0.0, np.pi / 2.0)
    return PolicyVisibleInput(
        1,
        1,
        bridge.B2_PHASE_INDEX,
        0,
        5,
        "ok",
        np.zeros((192, 256, 3), np.uint8),
        np.ones((192, 256), dtype="<f4"),
        np.ones((192, 256), dtype=np.bool_),
        np.asarray((200.0, 200.0, 127.5, 95.5), dtype="<f8"),
        np.eye(4, dtype="<f8"),
        proprioception,
        np.zeros((4, 16), dtype="<f8"),
        np.asarray((False, False, False, True), dtype=np.bool_),
    )


def _observation():
    from hwr.core.embodied import (
        DualArmObservation,
        DualArmProprioception,
        NaturalLanguageInstruction,
    )
    from hwr.core.types import SafetyState

    proprioception = DualArmProprioception(
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        (0.0,) * 6,
        0.0,
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0),
        (0.0,) * 6,
    )
    return DualArmObservation(
        100,
        2,
        "task/v1",
        NaturalLanguageInstruction("test"),
        proprioception,
        (),
        (),
        SafetyState.OK,
    )


def _fake_backend():
    arrays = SimpleNamespace(
        body_mass=np.asarray((1.0,)),
        body_inertia=np.asarray(((1.0, 1.0, 1.0),)),
        geom_friction=np.asarray(((1.0, 0.1, 0.1),)),
        light_diffuse=np.asarray(((1.0, 1.0, 1.0),)),
        mat_rgba=np.asarray(((1.0, 1.0, 1.0, 1.0),)),
        cam_pos=np.asarray(((0.0, 0.0, 0.0),)),
        cam_quat=np.asarray(((1.0, 0.0, 0.0, 0.0),)),
        cam_fovy=np.asarray((60.0,)),
    )
    data = SimpleNamespace(ctrl=np.asarray((0.0, 0.0)))
    ledger = SimpleNamespace(report=lambda: {"count": 1})
    safety = SimpleNamespace(limits=SafetyLimits(0.45, 1.0, 1.0))
    return SimpleNamespace(
        model=arrays,
        data=data,
        _left_targets=np.asarray((0.0,) * 6),
        _right_targets=np.asarray((0.0,) * 6),
        _action_queue=[],
        _observation_queue=[],
        _timestamp_ns=lambda: 100,
        _sequence=2,
        _steps=5,
        result=lambda: None,
        _placement=SimpleNamespace(stable_steps=0),
        _left_contact_steps=0,
        _right_contact_steps=0,
        _simultaneous_contact_steps=0,
        _concurrent_steps=0,
        _maximum_concurrent_steps=0,
        _severe_collision_count=0,
        _maximum_forbidden_force=0.0,
        _maximum_forbidden_pair=None,
        _episode_seed=7,
        _step_left_contact=False,
        _step_right_contact=False,
        _initial_target_distance=1.0,
        _maximum_controlled_target_progress=0.0,
        _maximum_controlled_articulation_progress=0.0,
        _previous_potential=0.0,
        _randomization={"action_latency_steps": 1, "observation_latency_steps": 2},
        _rng=SimpleNamespace(getstate=lambda: ("fake", 1)),
        _camera_rendering_enabled=True,
        _cached_cameras=(),
        safety=safety,
        contact_ledger=ledger,
    )


class _FakeGraph:
    def report(self):
        return {"count": 2}


class _HeadlessRenderer:
    def __init__(self, model, *, width, height):
        self.width = width
        self.height = height

    def rgb(self, data, camera_name, *, timestamp_ns, frame_index, camera_id=None):
        from hwr.core.types import CameraFrame

        return CameraFrame(
            camera_id or camera_name,
            timestamp_ns,
            frame_index,
            self.width,
            self.height,
            "rgb8",
            payload=bytes(self.width * self.height * 3),
        )

    def depth(
        self, data, camera_name, *, timestamp_ns, frame_index, camera_id=None
    ):
        from hwr.core.types import CameraFrame

        payload = np.ones(
            (self.height, self.width), dtype=np.float32
        ).tobytes()
        return CameraFrame(
            camera_id or camera_name,
            timestamp_ns,
            frame_index,
            self.width,
            self.height,
            "depth32f",
            payload=payload,
        )

    def close(self):
        return None
