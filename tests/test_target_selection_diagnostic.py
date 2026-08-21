from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hwr.adapters.mujoco.target_selection_diagnostic import (
    ACQUISITION_PHASES,
    MainEventTracker,
    TargetSelectionDiagnostic,
    _AcquisitionState,
    _acquisition_phase,
    _input_failure,
    policy_input_bytes,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.core.embodied import (
    DualArmObservation,
    DualArmProprioception,
    FrameCameraCalibration,
    NaturalLanguageInstruction,
)
from hwr.core.types import CameraFrame, SafetyState
from hwr.eval.target_selection import deserialize_policy_input
from hwr.eval.target_selection_safety import evaluate_safety_guards


def _period(
    *,
    grasps=(),
    contacts=(),
    settling: bool = False,
) -> dict[str, object]:
    return {
        "reset_settling_excluded": settling,
        "substeps": [
            {
                "same_object_dual_arm_grasps": list(grasps),
                "same_entity_dual_arm_contacts": list(contacts),
            }
        ],
    }


def test_main_event_requires_dual_pad_grasp_for_manipulated_object() -> None:
    entity = "manipulated_object:cup"
    tracker = MainEventTracker()

    tracker.update(_period(contacts=(entity,)), {entity: (0.0, 0.0, 0.0)}, {entity: (0.02, 0.0, 0.0)})
    assert tracker.event is False

    tracker.update(_period(grasps=(entity,)), {entity: (0.02, 0.0, 0.0)}, {entity: (0.025, 0.0, 0.0)})
    assert tracker.event is False
    tracker.update(_period(grasps=(entity,)), {entity: (0.025, 0.0, 0.0)}, {entity: (0.035, 0.0, 0.0)})
    assert tracker.event is True
    assert tracker.entities == {entity}


def test_articulation_uses_same_entity_contact_and_settling_is_excluded() -> None:
    entity = "articulation:drawer"
    tracker = MainEventTracker()

    tracker.update(_period(contacts=(entity,), settling=True), {entity: 0.0}, {entity: 0.02})
    assert tracker.event is False
    tracker.update(_period(contacts=(entity,)), {entity: 0.02}, {entity: 0.025})
    tracker.update(_period(contacts=(entity,)), {entity: 0.025}, {entity: 0.035})

    assert tracker.event is True
    assert tracker.maximum_motion[entity] == pytest.approx(0.015)


def test_policy_input_bridge_serializes_only_frozen_visible_fields() -> None:
    rgb = np.zeros((192, 256, 3), np.uint8)
    depth = np.ones((192, 256), np.float32)
    proprioception = DualArmProprioception(
        left_joint_position=(0.0,) * 6,
        left_joint_velocity=(0.0,) * 6,
        right_joint_position=(0.0,) * 6,
        right_joint_velocity=(0.0,) * 6,
        left_gripper_position=0.1,
        right_gripper_position=0.2,
        base_pose=(1.0, 2.0, 0.3),
        base_twist=(0.0, 0.0),
        imu=(0.0,) * 6,
    )
    observation = DualArmObservation(
        timestamp_ns=1_000,
        sequence_id=4,
        task_id="private-task",
        instruction=NaturalLanguageInstruction("private instruction"),
        proprioception=proprioception,
        cameras=(
            CameraFrame("head_rgb", 1_000, 4, 256, 192, "rgb8", payload=rgb.tobytes()),
            CameraFrame("head_depth", 1_000, 4, 256, 192, "depth32f", payload=depth.tobytes()),
        ),
        camera_calibrations=(
            FrameCameraCalibration("head_rgb", (200.0, 200.0, 127.5, 95.5), tuple(np.eye(4).flat)),
            FrameCameraCalibration("head_depth", (200.0, 200.0, 127.5, 95.5), tuple(np.eye(4).flat)),
        ),
        safety_state=SafetyState.OK,
    )

    payload = policy_input_bytes(observation, [], [], 9, phase_index=2, phase_step=3)
    restored = deserialize_policy_input(payload)

    assert restored.base_pose == (1.0, 2.0, 0.3)
    assert restored.policy_rng_seed == 9
    assert b"private-task" not in payload
    assert b"private instruction" not in payload


def test_acquisition_phase_identity_and_forward_projection_are_frozen() -> None:
    assert _acquisition_phase(0) == (0, "A0_stable", 0)
    assert _acquisition_phase(ACQUISITION_PHASES[0][1]) == (
        1,
        "A1_panorama",
        0,
    )
    state = _AcquisitionState((1.0, 2.0, np.pi / 2.0))
    value = _input_with_pose((2.3, 2.5, np.pi / 2.0))

    action, _ = state.action("A2_forward", policy_input_bytes(
        value, [], [], 9, phase_index=2, phase_step=0
    ))

    assert action[0] == pytest.approx(0.12)


@pytest.mark.parametrize(
    "task_id",
    (
        "tidy_living_room_3d/v1",
        "clear_dining_table_3d/v1",
        "store_kitchen_items_3d/v1",
    ),
)
def test_pure_latency_sampler_matches_formal_backend(task_id: str) -> None:
    root = Path(__file__).resolve().parents[1]
    tasks, bindings = load_default_formal_household_catalogs(root)
    diagnostic = TargetSelectionDiagnostic(tasks[task_id], bindings[task_id])
    expected = diagnostic.sample_latencies(20264102)
    backend = diagnostic._backend()
    try:
        backend.reset(seed=20264102, task_id=task_id)
        randomization = backend.task_audit()["randomization"]
    finally:
        backend.close()

    assert expected == (
        int(randomization["observation_latency_steps"]),
        int(randomization["action_latency_steps"]),
    )


def test_latency_warmup_repeats_are_allowed_but_partial_or_backward_identity_fails() -> None:
    observation = _input_with_pose((0.0, 0.0, 0.0))
    payload = policy_input_bytes(
        observation, [], [], 9, phase_index=1, phase_step=0
    )
    backend = type("Backend", (), {"_timestamp_ns": lambda self: 1_000})()

    assert _input_failure(
        backend,
        observation,
        payload,
        supported_only=True,
        previous_identity=(1_000, 4),
    ) is None
    assert _input_failure(
        backend,
        observation,
        payload,
        supported_only=True,
        previous_identity=(999, 4),
    ) == "nonmonotonic_observation"
    assert _input_failure(
        backend,
        observation,
        payload,
        supported_only=True,
        previous_identity=(1_000, 5),
    ) == "nonmonotonic_observation"


def _input_with_pose(pose: tuple[float, float, float]) -> DualArmObservation:
    rgb = np.zeros((192, 256, 3), np.uint8)
    depth = np.full((192, 256), 5.0, np.float32)
    proprioception = DualArmProprioception(
        left_joint_position=(0.0,) * 6,
        left_joint_velocity=(0.0,) * 6,
        right_joint_position=(0.0,) * 6,
        right_joint_velocity=(0.0,) * 6,
        left_gripper_position=0.0,
        right_gripper_position=0.0,
        base_pose=pose,
        base_twist=(0.0, 0.0),
        imu=(0.0,) * 6,
    )
    return DualArmObservation(
        timestamp_ns=1_000,
        sequence_id=4,
        task_id="private-task",
        instruction=NaturalLanguageInstruction("private instruction"),
        proprioception=proprioception,
        cameras=(
            CameraFrame("head_rgb", 1_000, 4, 256, 192, "rgb8", payload=rgb.tobytes()),
            CameraFrame("head_depth", 1_000, 4, 256, 192, "depth32f", payload=depth.tobytes()),
        ),
        camera_calibrations=(
            FrameCameraCalibration("head_rgb", (200.0, 200.0, 127.5, 95.5), tuple(np.eye(4).flat)),
            FrameCameraCalibration("head_depth", (200.0, 200.0, 127.5, 95.5), tuple(np.eye(4).flat)),
        ),
        safety_state=SafetyState.OK,
    )


def test_safety_guards_keep_force_and_impulse_separate_and_require_support() -> None:
    records = [
        _safety_record(task, observation, action)
        for task in (
            "tidy_living_room_3d/v1",
            "clear_dining_table_3d/v1",
            "store_kitchen_items_3d/v1",
        )
        for observation in (1, 2)
        for action in (1, 2, 3)
    ]

    report = evaluate_safety_guards(records)

    assert report["hard_guard"]["passed"] is True
    assert report["target_contact_intensity_guard"]["supported"] is True
    assert report["target_contact_intensity_guard"]["passed"] is True
    floor = report["non_target_allowed_contact_guard"]["categories"][
        "floor_support"
    ]
    assert set(floor) == {"peak_force", "cumulative_impulse"}
    assert floor["peak_force"]["point_ratio"] == pytest.approx(1.0)
    assert floor["cumulative_impulse"]["point_ratio"] == pytest.approx(1.0)

    insufficient = evaluate_safety_guards(records[:3])
    assert insufficient["target_contact_intensity_guard"]["supported"] is False
    assert insufficient["target_contact_intensity_guard"]["passed"] is False


def _safety_record(task: str, observation: int, action: int) -> dict[str, object]:
    categories = {
        "floor_support": {
            "category_peak_force": 2.0,
            "cumulative_impulse": 3.0,
        },
        "target_container": {
            "category_peak_force": 0.0,
            "cumulative_impulse": 0.0,
        },
    }
    branch = {
        "invalid_force_count": 0,
        "severe_collision_count": 0,
        "stale_action_applied_count": 0,
        "p40_conservation": {"maximum_absolute_difference": 0.0},
        "action_bounds_valid": True,
        "main_event_entities": ["manipulated_object:item"],
        "entity_contact_graph": {
            "legacy_p40_categories": categories,
            "robot_environment_edges": [
                {
                    "entity": "manipulated_object:item",
                    "substep_peak_force": 4.0,
                    "cumulative_impulse": 2.0,
                    "contact_duration_seconds": 0.5,
                }
            ],
        },
    }
    return {
        "domain": "supported",
        "task_id": task,
        "observation_latency_steps": observation,
        "action_latency_steps": action,
        "candidate": branch,
        "control": branch,
    }
