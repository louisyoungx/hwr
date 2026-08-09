from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from hwr.adapters.mujoco import (  # noqa: E402
    Mujoco3DBackend,
    Mujoco3DConfig,
    inspect_robot_model,
    run_contact_grasp_trial,
)
from hwr.core.types import ActionFrame  # noqa: E402


MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/mujoco/mobile_manipulator_smoke.xml"


def _action(observation, *, linear: float = 0.0) -> ActionFrame:
    return ActionFrame(
        created_at_ns=observation.timestamp_ns,
        valid_from_ns=observation.timestamp_ns,
        valid_until_ns=observation.timestamp_ns + 100_000_000,
        source="test_policy",
        base_linear=linear,
        arm_command=(0.0,) * 6,
    )


def test_compiled_robot_has_required_dynamics_and_sensors() -> None:
    backend = Mujoco3DBackend(Mujoco3DConfig(model_path=MODEL_PATH, camera_width=80, camera_height=60))
    try:
        report = inspect_robot_model(backend.model)
    finally:
        backend.close()

    assert report.valid
    assert report.wheel_joint_count == 4
    assert report.arm_joint_count == 6
    assert report.finger_joint_count == 2
    assert report.policy_cameras == ("head_rgb", "head_depth", "wrist_rgb")
    assert report.invalid_dynamic_bodies == ()
    assert report.equality_constraint_count == 0
    assert report.gravity == (0.0, 0.0, -9.81)


def test_backend_exposes_pixels_without_privileged_features() -> None:
    backend = Mujoco3DBackend(Mujoco3DConfig(model_path=MODEL_PATH, camera_width=80, camera_height=60))
    try:
        observation = backend.reset(seed=4, task_id=backend.config.task_id)
    finally:
        backend.close()

    assert len(observation.joint_position) == 6
    assert observation.task_stage == "instruction_following"
    assert observation.features == {}
    assert tuple(camera.camera_id for camera in observation.cameras) == (
        "head_rgb",
        "head_depth",
        "wrist_rgb",
    )
    head_rgb, head_depth, wrist_rgb = observation.cameras
    assert len(head_rgb.payload or b"") == 80 * 60 * 3
    assert len(head_depth.payload or b"") == 80 * 60 * 4
    assert len(wrist_rgb.payload or b"") == 80 * 60 * 3
    depth = np.frombuffer(head_depth.payload, dtype=np.float32)
    assert np.isfinite(depth).all()
    assert depth.max() > depth.min() > 0


def test_four_wheel_actuation_moves_physical_base_forward() -> None:
    backend = Mujoco3DBackend(Mujoco3DConfig(model_path=MODEL_PATH, camera_width=64, camera_height=48))
    try:
        observation = backend.reset(seed=2, task_id=backend.config.task_id)
        start_x = observation.base_pose[0]
        for _ in range(5):
            outcome = backend.apply(_action(observation, linear=0.12))
            observation = outcome.observation
    finally:
        backend.close()

    assert observation.base_pose[0] > start_x + 0.015
    assert outcome.info["physics_contacts"] > 0
    assert outcome.info["applied_action"].source == "test_policy"


def test_gripper_lifts_object_using_bilateral_contacts_without_weld() -> None:
    backend = Mujoco3DBackend(
        Mujoco3DConfig(
            model_path=MODEL_PATH,
            camera_width=32,
            camera_height=24,
            max_steps=400,
        )
    )
    try:
        report = run_contact_grasp_trial(backend, seed=11)
    finally:
        backend.close()

    assert report.success
    assert report.equality_constraint_count == 0
    assert report.bilateral_contact_steps >= 20
    assert report.maximum_left_normal_force > 0
    assert report.maximum_right_normal_force > 0
    assert report.final_object_position[2] - report.initial_object_position[2] >= 0.25
    assert report.action_source == "privileged_3d_expert"
