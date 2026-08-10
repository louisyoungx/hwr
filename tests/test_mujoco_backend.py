from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from hwr.adapters.mujoco import (  # noqa: E402
    Mujoco3DBackend,
    Mujoco3DConfig,
    PrivilegedCartesianExpert,
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
    assert report.manipulator_count == 2
    assert report.arm_joint_count == 12
    assert report.finger_joint_count == 4
    assert report.central_body_present
    assert report.top_camera_present
    assert report.overall_height_m == pytest.approx(1.60)
    assert report.arm_mount_y_m == pytest.approx((-0.31, 0.31))
    assert report.finger_joint_travel_m == pytest.approx((0.095,) * 4)
    assert report.gripper_open_gaps_m == pytest.approx((0.22, 0.22))
    assert report.gripper_closed_gaps_m == pytest.approx((0.03, 0.03))
    assert report.policy_cameras == ("head_rgb", "head_depth", "wrist_rgb")
    assert report.invalid_dynamic_bodies == ()
    assert report.equality_constraint_count == 0
    assert report.gravity == (0.0, 0.0, -9.81)


@pytest.mark.parametrize("side", ["right", "left"])
def test_gripper_uses_slender_pincer_links_and_contact_pads(side: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    palm_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_gripper_palm_shell"
    )
    assert palm_id >= 0
    assert model.geom_type[palm_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_size[palm_id, 1] <= 0.09

    for finger in ("left", "right"):
        prefix = f"{side}_gripper_{finger}"
        knuckle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_knuckle"
        )
        assert knuckle_id >= 0
        assert model.geom_type[knuckle_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        for segment in ("proximal", "distal", "tip"):
            segment_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_{segment}"
            )
            assert segment_id >= 0
            assert model.geom_type[segment_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
        pad_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_pad"
        )
        assert pad_id >= 0
        assert model.geom_type[pad_id] == mujoco.mjtGeom.mjGEOM_BOX
        assert model.geom_size[pad_id, 1] <= 0.012

    obsolete_backstop = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_gripper_pull_backstop"
    )
    assert obsolete_backstop == -1


def test_secondary_arm_is_dynamic_and_holds_its_stowed_pose() -> None:
    backend = Mujoco3DBackend(
        Mujoco3DConfig(model_path=MODEL_PATH, camera_width=32, camera_height=24)
    )
    try:
        observation = backend.reset(seed=7, task_id=backend.config.task_id)
        initial = tuple(
            float(backend.data.qpos[backend.model.jnt_qposadr[joint_id]])
            for joint_id in backend.bundle.ids.secondary_arm_joints
        )
        for _ in range(5):
            observation = backend.apply(_action(observation, linear=0.05)).observation
        final = tuple(
            float(backend.data.qpos[backend.model.jnt_qposadr[joint_id]])
            for joint_id in backend.bundle.ids.secondary_arm_joints
        )
    finally:
        backend.close()

    assert len(backend.bundle.ids.secondary_arm_actuators) == 6
    assert final == pytest.approx(initial, abs=0.07)


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


def test_cartesian_expert_can_weight_orientation_without_changing_action_contract() -> None:
    backend = Mujoco3DBackend(
        Mujoco3DConfig(model_path=MODEL_PATH, camera_width=32, camera_height=24)
    )
    try:
        observation = backend.reset(seed=4, task_id=backend.config.task_id)
        expert = PrivilegedCartesianExpert(backend)
        expert.set_orientation_target(np.eye(3))
        action = expert.action(
            observation,
            target_position=expert.site_position(),
            gripper_target=0.0,
            orientation_weight=0.10,
        )
    finally:
        backend.close()

    assert len(action.arm_command) == 6
    assert np.isfinite(action.arm_command).all()


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


@pytest.mark.parametrize("seed", [0, 1, 2, 11, 17])
def test_gripper_lifts_object_using_bilateral_contacts_without_weld(seed: int) -> None:
    backend = Mujoco3DBackend(
        Mujoco3DConfig(
            model_path=MODEL_PATH,
            camera_width=32,
            camera_height=24,
            max_steps=400,
        )
    )
    try:
        report = run_contact_grasp_trial(backend, seed=seed)
    finally:
        backend.close()

    assert report.success
    assert report.equality_constraint_count == 0
    assert report.bilateral_contact_steps >= 20
    assert report.maximum_left_normal_force > 0
    assert report.maximum_right_normal_force > 0
    assert report.final_object_position[2] - report.initial_object_position[2] >= 0.25
    assert report.action_source == "privileged_3d_expert"
