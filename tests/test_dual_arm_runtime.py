from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from hwr.adapters.mujoco import MujocoDualArmBackend, MujocoDualArmConfig  # noqa: E402
from hwr.core.embodied import DualArmAction, DualArmActionFrame  # noqa: E402


MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/mujoco/mobile_manipulator_smoke.xml"


def _frame(
    timestamp_ns: int,
    *,
    left: tuple[float, ...] = (0.0,) * 6,
    right: tuple[float, ...] = (0.0,) * 6,
    left_gripper: float = 0.0,
    right_gripper: float = 0.0,
    source: str = "test_actor",
) -> DualArmActionFrame:
    return DualArmActionFrame(
        created_at_ns=timestamp_ns,
        valid_from_ns=timestamp_ns,
        valid_until_ns=timestamp_ns + 100_000_000,
        source=source,
        action=DualArmAction(
            base_linear=0.0,
            base_angular=0.0,
            left_arm=left,
            right_arm=right,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
        ),
    )


def _backend() -> MujocoDualArmBackend:
    return MujocoDualArmBackend(
        MujocoDualArmConfig(
            model_path=MODEL_PATH,
            camera_width=40,
            camera_height=30,
            max_steps=20,
        )
    )


def test_observation_exposes_both_arms_and_four_deployable_cameras() -> None:
    backend = _backend()
    try:
        observation = backend.reset(seed=3, task_id=backend.config.task_id)
    finally:
        backend.close()

    proprioception = observation.proprioception
    assert len(proprioception.left_joint_position) == 6
    assert len(proprioception.right_joint_position) == 6
    assert len(proprioception.left_joint_velocity) == 6
    assert len(proprioception.right_joint_velocity) == 6
    assert tuple(camera.camera_id for camera in observation.cameras) == (
        "head_rgb",
        "head_depth",
        "left_wrist_rgb",
        "right_wrist_rgb",
    )
    assert tuple(value.camera_id for value in observation.camera_calibrations) == (
        "head_rgb",
        "head_depth",
        "left_wrist_rgb",
        "right_wrist_rgb",
    )
    assert all(value.intrinsics[0] > 0.0 for value in observation.camera_calibrations)
    assert not np.allclose(
        observation.camera_calibrations[2].robot_from_camera,
        observation.camera_calibrations[3].robot_from_camera,
    )
    assert observation.instruction.text == "Control both robotic arms simultaneously to complete the task"
    assert not hasattr(observation, "features")
    assert not hasattr(observation, "task_stage")
    assert not hasattr(observation, "reward")


def test_one_action_updates_left_and_right_arm_and_gripper_targets() -> None:
    backend = _backend()
    try:
        observation = backend.reset(seed=5, task_id=backend.config.task_id)
        initial_left = backend._left_targets.copy()  # noqa: SLF001
        initial_right = backend._right_targets.copy()  # noqa: SLF001
        outcome = backend.apply(
            _frame(
                observation.timestamp_ns,
                left=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
                right=(-0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
                left_gripper=0.25,
                right_gripper=0.75,
            )
        )
        moved_left = backend._left_targets.copy()  # noqa: SLF001
        moved_right = backend._right_targets.copy()  # noqa: SLF001
        left_gripper = backend.data.ctrl[
            list(backend.bundle.ids.secondary_finger_actuators)
        ]
        right_gripper = backend.data.ctrl[list(backend.bundle.ids.finger_actuators)]
    finally:
        backend.close()

    assert not np.allclose(moved_left, initial_left)
    assert not np.allclose(moved_right, initial_right)
    assert left_gripper == pytest.approx((0.02375, 0.02375))
    assert right_gripper == pytest.approx((0.07125, 0.07125))
    assert outcome.info["applied_action"].source == "test_actor"
    assert outcome.observation.sequence_id == 1


def test_resolved_rate_ik_maps_normalized_tool_twist_without_task_truth() -> None:
    backend = _backend()
    try:
        backend.reset(seed=6, task_id=backend.config.task_id)
        joint_ids = backend.bundle.ids.secondary_arm_joints
        velocity = backend._resolved_joint_velocity(  # noqa: SLF001
            (0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            joint_ids,
            backend._left_tool_site,  # noqa: SLF001
        )
        jacobian = np.zeros((3, backend.model.nv))
        mujoco.mj_jacSite(
            backend.model,
            backend.data,
            jacobian,
            None,
            backend._left_tool_site,  # noqa: SLF001
        )
        dofs = [backend.model.jnt_dofadr[joint] for joint in joint_ids]
        achieved_linear = jacobian[:, dofs] @ velocity
    finally:
        backend.close()

    assert achieved_linear[0] > 0.0
    assert abs(achieved_linear[1]) < achieved_linear[0]


def test_expired_action_stops_both_arms_and_holds_measured_grippers() -> None:
    backend = _backend()
    try:
        observation = backend.reset(seed=7, task_id=backend.config.task_id)
        outcome = backend.apply(
            _frame(
                observation.timestamp_ns,
                left_gripper=0.4,
                right_gripper=0.7,
            )
        )
        measured = (
            outcome.observation.proprioception.left_gripper_position,
            outcome.observation.proprioception.right_gripper_position,
        )
        expired = DualArmActionFrame(
            created_at_ns=0,
            valid_from_ns=0,
            valid_until_ns=1,
            source="stale_actor",
            action=DualArmAction(0.3, 0.2, (1.0,) * 6, (-1.0,) * 6, 0.0, 0.0),
        )
        stopped = backend.apply(expired)
    finally:
        backend.close()

    applied = stopped.info["applied_action"]
    assert applied.source == "safety"
    assert applied.action.left_arm == (0.0,) * 6
    assert applied.action.right_arm == (0.0,) * 6
    assert (applied.action.left_gripper, applied.action.right_gripper) == pytest.approx(
        measured
    )
    assert stopped.events[0].details["reason"] == "outside_validity_window"


def test_predictive_safety_rejects_motion_before_physics_commit() -> None:
    backend = _backend()
    try:
        observation = backend.reset(seed=8, task_id=backend.config.task_id)
        backend._predictive_safety_enabled = lambda: True  # type: ignore[method-assign]  # noqa: SLF001
        backend._predictive_safety_violation = lambda: True  # type: ignore[method-assign]  # noqa: SLF001
        outcome = backend.apply(
            _frame(
                observation.timestamp_ns,
                left=(1.0,) * 6,
                right=(-1.0,) * 6,
            )
        )
    finally:
        backend.close()

    applied = outcome.info["applied_action"]
    assert applied.source == "safety"
    assert applied.action.left_arm == (0.0,) * 6
    assert applied.action.right_arm == (0.0,) * 6
    assert outcome.info["safety_intervened"] is True
    assert outcome.info["physics_advanced"] is False
    assert outcome.events[-1].details["reason"] == "predicted_severe_collision"


def test_runtime_rejects_legacy_single_arm_action_frame() -> None:
    from hwr.core.types import ActionFrame

    backend = _backend()
    try:
        backend.reset(seed=9, task_id=backend.config.task_id)
        with pytest.raises(TypeError, match="DualArmActionFrame"):
            backend.apply(ActionFrame(0, 0, 1, "legacy", arm_command=(0.0,) * 6))
    finally:
        backend.close()


def test_normalized_gripper_one_physically_closes_both_pincers() -> None:
    backend = _backend()
    try:
        observation = backend.reset(seed=10, task_id=backend.config.task_id)
        left_geoms = tuple(
            mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("left_gripper_left_pad", "left_gripper_right_pad")
        )
        right_geoms = tuple(
            mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("right_gripper_left_pad", "right_gripper_right_pad")
        )
        separation = lambda geoms: np.linalg.norm(
            backend.data.geom_xpos[geoms[0]] - backend.data.geom_xpos[geoms[1]]
        )
        open_separation = (separation(left_geoms), separation(right_geoms))
        for _ in range(12):
            outcome = backend.apply(
                _frame(
                    observation.timestamp_ns,
                    left_gripper=1.0,
                    right_gripper=1.0,
                )
            )
            observation = outcome.observation
        closed_separation = (separation(left_geoms), separation(right_geoms))
    finally:
        backend.close()

    assert closed_separation[0] < open_separation[0] - 0.05
    assert closed_separation[1] < open_separation[1] - 0.05
