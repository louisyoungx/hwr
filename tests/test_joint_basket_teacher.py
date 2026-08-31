from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.bimanual_teacher import BASKET_TASK_ID
from hwr.adapters.mujoco.joint_basket_acquire import (
    _pad_balance_correction,
    acquire_feedback,
)
from hwr.adapters.mujoco.joint_basket_planner import (
    _planning_penetration,
    plan_joint_grasp,
)
from hwr.adapters.mujoco.names import FINGER_TRAVEL
from hwr.adapters.mujoco.joint_basket_teacher import (
    JOINT_TEACHER_SOURCE,
    JointBasketMotionTeacher,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_bimanual_training_catalogs,
)
from hwr.apps.evaluate_joint_basket_teacher import (
    COHORT_SEEDS,
    DEFAULT_SEEDS,
    _report,
)
from hwr.train.bimanual_runtime import dual_arm_action_frame


ROOT = Path(__file__).resolve().parents[1]


def _backend() -> MujocoBimanualTaskBackend:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    return MujocoBimanualTaskBackend(
        tasks[BASKET_TASK_ID],
        bindings[BASKET_TASK_ID],
        camera_width=16,
        camera_height=12,
    )


def test_joint_planner_is_non_mutating_and_plans_bilateral_contact() -> None:
    backend = _backend()
    try:
        backend.reset(seed=19_001, task_id=BASKET_TASK_ID)
        qpos = backend.data.qpos.copy()
        qvel = backend.data.qvel.copy()
        controls = backend.data.ctrl.copy()

        plan = plan_joint_grasp(backend, seed=19_001)
    finally:
        backend.close()

    assert len(plan.waypoints) == 9
    assert plan.maximum_pad_distance <= 0.004
    np.testing.assert_array_equal(backend.data.qpos, qpos)
    np.testing.assert_array_equal(backend.data.qvel, qvel)
    np.testing.assert_array_equal(backend.data.ctrl, controls)


def test_joint_plan_path_and_closed_endpoint_avoid_non_grasp_payload_contacts() -> None:
    backend = _backend()
    try:
        backend.reset(seed=19_001, task_id=BASKET_TASK_ID)
        plan = plan_joint_grasp(backend, seed=19_001)
        model = backend.model
        work = mujoco.MjData(model)
        base_address = model.jnt_qposadr[backend.bundle.ids.base_joint]
        for left, right in plan.waypoints:
            mujoco.mj_copyData(work, model, backend.data)
            work.qpos[base_address] = plan.base_x
            for values, joints in (
                (left, backend.bundle.ids.secondary_arm_joints),
                (right, backend.bundle.ids.arm_joints),
            ):
                work.qpos[
                    [model.jnt_qposadr[joint] for joint in joints]
                ] = values
            mujoco.mj_forward(model, work)
            assert _planning_penetration(backend, work) <= 0.002
        for joint in (
            *backend.bundle.ids.secondary_finger_joints,
            *backend.bundle.ids.finger_joints,
        ):
            work.qpos[model.jnt_qposadr[joint]] = 0.98 * FINGER_TRAVEL
        mujoco.mj_forward(model, work)
        assert _planning_penetration(backend, work) <= 1e-4
    finally:
        backend.close()


def test_joint_teacher_declares_full_task_and_emits_canonical_action() -> None:
    backend = _backend()
    try:
        observation = backend.reset(seed=19_001, task_id=BASKET_TASK_ID)
        teacher = JointBasketMotionTeacher(backend, seed=19_001)

        output = teacher.action(observation)
    finally:
        backend.close()

    assert JointBasketMotionTeacher.implemented_task_phases == {
        "approach",
        "acquire",
        "secure",
        "lift",
        "target_transport",
        "place",
        "release",
        "stabilize",
    }
    assert len(output.action.vector()) == 16
    assert max(abs(value) for value in output.action.vector()[2:14]) <= 0.35


def test_acquire_feedback_targets_live_pad_midpoints_and_waits_to_close() -> None:
    backend = _backend()
    try:
        backend.reset(seed=19_001, task_id=BASKET_TASK_ID)
        plan = plan_joint_grasp(backend, seed=19_001)
        feedback = acquire_feedback(
            backend,
            target_rotations=(
                plan.left_site_rotation,
                plan.right_site_rotation,
            ),
            target_handle_from_midpoints=(
                plan.left_handle_from_pad_midpoint,
                plan.right_handle_from_pad_midpoint,
            ),
            current_grippers=(0.8, 0.7),
        )
        ids = backend.task_ids
        for arm, site, pads, handle, target_offset in zip(
            feedback.arms,
            (ids.left_grasp_site, ids.right_grasp_site),
            (ids.left_pads, ids.right_pads),
            (ids.left_interaction_geom, ids.right_interaction_geom),
            (
                plan.left_handle_from_pad_midpoint,
                plan.right_handle_from_pad_midpoint,
            ),
            strict=True,
        ):
            site_position = backend.data.site_xpos[site]
            site_rotation = backend.data.site_xmat[site].reshape(3, 3)
            pad_positions = [
                backend.data.geom_xpos[pad] for pad in sorted(pads)
            ]
            midpoint = 0.5 * (pad_positions[0] + pad_positions[1])
            local_site_to_midpoint = site_rotation.T @ (
                midpoint - site_position
            )
            predicted_midpoint = (
                arm.target_site_position
                + arm.target_site_rotation @ local_site_to_midpoint
            )
            desired_midpoint = (
                backend.data.geom_xpos[handle]
                - arm.target_site_rotation @ target_offset
            )
            np.testing.assert_allclose(
                predicted_midpoint,
                desired_midpoint,
                atol=1e-12,
            )
        assert feedback.grippers == (0.0, 0.0)
    finally:
        backend.close()


def test_acquire_feedback_closes_incrementally_when_joint_plan_is_centered() -> None:
    backend = _backend()
    try:
        backend.reset(seed=19_001, task_id=BASKET_TASK_ID)
        plan = plan_joint_grasp(backend, seed=19_001)
        model = backend.model
        base_address = model.jnt_qposadr[backend.bundle.ids.base_joint]
        backend.data.qpos[base_address] = plan.base_x
        for values, joints in (
            (plan.left_joint_target, backend.bundle.ids.secondary_arm_joints),
            (plan.right_joint_target, backend.bundle.ids.arm_joints),
        ):
            backend.data.qpos[
                [model.jnt_qposadr[joint] for joint in joints]
            ] = values
        mujoco.mj_forward(model, backend.data)

        feedback = acquire_feedback(
            backend,
            target_rotations=(
                plan.left_site_rotation,
                plan.right_site_rotation,
            ),
            target_handle_from_midpoints=(
                plan.left_handle_from_pad_midpoint,
                plan.right_handle_from_pad_midpoint,
            ),
            current_grippers=(0.0, 0.0),
        )
    finally:
        backend.close()

    assert all(arm.centered for arm in feedback.arms)
    assert 0.0 < feedback.grippers[0] <= 0.06
    assert 0.0 < feedback.grippers[1] <= 0.06


def test_pad_balance_moves_toward_the_more_distant_pad() -> None:
    correction = _pad_balance_correction(
        (np.asarray((0.0, -0.1, 0.0)), np.asarray((0.0, 0.1, 0.0))),
        (-0.003, 0.001),
    )

    np.testing.assert_allclose(correction, (0.0, -0.002, 0.0))


def test_secure_handoff_executes_lift_action_and_preserves_contact() -> None:
    backend = _backend()
    safety_interventions = 0
    executed_lift = False
    try:
        observation = backend.reset(seed=19_001, task_id=BASKET_TASK_ID)
        backend.set_camera_rendering(False)
        teacher = JointBasketMotionTeacher(backend, seed=19_001)
        for _ in range(360):
            output = teacher.action(observation)
            outcome = backend.apply(
                dual_arm_action_frame(
                    observation.timestamp_ns,
                    output.action,
                    source=JOINT_TEACHER_SOURCE,
                )
            )
            observation = outcome.observation
            safety_interventions += int(outcome.info["safety_intervened"])
            audit = backend.task_audit()
            if output.stage == "lift":
                executed_lift = True
                break
    finally:
        backend.close()

    assert executed_lift
    assert teacher.stage == "lift"
    assert audit["maximum_concurrent_steps"] >= 16
    assert audit["metrics"]["left_contact"] == 1.0
    assert audit["metrics"]["right_contact"] == 1.0
    assert audit["severe_collision_count"] == 0
    assert safety_interventions == 0


def test_r0020_report_only_advances_l0_after_small_cohort() -> None:
    single = _report(
        seeds=DEFAULT_SEEDS,
        episodes=(_episode(success=True),),
        source_commit="a" * 40,
        source_worktree_dirty=True,
        elapsed_seconds=1.0,
    )
    cohort = _report(
        seeds=COHORT_SEEDS,
        episodes=tuple(_episode(success=index < 3) for index in range(4)),
        source_commit="a" * 40,
        source_worktree_dirty=True,
        elapsed_seconds=1.0,
    )

    assert single["decision"] == "validated_development"
    assert single["single_seed_gate_passed"] is True
    assert single["l0_gate_passed"] is False
    assert cohort["decision"] == "validated_development"
    assert cohort["l0_gate_passed"] is True
    assert cohort["confirmation_evidence"] == {
        "status": "not_run",
        "valid": None,
    }


def test_controller_failure_is_a_valid_failed_episode() -> None:
    failed = _episode(success=False)
    failed["controller_failure"] = "joint grasp planner found no bilateral grasp"

    report = _report(
        seeds=DEFAULT_SEEDS,
        episodes=(failed,),
        source_commit="a" * 40,
        source_worktree_dirty=True,
        elapsed_seconds=1.0,
    )

    assert report["decision"] == "abandoned"
    assert report["single_seed_gate_passed"] is False


def _episode(*, success: bool) -> dict[str, object]:
    return {
        "success": success,
        "valid_episode_result": True,
        "safety_intervention_count": 0,
        "audit": {"severe_collision_count": 0},
    }
