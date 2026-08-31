from __future__ import annotations

from pathlib import Path

import numpy as np

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.bimanual_teacher import BASKET_TASK_ID
from hwr.adapters.mujoco.joint_basket_planner import plan_joint_grasp
from hwr.adapters.mujoco.joint_basket_teacher import JointBasketMotionTeacher
from hwr.adapters.mujoco.training_catalog import (
    load_default_bimanual_training_catalogs,
)
from hwr.apps.evaluate_joint_basket_teacher import (
    COHORT_SEEDS,
    DEFAULT_SEEDS,
    _report,
)


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
