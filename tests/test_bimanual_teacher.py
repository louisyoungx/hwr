from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.bimanual_teacher import (
    BASKET_TASK_ID,
    PrivilegedBasketTeacher,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_bimanual_training_catalogs,
)
from hwr.apps.evaluate_bimanual_teacher import (
    CONFIRMATION_EPISODES,
    CONFIRMATION_OUTPUT,
    CONFIRMATION_SEED_START,
    CONFIRMATION_SEED_STRIDE,
    REQUIRED_TEACHER_TASK_PHASES,
    _confirmation_qualification,
    _exit_code,
    _output_path,
    _require_confirmation_preconditions,
    _report,
    _seeds,
    build_parser,
)
from hwr.train.bimanual_runtime import dual_arm_action_frame


ROOT = Path(__file__).resolve().parents[1]


def _confirmation_episode(
    *,
    controller: str,
    seed: int,
    success: bool,
) -> dict[str, object]:
    return {
        "controller": controller,
        "seed": seed,
        "episode_completed": True,
        "physics_episode_completed": True,
        "valid_episode_result": True,
        "controller_failure": None,
        "success": success,
        "audit": {
            "severe_collision_count": 0,
            "maximum_concurrent_steps": 10,
        },
        "safety_intervention_count": 0,
        "first_contact_step": {"bilateral": 1},
        "termination_reason": (
            "bimanual_task_success" if success else "bimanual_task_timeout"
        ),
    }


def _backend() -> MujocoBimanualTaskBackend:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    return MujocoBimanualTaskBackend(
        tasks[BASKET_TASK_ID],
        bindings[BASKET_TASK_ID],
        camera_width=16,
        camera_height=12,
    )


def test_privileged_grasp_planning_does_not_mutate_authoritative_state() -> None:
    backend = _backend()
    try:
        backend.reset(seed=19_001, task_id=BASKET_TASK_ID)
        teacher = PrivilegedBasketTeacher(backend, seed=19_001)
        qpos = backend.data.qpos.copy()
        qvel = backend.data.qvel.copy()
        controls = backend.data.ctrl.copy()

        targets = teacher._plan_grasp()

        assert all(target.shape == (6,) for target in targets)
        np.testing.assert_array_equal(backend.data.qpos, qpos)
        np.testing.assert_array_equal(backend.data.qvel, qvel)
        np.testing.assert_array_equal(backend.data.ctrl, controls)
    finally:
        backend.close()


def test_teacher_produces_real_bilateral_contact_before_transport_failure() -> None:
    backend = _backend()
    try:
        observation = backend.reset(seed=19_001, task_id=BASKET_TASK_ID)
        backend.set_camera_rendering(False)
        teacher = PrivilegedBasketTeacher(backend, seed=19_001)
        safety_interventions = 0
        for _ in range(backend.task.max_steps):
            output = teacher.action(observation)
            assert len(output.action.vector()) == 16
            assert max(abs(value) for value in output.action.vector()[2:14]) <= (
                0.35 + 1.0e-12
            )
            outcome = backend.apply(
                dual_arm_action_frame(
                    observation.timestamp_ns,
                    output.action,
                    source="test_r0019_teacher",
                )
            )
            observation = outcome.observation
            safety_interventions += int(outcome.info["safety_intervened"])
            if outcome.terminated or outcome.truncated:
                break
        audit = backend.task_audit()
    finally:
        backend.close()

    assert audit["maximum_concurrent_steps"] >= 10
    assert audit["simultaneous_contact_steps"] > 0
    assert audit["severe_collision_count"] == 0
    assert safety_interventions == 0
    assert teacher.failure_stage == "transport_contact_lost"


def test_confirmation_seed_domain_and_decision_are_frozen() -> None:
    arguments = argparse.Namespace(
        mode="confirmation",
        controller="paired",
        seed=None,
    )
    seeds = _seeds(arguments)

    assert len(seeds) == CONFIRMATION_EPISODES
    assert seeds[0] == CONFIRMATION_SEED_START
    assert seeds[-1] == (
        CONFIRMATION_SEED_START
        + (CONFIRMATION_EPISODES - 1) * CONFIRMATION_SEED_STRIDE
    )
    with pytest.raises(ValueError, match="cannot be overridden"):
        _seeds(
            argparse.Namespace(
                mode="confirmation",
                controller="paired",
                seed=[1],
            )
        )
    episodes = tuple(
        _confirmation_episode(
            controller=controller,
            seed=seed,
            success=(controller == "teacher" and index < 80),
        )
        for index, seed in enumerate(seeds)
        for controller in ("baseline", "teacher")
    )
    original_phases = PrivilegedBasketTeacher.implemented_task_phases
    PrivilegedBasketTeacher.implemented_task_phases = REQUIRED_TEACHER_TASK_PHASES
    try:
        report = _report(
            mode="confirmation",
            controllers=("baseline", "teacher"),
            seeds=seeds,
            episodes=episodes,
            source_commit="a" * 40,
            source_files={},
            source_worktree_dirty=False,
            elapsed_seconds=1.0,
            confirmation_qualification={"status": "validated"},
        )
    finally:
        PrivilegedBasketTeacher.implemented_task_phases = original_phases

    assert report["decision"] == "validated_development"
    assert report["confirmation_evidence"] == {
        "status": "completed",
        "valid": True,
        "errors": [],
    }


@pytest.mark.parametrize(
    ("episodes_to_remove", "source_worktree_dirty", "expected_error"),
    (
        (1, False, "confirmation_episode_pairs_incomplete"),
        (0, True, "source_worktree_dirty"),
    ),
)
def test_confirmation_rejects_incomplete_or_dirty_evidence(
    episodes_to_remove: int,
    source_worktree_dirty: bool,
    expected_error: str,
) -> None:
    seeds = tuple(
        CONFIRMATION_SEED_START + index * CONFIRMATION_SEED_STRIDE
        for index in range(CONFIRMATION_EPISODES)
    )
    episodes = tuple(
        _confirmation_episode(
            controller=controller,
            seed=seed,
            success=controller == "teacher",
        )
        for seed in seeds
        for controller in ("baseline", "teacher")
    )

    original_phases = PrivilegedBasketTeacher.implemented_task_phases
    PrivilegedBasketTeacher.implemented_task_phases = REQUIRED_TEACHER_TASK_PHASES
    try:
        report = _report(
            mode="confirmation",
            controllers=("baseline", "teacher"),
            seeds=seeds,
            episodes=(
                episodes[:-episodes_to_remove] if episodes_to_remove else episodes
            ),
            source_commit="a" * 40,
            source_files={},
            source_worktree_dirty=source_worktree_dirty,
            elapsed_seconds=1.0,
            confirmation_qualification={"status": "validated"},
        )
    finally:
        PrivilegedBasketTeacher.implemented_task_phases = original_phases

    assert report["decision"] == "invalid"
    assert expected_error in report["confirmation_evidence"]["errors"]


def test_confirmation_refuses_incomplete_teacher_and_dirty_worktree() -> None:
    with pytest.raises(RuntimeError, match="source_worktree_dirty"):
        _require_confirmation_preconditions(
            "confirmation",
            ("baseline", "teacher"),
            True,
            output=ROOT / CONFIRMATION_OUTPUT,
            root=ROOT,
            output_exists=False,
        )
    with pytest.raises(RuntimeError, match="teacher_missing_task_phases"):
        _require_confirmation_preconditions(
            "confirmation",
            ("baseline", "teacher"),
            False,
            output=ROOT / CONFIRMATION_OUTPUT,
            root=ROOT,
            output_exists=False,
        )
    with pytest.raises(RuntimeError, match="output_already_exists"):
        original_phases = PrivilegedBasketTeacher.implemented_task_phases
        PrivilegedBasketTeacher.implemented_task_phases = REQUIRED_TEACHER_TASK_PHASES
        try:
            _require_confirmation_preconditions(
                "confirmation",
                ("baseline", "teacher"),
                False,
                output=ROOT / CONFIRMATION_OUTPUT,
                root=ROOT,
                output_exists=True,
            )
        finally:
            PrivilegedBasketTeacher.implemented_task_phases = original_phases

    _require_confirmation_preconditions(
        "development",
        ("baseline", "teacher"),
        True,
        output=ROOT / CONFIRMATION_OUTPUT,
        root=ROOT,
        output_exists=False,
    )


def test_output_default_is_selected_from_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "confirmation"])

    assert arguments.output is None
    assert _output_path(ROOT, arguments.mode, arguments.output) == (
        ROOT / CONFIRMATION_OUTPUT
    )
    assert _output_path(ROOT, "development", None) == (
        ROOT / "runs/research-loop/0019/development/latest.json"
    )


def test_confirmation_requires_frozen_output_path() -> None:
    original_phases = PrivilegedBasketTeacher.implemented_task_phases
    PrivilegedBasketTeacher.implemented_task_phases = REQUIRED_TEACHER_TASK_PHASES
    try:
        with pytest.raises(RuntimeError, match="output_path_mismatch"):
            _require_confirmation_preconditions(
                "confirmation",
                ("baseline", "teacher"),
                False,
                output=ROOT / "runs/research-loop/0019/confirmation/retry.json",
                root=ROOT,
                output_exists=False,
            )
    finally:
        PrivilegedBasketTeacher.implemented_task_phases = original_phases


def test_confirmation_rejects_unfrozen_domain_and_unpaired_controllers() -> None:
    seeds = tuple(
        CONFIRMATION_SEED_START + index * CONFIRMATION_SEED_STRIDE
        for index in range(CONFIRMATION_EPISODES)
    )
    report = _report(
        mode="confirmation",
        controllers=("teacher",),
        seeds=seeds[1:],
        episodes=(),
        source_commit="a" * 40,
        source_files={},
        source_worktree_dirty=False,
        elapsed_seconds=1.0,
    )

    assert report["decision"] == "invalid"
    assert report["confirmation_evidence"]["errors"] == [
        "confirmation_qualification_invalid",
        "confirmation_seed_domain_mismatch",
        "confirmation_controllers_not_paired",
        "teacher_missing_task_phases:lift,place,release,stabilize,target_transport",
        "confirmation_episode_pairs_incomplete",
    ]


def test_development_report_marks_confirmation_not_run_and_candidate_invalid() -> None:
    report = _report(
        mode="development",
        controllers=("baseline", "teacher"),
        seeds=(19_001,),
        episodes=(
            _confirmation_episode(
                controller="baseline",
                seed=19_001,
                success=False,
            ),
            _confirmation_episode(
                controller="teacher",
                seed=19_001,
                success=False,
            ),
        ),
        source_commit="a" * 40,
        source_files={},
        source_worktree_dirty=True,
        elapsed_seconds=1.0,
    )

    assert report["decision"] == "invalid"
    assert report["l0_gate_passed"] is False
    assert report["confirmation_evidence"] == {
        "status": "not_run",
        "valid": None,
        "errors": [],
    }
    assert report["implementation_evidence"]["errors"] == [
        "teacher_missing_task_phases:lift,place,release,stabilize,target_transport"
    ]


def test_complete_teacher_without_development_success_is_abandoned() -> None:
    original_phases = PrivilegedBasketTeacher.implemented_task_phases
    PrivilegedBasketTeacher.implemented_task_phases = REQUIRED_TEACHER_TASK_PHASES
    try:
        report = _report(
            mode="development",
            controllers=("teacher",),
            seeds=(19_001,),
            episodes=(
                _confirmation_episode(
                    controller="teacher",
                    seed=19_001,
                    success=False,
                ),
            ),
            source_commit="a" * 40,
            source_files={},
            source_worktree_dirty=False,
            elapsed_seconds=1.0,
        )
    finally:
        PrivilegedBasketTeacher.implemented_task_phases = original_phases

    assert report["decision"] == "abandoned"
    assert report["l0_gate_passed"] is False


def test_invalid_report_uses_nonzero_cli_exit() -> None:
    assert _exit_code({"decision": "validated_development"}) == 0
    assert _exit_code({"decision": "invalid"}) == 2
    assert _exit_code({"decision": "abandoned"}) == 2


def test_confirmation_rejects_controller_failure_episode() -> None:
    seeds = tuple(
        CONFIRMATION_SEED_START + index * CONFIRMATION_SEED_STRIDE
        for index in range(CONFIRMATION_EPISODES)
    )
    episodes = [
        _confirmation_episode(
            controller=controller,
            seed=seed,
            success=controller == "teacher",
        )
        for seed in seeds
        for controller in ("baseline", "teacher")
    ]
    episodes[-1]["episode_completed"] = False
    episodes[-1]["valid_episode_result"] = False
    episodes[-1]["infrastructure_error"] = {
        "type": "RuntimeError",
        "message": "backend failed",
    }
    original_phases = PrivilegedBasketTeacher.implemented_task_phases
    PrivilegedBasketTeacher.implemented_task_phases = REQUIRED_TEACHER_TASK_PHASES
    try:
        report = _report(
            mode="confirmation",
            controllers=("baseline", "teacher"),
            seeds=seeds,
            episodes=episodes,
            source_commit="a" * 40,
            source_files={},
            source_worktree_dirty=False,
            elapsed_seconds=1.0,
            confirmation_qualification={"status": "validated"},
        )
    finally:
        PrivilegedBasketTeacher.implemented_task_phases = original_phases

    assert report["decision"] == "invalid"
    assert "confirmation_episode_execution_incomplete" in (
        report["confirmation_evidence"]["errors"]
    )


def test_confirmation_qualification_requires_matching_successful_development(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "qualification.json"
    report_path.write_text(
        """{
  "mode": "development",
  "task_id": "carry_living_room_basket/v1",
  "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_worktree_dirty": false,
  "source_files": {"teacher.py": "hash"},
  "implementation_evidence": {"valid": true},
  "decision": "validated_development",
  "summaries": {"teacher": {"successes": 1}},
  "confirmation_evidence": {"status": "not_run"},
  "run_status": {"completed": true}
}
""",
        encoding="utf-8",
    )

    qualification = _confirmation_qualification(
        mode="confirmation",
        report_path=report_path,
        root=tmp_path,
        source_commit="a" * 40,
        source_files={"teacher.py": "hash"},
    )

    assert qualification["status"] == "validated"
    with pytest.raises(RuntimeError, match="source_commit_mismatch"):
        _confirmation_qualification(
            mode="confirmation",
            report_path=report_path,
            root=tmp_path,
            source_commit="b" * 40,
            source_files={"teacher.py": "hash"},
        )
