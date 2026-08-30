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
    CONFIRMATION_SEED_START,
    CONFIRMATION_SEED_STRIDE,
    _require_clean_confirmation_source,
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
    report = _report(
        mode="confirmation",
        controllers=("baseline", "teacher"),
        seeds=seeds,
        episodes=episodes,
        source_commit="a" * 40,
        source_files={},
        source_worktree_dirty=False,
        elapsed_seconds=1.0,
    )

    assert report["decision"] == "validated_development"
    assert report["confirmation_evidence"] == {"valid": True, "errors": []}


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

    report = _report(
        mode="confirmation",
        controllers=("baseline", "teacher"),
        seeds=seeds,
        episodes=episodes[:-episodes_to_remove] if episodes_to_remove else episodes,
        source_commit="a" * 40,
        source_files={},
        source_worktree_dirty=source_worktree_dirty,
        elapsed_seconds=1.0,
    )

    assert report["decision"] == "invalid"
    assert expected_error in report["confirmation_evidence"]["errors"]


def test_confirmation_refuses_to_run_from_dirty_worktree() -> None:
    with pytest.raises(RuntimeError, match="clean committed"):
        _require_clean_confirmation_source("confirmation", True)

    _require_clean_confirmation_source("development", True)


def test_output_default_is_selected_from_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "confirmation"])

    assert arguments.output is None


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
        "confirmation_controllers_not_paired",
        "confirmation_seed_domain_mismatch",
        "confirmation_episode_pairs_incomplete",
    ]
