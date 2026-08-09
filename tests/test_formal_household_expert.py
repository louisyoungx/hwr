from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from hwr.adapters.mujoco import (  # noqa: E402
    MujocoHouseholdBackend,
    PrivilegedHouseholdExpert,
    load_mujoco_task_bindings,
)
from hwr.scenarios.formal3d import load_formal_3d_tasks  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TASKS = load_formal_3d_tasks(ROOT / "configs/tasks/formal_3d_v1.json")
BINDINGS = load_mujoco_task_bindings(
    ROOT / "configs/adapters/mujoco/formal_3d_v1.json", root=ROOT
)


def test_living_room_expert_completes_two_contact_only_pick_and_place_cycles() -> None:
    task_id = "tidy_living_room_3d/v1"
    backend = MujocoHouseholdBackend(
        TASKS[task_id], BINDINGS[task_id], camera_width=2, camera_height=2
    )
    try:
        observation = backend.reset(seed=301, task_id=task_id)
        expert = PrivilegedHouseholdExpert(backend)
        for _ in range(TASKS[task_id].max_steps):
            output = expert.action(observation)
            outcome = backend.apply(output.action)
            observation = outcome.observation
            if outcome.terminated or outcome.truncated:
                break
        result = backend.result()
        audit = backend.audit_snapshot()
    finally:
        backend.close()

    assert result is not None and result.success
    assert expert.failed is False
    assert audit["stable_steps"] >= 40
    assert audit["severe_collision_count"] == 0
    assert all(value["inside_target"] for value in audit["objects"].values())
    assert all(value["bilateral_contact_steps"] > 0 for value in audit["objects"].values())
    assert backend.model.neq == 0
