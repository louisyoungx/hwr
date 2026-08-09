"""Reproducible contact-only grasp trial for the 3D physics baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import mujoco

from hwr.adapters.mujoco.backend import Mujoco3DBackend
from hwr.adapters.mujoco.contact import GraspContactMonitor, GraspContactSample
from hwr.adapters.mujoco.expert import PrivilegedCartesianExpert
from hwr.core.types import ObservationFrame


@dataclass(frozen=True)
class ContactTrialFrame:
    stage: str
    stage_step: int
    observation: ObservationFrame
    contact: GraspContactSample
    object_position: tuple[float, float, float]


@dataclass(frozen=True)
class ContactGraspReport:
    seed: int
    action_source: str
    initial_object_position: tuple[float, float, float]
    final_object_position: tuple[float, float, float]
    maximum_object_height: float
    bilateral_contact_steps: int
    maximum_left_normal_force: float
    maximum_right_normal_force: float
    equality_constraint_count: int
    success: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FrameCallback = Callable[[ContactTrialFrame], None]


def run_contact_grasp_trial(
    backend: Mujoco3DBackend,
    *,
    seed: int,
    frame_callback: FrameCallback | None = None,
) -> ContactGraspReport:
    observation = backend.reset(seed=seed, task_id=backend.config.task_id)
    expert = PrivilegedCartesianExpert(backend)
    monitor = GraspContactMonitor(backend.model, object_geom="smoke_object_geom")
    object_body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "smoke_object",
    )
    initial_position = _body_position(backend, object_body_id)
    object_x, object_y, _ = initial_position
    stages = (
        ("approach_above", (object_x, object_y, 0.32), 0.0, 100),
        ("descend_open", (object_x, object_y, 0.12), 0.0, 100),
        ("close_on_contact", (object_x, object_y, 0.12), 1.0, 50),
        ("lift_while_closed", (object_x - 0.05, object_y, 0.50), 1.0, 100),
    )
    maximum_height = initial_position[2]
    bilateral_steps = 0
    maximum_left_force = 0.0
    maximum_right_force = 0.0
    for stage, target, gripper_target, step_count in stages:
        for stage_step in range(step_count):
            action = expert.action(
                observation,
                target_position=target,
                gripper_target=gripper_target,
            )
            observation = backend.apply(action).observation
            position = _body_position(backend, object_body_id)
            contact = monitor.sample(backend.data)
            maximum_height = max(maximum_height, position[2])
            bilateral_steps += int(contact.bilateral)
            maximum_left_force = max(maximum_left_force, contact.left_normal_force)
            maximum_right_force = max(maximum_right_force, contact.right_normal_force)
            if frame_callback is not None:
                frame_callback(
                    ContactTrialFrame(
                        stage=stage,
                        stage_step=stage_step,
                        observation=observation,
                        contact=contact,
                        object_position=position,
                    )
                )
    final_position = _body_position(backend, object_body_id)
    success = (
        maximum_height - initial_position[2] >= 0.25
        and final_position[2] - initial_position[2] >= 0.25
        and bilateral_steps >= 20
        and backend.model.neq == 0
    )
    return ContactGraspReport(
        seed=seed,
        action_source=expert.source,
        initial_object_position=initial_position,
        final_object_position=final_position,
        maximum_object_height=maximum_height,
        bilateral_contact_steps=bilateral_steps,
        maximum_left_normal_force=maximum_left_force,
        maximum_right_normal_force=maximum_right_force,
        equality_constraint_count=backend.model.neq,
        success=success,
    )


def _body_position(backend: Mujoco3DBackend, body_id: int) -> tuple[float, float, float]:
    return tuple(float(value) for value in backend.data.xpos[body_id])
