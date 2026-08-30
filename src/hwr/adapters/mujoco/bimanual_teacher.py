"""Privileged L0 teacher and generic-primitive baseline for the basket task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.names import FINGER_TRAVEL
from hwr.adapters.mujoco.target_selection_diagnostic import policy_input_bytes
from hwr.core.embodied import DualArmAction, DualArmObservation
from hwr.eval.target_selection import (
    PHASES,
    POST_SELECTION_STEPS,
    Candidate,
    phase_for_step,
    primitive_action,
)


BASKET_TASK_ID = "carry_living_room_basket/v1"
TEACHER_SOURCE = "r0019_privileged_basket_teacher"
BASELINE_SOURCE = "r0019_generic_b0_b7_baseline"
PLANNER_GRIPPER = 0.85
SECURE_GRIPPER = 0.95


@dataclass(frozen=True)
class TeacherOutput:
    action: DualArmAction
    stage: str


class PrivilegedBasketTeacher:
    """Closed-loop development teacher using private geometry and contact state."""

    def __init__(self, backend: MujocoBimanualTaskBackend, *, seed: int) -> None:
        if backend.task.task_id != BASKET_TASK_ID:
            raise ValueError("R0019 teacher only supports the living-room basket task")
        self.backend = backend
        self.seed = int(seed)
        self.stage = "approach_base"
        self.stage_step = 0
        self.failure_stage: str | None = None
        self._joint_targets: tuple[np.ndarray, np.ndarray] | None = None

    def action(self, observation: DualArmObservation) -> TeacherOutput:
        audit = self.backend.task_audit()
        metrics = audit["metrics"]
        if self.stage == "approach_base":
            if observation.proprioception.base_pose[0] < 0.073:
                return TeacherOutput(
                    _action(base_linear=0.06, gripper=0.0),
                    self.stage,
                )
            self._joint_targets = self._plan_grasp()
            self._advance("acquire")
        if self.stage == "acquire":
            if int(audit["concurrent_steps"]) >= 10:
                self._advance("secure")
            else:
                return TeacherOutput(
                    self._joint_tracking_action(PLANNER_GRIPPER),
                    self.stage,
                )
        if self.stage == "secure":
            if self.stage_step >= 40:
                self._advance("transport")
            else:
                self.stage_step += 1
                return TeacherOutput(
                    self._joint_tracking_action(SECURE_GRIPPER),
                    self.stage,
                )
        if self.stage == "transport":
            if not bool(metrics["left_contact"] and metrics["right_contact"]):
                self.failure_stage = "transport_contact_lost"
                self._advance("failed_hold")
            else:
                self.stage_step += 1
                return TeacherOutput(
                    _action(
                        left_arm=(0.10, 0.0, 0.0, 0.0, 0.0, 0.0),
                        right_arm=(0.10, 0.0, 0.0, 0.0, 0.0, 0.0),
                        gripper=SECURE_GRIPPER,
                    ),
                    self.stage,
                )
        return TeacherOutput(
            _action(
                gripper=(
                    observation.proprioception.left_gripper_position,
                    observation.proprioception.right_gripper_position,
                )
            ),
            self.stage,
        )

    def _advance(self, stage: str) -> None:
        self.stage = stage
        self.stage_step = 0

    def _plan_grasp(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._plan_arm(
                arm="left",
                joint_ids=self.backend.bundle.ids.secondary_arm_joints,
                finger_ids=self.backend.bundle.ids.secondary_finger_joints,
                pad_names=("left_gripper_left_pad", "left_gripper_right_pad"),
                handle_id=self.backend.task_ids.left_interaction_geom,
                rng_seed=self.seed + 85,
            ),
            self._plan_arm(
                arm="right",
                joint_ids=self.backend.bundle.ids.arm_joints,
                finger_ids=self.backend.bundle.ids.finger_joints,
                pad_names=("right_gripper_left_pad", "right_gripper_right_pad"),
                handle_id=self.backend.task_ids.right_interaction_geom,
                rng_seed=self.seed + 10_085,
            ),
        )

    def _plan_arm(
        self,
        *,
        arm: Literal["left", "right"],
        joint_ids: tuple[int, ...],
        finger_ids: tuple[int, ...],
        pad_names: tuple[str, str],
        handle_id: int,
        rng_seed: int,
    ) -> np.ndarray:
        model = self.backend.model
        source = self.backend.data
        pad_ids = tuple(
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in pad_names
        )
        site_id = (
            self.backend.task_ids.left_grasp_site
            if arm == "left"
            else self.backend.task_ids.right_grasp_site
        )
        qpos_addresses = np.asarray(
            [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
        )
        lower = np.asarray(
            [float(model.jnt_range[joint_id][0]) + 0.01 for joint_id in joint_ids]
        )
        upper = np.asarray(
            [float(model.jnt_range[joint_id][1]) - 0.01 for joint_id in joint_ids]
        )
        mean = source.qpos[qpos_addresses].copy()
        standard_deviation = np.asarray((0.5, 0.5, 0.7, 0.8, 0.6, 0.8))
        rng = np.random.default_rng(rng_seed)
        work = mujoco.MjData(model)
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for _ in range(12):
            population = np.vstack(
                (mean, rng.normal(mean, standard_deviation, size=(255, 6)))
            )
            population = np.clip(population, lower, upper)
            scored = []
            for joint_position in population:
                mujoco.mj_copyData(work, model, source)
                work.qpos[qpos_addresses] = joint_position
                for joint_id in finger_ids:
                    work.qpos[model.jnt_qposadr[joint_id]] = (
                        PLANNER_GRIPPER * FINGER_TRAVEL
                    )
                mujoco.mj_forward(model, work)
                distances = np.asarray(
                    [
                        _geom_distance(model, work, pad_id, handle_id)
                        for pad_id in pad_ids
                    ]
                )
                site_distance = float(
                    np.linalg.norm(
                        work.site_xpos[site_id] - work.geom_xpos[handle_id]
                    )
                )
                objective = float(
                    np.square(distances + 0.001).sum()
                    + 0.05 * max(float(distances.max()), 0.0) ** 2
                    + 0.0005 * site_distance**2
                )
                scored.append((objective, joint_position.copy(), distances))
            scored.sort(key=lambda value: value[0])
            elite = scored[:24]
            mean = np.mean([value[1] for value in elite], axis=0)
            standard_deviation = np.maximum(
                np.std([value[1] for value in elite], axis=0),
                0.005,
            )
            if best is None or elite[0][0] < best[0]:
                best = elite[0]
            if best[2].max() < 0.0005:
                break
        if best is None or best[2].max() >= 0.005:
            raise RuntimeError(f"{arm} grasp planning did not find bilateral pad contact")
        return best[1]

    def _joint_tracking_action(self, gripper: float) -> DualArmAction:
        if self._joint_targets is None:
            raise RuntimeError("teacher grasp plan is unavailable")
        commands = []
        base_rotation = self.backend.data.xmat[
            self.backend.bundle.ids.base_body
        ].reshape(3, 3)
        for target, joint_ids, site_id in (
            (
                self._joint_targets[0],
                self.backend.bundle.ids.secondary_arm_joints,
                self.backend.task_ids.left_grasp_site,
            ),
            (
                self._joint_targets[1],
                self.backend.bundle.ids.arm_joints,
                self.backend.task_ids.right_grasp_site,
            ),
        ):
            current = np.asarray(
                [
                    self.backend.data.qpos[self.backend.model.jnt_qposadr[joint_id]]
                    for joint_id in joint_ids
                ]
            )
            desired_joint_velocity = _clip_norm(1.5 * (target - current), 0.30)
            jacobian_position = np.zeros((3, self.backend.model.nv))
            jacobian_rotation = np.zeros((3, self.backend.model.nv))
            mujoco.mj_jacSite(
                self.backend.model,
                self.backend.data,
                jacobian_position,
                jacobian_rotation,
                site_id,
            )
            dofs = [self.backend.model.jnt_dofadr[joint_id] for joint_id in joint_ids]
            jacobian = np.vstack(
                (jacobian_position[:, dofs], jacobian_rotation[:, dofs])
            )
            backend_inverse = jacobian.T @ np.linalg.inv(
                jacobian @ jacobian.T
                + np.eye(6) * self.backend.config.ik_damping**2
            )
            twist = np.linalg.lstsq(
                backend_inverse, desired_joint_velocity, rcond=1.0e-5
            )[0]
            command = np.concatenate(
                (
                    base_rotation.T
                    @ twist[:3]
                    / self.backend.config.max_tool_linear_velocity,
                    base_rotation.T
                    @ twist[3:]
                    / self.backend.config.max_tool_angular_velocity,
                )
            )
            maximum = float(np.abs(command).max())
            if maximum > 0.35:
                command *= 0.35 / maximum
            commands.append(tuple(float(value) for value in command))
        return _action(
            left_arm=commands[0],
            right_arm=commands[1],
            gripper=gripper,
        )


class GenericBasketPrimitiveBaseline:
    """Run the historical generic B0-B7 controller on a truth-bound basket candidate."""

    def __init__(self, backend: MujocoBimanualTaskBackend, *, seed: int) -> None:
        if backend.task.task_id != BASKET_TASK_ID:
            raise ValueError("R0019 baseline only supports the living-room basket task")
        self.backend = backend
        self.seed = int(seed)
        self.step = 0
        self.history: list[tuple[float, ...]] = []
        self.available: list[bool] = []
        self.acquisition_pose: tuple[float, float, float] | None = None
        self.candidate: Candidate | None = None

    def reset(self, observation: DualArmObservation) -> None:
        self.step = 0
        self.history.clear()
        self.available.clear()
        self.acquisition_pose = tuple(observation.proprioception.base_pose)
        base_position = self.backend.data.xpos[
            self.backend.bundle.ids.base_body
        ].copy()
        base_rotation = self.backend.data.xmat[
            self.backend.bundle.ids.base_body
        ].reshape(3, 3)
        handles = (
            self.backend.data.geom_xpos[
                self.backend.task_ids.left_interaction_geom
            ],
            self.backend.data.geom_xpos[
                self.backend.task_ids.right_interaction_geom
            ],
        )
        center = base_rotation.T @ (0.5 * (handles[0] + handles[1]) - base_position)
        self.candidate = Candidate(
            tuple(float(value) for value in center),
            (-1.0, 0.0, 0.0),
            0.40,
            0.10,
            1,
            1,
            0,
            0,
            0,
        )

    def action(self, observation: DualArmObservation) -> TeacherOutput:
        if self.candidate is None or self.acquisition_pose is None:
            self.reset(observation)
        if self.step >= POST_SELECTION_STEPS:
            stage = "post_B7_hold"
            action = _action(
                gripper=(
                    observation.proprioception.left_gripper_position,
                    observation.proprioception.right_gripper_position,
                )
            )
        else:
            stage, local_step = phase_for_step(self.step)
            payload = policy_input_bytes(
                observation,
                self.history,
                self.available,
                self.seed ^ 0x5A17,
                phase_index=5 + _phase_index(stage),
                phase_step=local_step,
            )
            action = DualArmAction.from_vector(
                primitive_action(
                    payload,
                    self.candidate,
                    self.acquisition_pose,
                    self.step,
                )
            )
        self.step += 1
        return TeacherOutput(action, stage)

    def record_applied(self, action: DualArmAction) -> None:
        self.history.append(tuple(action.vector()))
        self.available.append(True)
        del self.history[:-4]
        del self.available[:-4]


def _phase_index(stage: str) -> int:
    return next(index for index, (name, _) in enumerate(PHASES) if name == stage)


def _geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first: int,
    second: int,
) -> float:
    segment = np.zeros(6)
    return float(mujoco.mj_geomDistance(model, data, first, second, 2.0, segment))


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or norm <= maximum:
        return vector
    return vector * maximum / norm


def _action(
    *,
    base_linear: float = 0.0,
    base_angular: float = 0.0,
    left_arm: tuple[float, ...] = (0.0,) * 6,
    right_arm: tuple[float, ...] = (0.0,) * 6,
    gripper: float | tuple[float, float] = 0.0,
) -> DualArmAction:
    grippers = (gripper, gripper) if isinstance(gripper, float) else gripper
    return DualArmAction(
        base_linear,
        base_angular,
        left_arm,
        right_arm,
        float(grippers[0]),
        float(grippers[1]),
    )
