"""Joint grasp configuration and path planning for the R0020 teacher."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.bimanual_teacher import BasketTeacherError
from hwr.adapters.mujoco.names import FINGER_TRAVEL


GRASP_GRIPPER = 0.98
NOMINAL_GRASP = np.asarray((0.0, 1.037, -0.453, 0.0, -0.524, 0.0))


@dataclass(frozen=True)
class JointGraspPlan:
    base_x: float
    left_joint_target: np.ndarray
    right_joint_target: np.ndarray
    waypoints: tuple[tuple[np.ndarray, np.ndarray], ...]
    maximum_pad_distance: float
    path_minimum_clearance: float


def plan_joint_grasp(
    backend: MujocoBimanualTaskBackend,
    *,
    seed: int,
) -> JointGraspPlan:
    model = backend.model
    source = backend.data
    left_joints = backend.bundle.ids.secondary_arm_joints
    right_joints = backend.bundle.ids.arm_joints
    all_joints = (*left_joints, *right_joints)
    addresses = np.asarray([model.jnt_qposadr[joint] for joint in all_joints])
    lower = np.asarray([model.jnt_range[joint][0] + 0.01 for joint in all_joints])
    upper = np.asarray([model.jnt_range[joint][1] - 0.01 for joint in all_joints])
    base_address = int(model.jnt_qposadr[backend.bundle.ids.base_joint])
    start_x = float(source.qpos[base_address])
    mean = np.concatenate(([0.055], NOMINAL_GRASP, NOMINAL_GRASP))
    standard_deviation = np.concatenate(
        ([0.035], np.tile((0.30, 0.35, 0.45, 0.50, 0.40, 0.50), 2))
    )
    rng = np.random.default_rng(seed ^ 0x20C0FFEE)
    work = mujoco.MjData(model)
    best: tuple[float, np.ndarray, float] | None = None
    for _ in range(14):
        population = np.vstack(
            (mean, rng.normal(mean, standard_deviation, size=(767, 13)))
        )
        population[:, 0] = np.clip(population[:, 0], start_x - 0.03, 0.11)
        population[:, 1:] = np.clip(population[:, 1:], lower, upper)
        scored = [
            _score_grasp_candidate(
                backend,
                work,
                candidate,
                addresses,
                base_address,
            )
            for candidate in population
        ]
        elite_indices = np.argsort([row[0] for row in scored])[:48]
        elite = population[elite_indices]
        mean = elite.mean(axis=0)
        standard_deviation = np.maximum(elite.std(axis=0), 0.003)
        candidate_best = scored[int(elite_indices[0])]
        if best is None or candidate_best[0] < best[0]:
            best = candidate_best
        if best[2] <= 0.001:
            break
    if best is None or best[2] > 0.004:
        raise BasketTeacherError("joint grasp planner found no bilateral grasp")
    target = best[1]
    left_target = target[1:7].copy()
    right_target = target[7:13].copy()
    current_left = source.qpos[
        [model.jnt_qposadr[joint] for joint in left_joints]
    ].copy()
    current_right = source.qpos[
        [model.jnt_qposadr[joint] for joint in right_joints]
    ].copy()
    waypoints = tuple(
        (
            current_left + alpha * (left_target - current_left),
            current_right + alpha * (right_target - current_right),
        )
        for alpha in np.linspace(0.12, 1.0, 9)
    )
    clearance = _validate_grasp_path(
        backend,
        work,
        base_x=float(target[0]),
        waypoints=waypoints,
        base_address=base_address,
        left_joints=left_joints,
        right_joints=right_joints,
    )
    return JointGraspPlan(
        base_x=float(target[0]),
        left_joint_target=left_target,
        right_joint_target=right_target,
        waypoints=waypoints,
        maximum_pad_distance=float(best[2]),
        path_minimum_clearance=clearance,
    )


def _score_grasp_candidate(
    backend: MujocoBimanualTaskBackend,
    work: mujoco.MjData,
    candidate: np.ndarray,
    addresses: np.ndarray,
    base_address: int,
) -> tuple[float, np.ndarray, float]:
    model = backend.model
    mujoco.mj_copyData(work, model, backend.data)
    work.qpos[base_address] = candidate[0]
    work.qpos[addresses] = candidate[1:]
    for joint in (
        *backend.bundle.ids.secondary_finger_joints,
        *backend.bundle.ids.finger_joints,
    ):
        work.qpos[model.jnt_qposadr[joint]] = GRASP_GRIPPER * FINGER_TRAVEL
    mujoco.mj_forward(model, work)
    distances = _pad_distances(backend, work)
    maximum = float(distances.max())
    left_worst = float(distances[:2].max())
    right_worst = float(distances[2:].max())
    target_distance = distances + 0.0015
    collision = _forbidden_penetration(backend, work)
    symmetry = abs(left_worst - right_worst)
    score = float(
        np.square(target_distance).sum()
        + 8.0 * max(maximum, 0.0) ** 2
        + 2.0 * symmetry**2
        + 25.0 * collision**2
        + 1.0e-5
        * np.square(candidate[1:] - np.tile(NOMINAL_GRASP, 2)).sum()
    )
    return score, candidate.copy(), maximum


def _validate_grasp_path(
    backend: MujocoBimanualTaskBackend,
    work: mujoco.MjData,
    *,
    base_x: float,
    waypoints: tuple[tuple[np.ndarray, np.ndarray], ...],
    base_address: int,
    left_joints: tuple[int, ...],
    right_joints: tuple[int, ...],
) -> float:
    model = backend.model
    minimum_clearance = float("inf")
    for left, right in waypoints:
        mujoco.mj_copyData(work, model, backend.data)
        work.qpos[base_address] = base_x
        for values, joints in ((left, left_joints), (right, right_joints)):
            work.qpos[[model.jnt_qposadr[joint] for joint in joints]] = values
        for joint in (
            *backend.bundle.ids.secondary_finger_joints,
            *backend.bundle.ids.finger_joints,
        ):
            work.qpos[model.jnt_qposadr[joint]] = 0.0
        mujoco.mj_forward(model, work)
        penetration = _forbidden_penetration(backend, work)
        if penetration > 0.002:
            raise BasketTeacherError("joint grasp path intersects scene geometry")
        minimum_clearance = min(minimum_clearance, -penetration)
    return minimum_clearance


def _pad_distances(
    backend: MujocoBimanualTaskBackend,
    data: mujoco.MjData,
) -> np.ndarray:
    ids = backend.task_ids
    pairs = (
        *((pad, ids.left_interaction_geom) for pad in sorted(ids.left_pads)),
        *((pad, ids.right_interaction_geom) for pad in sorted(ids.right_pads)),
    )
    segment = np.zeros(6)
    return np.asarray(
        [
            mujoco.mj_geomDistance(
                backend.model,
                data,
                pad,
                handle,
                2.0,
                segment,
            )
            for pad, handle in pairs
        ]
    )


def _forbidden_penetration(
    backend: MujocoBimanualTaskBackend,
    data: mujoco.MjData,
) -> float:
    ids = backend.task_ids
    penetration = 0.0
    for contact in data.contact[: data.ncon]:
        first, second = int(contact.geom1), int(contact.geom2)
        robot_first = first in ids.robot_geoms
        robot_second = second in ids.robot_geoms
        if robot_first == robot_second:
            continue
        other = second if robot_first else first
        if other in ids.allowed_robot_contacts:
            continue
        penetration = max(penetration, max(0.0, -float(contact.dist)))
    return penetration
