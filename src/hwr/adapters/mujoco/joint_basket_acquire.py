"""Online pad/handle feedback for the R0020 acquire phase."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.names import FINGER_TRAVEL


MIDPOINT_TOLERANCE_METERS = 0.012
ROTATION_TOLERANCE_RADIANS = 0.10
PAD_TARGET_DISTANCE_METERS = -0.001
MAXIMUM_GRIPPER_INCREMENT = 0.06
PAD_BALANCE_ENTRY_DISTANCE_METERS = 0.015
MAXIMUM_PAD_BALANCE_METERS = 0.003


@dataclass(frozen=True)
class ArmAcquireGeometry:
    target_site_position: np.ndarray
    target_site_rotation: np.ndarray
    pad_distances: tuple[float, float]
    pad_contacts: tuple[bool, bool]
    midpoint_error: np.ndarray
    rotation_error_radians: float

    @property
    def centered(self) -> bool:
        return (
            float(np.linalg.norm(self.midpoint_error))
            <= MIDPOINT_TOLERANCE_METERS
            and self.rotation_error_radians <= ROTATION_TOLERANCE_RADIANS
        )

    @property
    def bilateral_contact(self) -> bool:
        return all(self.pad_contacts)


@dataclass(frozen=True)
class AcquireFeedback:
    targets: tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]
    grippers: tuple[float, float]
    arms: tuple[ArmAcquireGeometry, ArmAcquireGeometry]

    @property
    def bilateral_contact(self) -> bool:
        return all(arm.bilateral_contact for arm in self.arms)


def acquire_feedback(
    backend: MujocoBimanualTaskBackend,
    *,
    target_rotations: tuple[np.ndarray, np.ndarray],
    target_handle_from_midpoints: tuple[np.ndarray, np.ndarray],
    current_grippers: tuple[float, float],
) -> AcquireFeedback:
    ids = backend.task_ids
    pairs = backend._contact_pairs()
    arms = tuple(
        _arm_geometry(
            backend,
            pad_ids=tuple(sorted(pads)),
            handle_id=handle,
            site_id=site,
            target_rotation=target_rotation,
            target_handle_offset=target_handle_offset,
            contact_pairs=pairs,
        )
        for pads, handle, site, target_rotation, target_handle_offset in (
            (
                ids.left_pads,
                ids.left_interaction_geom,
                ids.left_grasp_site,
                target_rotations[0],
                target_handle_from_midpoints[0],
            ),
            (
                ids.right_pads,
                ids.right_interaction_geom,
                ids.right_grasp_site,
                target_rotations[1],
                target_handle_from_midpoints[1],
            ),
        )
    )
    if not all(arm.centered for arm in arms):
        grippers = (0.0, 0.0)
    else:
        grippers = tuple(
            _next_gripper(current, arm)
            for current, arm in zip(current_grippers, arms, strict=True)
        )
    return AcquireFeedback(
        targets=tuple(
            (arm.target_site_position, arm.target_site_rotation) for arm in arms
        ),
        grippers=grippers,
        arms=arms,
    )


def _arm_geometry(
    backend: MujocoBimanualTaskBackend,
    *,
    pad_ids: tuple[int, int],
    handle_id: int,
    site_id: int,
    target_rotation: np.ndarray,
    target_handle_offset: np.ndarray,
    contact_pairs: set[frozenset[int]],
) -> ArmAcquireGeometry:
    data = backend.data
    pad_positions = tuple(data.geom_xpos[pad].copy() for pad in pad_ids)
    midpoint = 0.5 * (pad_positions[0] + pad_positions[1])
    handle_position = data.geom_xpos[handle_id].copy()
    site_position = data.site_xpos[site_id].copy()
    site_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    local_site_to_midpoint = site_rotation.T @ (midpoint - site_position)
    distances = tuple(
        _geom_distance(backend, pad, handle_id) for pad in pad_ids
    )
    target_midpoint = handle_position - target_rotation @ target_handle_offset
    target_midpoint += _pad_balance_correction(pad_positions, distances)
    target_position = target_midpoint - target_rotation @ local_site_to_midpoint
    contacts = tuple(
        frozenset((pad, handle_id)) in contact_pairs for pad in pad_ids
    )
    return ArmAcquireGeometry(
        target_site_position=target_position,
        target_site_rotation=target_rotation,
        pad_distances=distances,
        pad_contacts=contacts,
        midpoint_error=target_midpoint - midpoint,
        rotation_error_radians=_rotation_angle(target_rotation @ site_rotation.T),
    )


def _pad_balance_correction(
    pad_positions: tuple[np.ndarray, np.ndarray],
    distances: tuple[float, float],
) -> np.ndarray:
    if max(distances) > PAD_BALANCE_ENTRY_DISTANCE_METERS:
        return np.zeros(3)
    axis = pad_positions[1] - pad_positions[0]
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        return np.zeros(3)
    magnitude = float(
        np.clip(
            0.5 * (distances[0] - distances[1]),
            -MAXIMUM_PAD_BALANCE_METERS,
            MAXIMUM_PAD_BALANCE_METERS,
        )
    )
    return axis * (magnitude / norm)


def _next_gripper(current: float, arm: ArmAcquireGeometry) -> float:
    if arm.bilateral_contact:
        return 1.0
    remaining = max(arm.pad_distances) - PAD_TARGET_DISTANCE_METERS
    increment = np.clip(
        remaining / FINGER_TRAVEL,
        0.0,
        MAXIMUM_GRIPPER_INCREMENT,
    )
    return float(np.clip(current + increment, 0.0, 1.0))


def _geom_distance(
    backend: MujocoBimanualTaskBackend,
    first: int,
    second: int,
) -> float:
    segment = np.zeros(6)
    return float(
        mujoco.mj_geomDistance(
            backend.model,
            backend.data,
            first,
            second,
            2.0,
            segment,
        )
    )


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))
