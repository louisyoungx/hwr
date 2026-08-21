"""Pure contracts for the frozen R0001-P41-E2 target-index diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np

INPUT_SCHEMA = "hwr.p41-target-index-input/v1"
CANDIDATE_SCHEMA = "hwr.p41-target-candidates/v1"
PLAN_SCHEMA = "hwr.p41-target-selection-plan/v1"
TERMINAL_SCHEMA = "hwr.p41-target-selection-terminal/v1"
POWER_SCHEMA = "hwr.p41-target-selection-power/v1"
PROPOSAL_ID = "R0001-P41-E2"
TASK_IDS = ("tidy_living_room_3d/v1", "clear_dining_table_3d/v1",
            "store_kitchen_items_3d/v1")
UNIFORM_DOMAIN = b"R0001-P41-E2|uniform-index-v1"
PHASES = (
    ("B0_orient", 100),
    ("B1_approach", 300),
    ("B2_preposition", 100),
    ("B3_contact_approach", 50),
    ("B4_close", 20),
    ("B5_pull", 30),
    ("B6_retract", 50),
    ("B7_stop", 10),
)
POST_SELECTION_STEPS = sum(length for _, length in PHASES)
ACQUISITION_STEPS = 995
PLANNED_HORIZON = ACQUISITION_STEPS + POST_SELECTION_STEPS
POWER_PAIR_COUNTS = (36, 54, 72, 90, 108)
POWER_TRIALS = 10_000
POWER_SEED = 20_264_102
INPUT_HEADER = struct.Struct("<qqiiQ")
INPUT_ARRAY_SPECS = (
    ("head_rgb_uint8", np.dtype("u1"), (192, 256, 3)),
    ("head_depth_m", np.dtype("<f4"), (192, 256)),
    ("head_depth_valid", np.dtype("u1"), (192, 256)),
    ("head_camera_intrinsics", np.dtype("<f8"), (4,)),
    ("robot_from_head_camera", np.dtype("<f8"), (4, 4)),
    ("proprioception", np.dtype("<f8"), (37,)),
    ("executed_action_history", np.dtype("<f8"), (4, 16)),
    ("history_available", np.dtype("u1"), (4,)),
)
SAFETY_STATES = ("ok", "degraded", "stopped", "emergency_stop")
ACTION_MINIMUM = np.asarray((-0.18, -0.50, *(-0.35,) * 12, 0.0, 0.0))
ACTION_MAXIMUM = np.asarray((0.18, 0.50, *(0.35,) * 12, 1.0, 1.0))


class TargetSelectionContractError(ValueError):
    """Raised when serialized policy input or frozen evidence is invalid."""


@dataclass(frozen=True)
class PolicyVisibleInput:
    observation_timestamp_ns: int
    sequence_id: int
    phase_index: int
    phase_step: int
    policy_rng_seed: int
    safety_state: str
    head_rgb_uint8: np.ndarray
    head_depth_m: np.ndarray
    head_depth_valid: np.ndarray
    head_camera_intrinsics: np.ndarray
    robot_from_head_camera: np.ndarray
    proprioception: np.ndarray
    executed_action_history: np.ndarray
    history_available: np.ndarray

    @property
    def base_pose(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.proprioception[26:29])

    @property
    def left_joint_position(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.proprioception[:6])

    @property
    def right_joint_position(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.proprioception[12:18])


@dataclass(frozen=True)
class RawCandidate:
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    width: float
    prominence: float
    support_count: int
    frame_ordinal: int
    row: int
    column: int


@dataclass(frozen=True)
class Candidate:
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    width: float
    prominence: float
    support_count: int
    view_count: int
    first_frame: int
    first_row: int
    first_column: int

    def canonical_key(self) -> tuple[int, ...]:
        return (
            *_quantize(self.center, 1_000.0),
            *_quantize(self.normal, 10_000.0),
            int(round(self.width * 1_000.0)),
            self.first_frame,
            self.first_row,
            self.first_column,
        )

    def canonical_record(self) -> tuple[int, ...]:
        return (
            *self.canonical_key(),
            int(round(self.prominence * 1_000.0)),
            self.support_count,
            self.view_count,
        )


@dataclass(frozen=True)
class CandidateSet:
    input_sha256: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    canonical_bytes: bytes
    candidate_set_sha256: str


def serialize_policy_input(value: PolicyVisibleInput) -> bytes:
    if value.observation_timestamp_ns < 0 or value.sequence_id < 0:
        raise TargetSelectionContractError("timestamp and sequence must be nonnegative")
    if value.phase_index < 0 or value.phase_step < 0:
        raise TargetSelectionContractError("phase state must be nonnegative")
    if not 0 <= value.policy_rng_seed < 2**64:
        raise TargetSelectionContractError("policy RNG seed must be uint64")
    try:
        safety_index = SAFETY_STATES.index(str(value.safety_state))
    except ValueError as error:
        raise TargetSelectionContractError("unknown safety state") from error
    arrays = []
    for name, dtype, shape in INPUT_ARRAY_SPECS:
        array = np.asarray(getattr(value, name))
        if array.shape != shape:
            raise TargetSelectionContractError(f"{name} shape differs")
        if name == "head_depth_valid" or name == "history_available":
            array = array.astype(dtype, copy=False)
        elif array.dtype != dtype:
            raise TargetSelectionContractError(f"{name} dtype differs")
        array = np.ascontiguousarray(array, dtype=dtype)
        if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
            raise TargetSelectionContractError(f"{name} contains nonfinite values")
        arrays.append(array.tobytes(order="C"))
    header = INPUT_HEADER.pack(
        int(value.observation_timestamp_ns),
        int(value.sequence_id),
        int(value.phase_index),
        int(value.phase_step),
        int(value.policy_rng_seed),
    )
    return b"".join(
        (INPUT_SCHEMA.encode("ascii"), b"\0", header, bytes((safety_index,)), *arrays)
    )


def deserialize_policy_input(payload: bytes) -> PolicyVisibleInput:
    prefix = INPUT_SCHEMA.encode("ascii") + b"\0"
    if not payload.startswith(prefix):
        raise TargetSelectionContractError("policy input schema differs")
    offset = len(prefix)
    timestamp, sequence, phase_index, phase_step, policy_seed = INPUT_HEADER.unpack_from(
        payload, offset
    )
    offset += INPUT_HEADER.size
    if offset >= len(payload) or payload[offset] >= len(SAFETY_STATES):
        raise TargetSelectionContractError("serialized safety state differs")
    safety_state = SAFETY_STATES[payload[offset]]
    offset += 1
    arrays: dict[str, np.ndarray] = {}
    for name, dtype, shape in INPUT_ARRAY_SPECS:
        size = math.prod(shape) * dtype.itemsize
        end = offset + size
        if end > len(payload):
            raise TargetSelectionContractError("serialized policy input is truncated")
        array = np.frombuffer(payload[offset:end], dtype=dtype).reshape(shape)
        array = array.astype(np.bool_, copy=False) if name.endswith(("valid", "available")) else array
        array.setflags(write=False)
        arrays[name] = array
        offset = end
    if offset != len(payload):
        raise TargetSelectionContractError("serialized policy input has trailing bytes")
    return PolicyVisibleInput(
        timestamp, sequence, phase_index, phase_step, policy_seed, safety_state,
        **arrays,
    )


def input_sha256(payload: bytes) -> str:
    deserialize_policy_input(payload)
    return hashlib.sha256(payload).hexdigest()


def corridor_obstacle_count(payload: bytes) -> int:
    value = deserialize_policy_input(payload)
    rows, columns = np.nonzero(value.head_depth_valid)
    depth = value.head_depth_m[rows, columns].astype(np.float64)
    valid = np.isfinite(depth) & (depth >= 0.10) & (depth <= 5.00)
    points = _camera_points(
        rows[valid], columns[valid], depth[valid], value.head_camera_intrinsics
    )
    points = _transform_points(value.robot_from_head_camera, points)
    return int(
        np.count_nonzero(
            (points[:, 0] >= 0.20)
            & (points[:, 0] <= 0.65)
            & (np.abs(points[:, 1]) <= 0.38)
            & (points[:, 2] >= -0.18)
            & (points[:, 2] <= 1.15)
        )
    )


def generate_candidate_set(
    keyframes: Sequence[bytes],
    *,
    acquisition_base_pose: Sequence[float],
    final_input: bytes,
) -> CandidateSet:
    origin = _pose(acquisition_base_pose)
    hashes = tuple(input_sha256(payload) for payload in (*keyframes, final_input))
    raw: list[RawCandidate] = []
    for ordinal, payload in enumerate(keyframes):
        frame = deserialize_policy_input(payload)
        raw.extend(_frame_candidates(frame, origin, ordinal))
    merged = _merge_candidates(raw)
    ordered = sorted(
        merged,
        key=lambda item: (
            -item.support_count,
            -item.view_count,
            -int(round(item.prominence * 1_000_000.0)),
            *_quantize(item.center, 1_000.0),
            item.first_frame,
            item.first_row,
            item.first_column,
        ),
    )[:64]
    ordered = tuple(sorted(ordered, key=Candidate.canonical_key))
    document = {
        "schema_version": CANDIDATE_SCHEMA,
        "acquisition_input_sha256": list(hashes),
        "candidate_count": len(ordered),
        "candidates": [list(item.canonical_record()) for item in ordered],
    }
    canonical = json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return CandidateSet(
        hashes, ordered, canonical, hashlib.sha256(canonical).hexdigest()
    )


def candidate_scores(
    candidates: CandidateSet,
    final_base_pose: Sequence[float],
    *,
    acquisition_base_pose: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[float, ...]:
    base = _acquisition_from_robot(
        acquisition_base_pose, final_base_pose
    )[:2, 3]
    centers = np.asarray([item.center for item in candidates.candidates], np.float64)
    scores = []
    for index, item in enumerate(candidates.candidates):
        if len(centers) == 1:
            nearest = 0.40
        else:
            distances = np.linalg.norm(centers[:, :3] - centers[index, :3], axis=1)
            nearest = float(np.min(distances[np.arange(len(centers)) != index]))
        horizontal_range = float(np.linalg.norm(centers[index, :2] - base))
        s_prom = _clip((item.prominence - 0.025) / 0.125, 0.0, 1.0)
        s_size = math.exp(-0.5 * ((item.width - 0.14) / 0.08) ** 2)
        s_view = min(item.view_count / 4.0, 1.0)
        s_iso = _clip((nearest - 0.08) / 0.32, 0.0, 1.0)
        s_range = _clip((3.0 - horizontal_range) / 2.0, 0.0, 1.0)
        scores.append(
            0.30 * s_prom
            + 0.25 * s_size
            + 0.20 * s_view
            + 0.15 * s_iso
            + 0.10 * s_range
        )
    return tuple(scores)


def select_candidate_index(
    candidates: CandidateSet,
    final_base_pose: Sequence[float],
    *,
    acquisition_base_pose: Sequence[float] = (0.0, 0.0, 0.0),
) -> int:
    if not candidates.candidates:
        return -1
    scores = candidate_scores(
        candidates,
        final_base_pose,
        acquisition_base_pose=acquisition_base_pose,
    )
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def select_control_index(candidates: CandidateSet, policy_rng_seed: int) -> int:
    count = len(candidates.candidates)
    if count == 0:
        return -1
    if not 0 <= policy_rng_seed < 2**64:
        raise TargetSelectionContractError("control seed must be uint64")
    limit = (2**256 // count) * count
    for counter in range(2**64):
        digest = hashlib.sha256(
            UNIFORM_DOMAIN
            + struct.pack("<Q", policy_rng_seed)
            + bytes.fromhex(candidates.candidate_set_sha256)
            + struct.pack("<Q", counter)
        ).digest()
        integer = int.from_bytes(digest, "big")
        if integer < limit:
            return integer % count
    raise TargetSelectionContractError("uniform-index rejection did not terminate")


def primitive_action(
    serialized_input: bytes,
    candidate: Candidate | None,
    acquisition_base_pose: Sequence[float],
    post_selection_step: int,
) -> tuple[float, ...]:
    value = deserialize_policy_input(serialized_input)
    if not 0 <= post_selection_step < POST_SELECTION_STEPS:
        raise TargetSelectionContractError("post-selection step is outside horizon")
    if candidate is None or value.safety_state != "ok":
        return _hold_action(value)
    phase, local_step = phase_for_step(post_selection_step)
    origin = _pose(acquisition_base_pose)
    base_in_acquisition = _acquisition_from_robot(origin, value.base_pose)
    base = base_in_acquisition[:3, 3]
    point = np.asarray(candidate.center, np.float64)
    forward = point[:2] - base[:2]
    horizontal_range = float(np.linalg.norm(forward))
    if horizontal_range < 0.35:
        return _hold_action(value)
    forward /= horizontal_range
    normal = np.asarray((-forward[0], -forward[1], 0.0))
    lateral = np.asarray((-forward[1], forward[0], 0.0))
    vertical = np.asarray((0.0, 0.0, 1.0))
    spacing_pre = _clip(candidate.width + 0.12, 0.18, 0.34)
    spacing_contact = _clip(candidate.width + 0.04, 0.10, 0.24)
    left_pre = point + 0.18 * normal + 0.5 * spacing_pre * lateral + 0.05 * vertical
    right_pre = point + 0.18 * normal - 0.5 * spacing_pre * lateral + 0.05 * vertical
    left_contact = point + 0.015 * normal + 0.5 * spacing_contact * lateral
    right_contact = point + 0.015 * normal - 0.5 * spacing_contact * lateral
    heading = math.atan2(float(forward[1]), float(forward[0]))
    base_yaw = _wrap(float(value.base_pose[2] - origin[2]))
    heading_error = _wrap(heading - base_yaw)
    base_linear = 0.0
    base_angular = 0.0
    left_target = right_target = None
    velocity_max = 0.0
    gripper = 0.0
    if phase == "B0_orient":
        base_angular = _clip(heading_error, -0.35, 0.35)
    elif phase == "B1_approach":
        base_linear = (
            0.0
            if abs(heading_error) > 0.35
            else _clip(0.6 * (horizontal_range - 0.85), 0.0, 0.12)
        )
        base_angular = _clip(heading_error, -0.25, 0.25)
    elif phase == "B2_preposition":
        left_target, right_target, velocity_max = left_pre, right_pre, 0.08
    elif phase == "B3_contact_approach":
        left_target, right_target, velocity_max = left_contact, right_contact, 0.03
    elif phase == "B4_close":
        left_target, right_target, velocity_max = left_contact, right_contact, 0.02
        gripper = 0.75 * (local_step + 1) / 20.0
    elif phase == "B5_pull":
        offset = 0.08 * normal + 0.02 * vertical
        left_target, right_target, velocity_max = (
            left_contact + offset,
            right_contact + offset,
            0.04,
        )
        gripper = 0.75
    elif phase == "B6_retract":
        offset = 0.05 * normal
        left_target, right_target, velocity_max = (
            left_pre + offset,
            right_pre + offset,
            0.06,
        )
        gripper = 0.75 if local_step < 30 else 0.75 * (49 - local_step) / 19.0
    left_velocity = right_velocity = np.zeros(3, np.float64)
    if left_target is not None and right_target is not None:
        left_tool, right_tool = tool_positions_in_acquisition(value, origin)
        left_velocity = _clip_norm(2.0 * (left_target - left_tool), velocity_max)
        right_velocity = _clip_norm(2.0 * (right_target - right_tool), velocity_max)
    action = np.asarray(
        (
            base_linear, base_angular,
            *(left_velocity / 0.30), 0.0, 0.0, 0.0,
            *(right_velocity / 0.30), 0.0, 0.0, 0.0,
            gripper, gripper,
        ),
        np.float64,
    )
    return tuple(float(item) for item in np.clip(action, ACTION_MINIMUM, ACTION_MAXIMUM))


def phase_for_step(step: int) -> tuple[str, int]:
    offset = 0
    for name, length in PHASES:
        if step < offset + length:
            return name, step - offset
        offset += length
    raise TargetSelectionContractError("primitive phase step is outside horizon")


def tool_positions_in_acquisition(
    value: PolicyVisibleInput, acquisition_base_pose: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    transform = _acquisition_from_robot(acquisition_base_pose, value.base_pose)
    return (
        _transform_point(transform, _tool_position(value.left_joint_position, 0.31)),
        _transform_point(transform, _tool_position(value.right_joint_position, -0.31)),
    )


def _frame_candidates(
    frame: PolicyVisibleInput, acquisition_base_pose: tuple[float, float, float],
    ordinal: int,
) -> list[RawCandidate]:
    depth = frame.head_depth_m
    valid = frame.head_depth_valid & np.isfinite(depth) & (depth >= 0.10) & (depth <= 5.00)
    transform = _acquisition_from_robot(acquisition_base_pose, frame.base_pose)
    camera_transform = transform @ frame.robot_from_head_camera
    camera_origin = camera_transform[:3, 3]
    base_center = transform[:3, 3]
    result = []
    for row in range(12, 180, 4):
        for column in range(12, 244, 4):
            center_depth = depth[row - 2 : row + 3, column - 2 : column + 3]
            center_valid = valid[row - 2 : row + 3, column - 2 : column + 3]
            ring_depth = depth[row - 10 : row + 11, column - 10 : column + 11]
            ring_valid = valid[row - 10 : row + 11, column - 10 : column + 11].copy()
            ring_valid[6:15, 6:15] = False
            if center_valid.sum() < 20 or ring_valid.sum() < 240:
                continue
            center_values = center_depth[center_valid].astype(np.float64)
            center_z = float(np.median(center_values))
            prominence = float(np.median(ring_depth[ring_valid])) - center_z
            if not 0.025 <= prominence <= 0.45:
                continue
            if float(np.quantile(center_values, 0.90) - np.quantile(center_values, 0.10)) > 0.04:
                continue
            patch_depth = depth[row - 10 : row + 11, column - 10 : column + 11]
            patch_valid = valid[row - 10 : row + 11, column - 10 : column + 11]
            patch_valid &= np.abs(patch_depth - center_z) <= max(0.025, 0.015 * center_z)
            rows, columns = np.nonzero(patch_valid)
            if len(rows) < 24:
                continue
            rows, columns = rows + row - 10, columns + column - 10
            points = _camera_points(
                rows, columns, patch_depth[patch_valid].astype(np.float64),
                frame.head_camera_intrinsics,
            )
            points = _transform_points(camera_transform, points)
            points = points[~_robot_self_mask(points, frame, acquisition_base_pose)]
            if len(points) < 24:
                continue
            candidate = _candidate_from_points(
                points, camera_origin, base_center, prominence, ordinal, row, column
            )
            if candidate is not None:
                result.append(candidate)
    return result


def _candidate_from_points(
    points: np.ndarray, camera: np.ndarray, base: np.ndarray, prominence: float,
    ordinal: int, row: int, column: int,
) -> RawCandidate | None:
    center = np.median(points, axis=0)
    horizontal = float(np.linalg.norm(center[:2] - base[:2]))
    if not -0.18 <= center[2] <= 1.30 or not 0.35 <= horizontal <= 4.00:
        return None
    covariance = np.cov(points - points.mean(axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    total = float(eigenvalues.sum())
    if total <= 0.0 or float(eigenvalues[0] / total) > 0.12:
        return None
    normal = eigenvectors[:, 0]
    if float(np.dot(normal, camera - center)) < 0.0:
        normal = -normal
    spans = [
        float(np.quantile((points - center) @ eigenvectors[:, index], 0.95)
              - np.quantile((points - center) @ eigenvectors[:, index], 0.05))
        for index in (1, 2)
    ]
    width = max(spans)
    if not 0.035 <= width <= 0.40:
        return None
    return RawCandidate(
        tuple(float(item) for item in center),
        tuple(float(item) for item in normal / np.linalg.norm(normal)),
        width, prominence, len(points), ordinal, row, column,
    )


def _merge_candidates(raw: Sequence[RawCandidate]) -> list[Candidate]:
    adjacency = [set() for _ in raw]
    for left in range(len(raw)):
        for right in range(left):
            distance = np.linalg.norm(
                np.asarray(raw[left].center) - np.asarray(raw[right].center)
            )
            cosine = float(np.dot(raw[left].normal, raw[right].normal))
            if (
                distance <= 0.08
                and cosine >= 0.80
                and abs(raw[left].width - raw[right].width) <= 0.10
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    seen: set[int] = set()
    candidates = []
    for start in range(len(raw)):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(raw[current])
            for neighbor in sorted(adjacency[current] - seen):
                seen.add(neighbor)
                stack.append(neighbor)
        views = {item.frame_ordinal for item in component}
        if len(views) < 2:
            continue
        normal = np.sum([item.normal for item in component], axis=0)
        if np.linalg.norm(normal) == 0.0:
            continue
        first = min(
            (item.frame_ordinal, item.row, item.column) for item in component
        )
        candidates.append(
            Candidate(
                tuple(float(item) for item in np.median(
                    [item.center for item in component], axis=0
                )),
                tuple(float(item) for item in normal / np.linalg.norm(normal)),
                float(np.median([item.width for item in component])),
                max(item.prominence for item in component),
                sum(item.support_count for item in component),
                len(views), *first,
            )
        )
    return candidates


def _robot_self_mask(
    points: np.ndarray, value: PolicyVisibleInput,
    acquisition_base_pose: Sequence[float],
) -> np.ndarray:
    transform = _acquisition_from_robot(acquisition_base_pose, value.base_pose)
    robot_points = _transform_points(np.linalg.inv(transform), points)
    base = (
        (np.abs(robot_points[:, 0] + 0.01) <= 0.36)
        & (np.abs(robot_points[:, 1]) <= 0.29)
        & (robot_points[:, 2] >= -0.21)
        & (robot_points[:, 2] <= 1.38)
    )
    mask = base.copy()
    for joints, lateral in (
        (value.left_joint_position, 0.31),
        (value.right_joint_position, -0.31),
    ):
        chain = _arm_chain(joints, lateral)
        for start, end, radius in zip(chain[:-1], chain[1:], (0.10, 0.065, 0.059, 0.055, 0.052, 0.13)):
            mask |= _point_segment_distance(robot_points, start, end) < radius + 0.06
    return mask


def _arm_chain(joints: Sequence[float], lateral: float) -> list[np.ndarray]:
    transforms = np.eye(4)
    points = []
    specifications = (
        ((0.02, lateral, 0.82), "z", joints[0]),
        ((0.0, 0.0, 0.13), "y", joints[1]),
        ((0.31, 0.0, 0.0), "y", joints[2]),
        ((0.27, 0.0, 0.0), "x", joints[3]),
        ((0.09, 0.0, 0.0), "y", joints[4]),
        ((0.08, 0.0, 0.0), "x", joints[5]),
        ((0.255, 0.0, -0.045), None, 0.0),
    )
    for translation, axis, angle in specifications:
        transforms = transforms @ _translation(translation)
        points.append(transforms[:3, 3].copy())
        if axis is not None:
            transforms = transforms @ _rotation(axis, float(angle))
    return points


def _tool_position(joints: Sequence[float], lateral: float) -> np.ndarray:
    return _arm_chain(joints, lateral)[-1]


def _hold_action(value: PolicyVisibleInput) -> tuple[float, ...]:
    return (
        0.0, 0.0, *(0.0,) * 12,
        float(value.proprioception[24]), float(value.proprioception[25]),
    )


def _camera_points(
    rows: np.ndarray, columns: np.ndarray, depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    fx, fy, cx, cy = (float(item) for item in intrinsics)
    return np.column_stack(
        ((columns - cx) * depth / fx, (rows - cy) * depth / fy, depth)
    )


def _acquisition_from_robot(
    acquisition_pose: Sequence[float], robot_pose: Sequence[float]
) -> np.ndarray:
    acquisition = _pose(acquisition_pose)
    robot = _pose(robot_pose)
    yaw = robot[2] - acquisition[2]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    transform = np.asarray(
        (
            (cosine, -sine, 0.0, 0.0),
            (sine, cosine, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        np.float64,
    )
    delta = np.asarray(robot[:2]) - np.asarray(acquisition[:2])
    origin_rotation = np.asarray(
        ((math.cos(acquisition[2]), math.sin(acquisition[2])),
         (-math.sin(acquisition[2]), math.cos(acquisition[2]))),
        np.float64,
    )
    transform[:2, 3] = origin_rotation @ delta
    return transform


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def _point_segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator == 0.0:
        return np.linalg.norm(points - start, axis=1)
    fraction = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
    closest = start + fraction[:, None] * delta
    return np.linalg.norm(points - closest, axis=1)


def _translation(values: Sequence[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = values
    return result


def _rotation(axis: str, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    rotations = {
        "x": ((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)),
        "y": ((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)),
        "z": ((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)),
    }
    result = np.eye(4)
    result[:3, :3] = rotations[axis]
    return result


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum or norm == 0.0 else vector * maximum / norm


def _pose(value: Sequence[float]) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise TargetSelectionContractError("base pose must have three finite values")
    return result


def _quantize(values: Sequence[float], scale: float) -> tuple[int, ...]:
    return tuple(int(round(float(item) * scale)) for item in values)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))
