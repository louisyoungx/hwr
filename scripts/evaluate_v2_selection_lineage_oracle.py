#!/usr/bin/env python3
"""Blind, source-disjoint R0001-P83 v2 selection-lineage oracle."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, multiprocessing, os
import resource, stat, struct, sys, threading, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence
if os.environ.get("HWR_P83_ISOLATED") == "1" and importlib.util.find_spec("hwr") is not None:
    raise RuntimeError("hwr_import_visible")
import numpy as np
PROPOSAL_ID = "R0001-P83"
PLAN_SCHEMA = "hwr.p83-blind-plan/v1"
RECEIPT_SCHEMA = "hwr.p83-blind-selection-receipt/v1"
INPUT_SCHEMA = "hwr.p41-target-index-input/v1"
CANDIDATE_VISIBLE_SCHEMA = "hwr.p50-candidate-visible-input/v1"
CANDIDATE_SCHEMA = "hwr.p79-target-candidates/v2"
INPUT_HEADER = struct.Struct("<qqiiQ")
ARRAY_SPECS = (
    ("head_rgb_uint8", np.dtype("u1"), (192, 256, 3)),
    ("head_depth_m", np.dtype("<f4"), (192, 256)),
    ("head_depth_valid", np.dtype("u1"), (192, 256)),
    ("head_camera_intrinsics", np.dtype("<f8"), (4,)),
    ("robot_from_head_camera", np.dtype("<f8"), (4, 4)),
    ("proprioception", np.dtype("<f8"), (37,)),
    ("executed_action_history", np.dtype("<f8"), (4, 16)),
    ("history_available", np.dtype("u1"), (4,)))
SAFETY_STATES = ("ok", "degraded", "stopped", "emergency_stop")
SCORE_WEIGHTS = (0.30, 0.25, 0.20, 0.15, 0.10)
TIE_BREAK_ID = "maximum-score-then-lowest-index/v1"
_LAST_PROCESS_TREE_PEAK_RSS_BYTES = 0
_LAST_READ_AUDIT_EVENTS: list[str] = []
_READ_AUDIT: "ReadAudit | None" = None
class OracleContractError(ValueError):
    """Raised when blind input or output violates the frozen contract."""
class ReadAudit:
    """Auxiliary audit for source-whitelisted file opens."""
    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self.events: list[str] = []
    def authorize(self, path: object, record: str | None = None) -> None:
        self._local.path = os.fsdecode(os.fspath(path))
        self._local.record = record
    def clear(self) -> None:
        self._local.path = None
        self._local.record = None
    def hook(self, event: str, arguments: tuple[object, ...]) -> None:
        if event != "open" or not arguments: return
        try:
            path = os.fsdecode(os.fspath(arguments[0]))
        except TypeError: return
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        reading = (("r" in mode or "+" in mode) if isinstance(mode, str)
                   else not isinstance(flags, int)
                   or (flags & os.O_ACCMODE) != os.O_WRONLY)
        if not reading: return
        try:
            authorized = self._local.path
        except AttributeError: return
        if path != authorized: return
        try:
            record = self._local.record
        except AttributeError: return
        if record is not None:
            with self._lock:
                self.events.append(record)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--plan-bytes", type=int, required=True)
    parser.add_argument("--plan-sha256", required=True); parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--worker-sha256", required=True)
    return parser
def run(arguments: argparse.Namespace) -> dict[str, object]:
    global _LAST_PROCESS_TREE_PEAK_RSS_BYTES
    started = time.perf_counter()
    _assert_runtime_isolation(); _prewarm_numpy()
    input_root = arguments.input_root.absolute(); plan_path = arguments.plan.absolute(); output = arguments.output.absolute()
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError(output)
    audit = ReadAudit()
    global _READ_AUDIT
    _READ_AUDIT = audit; sys.addaudithook(audit.hook)
    plan_bytes, plan_identity = stable_file_read(plan_path, {
        "bytes": arguments.plan_bytes, "sha256": arguments.plan_sha256},
        kind="sanitized_plan", logical_path="blind-plan.json",
        root=plan_path.parent)
    plan = json.loads(plan_bytes); _validate_plan(plan)
    receipt = rebuild_plan(plan, input_root=input_root,
                           plan_sha256=_sha256(plan_bytes),
                           worker_source_sha256=arguments.worker_sha256)
    receipt["read_audit"] = {"trust_role": "auxiliary",
        "audited_open_count": len(_LAST_READ_AUDIT_EVENTS),
        "expected_open_count": 1 + int(plan["input_file_count"]),
        "path_sequence_sha256": _sha256(
            json.dumps(sorted(_LAST_READ_AUDIT_EVENTS),
                       separators=(",", ":")).encode("utf-8")),
        "plan_fd_identity": plan_identity}
    receipt_bytes = canonical_json_bytes(receipt)
    _atomic_write(output, receipt_bytes)
    _LAST_PROCESS_TREE_PEAK_RSS_BYTES = max(
        _LAST_PROCESS_TREE_PEAK_RSS_BYTES, _peak_rss_bytes())
    return {"output_sha256": _sha256(receipt_bytes),
        "wall_seconds": time.perf_counter() - started,
        "process_tree_peak_rss_upper_bound_bytes": _LAST_PROCESS_TREE_PEAK_RSS_BYTES}
def rebuild_plan(plan: Mapping[str, object], *, input_root: Path,
                 plan_sha256: str,
                 worker_source_sha256: str) -> dict[str, object]:
    global _LAST_PROCESS_TREE_PEAK_RSS_BYTES, _LAST_READ_AUDIT_EVENTS
    _validate_plan(plan)
    jobs = [(episode, str(input_root)) for episode in plan["episodes"]]
    workers = min(len(jobs), os.cpu_count() or 1)
    if len(jobs) == 1: results = [_rebuild_job(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context("fork")) as executor:
            results = list(executor.map(_rebuild_job, jobs))
    episodes = [value["episode"] for value in results]; read_ledger = [entry for value in results for entry in value["read_ledger"]]
    peaks: dict[int, int] = {}
    for value in results:
        pid = int(value["worker_pid"])
        peaks[pid] = max(peaks.get(pid, 0), int(value["worker_peak_rss_bytes"]))
    _LAST_PROCESS_TREE_PEAK_RSS_BYTES = _peak_rss_bytes() + sum(peaks.values())
    parent_events = [] if _READ_AUDIT is None else list(_READ_AUDIT.events)
    _LAST_READ_AUDIT_EVENTS = parent_events if len(jobs) == 1 else [
        *parent_events, *(event for value in results
                          for event in value["audit_events"])]
    capture_count = sum(int(value["capture_count"]) for value in episodes); candidate_count = sum(int(value["candidate_count"]) for value in episodes)
    return {
        "schema_version": RECEIPT_SCHEMA, "proposal_id": PROPOSAL_ID,
        "status": "complete", "plan_sha256": plan_sha256,
        "worker_source_sha256": worker_source_sha256,
        "candidate_schema_version": CANDIDATE_SCHEMA,
        "score_weights": list(SCORE_WEIGHTS), "tie_break_id": TIE_BREAK_ID,
        "episode_count": len(episodes), "capture_count": capture_count,
        "candidate_count": candidate_count,
        "input_file_match_count": len(read_ledger),
        "execution": {"job_count": len(jobs), "worker_count": workers,
            "worker_process_count": len(peaks),
            "parallel_path_used": len(jobs) > 1},
        "read_ledger": read_ledger, "episodes": episodes,
        "mutation_evidence": mutation_evidence(episodes)}
def _rebuild_job(job: tuple[Mapping[str, object], str]) -> dict[str, object]:
    episode, input_root_value = job
    audit_offset = len(_READ_AUDIT.events) if _READ_AUDIT is not None else 0
    read_ledger: list[dict[str, object]] = []
    rebuilt = _rebuild_episode(episode, input_root=Path(input_root_value),
                               read_ledger=read_ledger, seen_paths=set())
    return {"episode": rebuilt, "read_ledger": read_ledger,
            "worker_pid": os.getpid(), "worker_peak_rss_bytes": _peak_rss_bytes(),
            "audit_events": ([] if _READ_AUDIT is None
                             else _READ_AUDIT.events[audit_offset:])}
def _rebuild_episode(episode: Mapping[str, object], *, input_root: Path,
                     read_ledger: list[dict[str, object]],
                     seen_paths: set[str]) -> dict[str, object]:
    payloads: list[bytes] = []; frames: list[dict[str, object]] = []
    visible_by_identity: dict[tuple[int, int], str] = {}
    for capture in episode["captures"]:
        policy = _read_bound_blob(input_root, capture["policy_input"], kind="policy_input",
                                  read_ledger=read_ledger, seen_paths=seen_paths)
        visible = _read_bound_blob(input_root, capture["candidate_visible_input"],
                                   kind="candidate_visible_input", read_ledger=read_ledger,
                                   seen_paths=seen_paths)
        frame = parse_policy_input(policy)
        expected_identity = tuple(int(value) for value in capture["observation_identity"])
        if frame["identity"] != expected_identity:
            raise OracleContractError("observation_identity")
        if candidate_visible_bytes(frame) != visible:
            raise OracleContractError("candidate_visible_bytes")
        visible_hash = _sha256(visible)
        if (expected_identity in visible_by_identity
                and visible_by_identity[expected_identity] != visible_hash):
            raise OracleContractError("observation_identity_reuse")
        visible_by_identity.setdefault(expected_identity, visible_hash)
        payloads.append(policy); frames.append(frame)
    candidates = generate_candidates(frames[:-1],
                                     acquisition_base_pose=episode["acquisition_base_pose"])
    input_hashes = tuple(_sha256(payload) for payload in payloads)
    canonical = candidate_document(input_hashes, candidates)
    scores = compute_scores(candidates, frames[-1]["base_pose"],
                            episode["acquisition_base_pose"])
    selected_index = select_index(scores)
    quantized = np.asarray([candidate_from_record(_candidate_record(value)) for value
                            in candidates], dtype=np.float64)
    canonical_scores = compute_scores(
        quantized, frames[-1]["base_pose"], episode["acquisition_base_pose"])
    order_mutation = candidate_document(input_hashes, candidates[::-1])
    moved_poses = (_world_poses_for_candidate(candidates[0],
        episode["acquisition_base_pose"], frames[-1]["base_pose"])
        if len(candidates) else (frames[-1]["base_pose"],) * 2)
    moved_scores = [compute_scores(candidates, pose, episode["acquisition_base_pose"])
                    for pose in moved_poses]
    weighted_scores = compute_scores(candidates, frames[-1]["base_pose"],
        episode["acquisition_base_pose"], weights=(0.29, 0.26, 0.20, 0.15, 0.10))
    tied = (scores[0], scores[0]) if scores else ()
    return {
        "episode_ordinal": episode["episode_ordinal"],
        "planned_episode_id": episode["planned_episode_id"], "task_id": episode["task_id"],
        "cell_id": episode["cell_id"], "replicate_ordinal": episode["replicate_ordinal"],
        "capture_count": len(payloads), "candidate_count": len(candidates),
        "candidate_canonical_ascii": canonical.decode("ascii"),
        "candidate_bytes": len(canonical), "candidate_sha256": _sha256(canonical),
        "score_bytes_sha256": score_hash(scores),
        "canonical_only_score_bytes_sha256": score_hash(canonical_scores),
        "selected_index": selected_index, "top_two_score_margin": top_two_margin(scores),
        "selected_canonical_identity": selected_identity(candidates, selected_index),
        "mutation_evidence": {
            "candidate_order_sha256": _sha256(order_mutation),
            "final_base_score_sha256": [score_hash(value) for value in moved_scores],
            "weight_score_sha256": score_hash(weighted_scores),
            "tie_break_baseline_index": select_index(tied),
            "tie_break_mutated_index": select_index(tied,
                tie_break="maximum-score-then-highest-index/mutation")}}
def parse_policy_input(payload: bytes) -> dict[str, object]:
    prefix = INPUT_SCHEMA.encode("ascii") + b"\0"
    if not payload.startswith(prefix):
        raise OracleContractError("policy_input_schema")
    offset = len(prefix)
    try:
        header = INPUT_HEADER.unpack_from(payload, offset)
    except struct.error as error:
        raise OracleContractError("policy_input_truncated") from error
    timestamp, sequence, phase_index, phase_step, policy_seed = header
    if min(timestamp, sequence, phase_index, phase_step) < 0 or policy_seed < 0:
        raise OracleContractError("policy_input_header")
    offset += INPUT_HEADER.size
    if offset >= len(payload) or payload[offset] >= len(SAFETY_STATES):
        raise OracleContractError("policy_input_safety")
    offset += 1
    arrays: dict[str, np.ndarray] = {}
    for name, dtype, shape in ARRAY_SPECS:
        size = math.prod(shape) * dtype.itemsize
        end = offset + size
        if end > len(payload):
            raise OracleContractError("policy_input_truncated")
        array = np.frombuffer(payload[offset:end], dtype=dtype).reshape(shape)
        if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
            raise OracleContractError("policy_input_nonfinite")
        if name.endswith(("valid", "available")) and np.any(array > 1):
            raise OracleContractError("policy_input_boolean")
        arrays[name] = array
        offset = end
    if offset != len(payload):
        raise OracleContractError("policy_input_trailing_bytes")
    proprioception = arrays["proprioception"]
    return {
        "identity": (int(timestamp), int(sequence)),
        "base_pose": tuple(float(value) for value in proprioception[26:29]),
        "joint_matrix": np.vstack((proprioception[:6], proprioception[12:18])),
        "lateral_offsets": np.asarray((0.31, -0.31), dtype=np.float64),
        "rgb": arrays["head_rgb_uint8"], "depth": arrays["head_depth_m"],
        "depth_valid": arrays["head_depth_valid"].astype(np.bool_, copy=False),
        "intrinsics": arrays["head_camera_intrinsics"],
        "robot_from_camera": arrays["robot_from_head_camera"],
        "proprioception": proprioception}
def candidate_visible_bytes(frame: Mapping[str, object]) -> bytes:
    proprioception = frame["proprioception"]
    selected = np.concatenate((proprioception[:6], proprioception[12:18],
                               proprioception[26:29]))
    return b"".join((CANDIDATE_VISIBLE_SCHEMA.encode("ascii"), b"\0",
                     np.ascontiguousarray(frame["rgb"]).tobytes(),
                     np.ascontiguousarray(frame["depth"]).tobytes(),
                     np.ascontiguousarray(frame["depth_valid"], dtype=np.uint8).tobytes(),
                     np.ascontiguousarray(frame["intrinsics"]).tobytes(),
                     np.ascontiguousarray(frame["robot_from_camera"]).tobytes(),
                     np.ascontiguousarray(selected, dtype="<f8").tobytes()))
def generate_candidates(frames: Sequence[Mapping[str, object]], *,
                        acquisition_base_pose: Sequence[float]) -> np.ndarray:
    origin = _pose(acquisition_base_pose)
    batches = []
    for ordinal, frame in enumerate(frames):
        batch = scan_frame(frame, origin, ordinal)
        if len(batch):
            batches.append(batch)
    raw = np.concatenate(batches) if batches else np.empty((0, 12), np.float64)
    merged = merge_components(raw)
    ranked = sorted(range(len(merged)), key=lambda index: (
        -int(merged[index, 8]), -int(merged[index, 9]),
        -int(round(merged[index, 7] * 1_000_000.0)),
        *_quantize(merged[index, :3], 1_000.0),
        *tuple(int(value) for value in merged[index, 10:13])))[:64]
    selected = merged[ranked] if ranked else np.empty((0, 13), np.float64)
    order = sorted(range(len(selected)),
                   key=lambda index: _candidate_key(selected[index]))
    return selected[order]
def scan_frame(frame: Mapping[str, object],
               acquisition_base_pose: tuple[float, float, float],
               ordinal: int) -> np.ndarray:
    depth = frame["depth"]
    valid = frame["depth_valid"] & np.isfinite(depth) & (depth >= 0.10) & (depth <= 5.00)
    transform = acquisition_from_robot(acquisition_base_pose, frame["base_pose"])
    camera_transform = transform @ frame["robot_from_camera"]
    context = (depth, valid, camera_transform, camera_transform[:3, 3],
               transform[:3, 3], frame, acquisition_base_pose, ordinal)
    result = [candidate for anchor in anchor_measurements(depth, valid)
              if (candidate := project_anchor(context, anchor)) is not None]
    return np.asarray(result, dtype=np.float64).reshape((-1, 12))
def anchor_measurements(depth: np.ndarray, valid: np.ndarray) -> list[tuple[int, int, float, float]]:
    windows = np.lib.stride_tricks.sliding_window_view(valid, (21, 21))[2:169:4, 2:233:4]
    ring = np.ones((21, 21), dtype=np.bool_); ring[6:15, 6:15] = False
    possible = np.argwhere((windows[..., 8:13, 8:13].sum((-2, -1)) >= 20)
                           & ((windows & ring).sum((-2, -1)) >= 240))
    measured = []
    for vertical, horizontal in possible:
        row, column = 12 + 4 * int(vertical), 12 + 4 * int(horizontal)
        center = depth[row - 2:row + 3, column - 2:column + 3]
        center_mask = valid[row - 2:row + 3, column - 2:column + 3]
        ring_depth = depth[row - 10:row + 11, column - 10:column + 11]
        ring_mask = valid[row - 10:row + 11, column - 10:column + 11].copy()
        ring_mask[6:15, 6:15] = False
        values = center[center_mask].astype(np.float64)
        center_z = float(np.median(values))
        prominence = float(np.median(ring_depth[ring_mask])) - center_z
        spread = float(np.quantile(values, 0.90) - np.quantile(values, 0.10))
        if 0.025 <= prominence <= 0.45 and spread <= 0.04:
            measured.append((row, column, center_z, prominence))
    return measured
def project_anchor(context: tuple[object, ...],
                   anchor: tuple[int, int, float, float]) -> np.ndarray | None:
    depth, valid, camera_transform, camera, base, frame, origin, ordinal = context
    row, column, center_z, prominence = anchor
    depth_patch = depth[row - 10:row + 11, column - 10:column + 11]
    support = valid[row - 10:row + 11, column - 10:column + 11].copy()
    support &= np.abs(depth_patch - center_z) <= max(0.025, 0.015 * center_z)
    rows, columns = np.nonzero(support)
    if len(rows) < 24:
        return None
    rows += row - 10; columns += column - 10
    cloud = camera_points(rows, columns, depth_patch[support].astype(np.float64),
                          frame["intrinsics"])
    cloud = transform_points(camera_transform, cloud)
    cloud = cloud[~robot_self_mask(cloud, frame, origin)]
    if len(cloud) < 24:
        return None
    return candidate_from_points(
        cloud, camera, base, prominence, ordinal, row, column)
def candidate_from_points(points: np.ndarray, camera: np.ndarray, base: np.ndarray,
                          prominence: float, ordinal: int, row: int,
                          column: int) -> np.ndarray | None:
    center = np.median(points, axis=0)
    horizontal = float(np.linalg.norm(center[:2] - base[:2]))
    if not -0.18 <= center[2] <= 1.30 or not 0.35 <= horizontal <= 4.00:
        return None
    surface = fit_surface(points, center, camera)
    if surface is None:
        return None
    normal, width = surface
    return np.asarray((*center, *normal, width, prominence, len(points),
                       ordinal, row, column), np.float64)
def fit_surface(points: np.ndarray, center: np.ndarray,
                camera: np.ndarray) -> tuple[np.ndarray, float] | None:
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.cov(points - points.mean(axis=0), rowvar=False))
    if (total := float(eigenvalues.sum())) <= 0.0 or eigenvalues[0] / total > 0.12:
        return None
    normal = eigenvectors[:, 0]
    normal *= -1.0 if np.dot(normal, camera - center) < 0.0 else 1.0
    coordinates = points - center
    width = max(
        float(np.quantile(coordinates @ eigenvectors[:, axis], 0.95)
              - np.quantile(coordinates @ eigenvectors[:, axis], 0.05))
        for axis in (1, 2))
    if not 0.035 <= width <= 0.40:
        return None
    return normal / np.linalg.norm(normal), width
def merge_components(raw: np.ndarray) -> np.ndarray:
    if not len(raw):
        return np.empty((0, 13), np.float64)
    parent = np.arange(len(raw))
    delta = raw[:, None, :3] - raw[None, :, :3]
    connected = ((np.linalg.norm(delta, axis=2) <= 0.08)
                 & ((raw[:, 3:6] @ raw[:, 3:6].T) >= 0.80)
                 & (np.abs(raw[:, None, 6] - raw[None, :, 6]) <= 0.10))
    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index
    for left, right in np.argwhere(np.tril(connected, k=-1)):
        left_root, right_root = root(int(left)), root(int(right))
        if left_root != right_root: parent[max(left_root, right_root)] = min(left_root, right_root)
    groups: dict[int, list[int]] = {}
    for index in range(len(raw)):
        groups.setdefault(root(index), []).append(index)
    result = []
    for indices in groups.values():
        rows = raw[indices]
        views = np.unique(rows[:, 9].astype(np.int64))
        normal = rows[:, 3:6].sum(axis=0)
        if len(views) < 2 or np.linalg.norm(normal) == 0.0:
            continue
        first = min(tuple(int(value) for value in row[9:12]) for row in rows)
        result.append(np.asarray((
            *np.median(rows[:, :3], axis=0), *(normal / np.linalg.norm(normal)),
            np.median(rows[:, 6]), np.max(rows[:, 7]), np.sum(rows[:, 8]),
            len(views), *first), np.float64))
    return np.asarray(result, dtype=np.float64).reshape((-1, 13))
def candidate_document(input_hashes: Sequence[str],
                       candidates: np.ndarray) -> bytes:
    document = {
        "schema_version": CANDIDATE_SCHEMA,
        "acquisition_input_sha256": list(input_hashes),
        "candidate_count": len(candidates),
        "candidates": [list(_candidate_record(value)) for value in candidates],
    }
    return json.dumps(document, ensure_ascii=True, separators=(",", ":"),
                      sort_keys=True).encode("ascii")
def compute_scores(candidates: np.ndarray, final_base_pose: Sequence[float],
                   acquisition_base_pose: Sequence[float], *,
                   weights: Sequence[float] = SCORE_WEIGHTS) -> tuple[float, ...]:
    if len(weights) != 5 or not all(math.isfinite(float(value)) for value in weights):
        raise OracleContractError("score_weights")
    base = acquisition_from_robot(acquisition_base_pose, final_base_pose)[:2, 3]
    if not len(candidates):
        return ()
    centers = candidates[:, :3]
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.full(len(candidates), 0.40) if len(candidates) == 1 else distances.min(axis=1)
    horizontal = np.linalg.norm(centers[:, :2] - base, axis=1)
    size_term = np.asarray([
        math.exp(-0.5 * ((width - 0.14) / 0.08) ** 2)
        for width in candidates[:, 6]], np.float64)
    terms = np.column_stack((
        np.clip((candidates[:, 7] - 0.025) / 0.125, 0.0, 1.0),
        size_term,
        np.minimum(candidates[:, 9] / 4.0, 1.0),
        np.clip((nearest - 0.08) / 0.32, 0.0, 1.0),
        np.clip((3.0 - horizontal) / 2.0, 0.0, 1.0)))
    return tuple(
        float(weights[0]) * float(row[0])
        + float(weights[1]) * float(row[1])
        + float(weights[2]) * float(row[2])
        + float(weights[3]) * float(row[3])
        + float(weights[4]) * float(row[4])
        for row in terms
    )
def select_index(scores: Sequence[float], *, tie_break: str = TIE_BREAK_ID) -> int:
    if not scores:
        return -1
    if tie_break == TIE_BREAK_ID:
        return int(np.argmax(np.asarray(scores, np.float64)))
    if tie_break == "maximum-score-then-highest-index/mutation":
        return len(scores) - 1 - int(np.argmax(np.asarray(scores[::-1], np.float64)))
    raise OracleContractError("tie_break")
def selected_identity(candidates: np.ndarray, selected_index: int) -> str | None:
    if selected_index == -1:
        if len(candidates):
            raise OracleContractError("selected_index")
        return None
    if not 0 <= selected_index < len(candidates):
        raise OracleContractError("selected_index")
    record = list(_candidate_record(candidates[selected_index]))
    return _sha256(json.dumps(record, separators=(",", ":")).encode("ascii"))
def candidate_from_record(record: Sequence[int]) -> np.ndarray:
    if len(record) != 13 or not all(isinstance(value, int) for value in record):
        raise OracleContractError("canonical_candidate_record")
    return np.asarray((*[float(value) / 1_000.0 for value in record[:3]],
        *[float(value) / 10_000.0 for value in record[3:6]],
        float(record[6]) / 1_000.0, float(record[10]) / 1_000.0,
        int(record[11]), int(record[12]), int(record[7]), int(record[8]),
        int(record[9])), np.float64)
def _candidate_key(candidate: np.ndarray) -> tuple[int, ...]:
    return (*_quantize(candidate[:3], 1_000.0),
            *_quantize(candidate[3:6], 10_000.0),
            int(round(candidate[6] * 1_000.0)),
            *tuple(int(value) for value in candidate[10:13]))
def _candidate_record(candidate: np.ndarray) -> tuple[int, ...]:
    return (*_candidate_key(candidate), int(round(candidate[7] * 1_000.0)),
            int(candidate[8]), int(candidate[9]))
def score_hash(scores: Sequence[float]) -> str:
    return _sha256(np.ascontiguousarray(scores, dtype="<f8").tobytes())
def top_two_margin(scores: Sequence[float]) -> float | None:
    if len(scores) < 2:
        return None
    ordered = sorted((float(value) for value in scores), reverse=True)
    return ordered[0] - ordered[1]
def camera_points(rows: np.ndarray, columns: np.ndarray, depth: np.ndarray,
                  intrinsics: np.ndarray) -> np.ndarray:
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    pixels -= intrinsics[2:4]
    pixels *= depth[:, None]
    pixels /= intrinsics[:2]
    return np.column_stack((pixels, depth))
def robot_self_mask(points: np.ndarray, frame: Mapping[str, object],
                    acquisition_base_pose: Sequence[float]) -> np.ndarray:
    transform = acquisition_from_robot(acquisition_base_pose, frame["base_pose"])
    robot_points = transform_points(np.linalg.inv(transform), points)
    mask = ((np.abs(robot_points[:, 0] + 0.01) <= 0.36)
            & (np.abs(robot_points[:, 1]) <= 0.29)
            & (robot_points[:, 2] >= -0.21)
            & (robot_points[:, 2] <= 1.38))
    chains = [arm_chain(joints, lateral)
              for joints, lateral in zip(frame["joint_matrix"], frame["lateral_offsets"])]
    starts = np.concatenate([chain[:-1] for chain in chains])
    ends = np.concatenate([chain[1:] for chain in chains])
    radii = np.tile(np.asarray((0.10, 0.065, 0.059, 0.055, 0.052, 0.13)), 2)
    mask |= np.any(point_segment_distances(robot_points, starts, ends)
                   < radii[None, :] + 0.06, axis=1)
    return mask
def arm_chain(joints: Sequence[float], lateral: float) -> list[np.ndarray]:
    offsets = ((0.02, lateral, 0.82), (0.0, 0.0, 0.13),
               (0.31, 0.0, 0.0), (0.27, 0.0, 0.0),
               (0.09, 0.0, 0.0), (0.08, 0.0, 0.0),
               (0.255, 0.0, -0.045))
    axes = ("z", "y", "y", "x", "y", "x", None)
    angles = (*joints, 0.0)
    pose = np.eye(4); positions = []
    for offset, axis, angle in zip(offsets, axes, angles):
        pose = pose @ rigid_translation(offset)
        positions.append(np.array(pose[:3, 3], copy=True))
        if axis:
            pose = pose @ joint_rotation(axis, float(angle))
    return positions
def acquisition_from_robot(acquisition_pose: Sequence[float],
                           robot_pose: Sequence[float]) -> np.ndarray:
    acquisition = _pose(acquisition_pose)
    robot = _pose(robot_pose)
    yaw = robot[2] - acquisition[2]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    transform = np.asarray(((cosine, -sine, 0.0, 0.0),
                            (sine, cosine, 0.0, 0.0),
                            (0.0, 0.0, 1.0, 0.0),
                            (0.0, 0.0, 0.0, 1.0)), np.float64)
    delta = np.asarray(robot[:2]) - np.asarray(acquisition[:2])
    rotation = np.asarray(((math.cos(acquisition[2]), math.sin(acquisition[2])),
                           (-math.sin(acquisition[2]), math.cos(acquisition[2]))),
                          np.float64)
    transform[:2, 3] = rotation @ delta
    return transform
def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]
def point_segment_distances(points: np.ndarray, starts: np.ndarray,
                            ends: np.ndarray) -> np.ndarray:
    deltas = ends - starts
    squared = np.einsum("ij,ij->i", deltas, deltas)
    offsets = points[:, None, :] - starts[None, :, :]
    fractions = np.einsum("nsi,si->ns", offsets, deltas)
    fractions = np.divide(fractions, squared, out=np.zeros_like(fractions),
                          where=squared[None, :] != 0.0)
    fractions = np.clip(fractions, 0.0, 1.0)
    closest = starts[None, :, :] + fractions[..., None] * deltas[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)
def _read_bound_blob(root: Path, descriptor: Mapping[str, object], *, kind: str,
                     read_ledger: list[dict[str, object]],
                     seen_paths: set[str]) -> bytes:
    relative = _validate_relative_path(descriptor.get("path"))
    path = root / relative
    relative_text = relative.as_posix()
    if relative_text in seen_paths:
        raise OracleContractError("input_path_duplicate")
    seen_paths.add(relative_text)
    content, identity = stable_file_read(
        path, descriptor, kind=kind, logical_path=relative_text, root=root)
    read_ledger.append(identity)
    return content
def stable_file_read(path: Path, descriptor: Mapping[str, object], *, kind: str,
                     logical_path: str, root: Path | None = None
                     ) -> tuple[bytes, dict[str, object]]:
    if root is None: root = path.parent
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise OracleContractError("path_escape_or_missing") from error
    if not relative.parts or ".." in relative.parts:
        raise OracleContractError("path_escape_or_missing")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directories: list[int] = []; descriptor_fd: int | None = None
    try:
        root_before = os.lstat(root)
        if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
            raise OracleContractError(f"{kind}_symlink_or_type")
        if _READ_AUDIT is not None: _READ_AUDIT.authorize(root)
        try: root_fd = os.open(root, directory_flags)
        finally:
            if _READ_AUDIT is not None: _READ_AUDIT.clear()
        directories.append(root_fd)
        root_open = os.fstat(root_fd)
        if (root_before.st_dev, root_before.st_ino) != (root_open.st_dev, root_open.st_ino):
            raise OracleContractError(f"{kind}_changed_during_read")
        parent_fd = root_fd
        for part in relative.parts[:-1]:
            before_directory = os.lstat(part, dir_fd=parent_fd)
            if stat.S_ISLNK(before_directory.st_mode) or not stat.S_ISDIR(before_directory.st_mode):
                raise OracleContractError(f"{kind}_symlink")
            if _READ_AUDIT is not None: _READ_AUDIT.authorize(part)
            try: next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            finally:
                if _READ_AUDIT is not None: _READ_AUDIT.clear()
            opened_directory = os.fstat(next_fd); directories.append(next_fd)
            if ((before_directory.st_dev, before_directory.st_ino)
                    != (opened_directory.st_dev, opened_directory.st_ino)):
                raise OracleContractError(f"{kind}_changed_during_read")
            parent_fd = next_fd
        leaf = relative.parts[-1]; before = os.lstat(leaf, dir_fd=parent_fd)
        if stat.S_ISLNK(before.st_mode): raise OracleContractError(f"{kind}_symlink")
        if not stat.S_ISREG(before.st_mode): raise OracleContractError(f"{kind}_type")
        if _READ_AUDIT is not None: _READ_AUDIT.authorize(leaf, logical_path)
        try:
            descriptor_fd = os.open(leaf, file_flags, dir_fd=parent_fd)
        finally:
            if _READ_AUDIT is not None: _READ_AUDIT.clear()
        opened = os.fstat(descriptor_fd); chunks = []
        while True:
            chunk = os.read(descriptor_fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk)
        after_read = os.fstat(descriptor_fd)
        after_path = os.lstat(leaf, dir_fd=parent_fd)
    except FileNotFoundError as error:
        raise OracleContractError(f"{kind}_missing") from error
    except OSError as error:
        raise OracleContractError(f"{kind}_open") from error
    finally:
        if descriptor_fd is not None: os.close(descriptor_fd)
        for directory in reversed(directories): os.close(directory)
    identities = [_stat_identity(value) for value in
                  (before, opened, after_read, after_path)]
    if len({tuple(value.values()) for value in identities}) != 1:
        raise OracleContractError(f"{kind}_changed_during_read")
    content = b"".join(chunks)
    if (len(content) != descriptor.get("bytes")
            or _sha256(content) != descriptor.get("sha256")):
        raise OracleContractError(f"{kind}_size_or_hash")
    return content, {
        "kind": kind, "path": logical_path, "bytes": len(content),
        "sha256": _sha256(content), "fd_identity": identities[1]}
def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {"device": int(value.st_dev), "inode": int(value.st_ino), "size": int(value.st_size)}
def _validate_plan(plan: Mapping[str, object]) -> None:
    required = {"schema_version", "proposal_id", "episode_count",
                "capture_count", "input_file_count", "episodes"}
    if not isinstance(plan, Mapping) or set(plan) != required:
        raise OracleContractError("plan_fields")
    if plan["schema_version"] != PLAN_SCHEMA or plan["proposal_id"] != PROPOSAL_ID:
        raise OracleContractError("plan_schema")
    episodes = plan["episodes"]
    if not isinstance(episodes, list) or len(episodes) != plan["episode_count"]:
        raise OracleContractError("episode_count")
    identifiers: list[str] = []
    paths: list[str] = []
    capture_total = 0
    for ordinal, episode in enumerate(episodes):
        _validate_episode(episode, ordinal)
        identifiers.append(str(episode["planned_episode_id"]))
        capture_total += len(episode["captures"])
        paths.extend(capture[name]["path"] for capture in episode["captures"]
                     for name in ("policy_input", "candidate_visible_input"))
    if len(identifiers) != len(set(identifiers)):
        raise OracleContractError("episode_duplicate")
    if len(paths) != len(set(paths)): raise OracleContractError("input_path_duplicate")
    if capture_total != plan["capture_count"]:
        raise OracleContractError("capture_count")
    if 2 * capture_total != plan["input_file_count"]:
        raise OracleContractError("input_file_count")
def _validate_episode(episode: Mapping[str, object], ordinal: int) -> None:
    required = {"episode_ordinal", "planned_episode_id", "task_id", "cell_id",
                "replicate_ordinal", "acquisition_base_pose", "captures"}
    if not isinstance(episode, Mapping) or set(episode) != required:
        raise OracleContractError("episode_fields")
    if episode["episode_ordinal"] != ordinal:
        raise OracleContractError("episode_order")
    _pose(episode["acquisition_base_pose"])
    captures = episode["captures"]
    if not isinstance(captures, list) or not captures:
        raise OracleContractError("capture_missing")
    for capture_ordinal, capture in enumerate(captures):
        required_capture = {"capture_ordinal", "final_input",
                            "observation_identity", "policy_input",
                            "candidate_visible_input"}
        if not isinstance(capture, Mapping) or set(capture) != required_capture:
            raise OracleContractError("capture_fields")
        if capture["capture_ordinal"] != capture_ordinal:
            raise OracleContractError("capture_order")
        identity = capture["observation_identity"]
        if (not isinstance(identity, list) or len(identity) != 2
                or not all(isinstance(value, int) and value >= 0
                           for value in identity)):
            raise OracleContractError("observation_identity")
        for name in ("policy_input", "candidate_visible_input"):
            _validate_descriptor(capture[name])
    final_flags = [capture["final_input"] for capture in captures]
    if any(type(value) is not bool for value in final_flags):
        raise OracleContractError("final_input_type")
    if sum(final_flags) != 1 or final_flags[-1] is not True:
        raise OracleContractError("final_input")
def _validate_descriptor(descriptor: Mapping[str, object]) -> None:
    if (not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "bytes", "sha256"}):
        raise OracleContractError("input_descriptor")
    _validate_relative_path(descriptor["path"])
    if not isinstance(descriptor["bytes"], int) or descriptor["bytes"] <= 0:
        raise OracleContractError("input_size")
    digest = descriptor["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)):
        raise OracleContractError("input_hash")
def _validate_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise OracleContractError("path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise OracleContractError("path_escape")
    return path
def mutation_evidence(episodes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    nonempty = [value for value in episodes if value["candidate_count"]]
    multi = [value for value in episodes if value["candidate_count"] > 1]
    return {
        "candidate_order_changed_count": sum(value["candidate_sha256"] !=
            value["mutation_evidence"]["candidate_order_sha256"] for value in multi),
        "candidate_order_denominator": len(multi),
        "final_base_score_changed_count": sum(any(value["score_bytes_sha256"] != digest
            for digest in value["mutation_evidence"]["final_base_score_sha256"])
            for value in nonempty),
        "weight_score_changed_count": sum(value["score_bytes_sha256"] !=
            value["mutation_evidence"]["weight_score_sha256"] for value in nonempty),
        "tie_break_flip_count": sum(value["mutation_evidence"]["tie_break_baseline_index"]
            == 0 and value["mutation_evidence"]["tie_break_mutated_index"] == 1
            for value in nonempty),
        "canonical_only_score_mismatch_count": sum(value["score_bytes_sha256"] !=
            value["canonical_only_score_bytes_sha256"] for value in nonempty),
        "nonempty_denominator": len(nonempty)}
def _assert_runtime_isolation() -> None:
    if importlib.util.find_spec("hwr") is not None:
        raise OracleContractError("hwr_import_visible")
def _prewarm_numpy() -> None:
    matrix = np.asarray(((2.0, 0.0), (0.0, 1.0)), np.float64)
    values = np.arange(25, dtype=np.float64).reshape(5, 5)
    np.linalg.eigh(matrix); np.linalg.inv(matrix); np.linalg.norm(matrix)
    np.quantile(matrix, (0.1, 0.9)); np.median(matrix); np.cov(matrix)
    np.lib.stride_tricks.sliding_window_view(values, (2, 2))
    np.unique(values); np.argwhere(values); np.column_stack((values, values))
    np.concatenate((values, values)); np.exp(values); np.ascontiguousarray(values)
def _world_poses_for_candidate(candidate: np.ndarray,
        acquisition_pose: Sequence[float], final_pose: Sequence[float]
        ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    origin = _pose(acquisition_pose)
    cosine, sine = math.cos(origin[2]), math.sin(origin[2])
    x = origin[0] + cosine * candidate[0] - sine * candidate[1]
    y = origin[1] + sine * candidate[0] + cosine * candidate[1]
    yaw = float(final_pose[2])
    return ((x, y, yaw), (x + 10.0 * cosine, y + 10.0 * sine, yaw))
def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
def rigid_translation(values: Sequence[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = values
    return result
def joint_rotation(axis: str, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    rotations = {"x": ((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)),
                 "y": ((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)),
                 "z": ((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1))}
    result = np.eye(4)
    result[:3, :3] = rotations[axis]
    return result
def _pose(value: Sequence[float]) -> tuple[float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise OracleContractError("base_pose") from error
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise OracleContractError("base_pose")
    return result
def _quantize(values: Sequence[float], scale: float) -> tuple[int, ...]:
    return tuple(int(round(float(value) * scale)) for value in values)
def _sha256(content: bytes) -> str: return hashlib.sha256(content).hexdigest()
def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss); return value if sys.platform == "darwin" else value * 1024
def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv)); print(json.dumps(result, sort_keys=True)); return 0
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OracleContractError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(3) from None
