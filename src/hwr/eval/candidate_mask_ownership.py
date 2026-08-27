"""Independent audit contracts for frozen R0001-P79-E1."""
from __future__ import annotations
import ast, copy, hashlib, json, re, sys
from collections import Counter
from dataclasses import dataclass
from types import FrameType
from typing import Iterable, Sequence
import numpy as np
from hwr.eval import target_selection
from hwr.eval.target_selection import (
    Candidate, CandidateSet, PolicyVisibleInput, RawCandidate,
    _acquisition_from_robot, _camera_points, _candidate_from_points,
    _merge_candidates, _pose, _quantize, _robot_self_mask, _transform_points,
    deserialize_policy_input, input_sha256)
PROPOSAL_ID = "R0001-P79-E1"; AUDIT_SCHEMA = "hwr.p79-candidate-mask-ownership-audit/v1"
CANDIDATE_SCHEMA_V2 = "hwr.p79-target-candidates/v2"
TRAVERSALS = ("row_major", "reverse_row_major", "column_major")
ROWS = tuple(range(12, 180, 4)); COLUMNS = tuple(range(12, 244, 4))
@dataclass(frozen=True)
class OracleFrame:
    raw: tuple[RawCandidate, ...]
    parent_sha256: str
    probe_count: int
    mutation_count: int
    probe_ledger_sha256: str
    probe_sha256_pair_counts: tuple[tuple[str, str, int], ...]
def generate_candidate_set_v2(keyframes: Sequence[bytes], *,
                              acquisition_base_pose: Sequence[float],
                              final_input: bytes) -> CandidateSet:
    origin = _pose(acquisition_base_pose); hashes = tuple(
        input_sha256(payload) for payload in (*keyframes, final_input))
    raw: list[RawCandidate] = []
    for ordinal, payload in enumerate(keyframes):
        frame = deserialize_policy_input(payload); raw.extend(
            _frame_candidates_v2(frame, origin, ordinal))
    merged = _merge_candidates(raw); ordered = sorted(merged, key=lambda item: (
        -item.support_count, -item.view_count,
        -int(round(item.prominence * 1_000_000.0)),
        *_quantize(item.center, 1_000.0),
        item.first_frame, item.first_row, item.first_column,
    ))[:64]
    ordered = tuple(sorted(ordered, key=Candidate.canonical_key)); document = {
        "schema_version": CANDIDATE_SCHEMA_V2,
        "acquisition_input_sha256": list(hashes),
        "candidate_count": len(ordered),
        "candidates": [list(item.canonical_record()) for item in ordered],
    }
    canonical = json.dumps(document, ensure_ascii=True, separators=(",", ":"),
                           sort_keys=True).encode("ascii")
    return CandidateSet(hashes, ordered, canonical, hashlib.sha256(canonical).hexdigest())
def _frame_candidates_v2(
    frame: PolicyVisibleInput, acquisition_base_pose: tuple[float, float, float],
    ordinal: int,
) -> list[RawCandidate]:
    depth = frame.head_depth_m; valid = (
        frame.head_depth_valid & np.isfinite(depth) & (depth >= 0.10) & (depth <= 5.00))
    transform = _acquisition_from_robot(acquisition_base_pose, frame.base_pose)
    camera_transform = transform @ frame.robot_from_head_camera
    camera_origin = camera_transform[:3, 3]; base_center = transform[:3, 3]
    result = []
    for row in range(12, 180, 4):
        for column in range(12, 244, 4):
            center_depth = depth[row - 2 : row + 3, column - 2 : column + 3]; center_valid = valid[row - 2 : row + 3, column - 2 : column + 3]
            ring_depth = depth[row - 10 : row + 11, column - 10 : column + 11]; ring_valid = valid[row - 10 : row + 11, column - 10 : column + 11].copy()
            ring_valid[6:15, 6:15] = False
            if center_valid.sum() < 20 or ring_valid.sum() < 240: continue
            center_values = center_depth[center_valid].astype(np.float64); center_z = float(np.median(center_values)); prominence = float(np.median(ring_depth[ring_valid])) - center_z
            if not 0.025 <= prominence <= 0.45: continue
            if float(np.quantile(center_values, 0.90) - np.quantile(center_values, 0.10)) > 0.04: continue
            patch_depth = depth[row - 10 : row + 11, column - 10 : column + 11]; patch_valid = valid[row - 10 : row + 11, column - 10 : column + 11]; patch_valid = patch_valid.copy()
            patch_valid &= np.abs(patch_depth - center_z) <= max(0.025, 0.015 * center_z)
            rows, columns = np.nonzero(patch_valid)
            if len(rows) < 24: continue
            rows, columns = rows + row - 10, columns + column - 10
            points = _camera_points(
                rows, columns, patch_depth[patch_valid].astype(np.float64),
                frame.head_camera_intrinsics)
            points = _transform_points(camera_transform, points); points = points[~_robot_self_mask(points, frame, acquisition_base_pose)]
            if len(points) < 24: continue
            candidate = _candidate_from_points(
                points, camera_origin, base_center, prominence, ordinal, row, column)
            if candidate is not None: result.append(candidate)
    return result
def audit_episode(
    keyframes: Sequence[bytes],
    *,
    acquisition_base_pose: Sequence[float],
    final_input: bytes,
) -> tuple[CandidateSet, dict[str, object]]:
    origin = target_selection._pose(acquisition_base_pose)
    frames = tuple(
        target_selection.deserialize_policy_input(payload)
        for payload in (*keyframes, final_input)
    )
    input_hashes = tuple(
        target_selection.input_sha256(payload)
        for payload in (*keyframes, final_input)
    )
    production, production_by_frame = _production_candidate_set(
        keyframes, acquisition_base_pose, final_input
    )
    oracle_raw = {name: [] for name in TRAVERSALS}
    frame_reports = []
    input_immutable = True
    for ordinal, frame in enumerate(frames):
        input_before = _sha256(frame.head_depth_valid.tobytes())
        if ordinal < len(keyframes):
            production_raw, production_mask = production_by_frame[ordinal]
        else:
            production_raw, production_mask = _production_frame(
                frame, origin, ordinal
            )
        input_after = _sha256(frame.head_depth_valid.tobytes())
        traversals = {
            name: oracle_frame_candidates(frame, origin, ordinal, name)
            for name in TRAVERSALS
        }
        if ordinal < len(keyframes):
            for name, result in traversals.items():
                oracle_raw[name].extend(result.raw)
        production_equal = (
            _raw_multiset(production_raw)
            == _raw_multiset(traversals["row_major"].raw)
        )
        traversal_equal = len({
            _raw_multiset_sha256(result.raw)
            for result in traversals.values()
        }) == 1
        immutable = (
            input_before == input_after
            and production_mask[0] == production_mask[1]
            and all(result.mutation_count == 0 for result in traversals.values())
        )
        input_immutable &= immutable
        frame_reports.append(
            {
                "frame_ordinal": ordinal,
                "observation_identity": [
                    frame.observation_timestamp_ns,
                    frame.sequence_id,
                ],
                "production_raw_count": len(production_raw),
                "production_raw_multiset_sha256": _raw_multiset_sha256(
                    production_raw
                ),
                "production_matches_oracle_row_major": production_equal,
                "production_parent_valid_sha256_before": production_mask[0],
                "production_parent_valid_sha256_after": production_mask[1],
                "production_parent_valid_byte_identical":
                    production_mask[0] == production_mask[1],
                "oracle_traversal_multisets_equal": traversal_equal,
                "input_head_depth_valid_sha256_before": input_before,
                "input_head_depth_valid_sha256_after": input_after,
                "input_head_depth_valid_byte_identical": immutable,
                "traversals": {
                    name: {
                        "raw_count": len(result.raw),
                        "raw_sequence_sha256": _raw_sequence_sha256(result.raw),
                        "raw_multiset_sha256": _raw_multiset_sha256(result.raw),
                        "parent_valid_sha256": result.parent_sha256,
                        "probe_count": result.probe_count,
                        "parent_mutation_count": result.mutation_count,
                        "probe_before_after_ledger_sha256": result.probe_ledger_sha256,
                        "probe_sha256_pair_counts": [
                            {
                                "before_sha256": before,
                                "after_sha256": after,
                                "count": count,
                            }
                            for before, after, count in result.probe_sha256_pair_counts
                        ],
                    }
                    for name, result in traversals.items()
                },
            }
        )
    final_sets = {
        name: candidate_set_from_raw(input_hashes, raw)
        for name, raw in oracle_raw.items()
    }
    production_raw_equal = all(
        frame["production_matches_oracle_row_major"] for frame in frame_reports
    ) and (
        _raw_multiset(
            candidate
            for ordinal in range(len(keyframes))
            for candidate in production_by_frame[ordinal][0]
        )
        == _raw_multiset(oracle_raw["row_major"])
    )
    traversal_raw_equal = all(
        frame["oracle_traversal_multisets_equal"] for frame in frame_reports
    ) and len({
        _raw_multiset_sha256(raw) for raw in oracle_raw.values()
    }) == 1
    final_bytes_equal = len({
        result.canonical_bytes for result in final_sets.values()
    }) == 1
    production_final_equal = (
        production.canonical_bytes
        == final_sets["row_major"].canonical_bytes
    )
    if not production_raw_equal:
        decision = "invalid"
    elif not traversal_raw_equal:
        decision = "rejected"
    elif not final_bytes_equal:
        decision = "inconclusive_secondary_order_dependence"
    elif not input_immutable or not production_final_equal:
        decision = "invalid"
    else:
        decision = "accepted as deterministic candidate-generator correction"
    checks = {
        "production_row_major_equals_oracle": production_raw_equal,
        "oracle_traversal_raw_multisets_equal": traversal_raw_equal,
        "oracle_parent_mask_mutation_count_zero": all(
            value["traversals"][name]["parent_mutation_count"] == 0
            for value in frame_reports
            for name in TRAVERSALS
        ),
        "input_head_depth_valid_byte_identical": input_immutable,
        "production_parent_valid_byte_identical": all(
            frame["production_parent_valid_byte_identical"]
            for frame in frame_reports
        ),
        "traversal_final_candidate_bytes_equal": final_bytes_equal,
        "production_final_equals_oracle_row_major": production_final_equal,
    }
    return production, {
        "schema_version": AUDIT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "decision": decision,
        "frame_count": len(frames),
        "frames": frame_reports,
        "traversals": {
            name: {
                "raw_count": len(raw),
                "raw_sequence_sha256": _raw_sequence_sha256(raw),
                "raw_multiset_sha256": _raw_multiset_sha256(raw),
                "candidate_count": len(final_sets[name].candidates),
                "candidate_set_sha256": final_sets[name].candidate_set_sha256,
            }
            for name, raw in oracle_raw.items()
        },
        "checks": {**checks, "passed": all(checks.values())},
    }
def _production_candidate_set(
    keyframes: Sequence[bytes],
    acquisition_base_pose: Sequence[float],
    final_input: bytes,
) -> tuple[CandidateSet, dict[int, tuple[tuple[RawCandidate, ...], tuple[str, str]]]]:
    raw_by_frame = {}
    initial_masks = {}
    def trace(frame: FrameType, event: str, argument: object):
        if (
            event == "line"
            and frame.f_code is _frame_candidates_v2.__code__
            and "valid" in frame.f_locals
        ):
            initial_masks.setdefault(
                int(frame.f_locals["ordinal"]),
                _sha256(frame.f_locals["valid"].tobytes()),
            )
        if (
            event == "return"
            and frame.f_code is _frame_candidates_v2.__code__
        ):
            ordinal = int(frame.f_locals["ordinal"])
            raw_by_frame[ordinal] = (
                tuple(argument),
                (
                    initial_masks[ordinal],
                    _sha256(frame.f_locals["valid"].tobytes()),
                ),
            )
        return trace
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        candidate_set = generate_candidate_set_v2(
            keyframes,
            acquisition_base_pose=acquisition_base_pose,
            final_input=final_input,
        )
    finally:
        sys.settrace(previous)
    if set(raw_by_frame) != set(range(len(keyframes))):
        raise RuntimeError("production candidate frame trace is incomplete")
    return candidate_set, raw_by_frame
def _production_frame(frame, origin, ordinal):
    initial = final = None
    def trace(current: FrameType, event: str, argument: object):
        nonlocal initial, final
        if current.f_code is _frame_candidates_v2.__code__:
            if event == "line" and "valid" in current.f_locals and initial is None:
                initial = _sha256(current.f_locals["valid"].tobytes())
            elif event == "return":
                final = _sha256(current.f_locals["valid"].tobytes())
        return trace
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        raw = tuple(_frame_candidates_v2(frame, origin, ordinal))
    finally:
        sys.settrace(previous)
    if initial is None or final is None:
        raise RuntimeError("production candidate frame trace is incomplete")
    return raw, (initial, final)
def oracle_frame_candidates(
    frame: PolicyVisibleInput,
    acquisition_base_pose: tuple[float, float, float],
    ordinal: int,
    traversal: str,
) -> OracleFrame:
    depth = frame.head_depth_m
    valid = (
        frame.head_depth_valid
        & np.isfinite(depth)
        & (depth >= 0.10)
        & (depth <= 5.00)
    )
    valid.setflags(write=False)
    parent_sha256 = _sha256(valid.tobytes())
    transform = target_selection._acquisition_from_robot(
        acquisition_base_pose, frame.base_pose
    )
    camera_transform = transform @ frame.robot_from_head_camera
    camera_origin = camera_transform[:3, 3]
    base_center = transform[:3, 3]
    raw = []
    mutation_count = 0
    ledger = hashlib.sha256()
    pairs: Counter[tuple[str, str]] = Counter()
    anchors = _anchors(traversal)
    for row, column in anchors:
        before = _sha256(valid.tobytes())
        candidate = _oracle_candidate_at(
            frame,
            depth,
            valid,
            camera_transform,
            camera_origin,
            base_center,
            acquisition_base_pose,
            ordinal,
            row,
            column,
        )
        after = _sha256(valid.tobytes())
        mutation_count += int(before != after)
        pairs[(before, after)] += 1
        ledger.update(
            json.dumps(
                [row, column, before, after],
                separators=(",", ":"),
            ).encode("ascii")
        )
        if candidate is not None:
            raw.append(candidate)
    return OracleFrame(
        tuple(raw),
        parent_sha256,
        len(anchors),
        mutation_count,
        ledger.hexdigest(),
        tuple((before, after, count) for (before, after), count in sorted(pairs.items())),
    )
def candidate_set_from_raw(
    input_hashes: Sequence[str],
    raw: Sequence[RawCandidate],
) -> CandidateSet:
    merged = target_selection._merge_candidates(raw)
    ordered = sorted(merged, key=lambda item: (
        -item.support_count, -item.view_count,
        -int(round(item.prominence * 1_000_000.0)),
        *target_selection._quantize(item.center, 1_000.0),
        item.first_frame, item.first_row, item.first_column,
    ))[:64]
    ordered = tuple(sorted(ordered, key=target_selection.Candidate.canonical_key))
    document = {
        "schema_version": CANDIDATE_SCHEMA_V2,
        "acquisition_input_sha256": list(input_hashes),
        "candidate_count": len(ordered),
        "candidates": [list(item.canonical_record()) for item in ordered],
    }
    canonical = json.dumps(document, ensure_ascii=True, separators=(",", ":"),
                           sort_keys=True).encode("ascii")
    return CandidateSet(tuple(input_hashes), ordered, canonical, _sha256(canonical))
def candidate_visible_bytes(value: PolicyVisibleInput) -> bytes:
    arrays = (
        value.head_rgb_uint8,
        value.head_depth_m,
        value.head_depth_valid,
        value.head_camera_intrinsics,
        value.robot_from_head_camera,
    )
    proprioception = np.asarray(value.proprioception, dtype="<f8")
    payloads = [np.ascontiguousarray(array).astype(
        np.uint8 if array is value.head_depth_valid else array.dtype,
        copy=False).tobytes() for array in arrays]
    selected = np.concatenate(
        (proprioception[:6], proprioception[12:18], proprioception[26:29])
    )
    return b"".join((b"hwr.p50-candidate-visible-input/v1\0", *payloads,
                     np.ascontiguousarray(selected, dtype="<f8").tobytes()))
def audit_legacy_source(source: str) -> dict[str, object]:
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name)
                           and target.id == "patch_valid"
                           for target in node.targets)
                   and isinstance(node.value, ast.Subscript)
                   and isinstance(node.value.value, ast.Name)
                   and node.value.value.id == "valid"]
    mutations = [node for node in ast.walk(tree)
                 if isinstance(node, ast.AugAssign)
                 and isinstance(node.target, ast.Name)
                 and node.target.id == "patch_valid"
                 and isinstance(node.op, ast.BitAnd)]
    passed = (
        len(assignments) == 1
        and len(mutations) == 1
        and assignments[0].lineno < mutations[0].lineno
    )
    return {"slice_assignment_count": len(assignments),
            "bitand_augmented_assignment_count": len(mutations),
            "assignment_precedes_mutation": passed, "passed": passed}
def audit_single_variable_source(
    current_source: str, legacy_source: str
) -> dict[str, bool]:
    current_frame = _function_node(current_source, "_frame_candidates_v2")
    legacy_frame = _function_node(legacy_source, "_frame_candidates")
    current_frame.name = legacy_frame.name
    ownership_count = _remove_ownership_copy(current_frame)
    current_generator = _function_node(current_source, "generate_candidate_set_v2")
    legacy_generator = _function_node(legacy_source, "generate_candidate_set")
    current_generator.name = legacy_generator.name
    for node in ast.walk(current_generator):
        if isinstance(node, ast.Name) and node.id == "_frame_candidates_v2":
            node.id = "_frame_candidates"
    schema_count = _normalize_schema(current_generator)
    frame_unchanged = _ast_equal(current_frame, legacy_frame)
    reduction_unchanged = _ast_equal(current_generator, legacy_generator)
    checks = {
        "one_ownership_copy_gate": ownership_count == 1,
        "frame_generator_otherwise_unchanged": frame_unchanged,
        "one_candidate_schema_change": schema_count == 1,
        "candidate_reduction_otherwise_unchanged": reduction_unchanged,
        "geometry_merge_ranking_selector_unchanged":
            frame_unchanged and reduction_unchanged,
    }
    return {**checks, "passed": all(checks.values())}
def boundary_fixture_audit() -> dict[str, object]:
    invalid = _fixture_input(valid=np.zeros((192, 256), dtype=np.bool_))
    flat = _fixture_input(timestamp=2, sequence=2)
    depth = np.ones((192, 256), dtype="<f4")
    valid = np.zeros((192, 256), dtype=np.bool_)
    row, column = 96, 200
    valid[row - 10 : row + 11, column - 10 : column + 11] = True
    depth[row - 2 : row + 3, column - 2 : column + 3] = 0.8
    single = _fixture_input(timestamp=3, sequence=3, depth=depth, valid=valid)
    nonfinite_depth = np.ones((192, 256), dtype="<f4")
    nonfinite_depth[row, column] = np.nan
    nonfinite = _fixture_input(
        timestamp=4, sequence=4, depth=nonfinite_depth
    )
    cases = {}
    for name, frame, expected in (
        ("all_invalid", invalid, 0),
        ("all_flat", flat, 0),
        ("single_acceptable_anchor", single, 1),
        ("nonfinite_depth", nonfinite, 0),
    ):
        production, production_mask = _production_frame(
            frame, (0.0, 0.0, 0.0), 0
        )
        oracle = {
            traversal: oracle_frame_candidates(
                frame, (0.0, 0.0, 0.0), 0, traversal
            )
            for traversal in TRAVERSALS
        }
        passed = (
            len(production) == expected
            and production_mask[0] == production_mask[1]
            and all(
                len(value.raw) == expected and value.mutation_count == 0
                for value in oracle.values()
            )
            and len({
                _raw_multiset_sha256(value.raw) for value in oracle.values()
            }) == 1
        )
        cases[name] = {
            "expected_raw_count": expected,
            "production_raw_count": len(production),
            "production_parent_mask_byte_identical":
                production_mask[0] == production_mask[1],
            "oracle_raw_counts": {
                traversal: len(value.raw)
                for traversal, value in oracle.items()
            },
            "passed": passed,
        }
    payload = target_selection.serialize_policy_input(single)
    candidate_set, duplicate = audit_episode(
        (payload, payload),
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=payload,
    )
    duplicate_passed = (
        duplicate["checks"]["passed"] is True
        and len(candidate_set.candidates) == 1
        and candidate_set.candidates[0].view_count == 2
    )
    cases["duplicate_observation_identity"] = {
        "candidate_count": len(candidate_set.candidates),
        "view_counts": [
            value.view_count for value in candidate_set.candidates
        ],
        "passed": duplicate_passed,
    }
    return {
        "cases": cases,
        "passed": all(value["passed"] for value in cases.values()),
    }
def overlap_fixture_audit() -> dict[str, object]:
    shape = (21, 21)
    conditions = {
        name: np.ones(shape, dtype=np.bool_)
        for name in ("top_left", "top_right", "bottom_left")
    }
    conditions["top_left"][:, 10:] = False
    slices = {
        "top_left": (slice(0, 21), slice(0, 21)),
        "top_right": (slice(0, 21), slice(10, 31)),
        "bottom_left": (slice(10, 31), slice(0, 21)),
    }
    orders = {
        "row_major": ("top_left", "top_right", "bottom_left"),
        "reverse_row_major": ("bottom_left", "top_right", "top_left"),
        "column_major": ("top_left", "bottom_left", "top_right"),
    }
    legacy = [
        _fixture_probe(name, order, slices, conditions, independent=False)
        for name, order in orders.items()
    ]
    corrected = [
        _fixture_probe(name, order, slices, conditions, independent=True)
        for name, order in orders.items()
    ]
    legacy_differs = len({
        _canonical_bytes(value["canonical_support"]) for value in legacy
    }) > 1
    corrected_equal = len({
        _canonical_bytes(value["canonical_support"]) for value in corrected
    }) == 1
    checks = {
        "overlapping_twenty_one_pixel_patches": True,
        "legacy_parent_mutation_positive": all(
            int(value["mutation_count"]) > 0 for value in legacy
        ),
        "legacy_traversal_support_differs": legacy_differs,
        "corrected_parent_mutation_zero": all(
            int(value["mutation_count"]) == 0 for value in corrected
        ),
        "corrected_traversal_support_equal": corrected_equal,
    }
    return {
        "orders": {name: list(value) for name, value in orders.items()},
        "legacy": legacy,
        "corrected": corrected,
        "checks": {**checks, "passed": all(checks.values())},
    }
def _oracle_candidate_at(
    frame,
    depth,
    valid,
    camera_transform,
    camera_origin,
    base_center,
    origin,
    ordinal,
    row,
    column,
) -> RawCandidate | None:
    center_depth = depth[row - 2 : row + 3, column - 2 : column + 3]
    center_valid = valid[row - 2 : row + 3, column - 2 : column + 3]
    ring_depth = depth[row - 10 : row + 11, column - 10 : column + 11]
    ring_valid = valid[row - 10 : row + 11, column - 10 : column + 11].copy()
    ring_valid[6:15, 6:15] = False
    if center_valid.sum() < 20 or ring_valid.sum() < 240:
        return None
    center_values = center_depth[center_valid].astype(np.float64)
    center_z = float(np.median(center_values))
    prominence = float(np.median(ring_depth[ring_valid])) - center_z
    if not 0.025 <= prominence <= 0.45:
        return None
    if float(
        np.quantile(center_values, 0.90)
        - np.quantile(center_values, 0.10)
    ) > 0.04:
        return None
    patch_depth = depth[row - 10 : row + 11, column - 10 : column + 11]
    valid_slice = valid[row - 10 : row + 11, column - 10 : column + 11]
    patch_valid = valid_slice & (
        np.abs(patch_depth - center_z) <= max(0.025, 0.015 * center_z)
    )
    rows, columns = np.nonzero(patch_valid)
    if len(rows) < 24:
        return None
    rows, columns = rows + row - 10, columns + column - 10
    points = target_selection._camera_points(
        rows,
        columns,
        patch_depth[patch_valid].astype(np.float64),
        frame.head_camera_intrinsics,
    )
    points = target_selection._transform_points(camera_transform, points)
    points = points[
        ~target_selection._robot_self_mask(points, frame, origin)
    ]
    if len(points) < 24:
        return None
    return target_selection._candidate_from_points(
        points,
        camera_origin,
        base_center,
        prominence,
        ordinal,
        row,
        column,
    )
def _anchors(traversal: str) -> tuple[tuple[int, int], ...]:
    row_major = tuple((row, column) for row in ROWS for column in COLUMNS)
    if traversal == "row_major":
        return row_major
    if traversal == "reverse_row_major":
        return tuple(reversed(row_major))
    if traversal == "column_major":
        return tuple((row, column) for column in COLUMNS for row in ROWS)
    raise ValueError("unknown candidate-mask traversal")
def _raw_record(candidate: RawCandidate) -> tuple[object, ...]:
    return (
        *(float(value).hex() for value in candidate.center),
        *(float(value).hex() for value in candidate.normal),
        float(candidate.width).hex(),
        float(candidate.prominence).hex(),
        candidate.support_count,
        candidate.frame_ordinal,
        candidate.row,
        candidate.column,
    )
def _raw_multiset(raw: Iterable[RawCandidate]) -> Counter[tuple[object, ...]]:
    return Counter(_raw_record(candidate) for candidate in raw)
def _raw_sequence_sha256(raw: Iterable[RawCandidate]) -> str:
    return _sha256(_canonical_bytes([_raw_record(value) for value in raw]))
def _raw_multiset_sha256(raw: Iterable[RawCandidate]) -> str:
    return _sha256(_canonical_bytes(sorted(_raw_multiset(raw).items())))
def _fixture_probe(traversal, order, slices, conditions, *, independent):
    parent = np.ones((31, 31), dtype=np.bool_)
    initial = _sha256(parent.tobytes())
    mutation_count = 0
    support = []
    for patch_name in order:
        before = _sha256(parent.tobytes())
        patch = parent[slices[patch_name]]
        if independent:
            patch = patch.copy()
        patch &= conditions[patch_name]
        after = _sha256(parent.tobytes())
        mutation_count += int(before != after)
        support.append([patch_name, int(patch.sum())])
    return {
        "traversal": traversal,
        "order": list(order),
        "support_sequence": support,
        "canonical_support": sorted(support),
        "parent_sha256_before": initial,
        "parent_sha256_after": _sha256(parent.tobytes()),
        "mutation_count": mutation_count,
    }
def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
def _fixture_input(
    *,
    timestamp: int = 1,
    sequence: int = 1,
    depth: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> PolicyVisibleInput:
    proprioception = np.zeros(37, dtype="<f8")
    proprioception[24:26] = 0.25
    return PolicyVisibleInput(
        timestamp,
        sequence,
        1,
        0,
        17,
        "ok",
        np.zeros((192, 256, 3), dtype=np.uint8),
        np.ones((192, 256), dtype="<f4") if depth is None else depth,
        np.ones((192, 256), dtype=np.bool_) if valid is None else valid,
        np.asarray((80.0, 80.0, 127.5, 95.5), dtype="<f8"),
        np.eye(4, dtype="<f8"),
        proprioception,
        np.zeros((4, 16), dtype="<f8"),
        np.asarray((False, False, False, True), dtype=np.bool_),
    )
def _function_node(source: str, name: str) -> ast.FunctionDef:
    values = [node for node in ast.parse(source).body
              if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(values) != 1: raise ValueError(f"candidate source function {name} differs")
    return copy.deepcopy(values[0])
def _remove_ownership_copy(function: ast.FunctionDef) -> int:
    matches = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and ast.unparse(node) == "patch_valid = patch_valid.copy()"
        ):
            matches.append(node)
    if len(matches) == 1:
        for node in ast.walk(function):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list) and matches[0] in value:
                    setattr(node, field, [item for item in value
                                          if item is not matches[0]])
    return len(matches)
def _normalize_schema(function: ast.FunctionDef) -> int:
    matches = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Dict):
            continue
        for index, key in enumerate(node.keys):
            if isinstance(key, ast.Constant) and key.value == "schema_version":
                matches.append((node, index))
    if len(matches) == 1:
        node, index = matches[0]
        node.values[index] = ast.Name(id="CANDIDATE_SCHEMA", ctx=ast.Load())
    return len(matches)
def _ast_equal(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)
def _sha256(content: bytes) -> str: return hashlib.sha256(content).hexdigest()
def aggregate_worker_memory(results) -> dict[str, object]:
    peaks = {}
    for result in results:
        pid, rss = int(result["worker_pid"]), int(result["worker_peak_rss_bytes"])
        peaks[pid] = max(peaks.get(pid, 0), rss)
    return {"worker_peak_rss_bytes_by_pid":
            {str(pid): peaks[pid] for pid in sorted(peaks)},
        "child_peak_rss_sum_bytes": sum(peaks.values())}
def bank_process_tree_memory(first, second, parent: int, pytest_peak: int):
    child = max(int(first["child_peak_rss_sum_bytes"]),
                int(second["child_peak_rss_sum_bytes"]))
    return {"parent_peak_rss_bytes": parent, "builds": [first, second],
            "maximum_bank_child_peak_rss_sum_bytes": child,
            "bank_process_tree_peak_rss_upper_bound_bytes": parent + child,
            "validation_pytest_child_peak_rss_bytes": pytest_peak}
def parse_pytest_receipt(output: bytes, returncode: int, allowed,
                         expected_passed: int, expected_skipped: int):
    lines = output.decode("utf-8", errors="replace").splitlines()
    event = lambda prefix: sorted(line[len(prefix):].split(" - ", 1)[0].strip()
                                  for line in lines if line.startswith(prefix))
    failed = event("FAILED "); errors = event("ERROR "); xpassed = event("XPASS ")
    summary = next((line for line in reversed(lines) if re.search(
        r"\d+ (?:failed|passed|skipped|xpassed|errors?)", line)), "")
    counts = {name: int(count) for count, name in re.findall(
        r"(\d+) (failed|passed|skipped|xpassed|errors?)", summary)}
    gate = (returncode == 1 and failed == sorted(allowed)
            and counts.get("failed") == len(allowed)
            and counts.get("passed") == expected_passed
            and counts.get("skipped", 0) == expected_skipped
            and counts.get("xpassed", 0) == 0
            and counts.get("error", 0) + counts.get("errors", 0) == 0
            and not errors and not xpassed)
    return {"returncode": returncode, "passed": counts.get("passed", 0),
            "skipped": counts.get("skipped", 0), "failed_ids": failed,
            "expected_passed": expected_passed, "expected_skipped": expected_skipped,
            "output_sha256": _sha256(output), "gate_passed": gate}
