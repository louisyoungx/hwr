"""Exact report-only funnel instrumentation for frozen R0001-P50-E2."""

from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import textwrap
from collections import Counter
from dataclasses import replace
from types import FrameType
from typing import Callable, Mapping, Sequence

import numpy as np

from hwr.eval import target_selection
from hwr.eval.target_selection import (
    Candidate,
    CandidateSet,
    PolicyVisibleInput,
    RawCandidate,
    candidate_scores,
    deserialize_policy_input,
    generate_candidate_set,
    select_candidate_index,
)

FUNNEL_SCHEMA = "hwr.p50-candidate-funnel/v1"
CANDIDATE_VISIBLE_SCHEMA = "hwr.p50-candidate-visible-input/v1"
ANCHOR_REJECTION_STAGES = (
    "center_ring_validity",
    "prominence",
    "center_depth_spread",
    "patch_support_before_self_mask",
    "support_after_self_mask",
    "height_range",
    "planarity",
    "width",
)
ANCHOR_TERMINAL_STAGES = (*ANCHOR_REJECTION_STAGES, "raw_candidate_accepted")
COMPONENT_TERMINAL_STAGES = (
    "view_count_lt_2_rejection",
    "aggregate_normal_zero_rejection",
    "pre_top64_candidate",
)


class CandidateFunnelContractError(ValueError):
    """Raised when exact formal-gate instrumentation cannot be proven."""


def candidate_visible_bytes(value: PolicyVisibleInput) -> bytes:
    """Serialize exactly the fields consumed by the formal candidate generator."""
    arrays = (
        value.head_rgb_uint8,
        value.head_depth_m,
        value.head_depth_valid,
        value.head_camera_intrinsics,
        value.robot_from_head_camera,
    )
    proprioception = np.asarray(value.proprioception, dtype="<f8")
    candidate_proprioception = np.concatenate(
        (proprioception[:6], proprioception[12:18], proprioception[26:29])
    )
    payloads = [
        np.ascontiguousarray(array).astype(
            np.uint8 if array is value.head_depth_valid else array.dtype,
            copy=False,
        ).tobytes()
        for array in arrays
    ]
    return b"".join(
        (
            CANDIDATE_VISIBLE_SCHEMA.encode("ascii"),
            b"\0",
            *payloads,
            np.ascontiguousarray(candidate_proprioception, dtype="<f8").tobytes(),
        )
    )


def candidate_gate_source_identity() -> dict[str, object]:
    functions = {
        name: getattr(target_selection, name)
        for name in (
            "_frame_candidates",
            "_candidate_from_points",
            "_merge_candidates",
            "generate_candidate_set",
        )
    }
    return {
        "functions": {
            name: _identity(inspect.getsource(function).encode())
            for name, function in functions.items()
        },
        "anchor_rejection_stage_count": len(_anchor_line_contract()),
        "component_terminal_stage_count": len(COMPONENT_TERMINAL_STAGES),
        "ranking_assignment_count": len(_ranking_line_contract()),
        "instrumentation": "python-line-trace-over-formal-generator-functions",
        "thresholds_or_gate_math_duplicated": False,
    }


def analyze_frame_funnel(
    payload: bytes,
    acquisition_base_pose: tuple[float, float, float],
    frame_ordinal: int,
) -> tuple[tuple[RawCandidate, ...], dict[str, object]]:
    frame = deserialize_policy_input(payload)
    raw, terminal_by_anchor = _trace_frame_candidates(
        frame, acquisition_base_pose, frame_ordinal
    )
    expected = len(range(12, 180, 4)) * len(range(12, 244, 4))
    counts = Counter(terminal_by_anchor.values())
    if len(terminal_by_anchor) != expected or set(counts) - set(
        ANCHOR_TERMINAL_STAGES
    ):
        raise CandidateFunnelContractError(
            "formal frame trace did not classify every enumerated anchor"
        )
    rejections = {
        stage: int(counts.get(stage, 0)) for stage in ANCHOR_REJECTION_STAGES
    }
    report = {
        "frame_ordinal": frame_ordinal,
        "observation_identity": [
            frame.observation_timestamp_ns,
            frame.sequence_id,
        ],
        "candidate_visible_sha256": _sha256(candidate_visible_bytes(frame)),
        "enumerated_anchor_count": expected,
        "first_rejection_counts": rejections,
        "stages": _ordered_stage_rows(expected, rejections),
        "raw_candidate_count": len(raw),
        "conservation": {
            "left": expected,
            "right": sum(rejections.values()) + len(raw),
            "passed": expected == sum(rejections.values()) + len(raw),
        },
    }
    return raw, report


def classify_frame_anchor(
    payload: bytes,
    acquisition_base_pose: tuple[float, float, float],
    *,
    frame_ordinal: int,
    row: int,
    column: int,
) -> str:
    frame = deserialize_policy_input(payload)
    _, terminal_by_anchor = _trace_frame_candidates(
        frame, acquisition_base_pose, frame_ordinal
    )
    try:
        return terminal_by_anchor[(frame_ordinal, row, column)]
    except KeyError as error:
        raise CandidateFunnelContractError("requested anchor was not enumerated") from error


def _trace_frame_candidates(
    frame: PolicyVisibleInput,
    acquisition_base_pose: tuple[float, float, float],
    frame_ordinal: int,
) -> tuple[tuple[RawCandidate, ...], dict[tuple[int, int, int], str]]:
    line_stages = _anchor_line_contract()
    terminal_by_anchor: dict[tuple[int, int, int], str] = {}

    def trace(current: FrameType, event: str, argument: object):
        del argument
        if event != "line":
            return trace
        if current.f_code not in (
            target_selection._frame_candidates.__code__,
            target_selection._candidate_from_points.__code__,
        ):
            return trace
        stage = line_stages.get((current.f_code.co_name, current.f_lineno))
        if stage is not None:
            key = (
                int(current.f_locals["ordinal"]),
                int(current.f_locals["row"]),
                int(current.f_locals["column"]),
            )
            if key in terminal_by_anchor:
                raise CandidateFunnelContractError(
                    "anchor entered more than one terminal stage"
                )
            terminal_by_anchor[key] = stage
        return trace

    raw = tuple(_traced_call(trace, target_selection._frame_candidates,
                             frame, acquisition_base_pose, frame_ordinal))
    for candidate in raw:
        key = (candidate.frame_ordinal, candidate.row, candidate.column)
        if key in terminal_by_anchor:
            raise CandidateFunnelContractError(
                "accepted anchor already entered a rejection stage"
            )
        terminal_by_anchor[key] = "raw_candidate_accepted"
    return raw, terminal_by_anchor


def classify_candidate_points(
    points: np.ndarray,
    camera: np.ndarray,
    base: np.ndarray,
) -> str:
    line_stages = _anchor_line_contract()
    observed: list[str] = []

    def trace(current: FrameType, event: str, argument: object):
        del argument
        if (
            event == "line"
            and current.f_code is target_selection._candidate_from_points.__code__
        ):
            stage = line_stages.get((current.f_code.co_name, current.f_lineno))
            if stage is not None:
                observed.append(stage)
        return trace

    candidate = _traced_call(
        trace,
        target_selection._candidate_from_points,
        np.asarray(points, np.float64),
        np.asarray(camera, np.float64),
        np.asarray(base, np.float64),
        0.1,
        0,
        20,
        20,
    )
    if candidate is not None:
        if observed:
            raise CandidateFunnelContractError(
                "accepted point fixture entered a rejection stage"
            )
        return "raw_candidate_accepted"
    if len(observed) != 1:
        raise CandidateFunnelContractError(
            "formal point fixture did not enter exactly one rejection stage"
        )
    return observed[0]


def analyze_components(
    raw: Sequence[RawCandidate],
    unique_identity_by_frame: Mapping[int, int],
) -> dict[str, object]:
    if set(item.frame_ordinal for item in raw) - set(unique_identity_by_frame):
        raise CandidateFunnelContractError("raw candidate frame identity is missing")
    ordinal = _trace_components(tuple(raw))
    shadow_raw = tuple(
        replace(
            item,
            frame_ordinal=int(unique_identity_by_frame[item.frame_ordinal]),
        )
        for item in raw
    )
    shadow = _trace_components(shadow_raw)
    ordinal_views = ordinal.pop("_component_view_counts")
    shadow_views = shadow.pop("_component_view_counts")
    if len(ordinal_views) != len(shadow_views):
        raise CandidateFunnelContractError("shadow component partition differs")
    view_monotonic = all(
        shadow_count <= ordinal_count
        for ordinal_count, shadow_count in zip(
            ordinal_views, shadow_views, strict=True
        )
    )
    candidate_monotonic = (
        shadow["pre_top64_candidate_count"]
        <= ordinal["pre_top64_candidate_count"]
    )
    return {
        "connected_component_built_count": ordinal["component_count"],
        "ordinal": ordinal,
        "shadow": shadow,
        "ordinal_retained_candidate_count": min(
            ordinal["pre_top64_candidate_count"], 64
        ),
        "shadow_retained_candidate_count": min(
            shadow["pre_top64_candidate_count"], 64
        ),
        "view_count_monotonic": view_monotonic,
        "shadow_candidate_monotonic": candidate_monotonic,
    }


def analyze_candidate_funnel(
    keyframes: Sequence[bytes],
    *,
    acquisition_base_pose: Sequence[float],
    final_input: bytes,
    expected_candidate_bytes: bytes | None,
    expected_selected_index: int,
    expected_score_sha256: str | None = None,
    selection_permitted: bool = True,
) -> dict[str, object]:
    if len(tuple(acquisition_base_pose)) != 3:
        raise CandidateFunnelContractError("acquisition pose must have three values")
    frames = [deserialize_policy_input(payload) for payload in keyframes]
    identities: dict[tuple[int, int], tuple[str, int]] = {}
    frame_to_unique: dict[int, int] = {}
    for ordinal, frame in enumerate(frames):
        identity = (frame.observation_timestamp_ns, frame.sequence_id)
        visible_sha256 = _sha256(candidate_visible_bytes(frame))
        if identity in identities and identities[identity][0] != visible_sha256:
            raise CandidateFunnelContractError(
                "observation identity changed candidate-visible payload"
            )
        identities.setdefault(identity, (visible_sha256, len(identities)))
        frame_to_unique[ordinal] = identities[identity][1]
    raw: list[RawCandidate] = []
    frame_reports = []
    for ordinal, payload in enumerate(keyframes):
        candidates, report = analyze_frame_funnel(
            payload, tuple(float(value) for value in acquisition_base_pose), ordinal
        )
        raw.extend(candidates)
        frame_reports.append(report)
    component = analyze_components(raw, frame_to_unique)
    formal, ranking = _trace_ranking(
        keyframes,
        tuple(float(value) for value in acquisition_base_pose),
        final_input,
    )
    formal_bytes_equal = (
        expected_candidate_bytes is not None
        and formal.canonical_bytes == expected_candidate_bytes
    )
    final_value = deserialize_policy_input(final_input)
    selected = (
        select_candidate_index(
            formal,
            final_value.base_pose,
            acquisition_base_pose=acquisition_base_pose,
        )
        if selection_permitted
        else -1
    )
    scores = candidate_scores(
        formal,
        final_value.base_pose,
        acquisition_base_pose=acquisition_base_pose,
    )
    score_sha256 = _score_sha256(scores)
    anchor = _aggregate_anchor_reports(frame_reports)
    checks = {
        "anchor_conservation": bool(anchor["conservation"]["passed"]),
        "component_conservation": bool(
            component["ordinal"]["conservation"]["passed"]
            and component["shadow"]["conservation"]["passed"]
        ),
        "ranking_conservation": bool(ranking["conservation"]["passed"]),
        "formal_candidate_bytes": formal_bytes_equal,
        "selected_index": selected == expected_selected_index,
        "candidate_scores": (
            expected_score_sha256 is None
            or score_sha256 == expected_score_sha256
        ),
        "identity_consistency": True,
        "view_count_monotonic": bool(component["view_count_monotonic"]),
        "shadow_candidate_monotonic": bool(
            component["shadow_candidate_monotonic"]
        ),
    }
    return {
        "schema_version": FUNNEL_SCHEMA,
        "capture_count": len(keyframes),
        "unique_observation_shadow": {
            "unique_observation_count": len(identities),
            "unique_payload_count": len(
                {visible for visible, _ in identities.values()}
            ),
            "duplicate_observation_count": len(keyframes) - len(identities),
            "view_count_monotonic": component["view_count_monotonic"],
            "shadow_candidate_monotonic": component[
                "shadow_candidate_monotonic"
            ],
        },
        "frames": frame_reports,
        "anchor_ledger": anchor,
        "component_ledger": component,
        "ranking_ledger": ranking,
        "formal_candidate": {
            "candidate_count": len(formal.candidates),
            "canonical_sha256": formal.candidate_set_sha256,
            "canonical_byte_count": len(formal.canonical_bytes),
            "canonical_bytes_bit_identical": formal_bytes_equal,
            "selected_index": selected,
            "selected_index_bit_identical": selected == expected_selected_index,
            "score_bytes_sha256": score_sha256,
        },
        "last_nonempty_stage": _last_nonempty_stage(
            len(raw),
            component["ordinal"]["component_count"],
            component["ordinal"]["pre_top64_candidate_count"],
            ranking["retained_candidate_count"],
        ),
        "checks": {**checks, "passed": all(checks.values())},
    }


def _trace_components(raw: tuple[RawCandidate, ...]) -> dict[str, object]:
    contract = _component_line_contract()
    terminal_counts = Counter()
    component_views: list[int] = []

    def trace(current: FrameType, event: str, argument: object):
        del argument
        if (
            event != "line"
            or current.f_code is not target_selection._merge_candidates.__code__
        ):
            return trace
        line = current.f_lineno
        if line == contract["component_built"]:
            component_views.append(
                len({item.frame_ordinal for item in current.f_locals["component"]})
            )
        stage = contract["rejections"].get(line)
        if stage is not None:
            terminal_counts[stage] += 1
        return trace

    candidates = tuple(
        _traced_call(trace, target_selection._merge_candidates, raw)
    )
    component_count = len(component_views)
    accepted_count = len(candidates)
    report = {
        "component_count": component_count,
        "view_count_lt_2_rejection_count": int(
            terminal_counts["view_count_lt_2_rejection"]
        ),
        "aggregate_normal_zero_rejection_count": int(
            terminal_counts["aggregate_normal_zero_rejection"]
        ),
        "pre_top64_candidate_count": len(candidates),
        "stages": _ordered_stage_rows(
            component_count,
            {
                "view_count_lt_2_rejection": int(
                    terminal_counts["view_count_lt_2_rejection"]
                ),
                "aggregate_normal_zero_rejection": int(
                    terminal_counts["aggregate_normal_zero_rejection"]
                ),
            },
        ),
        "conservation": {
            "left": component_count,
            "right": sum(terminal_counts.values()) + accepted_count,
            "passed": component_count == sum(terminal_counts.values()) + accepted_count,
        },
        "_component_view_counts": component_views,
    }
    return report


def _trace_ranking(
    keyframes: Sequence[bytes],
    acquisition_base_pose: tuple[float, float, float],
    final_input: bytes,
) -> tuple[CandidateSet, dict[str, object]]:
    assignment_lines = _ranking_line_contract()
    snapshots: dict[int, tuple[tuple[int, ...], ...]] = {}

    def trace(current: FrameType, event: str, argument: object):
        del argument
        if (
            event == "line"
            and current.f_code is generate_candidate_set.__code__
            and current.f_lineno in assignment_lines
            and "ordered" in current.f_locals
        ):
            snapshots[current.f_lineno] = tuple(
                item.canonical_record() for item in current.f_locals["ordered"]
            )
        return trace

    formal = _traced_call(
        trace,
        generate_candidate_set,
        keyframes,
        acquisition_base_pose=acquisition_base_pose,
        final_input=final_input,
    )
    first_line, second_line = assignment_lines
    retained_ranked = snapshots.get(second_line)
    if retained_ranked is None:
        raise CandidateFunnelContractError(
            "formal ranking trace did not capture retained candidates"
        )
    canonical = tuple(item.canonical_record() for item in formal.candidates)
    if sorted(retained_ranked) != sorted(canonical):
        raise CandidateFunnelContractError(
            "canonical reorder changed candidate membership"
        )
    pre_count = _pre_top64_count_from_frame_trace(
        keyframes, acquisition_base_pose
    )
    retained = len(canonical)
    return formal, {
        "pre_top64_candidate_count": pre_count,
        "retained_candidate_count": retained,
        "truncated_candidate_count": pre_count - retained,
        "stages": [ranking_ledger(pre_count, retained)],
        "canonical_reorder_membership_unchanged": True,
        "formal_assignment_lines": [first_line, second_line],
        "conservation": {
            "left": pre_count,
            "right": retained + (pre_count - retained),
            "passed": pre_count == retained + (pre_count - retained),
        },
    }


def ranking_ledger(
    pre_top64_candidate_count: int,
    retained_candidate_count: int,
) -> dict[str, object]:
    if not (
        0 <= retained_candidate_count <= min(pre_top64_candidate_count, 64)
    ):
        raise CandidateFunnelContractError("ranking counts differ from top-64 gate")
    return {
        "stage": "top64",
        "input_count": pre_top64_candidate_count,
        "rejection_count": pre_top64_candidate_count - retained_candidate_count,
        "survival_count": retained_candidate_count,
    }


def _pre_top64_count_from_frame_trace(
    keyframes: Sequence[bytes],
    acquisition_base_pose: tuple[float, float, float],
) -> int:
    raw = []
    for ordinal, payload in enumerate(keyframes):
        raw.extend(
            target_selection._frame_candidates(
                deserialize_policy_input(payload),
                acquisition_base_pose,
                ordinal,
            )
        )
    return len(target_selection._merge_candidates(raw))


def _aggregate_anchor_reports(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    enumerated = sum(int(report["enumerated_anchor_count"]) for report in reports)
    raw = sum(int(report["raw_candidate_count"]) for report in reports)
    rejections = {
        stage: sum(
            int(report["first_rejection_counts"][stage]) for report in reports
        )
        for stage in ANCHOR_REJECTION_STAGES
    }
    return {
        "enumerated_anchor_count": enumerated,
        "first_rejection_counts": rejections,
        "stages": _ordered_stage_rows(enumerated, rejections),
        "raw_candidate_count": raw,
        "conservation": {
            "left": enumerated,
            "right": sum(rejections.values()) + raw,
            "passed": enumerated == sum(rejections.values()) + raw,
        },
    }


def _ordered_stage_rows(
    initial_count: int,
    rejection_counts: Mapping[str, int],
) -> list[dict[str, object]]:
    remaining = initial_count
    stages = []
    for stage, rejected in rejection_counts.items():
        rejected = int(rejected)
        if rejected < 0 or rejected > remaining:
            raise CandidateFunnelContractError("stage count cannot conserve")
        stages.append(
            {
                "stage": stage,
                "input_count": remaining,
                "rejection_count": rejected,
                "survival_count": remaining - rejected,
            }
        )
        remaining -= rejected
    return stages


def _anchor_line_contract() -> dict[tuple[str, int], str]:
    frame_continues = _node_lines(
        target_selection._frame_candidates, ast.Continue
    )
    point_returns = _none_return_lines(
        target_selection._candidate_from_points
    )
    if len(frame_continues) != 5 or len(point_returns) != 3:
        raise CandidateFunnelContractError(
            "formal anchor gate source structure differs"
        )
    lines = [
        *(("_frame_candidates", line) for line in frame_continues),
        *(("_candidate_from_points", line) for line in point_returns),
    ]
    return dict(zip(lines, ANCHOR_REJECTION_STAGES, strict=True))


def _component_line_contract() -> dict[str, object]:
    source, start, tree = _function_tree(target_selection._merge_candidates)
    del source
    view_assignment = [
        start + node.lineno - 1
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "views"
            for target in node.targets
        )
    ]
    terminal_lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Continue)
        ):
            expression = ast.unparse(node.test)
            if "views" in expression or "normal" in expression:
                terminal_lines.append(start + node.body[0].lineno - 1)
    if (
        len(view_assignment) != 1
        or len(terminal_lines) != 2
    ):
        raise CandidateFunnelContractError(
            "formal component gate source structure differs"
        )
    rejections = {
        terminal_lines[0]: "view_count_lt_2_rejection",
        terminal_lines[1]: "aggregate_normal_zero_rejection",
    }
    return {"component_built": view_assignment[0], "rejections": rejections}


def _ranking_line_contract() -> tuple[int, int]:
    source, start, tree = _function_tree(generate_candidate_set)
    del source
    lines = sorted(
        start + node.lineno - 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "ordered"
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        )
    )
    if len(lines) != 2:
        raise CandidateFunnelContractError(
            "formal ranking source structure differs"
        )
    return tuple(lines)


def _node_lines(function: Callable[..., object], kind: type[ast.AST]) -> list[int]:
    _, start, tree = _function_tree(function)
    return sorted(
        start + node.lineno - 1
        for node in ast.walk(tree)
        if isinstance(node, kind)
    )


def _none_return_lines(function: Callable[..., object]) -> list[int]:
    _, start, tree = _function_tree(function)
    return sorted(
        start + node.lineno - 1
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    )


def _function_tree(
    function: Callable[..., object],
) -> tuple[str, int, ast.Module]:
    lines, start = inspect.getsourcelines(function)
    source = textwrap.dedent("".join(lines))
    return source, start, ast.parse(source)


def _traced_call(trace, function, *args, **kwargs):
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        return function(*args, **kwargs)
    finally:
        sys.settrace(previous)


def _last_nonempty_stage(
    raw_count: int,
    component_count: int,
    pre_top64_count: int,
    retained_count: int,
) -> str:
    if retained_count:
        return "retained_candidate"
    if pre_top64_count:
        return "pre_top64_candidate"
    if component_count:
        return "connected_component"
    if raw_count:
        return "raw_candidate"
    return "enumerated_anchor"


def _score_sha256(scores: Sequence[float]) -> str:
    return _sha256(np.ascontiguousarray(scores, dtype="<f8").tobytes())


def _identity(content: bytes) -> dict[str, object]:
    return {"sha256": _sha256(content), "bytes": len(content)}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
