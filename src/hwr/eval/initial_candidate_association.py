"""Pure candidate-to-entity association for frozen R0001-P68-E1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from hwr.eval import target_selection
from hwr.eval.target_selection import Candidate, CandidateSet, RawCandidate

PROPOSAL_ID = "R0001-P68-E1"
REPORT_SCHEMA = "hwr.p68-initial-candidate-association-report/v1"
COMPATIBLE_RATIO = 0.80
HIGH_COMPATIBLE_COUNT = 18
LOW_COMPATIBLE_COUNT = 6
MAX_MIXED_COUNT = 2
TASK_COMPATIBLE_COUNT = 5
TASK_MAX_MIXED_COUNT = 1
UNKNOWN_LABELS = frozenset(("background", "unknown", "unknown_site"))


@dataclass(frozen=True)
class RawSupport:
    raw: RawCandidate
    rows: tuple[int, ...]
    columns: tuple[int, ...]


@dataclass(frozen=True)
class CandidateSupport:
    candidate: Candidate
    raw_support: tuple[RawSupport, ...]


def reconstruct_candidate_support(
    keyframes: Sequence[bytes],
    *,
    acquisition_base_pose: Sequence[float],
    final_input: bytes,
) -> tuple[CandidateSet, tuple[CandidateSupport, ...]]:
    official = target_selection.generate_candidate_set(
        keyframes,
        acquisition_base_pose=acquisition_base_pose,
        final_input=final_input,
    )
    require_legacy_v1_candidate_set(official)
    origin = target_selection._pose(acquisition_base_pose)
    raw = tuple(
        support
        for ordinal, payload in enumerate(keyframes)
        for support in _frame_support(
            target_selection.deserialize_policy_input(payload),
            origin,
            ordinal,
        )
    )
    components = _components(raw)
    merged = []
    for component in components:
        candidates = target_selection._merge_candidates(
            [support.raw for support in component]
        )
        if len(candidates) == 1:
            merged.append(CandidateSupport(candidates[0], tuple(component)))
        elif candidates:
            raise ValueError("candidate component reconstruction split")
    ordered = sorted(
        merged,
        key=lambda item: (
            -item.candidate.support_count,
            -item.candidate.view_count,
            -int(round(item.candidate.prominence * 1_000_000.0)),
            *target_selection._quantize(item.candidate.center, 1_000.0),
            item.candidate.first_frame,
            item.candidate.first_row,
            item.candidate.first_column,
        ),
    )[:64]
    ordered = tuple(sorted(ordered, key=lambda item: item.candidate.canonical_key()))
    if tuple(
        item.candidate.canonical_record() for item in ordered
    ) != tuple(candidate.canonical_record() for candidate in official.candidates):
        raise ValueError("candidate support reconstruction differs")
    return official, tuple(
        CandidateSupport(candidate, item.raw_support)
        for candidate, item in zip(official.candidates, ordered, strict=True)
    )


def require_legacy_v1_candidate_set(candidate_set: CandidateSet) -> None:
    try:
        document = json.loads(candidate_set.canonical_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("P68 candidate document is not canonical JSON") from error
    if document.get("schema_version") != target_selection.CANDIDATE_SCHEMA:
        raise ValueError("P68 requires legacy-v1 candidate schema")
    if canonical_bytes(document) != candidate_set.canonical_bytes:
        raise ValueError("P68 candidate document bytes differ")
    if hashlib.sha256(candidate_set.canonical_bytes).hexdigest() != (
        candidate_set.candidate_set_sha256
    ):
        raise ValueError("P68 candidate document hash differs")


def associate_candidates(
    supports: Sequence[CandidateSupport],
    segmentations: Sequence[np.ndarray],
    table: Mapping[str, object],
    allowed_labels: frozenset[str],
) -> list[dict[str, object]]:
    records = []
    for index, support in enumerate(supports):
        labels: Counter[str] = Counter()
        roles: Counter[str] = Counter()
        for raw in support.raw_support:
            segmentation = np.asarray(segmentations[raw.raw.frame_ordinal])
            if segmentation.shape != (192, 256, 2):
                raise ValueError("segmentation shape differs")
            values = segmentation[
                np.asarray(raw.rows, dtype=np.intp),
                np.asarray(raw.columns, dtype=np.intp),
            ]
            for object_id, object_type in values:
                entity = _segmentation_entity(
                    table, int(object_id), int(object_type)
                )
                labels[str(entity["label"])] += 1
                roles[str(entity["role"])] += 1
        total = sum(labels.values())
        if total != support.candidate.support_count:
            raise ValueError("candidate support count differs")
        compatible = sum(labels[label] for label in allowed_labels)
        incompatible = sum(
            count
            for label, count in labels.items()
            if label not in allowed_labels and label not in UNKNOWN_LABELS
        )
        compatible_ratio = compatible / total if total else 0.0
        incompatible_ratio = incompatible / total if total else 0.0
        if compatible_ratio >= COMPATIBLE_RATIO:
            classification = "stage_compatible"
        elif incompatible_ratio >= COMPATIBLE_RATIO:
            classification = "stage_incompatible"
        else:
            classification = "mixed_or_unknown"
        records.append(
            {
                "candidate_index": index,
                "candidate": asdict(support.candidate),
                "classification": classification,
                "allowed_labels": sorted(allowed_labels),
                "total_support_count": total,
                "compatible_support_count": compatible,
                "incompatible_support_count": incompatible,
                "compatible_ratio": compatible_ratio,
                "incompatible_ratio": incompatible_ratio,
                "label_counts": dict(sorted(labels.items())),
                "role_counts": dict(sorted(roles.items())),
                "raw_component_count": len(support.raw_support),
            }
        )
    return records


def classify_episode(
    *,
    task_id: str,
    planned_episode_id: str,
    selected_index: int,
    candidate_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not candidate_records:
        classification = "no_relevant_final_candidate"
        subtype = "candidate_set_empty"
    elif not 0 <= selected_index < len(candidate_records):
        raise ValueError("selected candidate index differs")
    else:
        selected = candidate_records[selected_index]["classification"]
        compatible_exists = any(
            item["classification"] == "stage_compatible"
            for item in candidate_records
        )
        if selected == "stage_compatible":
            classification = "stage_compatible_selected"
            subtype = None
        elif selected == "stage_incompatible" and compatible_exists:
            classification = "relevant_exists_but_distractor_selected"
            subtype = None
        elif selected == "stage_incompatible":
            classification = "no_relevant_final_candidate"
            subtype = "all_final_candidates_incompatible"
        else:
            classification = "mixed_or_unknown"
            subtype = None
    return {
        "task_id": task_id,
        "planned_episode_id": planned_episode_id,
        "selected_index": selected_index,
        "candidate_count": len(candidate_records),
        "classification": classification,
        "subtype": subtype,
        "candidates": list(candidate_records),
    }


def analyze_episode_records(
    records: Sequence[Mapping[str, object]],
    task_order: Sequence[str],
) -> dict[str, object]:
    if len(records) != 24:
        return _invalid("association cohort must contain 24 Episodes")
    identifiers = [str(record["planned_episode_id"]) for record in records]
    if len(set(identifiers)) != len(identifiers):
        return _invalid("association Episode IDs must be unique")
    counts = Counter(str(record["classification"]) for record in records)
    task_counts = {}
    for task_id in task_order:
        task_records = [
            record for record in records if record["task_id"] == task_id
        ]
        if len(task_records) != 8:
            return _invalid("each task must contain eight Episodes")
        task_counts[task_id] = dict(
            Counter(str(record["classification"]) for record in task_records)
        )
    compatible = counts["stage_compatible_selected"]
    mixed = counts["mixed_or_unknown"]
    high = (
        compatible >= HIGH_COMPATIBLE_COUNT
        and all(
            task_counts[task].get("stage_compatible_selected", 0)
            >= TASK_COMPATIBLE_COUNT
            for task in task_order
        )
        and mixed <= MAX_MIXED_COUNT
        and all(
            task_counts[task].get("mixed_or_unknown", 0)
            <= TASK_MAX_MIXED_COUNT
            for task in task_order
        )
    )
    if high:
        decision = "accepted as initial-association stopping-gate evidence"
    elif compatible <= LOW_COMPATIBLE_COUNT:
        decision = "accepted as selector-relevance stopping evidence"
    else:
        decision = "inconclusive"
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "decision": decision,
        "validation_error": None,
        "sample_unit": "Episode",
        "episode_count": len(records),
        "counts": dict(sorted(counts.items())),
        "task_counts": task_counts,
        "thresholds": {
            "candidate_compatible_ratio": COMPATIBLE_RATIO,
            "high_compatible_count": HIGH_COMPATIBLE_COUNT,
            "low_compatible_count": LOW_COMPATIBLE_COUNT,
            "maximum_mixed_count": MAX_MIXED_COUNT,
            "task_compatible_count": TASK_COMPATIBLE_COUNT,
            "task_maximum_mixed_count": TASK_MAX_MIXED_COUNT,
        },
        "checks": {
            "twenty_four_unique_episodes": True,
            "eight_episodes_per_task": True,
            "passed": True,
        },
        **_claim_flags(),
    }


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _segmentation_entity(
    table: Mapping[str, object],
    object_id: int,
    object_type: int,
) -> Mapping[str, object]:
    if object_id == -1 and object_type == -1:
        return table["background"]
    collections = {5: "geoms", 6: "sites"}
    name = collections.get(object_type)
    values = table.get(name, ()) if name is not None else ()
    if isinstance(values, Sequence) and 0 <= object_id < len(values):
        return values[object_id]
    return {"label": "unknown", "role": "unknown", "instance": None}


def _frame_support(frame, origin, ordinal: int) -> list[RawSupport]:
    depth = frame.head_depth_m
    valid = (
        frame.head_depth_valid
        & np.isfinite(depth)
        & (depth >= 0.10)
        & (depth <= 5.00)
    )
    transform = target_selection._acquisition_from_robot(
        origin, frame.base_pose
    )
    camera_transform = transform @ frame.robot_from_head_camera
    camera_origin = camera_transform[:3, 3]
    base_center = transform[:3, 3]
    result = []
    for row in range(12, 180, 4):
        for column in range(12, 244, 4):
            value = _raw_support_at(
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
            )
            if value is not None:
                result.append(value)
    return result


def _raw_support_at(
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
) -> RawSupport | None:
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
    patch_valid = valid[row - 10 : row + 11, column - 10 : column + 11]
    patch_valid &= (
        np.abs(patch_depth - center_z) <= max(0.025, 0.015 * center_z)
    )
    rows, columns = np.nonzero(patch_valid)
    if len(rows) < 24:
        return None
    rows = rows + row - 10
    columns = columns + column - 10
    points = target_selection._camera_points(
        rows,
        columns,
        patch_depth[patch_valid].astype(np.float64),
        frame.head_camera_intrinsics,
    )
    points = target_selection._transform_points(camera_transform, points)
    keep = ~target_selection._robot_self_mask(points, frame, origin)
    points, rows, columns = points[keep], rows[keep], columns[keep]
    if len(points) < 24:
        return None
    candidate = target_selection._candidate_from_points(
        points,
        camera_origin,
        base_center,
        prominence,
        ordinal,
        row,
        column,
    )
    if candidate is None:
        return None
    return RawSupport(
        candidate,
        tuple(int(value) for value in rows),
        tuple(int(value) for value in columns),
    )


def _components(raw: Sequence[RawSupport]) -> list[list[RawSupport]]:
    adjacency = [set() for _ in raw]
    for left in range(len(raw)):
        for right in range(left):
            left_candidate, right_candidate = raw[left].raw, raw[right].raw
            distance = np.linalg.norm(
                np.asarray(left_candidate.center)
                - np.asarray(right_candidate.center)
            )
            cosine = float(
                np.dot(left_candidate.normal, right_candidate.normal)
            )
            if (
                distance <= 0.08
                and cosine >= 0.80
                and abs(left_candidate.width - right_candidate.width) <= 0.10
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    seen: set[int] = set()
    components = []
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
        components.append(component)
    return components


def _invalid(reason: str) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "decision": "invalid",
        "validation_error": reason,
        "checks": {"passed": False},
        **_claim_flags(),
    }


def _claim_flags() -> dict[str, bool]:
    return {
        "training_executed": False,
        "policy_inference_executed": False,
        "capability_claim_allowed": False,
        "task_success_claim_allowed": False,
        "generalization_claim_allowed": False,
        "hardware_safety_claim_allowed": False,
        "entity_recognition_claim_allowed": False,
        "reachability_claim_allowed": False,
    }
