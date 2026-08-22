from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hwr.apps import aggregate_candidate_funnels
from hwr.eval.candidate_funnel import (
    ANCHOR_REJECTION_STAGES,
    CandidateFunnelContractError,
    analyze_candidate_funnel,
    analyze_components,
    analyze_frame_funnel,
    candidate_gate_source_identity,
    classify_frame_anchor,
    classify_candidate_points,
    ranking_ledger,
)
from hwr.eval.target_selection import (
    PolicyVisibleInput,
    RawCandidate,
    generate_candidate_set,
    serialize_policy_input,
)


def _input(
    *,
    timestamp: int = 1,
    sequence: int = 1,
    phase: int = 1,
    depth: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> PolicyVisibleInput:
    return PolicyVisibleInput(
        observation_timestamp_ns=timestamp,
        sequence_id=sequence,
        phase_index=phase,
        phase_step=0,
        policy_rng_seed=9,
        safety_state="ok",
        head_rgb_uint8=np.zeros((192, 256, 3), dtype=np.uint8),
        head_depth_m=(
            np.ones((192, 256), dtype="<f4") if depth is None else depth
        ),
        head_depth_valid=(
            np.ones((192, 256), dtype=np.bool_) if valid is None else valid
        ),
        head_camera_intrinsics=np.asarray(
            (200.0, 200.0, 127.5, 95.5), dtype="<f8"
        ),
        robot_from_head_camera=np.eye(4, dtype="<f8"),
        proprioception=np.zeros(37, dtype="<f8"),
        executed_action_history=np.zeros((4, 16), dtype="<f8"),
        history_available=np.zeros(4, dtype=np.bool_),
    )


def _raw(
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
    frame: int,
    row: int = 20,
) -> RawCandidate:
    return RawCandidate(center, normal, 0.1, 0.1, 30, frame, row, 20)


def test_gate_source_contract_is_derived_from_formal_functions() -> None:
    identity = candidate_gate_source_identity()

    assert set(identity["functions"]) == {
        "_frame_candidates",
        "_candidate_from_points",
        "_merge_candidates",
        "generate_candidate_set",
    }
    assert all(
        len(value["sha256"]) == 64 and value["bytes"] > 0
        for value in identity["functions"].values()
    )
    assert identity["anchor_rejection_stage_count"] == 8
    assert identity["component_terminal_stage_count"] == 3


def test_frame_ledger_uses_first_rejection_and_conserves_all_anchors() -> None:
    invalid = np.zeros((192, 256), dtype=np.bool_)
    payload = serialize_policy_input(_input(valid=invalid))

    _, report = analyze_frame_funnel(payload, (0.0, 0.0, 0.0), 0)

    assert report["enumerated_anchor_count"] == 42 * 58
    assert report["first_rejection_counts"]["center_ring_validity"] == 42 * 58
    assert report["raw_candidate_count"] == 0
    assert report["conservation"]["passed"] is True
    assert sum(report["first_rejection_counts"].values()) == 42 * 58
    assert report["stages"][0] == {
        "stage": "center_ring_validity",
        "input_count": 42 * 58,
        "rejection_count": 42 * 58,
        "survival_count": 0,
    }


def test_prominence_is_the_first_rejection_for_flat_valid_depth() -> None:
    payload = serialize_policy_input(_input())

    _, report = analyze_frame_funnel(payload, (0.0, 0.0, 0.0), 0)

    assert report["first_rejection_counts"]["prominence"] == 42 * 58
    assert report["first_rejection_counts"]["center_depth_spread"] == 0
    assert report["raw_candidate_count"] == 0


def test_remaining_pixel_gates_are_exact_first_rejections() -> None:
    row, column = 96, 128
    spread = np.full((192, 256), 1.2, dtype="<f4")
    spread[row - 2 : row + 3, column - 2 : column + 3] = np.linspace(
        0.96, 1.04, 25, dtype=np.float32
    ).reshape(5, 5)
    assert classify_frame_anchor(
        serialize_policy_input(_input(depth=spread)),
        (0.0, 0.0, 0.0),
        frame_ordinal=0,
        row=row,
        column=column,
    ) == "center_depth_spread"

    support = np.full((192, 256), 1.2, dtype="<f4")
    valid = np.ones((192, 256), dtype=np.bool_)
    support[row - 2 : row + 3, column - 2 : column + 3] = 1.0
    valid[row - 2, column - 2 : column + 3] = False
    assert classify_frame_anchor(
        serialize_policy_input(_input(depth=support, valid=valid)),
        (0.0, 0.0, 0.0),
        frame_ordinal=0,
        row=row,
        column=column,
    ) == "patch_support_before_self_mask"

    self_masked = np.full((192, 256), 0.30, dtype="<f4")
    self_masked[row - 2 : row + 3, column - 2 : column + 3] = 0.20
    assert classify_frame_anchor(
        serialize_policy_input(_input(depth=self_masked)),
        (0.0, 0.0, 0.0),
        frame_ordinal=0,
        row=row,
        column=column,
    ) == "support_after_self_mask"


def test_candidate_point_gates_hit_height_planarity_width_and_acceptance() -> None:
    rows, columns = np.indices((6, 6), dtype=np.float64)
    accepted = np.column_stack(
        (
            np.full(rows.size, 0.50),
            (columns.ravel() - 2.5) * 0.01,
            0.70 + (rows.ravel() - 2.5) * 0.01,
        )
    )
    too_near = accepted.copy()
    too_near[:, 0] = 0.10
    cube = np.asarray(
        [
            (0.45 + x, y, 0.65 + z)
            for x in (-0.05, 0.05)
            for y in (-0.05, 0.05)
            for z in (-0.05, 0.05)
        ],
        dtype=np.float64,
    )
    narrow = accepted.copy()
    narrow[:, 1:] *= 0.1

    assert classify_candidate_points(
        too_near, np.asarray((0.0, 0.0, 0.7)), np.zeros(3)
    ) == "height_range"
    assert classify_candidate_points(
        cube, np.asarray((0.0, 0.0, 0.7)), np.zeros(3)
    ) == "planarity"
    assert classify_candidate_points(
        narrow, np.asarray((0.0, 0.0, 0.7)), np.zeros(3)
    ) == "width"
    assert classify_candidate_points(
        accepted, np.asarray((0.0, 0.0, 0.7)), np.zeros(3)
    ) == "raw_candidate_accepted"


def test_component_ledger_tracks_first_terminal_stage_and_conserves() -> None:
    raw = (
        _raw((0.0, 0.0, 0.5), (1.0, 0.0, 0.0), 0),
        _raw((0.0, 0.0, 0.5), (1.0, 0.0, 0.0), 1),
        _raw((1.0, 0.0, 0.5), (1.0, 0.0, 0.0), 0),
    )

    report = analyze_components(raw, {0: 0, 1: 1})

    assert report["ordinal"]["component_count"] == 2
    assert report["ordinal"]["view_count_lt_2_rejection_count"] == 1
    assert report["ordinal"]["aggregate_normal_zero_rejection_count"] == 0
    assert report["ordinal"]["pre_top64_candidate_count"] == 1
    assert report["ordinal"]["conservation"]["passed"] is True
    assert report["ordinal"]["stages"] == [
        {
            "stage": "view_count_lt_2_rejection",
            "input_count": 2,
            "rejection_count": 1,
            "survival_count": 1,
        },
        {
            "stage": "aggregate_normal_zero_rejection",
            "input_count": 1,
            "rejection_count": 0,
            "survival_count": 1,
        },
    ]
    assert report["shadow"]["pre_top64_candidate_count"] == 1
    assert report["view_count_monotonic"] is True


def test_component_zero_normal_gate_is_observed_from_formal_merge() -> None:
    normals = (
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
    )
    raw = tuple(
        _raw((0.0, 0.0, 0.5), normal, index % 2, row=20 + index)
        for index, normal in enumerate(normals)
    )

    report = analyze_components(raw, {0: 0, 1: 1})

    assert report["ordinal"]["component_count"] == 1
    assert report["ordinal"]["aggregate_normal_zero_rejection_count"] == 1
    assert report["ordinal"]["pre_top64_candidate_count"] == 0
    assert report["ordinal"]["conservation"]["passed"] is True


def test_duplicate_observation_shadow_never_increases_views_or_candidates() -> None:
    raw = (
        _raw((0.0, 0.0, 0.5), (1.0, 0.0, 0.0), 0),
        _raw((0.0, 0.0, 0.5), (1.0, 0.0, 0.0), 1),
    )

    report = analyze_components(raw, {0: 0, 1: 0})

    assert report["ordinal"]["pre_top64_candidate_count"] == 1
    assert report["shadow"]["view_count_lt_2_rejection_count"] == 1
    assert report["shadow"]["pre_top64_candidate_count"] == 0
    assert report["view_count_monotonic"] is True
    assert report["shadow_candidate_monotonic"] is True


def test_ranking_ledger_enforces_top64_conservation() -> None:
    assert ranking_ledger(70, 64) == {
        "stage": "top64",
        "input_count": 70,
        "rejection_count": 6,
        "survival_count": 64,
    }
    with pytest.raises(CandidateFunnelContractError, match="top-64"):
        ranking_ledger(65, 65)


def test_full_funnel_is_deterministic_and_preserves_formal_candidate_bytes() -> None:
    first = serialize_policy_input(_input(timestamp=1, sequence=1, phase=1))
    duplicate = serialize_policy_input(_input(timestamp=1, sequence=1, phase=3))
    final = serialize_policy_input(_input(timestamp=2, sequence=2, phase=4))
    official = generate_candidate_set(
        (first, duplicate),
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=final,
    )

    report = analyze_candidate_funnel(
        (first, duplicate),
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=final,
        expected_candidate_bytes=official.canonical_bytes,
        expected_selected_index=-1,
    )
    replay = analyze_candidate_funnel(
        (first, duplicate),
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=final,
        expected_candidate_bytes=official.canonical_bytes,
        expected_selected_index=-1,
    )

    assert report == replay
    assert report["checks"]["passed"] is True
    assert report["formal_candidate"]["canonical_bytes_bit_identical"] is True
    assert report["unique_observation_shadow"]["unique_observation_count"] == 1
    assert report["unique_observation_shadow"]["unique_payload_count"] == 1
    assert report["anchor_ledger"]["conservation"]["passed"] is True
    assert report["component_ledger"]["ordinal"]["conservation"]["passed"] is True
    assert report["ranking_ledger"]["conservation"]["passed"] is True


def test_same_observation_identity_with_different_visible_payload_fails_closed() -> None:
    first_value = _input(timestamp=1, sequence=1)
    rgb = np.array(first_value.head_rgb_uint8, copy=True)
    rgb[0, 0, 0] = 1
    changed = replace(first_value, phase_index=3, head_rgb_uint8=rgb)

    with pytest.raises(
        CandidateFunnelContractError, match="observation identity changed"
    ):
        analyze_candidate_funnel(
            (
                serialize_policy_input(first_value),
                serialize_policy_input(changed),
            ),
            acquisition_base_pose=(0.0, 0.0, 0.0),
            final_input=serialize_policy_input(_input(timestamp=2, sequence=2)),
            expected_candidate_bytes=None,
            expected_selected_index=-1,
        )


def test_anchor_stage_names_are_frozen_and_ordered() -> None:
    assert ANCHOR_REJECTION_STAGES == (
        "center_ring_validity",
        "prominence",
        "center_depth_spread",
        "patch_support_before_self_mask",
        "support_after_self_mask",
        "height_range",
        "planarity",
        "width",
    )


def test_aggregate_requires_twenty_four_unique_episodes_and_twelve_cells() -> None:
    episodes = []
    for cell in range(12):
        for replicate in range(2):
            episodes.append(
                {
                    "planned_episode_id": f"{cell:02d}{replicate}" + "a" * 61,
                    "task_id": f"task-{cell // 4}",
                    "cell_id": f"cell-{cell:02d}",
                    "capture_enabled_disabled_identity": True,
                    "funnel": {
                        "checks": {"passed": True},
                        "anchor_ledger": {
                            "raw_candidate_count": 2,
                            "stages": [
                                {
                                    "stage": "prominence",
                                    "input_count": 10,
                                    "rejection_count": 6,
                                    "survival_count": 4,
                                }
                            ],
                        },
                        "component_ledger": {
                            "ordinal": {
                                "component_count": 1,
                                "stages": [
                                    {
                                        "stage": "view_count_lt_2_rejection",
                                        "input_count": 1,
                                        "rejection_count": 0,
                                        "survival_count": 1,
                                    }
                                ],
                            }
                        },
                        "ranking_ledger": {
                            "pre_top64_candidate_count": 1,
                            "stages": [
                                {
                                    "stage": "top64",
                                    "input_count": 1,
                                    "rejection_count": 0,
                                    "survival_count": 1,
                                }
                            ],
                        },
                        "formal_candidate": {"candidate_count": 1},
                    },
                }
            )

    report = aggregate_candidate_funnels(episodes)

    assert report["checks"]["passed"] is True
    assert report["cell_count"] == 12
    assert all(
        value["repeatable_descriptive_loss_stage"] == ["anchor.prominence"]
        for value in report["cells"]
    )
    assert report["weakest_task_cell"]["cell_id"] == "cell-00"
