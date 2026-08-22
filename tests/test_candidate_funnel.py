from __future__ import annotations

import hashlib
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import hwr.apps as app_helpers
from hwr.apps import aggregate_candidate_funnels
from hwr.apps import evaluate_candidate_acquisition as app
from hwr.eval import candidate_funnel, target_selection
from hwr.eval.candidate_funnel import (
    ANCHOR_REJECTION_STAGES,
    CandidateFunnelContractError,
    analyze_candidate_funnel,
    analyze_components,
    candidate_gate_source_identity,
    ranking_ledger,
)
from hwr.eval.target_selection import (
    PolicyVisibleInput,
    RawCandidate,
    generate_candidate_set,
    serialize_policy_input,
)

SALT = "0" * 64
TEST_COMMITMENT = hashlib.sha256(SALT.encode()).hexdigest()


def _plan_and_execution() -> tuple[dict[str, object], dict[str, object]]:
    cells = [
        {
            "cell_id": f"cell-{ordinal:02d}-obs-{observation}-action-{action}",
            "cell_ordinal": ordinal,
            "task_id": task,
            "observation_latency_steps": observation,
            "action_latency_steps": action,
            "replicate_count": 2,
        }
        for ordinal, (task, observation, action) in enumerate(app.FROZEN_CELLS)
    ]
    episodes = []
    for cell in cells:
        for replicate in range(2):
            identity = app.planned_episode_id(
                app.PLAN_ID, cell["task_id"], cell["cell_id"], replicate
            )
            episodes.append(
                {
                    "planned_episode_id": identity,
                    "task_id": cell["task_id"],
                    "cell_id": cell["cell_id"],
                    "cell_ordinal": cell["cell_ordinal"],
                    "replicate_ordinal": replicate,
                    "candidate_ordinal": replicate,
                    "environment_seed": app.derive_domain_seed(
                        SALT, "environment", identity
                    ),
                    "policy_rng_seed": app.derive_domain_seed(
                        SALT, "policy", identity
                    ),
                    "sampled_observation_latency_steps": cell[
                        "observation_latency_steps"
                    ],
                    "sampled_action_latency_steps": cell["action_latency_steps"],
                    "replacement": False,
                }
            )
    plan = {
        "schema_version": app.PLAN_SCHEMA,
        "proposal_id": app.PROPOSAL_ID,
        "mode": "formal",
        "plan_id": app.PLAN_ID,
        "salt_commitment": TEST_COMMITMENT,
        "salt_reveal": SALT,
        "commitment_verified": True,
        "seed_schema": app.SEED_SCHEMA,
        "role_enters_seed_derivation": False,
        "natural_evaluation_latency_rejection": True,
        "reset_latency_override_used": False,
        "replacement_seed_allowed": False,
        "maximum_candidate_ordinal": 95,
        "maximum_latency_sampler_calls": 1_152,
        "planned_control_steps_per_episode": app.ACQUISITION_STEPS,
        "planned_episode_count": 24,
        "planned_control_step_count": 24 * app.ACQUISITION_STEPS,
        "validation_replay_count": 24,
        "maximum_physical_acquisition_count": 48,
        "maximum_control_step_count": 48 * app.ACQUISITION_STEPS,
        "cells": cells,
        "episodes": episodes,
        "rejected_seed_audit": [],
    }
    records = [
        {
            **episode,
            "planned_latency": {
                "observation_steps": episode[
                    "sampled_observation_latency_steps"
                ],
                "action_steps": episode["sampled_action_latency_steps"],
            },
            "replacement": False,
        }
        for episode in episodes
    ]
    return plan, {
        "terminals": [dict(record) for record in records],
        "capsules": {"episodes": [dict(record) for record in records]},
    }


def _mock_e2_loader(monkeypatch, documents) -> None:
    monkeypatch.setattr(
        app_helpers, "_read_object", lambda path: documents[path.name]
    )
    monkeypatch.setattr(
        app_helpers, "_verify_artifacts", lambda root, manifest: None
    )
    monkeypatch.setattr(
        app_helpers, "candidate_commit_is_ancestor", lambda *args: True
    )
    monkeypatch.setattr(
        app_helpers, "_require_e1_source_identity", lambda *args: None
    )


def _load_e2(tmp_path: Path) -> None:
    app_helpers.analyze_candidate_capsule_directory(
        tmp_path,
        tmp_path,
        {},
        app.FROZEN_DOCUMENT_COMMIT,
        app.E1_SOURCE_NAMES,
        expected_plan_schema=app.PLAN_SCHEMA,
        expected_proposal_id=app.PROPOSAL_ID,
        expected_plan_id=app.PLAN_ID,
        expected_commitment=TEST_COMMITMENT,
        expected_cells=app.FROZEN_CELLS,
        acquisition_steps=app.ACQUISITION_STEPS,
    )


def _e2_documents(plan, execution) -> dict[str, object]:
    return {
        "manifest.json": {
            "status": "complete",
            "source_commit": "a" * 40,
            "frozen_document_commit": app.FROZEN_DOCUMENT_COMMIT,
            "frozen_document_commit_is_ancestor": True,
        },
        "report.json": {
            "decision": "accepted as immutable acquisition evidence contract"
        },
        "plan.json": plan,
        "capsules.json": {
            "capsule_count": 24,
            "episodes": execution["capsules"]["episodes"],
            "terminals": execution["terminals"],
        },
    }


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


def test_full_funnel_is_deterministic_and_traces_one_formal_call() -> None:
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
    assert report["ranking_ledger"]["formal_generator_call_count"] == 1
    assert report["ranking_ledger"]["formal_merge_call_count"] == 1
    assert report["all_capsule_input_count"] == 3
    assert report["candidate_keyframe_count"] == 2
    assert "capture_count" not in report
    assert report["all_capsule_inputs"]["includes_a4_final"] is True
    assert report["all_capsule_inputs"]["unique_observation_count"] == 2
    assert report["all_capsule_inputs"]["inputs"][-1]["a4_final_input"] is True
    assert sum(
        value["candidate_keyframe"]
        for value in report["all_capsule_inputs"]["inputs"]
    ) == 2
    assert report["unique_observation_shadow"]["unique_observation_count"] == 1
    assert report["unique_observation_shadow"]["unique_payload_count"] == 1
    assert report["anchor_ledger"]["conservation"]["passed"] is True
    assert report["component_ledger"]["ordinal"]["conservation"]["passed"] is True
    assert report["ranking_ledger"]["conservation"]["passed"] is True


def test_production_funnel_has_one_formal_generation_and_only_one_shadow_merge() -> None:
    first = serialize_policy_input(_input(timestamp=1, sequence=1, phase=1))
    second = serialize_policy_input(_input(timestamp=2, sequence=2, phase=3))
    final = serialize_policy_input(_input(timestamp=3, sequence=3, phase=4))
    official = generate_candidate_set(
        (first, second),
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=final,
    )
    calls = Counter()

    def profile(frame, event, argument):
        del argument
        if event == "call":
            if frame.f_code is target_selection.generate_candidate_set.__code__:
                calls["generate"] += 1
            elif frame.f_code is target_selection._merge_candidates.__code__:
                calls["merge"] += 1
        return profile

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        report = analyze_candidate_funnel(
            (first, second),
            acquisition_base_pose=(0.0, 0.0, 0.0),
            final_input=final,
            expected_candidate_bytes=official.canonical_bytes,
            expected_selected_index=-1,
        )
    finally:
        sys.setprofile(previous)

    assert calls == {"generate": 1, "merge": 2}
    assert report["ranking_ledger"]["formal_generator_call_count"] == 1
    assert report["ranking_ledger"]["formal_merge_call_count"] == 1


def _runtime_anchor_stage(
    *, payload: bytes | None = None, points: np.ndarray | None = None
) -> str:
    stages = candidate_funnel._anchor_line_contract()
    observed = []

    def trace(frame, event, argument):
        del argument
        if event == "line":
            stage = stages.get((frame.f_code.co_name, frame.f_lineno))
            if stage is not None:
                if frame.f_code is target_selection._candidate_from_points.__code__:
                    observed.append(stage)
                elif frame.f_locals.get("row") == 96 and frame.f_locals.get("column") == 128:
                    observed.append(stage)
        return trace

    if payload is not None:
        function = target_selection._frame_candidates
        arguments = (
            target_selection.deserialize_policy_input(payload),
            (0.0, 0.0, 0.0),
            0,
        )
    else:
        function = target_selection._candidate_from_points
        arguments = (points, np.asarray((0.0, 0.0, 0.7)), np.zeros(3), 0.1, 0, 96, 128)
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        function(*arguments)
    finally:
        sys.settrace(previous)
    assert len(observed) == 1
    return observed[0]


def test_all_eight_anchor_gates_have_runtime_traced_boundary_counterexamples() -> None:
    row, column = 96, 128
    invalid = np.zeros((192, 256), dtype=np.bool_)
    flat = np.ones((192, 256), dtype="<f4")
    spread = np.full((192, 256), 1.2, dtype="<f4")
    spread[row - 2 : row + 3, column - 2 : column + 3] = np.linspace(
        0.96, 1.04, 25, dtype=np.float32
    ).reshape(5, 5)
    support = np.full((192, 256), 1.2, dtype="<f4")
    support[row - 2 : row + 3, column - 2 : column + 3] = 1.0
    support_valid = np.ones((192, 256), dtype=np.bool_)
    support_valid[row - 2, column - 2 : column + 3] = False
    self_masked = np.full((192, 256), 0.30, dtype="<f4")
    self_masked[row - 2 : row + 3, column - 2 : column + 3] = 0.20
    rows, columns = np.indices((6, 6), dtype=np.float64)
    accepted = np.column_stack((
        np.full(rows.size, 0.50),
        (columns.ravel() - 2.5) * 0.01,
        0.70 + (rows.ravel() - 2.5) * 0.01,
    ))
    too_near = accepted.copy()
    too_near[:, 0] = 0.10
    cube = np.asarray([
        (0.45 + x, y, 0.65 + z)
        for x in (-0.05, 0.05)
        for y in (-0.05, 0.05)
        for z in (-0.05, 0.05)
    ])
    narrow = accepted.copy()
    narrow[:, 1:] *= 0.1
    cases = (
        ("center_ring_validity", {"payload": serialize_policy_input(_input(valid=invalid))}),
        ("prominence", {"payload": serialize_policy_input(_input(depth=flat))}),
        ("center_depth_spread", {"payload": serialize_policy_input(_input(depth=spread))}),
        ("patch_support_before_self_mask", {
            "payload": serialize_policy_input(_input(depth=support, valid=support_valid))
        }),
        ("support_after_self_mask", {
            "payload": serialize_policy_input(_input(depth=self_masked))
        }),
        ("height_range", {"points": too_near}),
        ("planarity", {"points": cube}),
        ("width", {"points": narrow}),
    )

    assert tuple(_runtime_anchor_stage(**arguments) for _, arguments in cases) == tuple(
        stage for stage, _ in cases
    )


def test_failed_online_candidate_is_only_labeled_counterfactual() -> None:
    keyframe = serialize_policy_input(_input(timestamp=1, sequence=1))
    final = serialize_policy_input(_input(timestamp=2, sequence=2, phase=4))

    report = analyze_candidate_funnel(
        (keyframe,),
        acquisition_base_pose=(0.0, 0.0, 0.0),
        final_input=final,
        expected_candidate_bytes=b"",
        expected_selected_index=-1,
        expected_score_sha256="e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855",
        selection_permitted=False,
    )

    assert report["formal_candidate"]["generated_online"] is False
    assert report["formal_candidate"]["candidate_count"] == 0
    assert report["offline_counterfactual_candidate"]["selected_index_not_used"] is True
    assert report["checks"]["passed"] is True


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


@pytest.mark.parametrize(
    "field",
    (
        "planned_episode_id",
        "task_id",
        "cell_id",
        "cell_ordinal",
        "replicate_ordinal",
        "candidate_ordinal",
        "environment_seed",
        "policy_rng_seed",
        "planned_latency",
        "replacement",
    ),
)
def test_e2_loader_rejects_plan_record_tampering(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, execution = _plan_and_execution()
    record = execution["capsules"]["episodes"][0]
    record[field] = (
        {"observation_steps": 2, "action_steps": 1}
        if field == "planned_latency"
        else True if field == "replacement" else "tampered"
    )
    _mock_e2_loader(monkeypatch, _e2_documents(plan, execution))

    with pytest.raises(CandidateFunnelContractError, match="ledger differs"):
        _load_e2(tmp_path)


@pytest.mark.parametrize(
    "tamper",
    (
        "delete",
        "environment_seed",
        "policy_rng_seed",
        "cell",
        "planned_episode_id",
    ),
)
def test_e2_loader_rejects_synchronized_plan_and_record_tampering(
    tamper: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, execution = _plan_and_execution()
    terminals = execution["terminals"]
    records = execution["capsules"]["episodes"]
    if tamper == "delete":
        del plan["episodes"][0]
        del terminals[0]
        del records[0]
    elif tamper in ("environment_seed", "policy_rng_seed"):
        for record in (plan["episodes"][0], terminals[0], records[0]):
            record[tamper] += 1
    elif tamper == "cell":
        for record in (plan["episodes"][0], terminals[0], records[0]):
            record["cell_id"] = "cell-11-obs-2-action-2"
            record["cell_ordinal"] = 11
    else:
        for record in (plan["episodes"][0], terminals[0], records[0]):
            record["planned_episode_id"] = "f" * 64
    _mock_e2_loader(monkeypatch, _e2_documents(plan, execution))

    with pytest.raises(CandidateFunnelContractError, match="plan contract"):
        _load_e2(tmp_path)


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
