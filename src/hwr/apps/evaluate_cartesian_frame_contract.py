"""Evaluate the frozen R0001-P51 Cartesian acquisition-to-base contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.eval import target_selection


MODULE_NAME = "hwr.apps.evaluate_cartesian_frame_contract"
PROPOSAL_ID = "R0001-P51"
REPORT_SCHEMA = "hwr.p51-cartesian-frame-contract-report/v1"
MANIFEST_SCHEMA = "hwr.p51-cartesian-frame-contract-artifacts/v1"
FAILURE_SCHEMA = "hwr.p51-cartesian-frame-contract-failure/v1"
FROZEN_DOCUMENT_COMMIT = "4385ceee2fffcbd23788b498d258747dc273465c"
VELOCITY_MAX = 0.08
ACQUISITION_YAWS = (0.0, math.pi / 3.0, -math.pi / 2.0)
RELATIVE_YAWS = (
    0.0,
    math.pi / 6.0,
    -math.pi / 6.0,
    math.pi / 2.0,
    -math.pi / 2.0,
    math.pi,
)
ACQUISITION_ERRORS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.3, -0.4, 0.5),
)
ARMS = ("left", "right")
TOLERANCE = 1e-12
PRIMITIVE_ACQUISITION_POSE = (2.0, -1.0, math.pi / 3.0)
PRIMITIVE_RELATIVE_YAWS = (0.0, math.pi / 2.0)
PRIMITIVE_PHASE_STEPS = (
    ("B0_orient", 0, 0.0, 0),
    ("B1_approach", 100, 0.0, 0),
    ("B2_preposition", 400, 0.08, 0),
    ("B3_contact_approach", 500, 0.03, 0),
    ("B4_close", 550, 0.02, 0),
    ("B4_close", 569, 0.02, 19),
    ("B5_pull", 570, 0.04, 0),
    ("B6_retract", 600, 0.06, 0),
    ("B6_retract", 640, 0.06, 40),
    ("B7_stop", 650, 0.0, 0),
)
PRIMITIVE_CANDIDATE = target_selection.Candidate(
    (1.40, 0.20, 0.75), (-1.0, 0.0, 0.0), 0.14, 0.15, 80, 4, 0, 20, 20
)
FROZEN_ACTION_MINIMUM = np.asarray((-0.18, -0.50, *(-0.35,) * 12, 0.0, 0.0))
FROZEN_ACTION_MAXIMUM = np.asarray((0.18, 0.50, *(0.35,) * 12, 1.0, 1.0))
CLAIM_FLAGS = {
    "training_executed": False,
    "policy_inference_executed": False,
    "closed_loop_physics_executed": False,
    "capability_claim_allowed": False,
    "task_success_claim_allowed": False,
    "generalization_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
}
UNCHANGED_FLAGS = {
    "candidate_generator_changed": False,
    "candidate_bytes_changed": False,
    "selector_changed": False,
    "acquisition_changed": False,
    "phase_changed": False,
    "target_changed": False,
    "velocity_cap_changed": False,
    "gripper_changed": False,
    "backend_changed": False,
    "safety_changed": False,
}
SOURCE_PATHS = (
    Path("src/hwr/eval/target_selection.py"),
    Path("src/hwr/apps/evaluate_cartesian_frame_contract.py"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--output",
        str(arguments.output),
    ]
    identities = _source_identities(root)
    try:
        _require_clean_source(root)
        evaluation = evaluate_contract()
        report = _build_report(source_commit, command, evaluation)
        artifacts = {"report.json": _json_bytes(report)}
        manifest = _manifest(
            source_commit,
            command,
            identities,
            artifacts,
            evaluation=evaluation,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "source_commit": source_commit,
            "decision": "invalid",
            "error_type": type(error).__name__,
            "error": str(error),
            **CLAIM_FLAGS,
            **UNCHANGED_FLAGS,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _manifest(
            source_commit,
            command,
            identities,
            artifacts,
            evaluation=None,
            status="failed",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
        raise
    return {
        "output": str(output),
        "decision": report["decision"],
        "cell_count": evaluation["cell_count"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
    }


def evaluate_contract() -> dict[str, object]:
    cells = []
    for acquisition_yaw in ACQUISITION_YAWS:
        for relative_yaw in RELATIVE_YAWS:
            current_base_yaw = acquisition_yaw + relative_yaw
            for acquisition_error in ACQUISITION_ERRORS:
                expected = _clip_norm(
                    2.0 * np.asarray(acquisition_error, np.float64),
                    VELOCITY_MAX,
                )
                legacy = expected.copy()
                candidate = target_selection.acquisition_error_to_base_velocity(
                    acquisition_error,
                    VELOCITY_MAX,
                    acquisition_yaw=acquisition_yaw,
                    current_base_yaw=current_base_yaw,
                )
                for arm in ARMS:
                    cells.append(
                        _cell(
                            arm,
                            acquisition_yaw,
                            relative_yaw,
                            acquisition_error,
                            expected,
                            legacy,
                            candidate,
                        )
                    )
    candidate_errors = [cell["candidate"]["errors"] for cell in cells]
    counterexamples = [
        cell
        for cell in cells
        if abs(abs(cell["relative_yaw"]) - math.pi / 2.0) <= TOLERANCE
        and np.linalg.norm(cell["acquisition_error"][:2]) > 0.0
    ]
    expected_count = (
        len(ACQUISITION_YAWS)
        * len(RELATIVE_YAWS)
        * len(ACQUISITION_ERRORS)
        * len(ARMS)
    )
    maxima = {
        name: max(float(errors[name]) for errors in candidate_errors)
        for name in ("maximum_absolute", "angular", "norm", "z")
    }
    checks = {
        "matrix_complete": len(cells) == expected_count,
        "both_arms_covered": {cell["arm"] for cell in cells} == set(ARMS),
        "all_values_finite": all(_cell_is_finite(cell) for cell in cells),
        "zero_relative_yaw_float64_bytes_identical": all(
            cell["candidate_legacy_float64_bytes_identical"]
            for cell in cells
            if cell["relative_yaw"] == 0.0
        ),
        "candidate_maximum_absolute_error_within_tolerance": (
            maxima["maximum_absolute"] <= TOLERANCE
        ),
        "candidate_angular_error_within_tolerance": (
            maxima["angular"] <= TOLERANCE
        ),
        "candidate_norm_error_within_tolerance": maxima["norm"] <= TOLERANCE,
        "candidate_z_error_within_tolerance": maxima["z"] <= TOLERANCE,
        "legacy_quarter_turn_counterexamples_rejected": bool(counterexamples)
        and all(
            cell["legacy"]["errors"]["angular"] >= math.pi / 2.0 - TOLERANCE
            for cell in counterexamples
        ),
    }
    primitive = evaluate_primitive_integration()
    checks["primitive_integration_passed"] = primitive["passed"]
    return {
        "formula": {
            "error": "e_A = target_A - tool_A",
            "rotation": "R_B_from_A = Rz(theta_A - theta_B)",
            "candidate": "v_B = R_B_from_A * clip_norm(2 * e_A, velocity_max)",
            "legacy": "v_B_legacy = clip_norm(2 * e_A, velocity_max)",
            "relative_yaw_definition": "theta_B - theta_A",
            "xy_only_rotation": True,
            "rotation_after_norm_clipping": True,
        },
        "matrix": {
            "acquisition_yaws": list(ACQUISITION_YAWS),
            "relative_yaws": list(RELATIVE_YAWS),
            "acquisition_errors": [list(value) for value in ACQUISITION_ERRORS],
            "arms": list(ARMS),
            "velocity_max": VELOCITY_MAX,
            "tolerance": TOLERANCE,
        },
        "cell_count": len(cells),
        "expected_cell_count": expected_count,
        "legacy_counterexample_count": len(counterexamples),
        "candidate_error_maxima": maxima,
        "checks": checks,
        "passed": all(checks.values()),
        "primitive_integration": primitive,
        "cells": cells,
    }


def evaluate_primitive_integration() -> dict[str, object]:
    calls = []
    original = target_selection.acquisition_error_to_base_velocity

    def traced(error, maximum, **yaws):
        calls.append((float(maximum), *(float(value) for value in yaws.values())))
        return original(error, maximum, **yaws)

    cases = [
        _primitive_case(phase, step, yaw, traced)
        for yaw in PRIMITIVE_RELATIVE_YAWS
        for phase, step, _, _ in PRIMITIVE_PHASE_STEPS
    ]
    holds = [_hold_case(None, "ok"), _hold_case(PRIMITIVE_CANDIDATE, "stopped")]
    manipulation = [case for case in cases if case["velocity_max"] > 0.0]
    expected_calls = [
        (cap, PRIMITIVE_ACQUISITION_POSE[2], PRIMITIVE_ACQUISITION_POSE[2] + yaw)
        for yaw in PRIMITIVE_RELATIVE_YAWS
        for _, _, cap, _ in PRIMITIVE_PHASE_STEPS
        if cap > 0.0
        for _ in ARMS
    ]
    checks = {
        "primitive_called_transform_for_both_arms": len(calls)
        == len(manipulation) * len(ARMS),
        "transform_arguments_preserved": calls == expected_calls,
        "phase_contract_preserved": all(case["phase_ok"] for case in cases),
        "target_contract_preserved": all(
            case["maximum_absolute_error"] <= TOLERANCE for case in manipulation
        ),
        "nonzero_relative_yaw_expected_action": all(
            case["maximum_absolute_error"] <= TOLERANCE
            for case in cases if case["relative_yaw"] != 0.0
        ),
        "zero_relative_yaw_legacy_bytes_identical": all(
            case["legacy_bytes_identical"] for case in cases
            if case["relative_yaw"] == 0.0
        ),
        "both_arms_match_expected": all(
            max(case["left_error"], case["right_error"]) <= TOLERANCE
            for case in manipulation
        ),
        "base_command_preserved": all(
            case["base_error"] <= TOLERANCE for case in cases
        ),
        "velocity_caps_preserved": all(case["cap_ok"] for case in cases),
        "arm_angular_commands_zero": all(case["angular_zero"] for case in cases),
        "gripper_contract_preserved": all(
            case["gripper_error"] <= TOLERANCE for case in cases
        ),
        "hold_contract_preserved": all(holds),
        "action_bounds_preserved": _bounds_contract(cases),
    }
    checks["only_frozen_formula_changed"] = all(
        value for name, value in checks.items()
        if name not in {
            "primitive_called_transform_for_both_arms",
            "transform_arguments_preserved",
            "nonzero_relative_yaw_expected_action",
            "zero_relative_yaw_legacy_bytes_identical",
        }
    )
    return {
        "fixture": {
            "acquisition_base_pose": list(PRIMITIVE_ACQUISITION_POSE),
            "relative_yaws": list(PRIMITIVE_RELATIVE_YAWS),
            "phase_steps": [list(item) for item in PRIMITIVE_PHASE_STEPS],
            "candidate": list(PRIMITIVE_CANDIDATE.canonical_record()),
            "tool_positions_acquisition_frame": [
                list(value) for value in _fixture_tools()
            ],
        },
        "case_count": len(cases),
        "hold_case_count": len(holds),
        "helper_call_count": len(calls),
        "checks": checks,
        "passed": all(checks.values()),
        "cases": cases,
    }


def _primitive_case(phase, step, relative_yaw, transform) -> dict[str, object]:
    actual = _invoke_primitive(step, relative_yaw, transform=transform)
    expected = _expected_primitive_action(step, relative_yaw)
    legacy = _invoke_primitive(step, relative_yaw, transform=_legacy_velocity)
    _, _, cap, local_step = next(
        item for item in PRIMITIVE_PHASE_STEPS if item[1] == step
    )
    observed_phase = target_selection.phase_for_step(step)
    return {
        "phase": phase,
        "post_selection_step": step,
        "relative_yaw": relative_yaw,
        "velocity_max": cap,
        "phase_ok": observed_phase == (phase, local_step),
        "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
        "base_error": float(np.max(np.abs(actual[:2] - expected[:2]))),
        "left_error": float(np.max(np.abs(actual[2:5] - expected[2:5]))),
        "right_error": float(np.max(np.abs(actual[8:11] - expected[8:11]))),
        "gripper_error": float(np.max(np.abs(actual[14:] - expected[14:]))),
        "legacy_bytes_identical": relative_yaw != 0.0
        or actual.astype("<f8").tobytes() == legacy.astype("<f8").tobytes(),
        "cap_ok": bool(
            np.linalg.norm(actual[2:5] * 0.30) <= cap + TOLERANCE
            and np.linalg.norm(actual[8:11] * 0.30) <= cap + TOLERANCE
        ),
        "angular_zero": bool(
            np.array_equal(actual[5:8], np.zeros(3))
            and np.array_equal(actual[11:14], np.zeros(3))
        ),
        "bounds_ok": bool(
            np.all(actual >= FROZEN_ACTION_MINIMUM)
            and np.all(actual <= FROZEN_ACTION_MAXIMUM)
        ),
    }


def _expected_primitive_action(step: int, relative_yaw: float) -> np.ndarray:
    phase, _, _, local_step = next(
        item for item in PRIMITIVE_PHASE_STEPS if item[1] == step
    )
    point = np.asarray(PRIMITIVE_CANDIDATE.center)
    forward = point[:2] / np.linalg.norm(point[:2])
    normal = np.asarray((-forward[0], -forward[1], 0.0))
    lateral = np.asarray((-forward[1], forward[0], 0.0))
    vertical = np.asarray((0.0, 0.0, 1.0))
    pre_spacing = min(max(PRIMITIVE_CANDIDATE.width + 0.12, 0.18), 0.34)
    contact_spacing = min(max(PRIMITIVE_CANDIDATE.width + 0.04, 0.10), 0.24)
    left_pre = point + 0.18 * normal + 0.5 * pre_spacing * lateral + 0.05 * vertical
    right_pre = point + 0.18 * normal - 0.5 * pre_spacing * lateral + 0.05 * vertical
    left_contact = point + 0.015 * normal + 0.5 * contact_spacing * lateral
    right_contact = point + 0.015 * normal - 0.5 * contact_spacing * lateral
    heading_error = math.atan2(
        math.sin(math.atan2(*forward[::-1]) - relative_yaw),
        math.cos(math.atan2(*forward[::-1]) - relative_yaw),
    )
    base_linear = base_angular = gripper = cap = 0.0
    left_target = right_target = None
    if phase == "B0_orient":
        base_angular = min(max(heading_error, -0.35), 0.35)
    elif phase == "B1_approach":
        if abs(heading_error) <= 0.35:
            base_linear = min(max(0.6 * (np.linalg.norm(point[:2]) - 0.85), 0.0), 0.12)
        base_angular = min(max(heading_error, -0.25), 0.25)
    elif phase == "B2_preposition":
        left_target, right_target, cap = left_pre, right_pre, 0.08
    elif phase == "B3_contact_approach":
        left_target, right_target, cap = left_contact, right_contact, 0.03
    elif phase == "B4_close":
        left_target, right_target, cap = left_contact, right_contact, 0.02
        gripper = 0.75 * (local_step + 1) / 20.0
    elif phase == "B5_pull":
        left_target = left_contact + 0.08 * normal + 0.02 * vertical
        right_target = right_contact + 0.08 * normal + 0.02 * vertical
        cap, gripper = 0.04, 0.75
    elif phase == "B6_retract":
        left_target, right_target = left_pre + 0.05 * normal, right_pre + 0.05 * normal
        cap = 0.06
        gripper = 0.75 if local_step < 30 else 0.75 * (49 - local_step) / 19.0
    left_velocity = right_velocity = np.zeros(3)
    if left_target is not None:
        left_tool, right_tool = _fixture_tools()
        left_velocity = _expected_velocity(left_target - left_tool, cap, relative_yaw)
        right_velocity = _expected_velocity(right_target - right_tool, cap, relative_yaw)
    action = np.asarray((
        base_linear, base_angular,
        *(left_velocity / 0.30), 0.0, 0.0, 0.0,
        *(right_velocity / 0.30), 0.0, 0.0, 0.0,
        gripper, gripper,
    ))
    return np.clip(action, FROZEN_ACTION_MINIMUM, FROZEN_ACTION_MAXIMUM)


def _expected_velocity(error, cap, relative_yaw) -> np.ndarray:
    velocity = _clip_norm(2.0 * error, cap)
    cosine, sine = math.cos(-relative_yaw), math.sin(-relative_yaw)
    return np.asarray((
        cosine * velocity[0] - sine * velocity[1],
        sine * velocity[0] + cosine * velocity[1],
        velocity[2],
    ))


def _invoke_primitive(
    step, relative_yaw, *, transform, candidate=PRIMITIVE_CANDIDATE,
    safety_state="ok",
) -> np.ndarray:
    original_transform = target_selection.acquisition_error_to_base_velocity
    original_tools = target_selection.tool_positions_in_acquisition
    target_selection.acquisition_error_to_base_velocity = transform
    target_selection.tool_positions_in_acquisition = lambda value, origin: _fixture_tools()
    try:
        payload = target_selection.serialize_policy_input(
            _primitive_input(relative_yaw, safety_state)
        )
        action = target_selection.primitive_action(
            payload, candidate, PRIMITIVE_ACQUISITION_POSE, step
        )
        return np.asarray(action)
    finally:
        target_selection.acquisition_error_to_base_velocity = original_transform
        target_selection.tool_positions_in_acquisition = original_tools


def _primitive_input(relative_yaw, safety_state) -> target_selection.PolicyVisibleInput:
    proprioception = np.zeros(37, dtype="<f8")
    proprioception[24:26] = (0.20, 0.80)
    proprioception[26:29] = (
        PRIMITIVE_ACQUISITION_POSE[0],
        PRIMITIVE_ACQUISITION_POSE[1],
        PRIMITIVE_ACQUISITION_POSE[2] + relative_yaw,
    )
    return target_selection.PolicyVisibleInput(
        50_000_000, 3, 5, 0, 17, safety_state,
        np.zeros((192, 256, 3), dtype=np.uint8),
        np.ones((192, 256), dtype="<f4"),
        np.ones((192, 256), dtype=np.bool_),
        np.asarray((200.0, 200.0, 127.5, 95.5), dtype="<f8"),
        np.eye(4, dtype="<f8"), proprioception,
        np.zeros((4, 16), dtype="<f8"),
        np.asarray((False, False, False, True), dtype=np.bool_),
    )


def _fixture_tools() -> tuple[np.ndarray, np.ndarray]:
    return np.asarray((1.0, 0.3, 0.7)), np.asarray((1.0, -0.3, 0.7))


def _legacy_velocity(error, maximum, **yaws) -> np.ndarray:
    del yaws
    return _clip_norm(2.0 * np.asarray(error), maximum)


def _hold_case(candidate, safety_state) -> bool:
    actual = _invoke_primitive(
        400, math.pi / 2.0, transform=_legacy_velocity,
        candidate=candidate, safety_state=safety_state,
    )
    expected = np.asarray((0.0, 0.0, *(0.0,) * 12, 0.20, 0.80))
    return actual.astype("<f8").tobytes() == expected.astype("<f8").tobytes()


def _bounds_contract(cases) -> bool:
    def oversized(error, maximum, **yaws):
        del error, maximum, yaws
        return np.asarray((1.0, -1.0, 1.0))

    actual = _invoke_primitive(400, math.pi / 2.0, transform=oversized)
    expected = np.asarray((
        0.0, 0.0, 0.35, -0.35, 0.35, 0.0, 0.0, 0.0,
        0.35, -0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0,
    ))
    constants_ok = (
        target_selection.ACTION_MINIMUM.astype("<f8").tobytes()
        == FROZEN_ACTION_MINIMUM.astype("<f8").tobytes()
        and target_selection.ACTION_MAXIMUM.astype("<f8").tobytes()
        == FROZEN_ACTION_MAXIMUM.astype("<f8").tobytes()
    )
    return constants_ok and all(case["bounds_ok"] for case in cases) and (
        actual.astype("<f8").tobytes() == expected.astype("<f8").tobytes()
    )


def _cell(
    arm: str,
    acquisition_yaw: float,
    relative_yaw: float,
    acquisition_error: Sequence[float],
    expected: np.ndarray,
    legacy: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    legacy_realized = _base_to_acquisition(legacy, relative_yaw)
    candidate_realized = _base_to_acquisition(candidate, relative_yaw)
    return {
        "arm": arm,
        "acquisition_yaw": acquisition_yaw,
        "current_base_yaw": acquisition_yaw + relative_yaw,
        "relative_yaw": relative_yaw,
        "acquisition_error": list(acquisition_error),
        "expected_acquisition_velocity": expected.tolist(),
        "legacy": _command_record(legacy, legacy_realized, expected),
        "candidate": _command_record(candidate, candidate_realized, expected),
        "candidate_legacy_float64_bytes_identical": (
            candidate.astype("<f8").tobytes() == legacy.astype("<f8").tobytes()
        ),
    }


def _command_record(
    command: np.ndarray,
    realized: np.ndarray,
    expected: np.ndarray,
) -> dict[str, object]:
    return {
        "base_frame_command": command.tolist(),
        "base_frame_command_float64_sha256": hashlib.sha256(
            command.astype("<f8").tobytes()
        ).hexdigest(),
        "realized_acquisition_frame_vector": realized.tolist(),
        "errors": {
            "maximum_absolute": float(np.max(np.abs(realized - expected))),
            "angular": _horizontal_angular_error(realized, expected),
            "norm": abs(float(np.linalg.norm(realized) - np.linalg.norm(expected))),
            "z": abs(float(realized[2] - expected[2])),
        },
    }


def _base_to_acquisition(vector: np.ndarray, relative_yaw: float) -> np.ndarray:
    if relative_yaw == 0.0:
        return vector.copy()
    cosine, sine = math.cos(relative_yaw), math.sin(relative_yaw)
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            vector[2],
        ),
        np.float64,
    )


def _horizontal_angular_error(actual: np.ndarray, expected: np.ndarray) -> float:
    cross = float(actual[0] * expected[1] - actual[1] * expected[0])
    dot = float(np.dot(actual[:2], expected[:2]))
    return abs(math.atan2(cross, dot))


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum or norm == 0.0 else vector * maximum / norm


def _cell_is_finite(cell: Mapping[str, object]) -> bool:
    values = (
        *cell["legacy"]["base_frame_command"],
        *cell["legacy"]["realized_acquisition_frame_vector"],
        *cell["legacy"]["errors"].values(),
        *cell["candidate"]["base_frame_command"],
        *cell["candidate"]["realized_acquisition_frame_vector"],
        *cell["candidate"]["errors"].values(),
    )
    return all(math.isfinite(float(value)) for value in values)


def _build_report(
    source_commit: str,
    command: Sequence[str],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "decision": (
            "accepted as Cartesian primitive correctness evidence"
            if evaluation["passed"]
            else "rejected"
        ),
        "evidence_scope": "deterministic analytic coordinate contract only",
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
        **evaluation,
    }


def _manifest(
    source_commit: str,
    command: Sequence[str],
    source_identities: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    *,
    evaluation: Mapping[str, object] | None,
    status: str,
) -> dict[str, object]:
    configuration = {
        "acquisition_yaws": list(ACQUISITION_YAWS),
        "relative_yaws": list(RELATIVE_YAWS),
        "acquisition_errors": [list(value) for value in ACQUISITION_ERRORS],
        "arms": list(ARMS),
        "velocity_max": VELOCITY_MAX,
        "tolerance": TOLERANCE,
        "primitive_acquisition_pose": list(PRIMITIVE_ACQUISITION_POSE),
        "primitive_relative_yaws": list(PRIMITIVE_RELATIVE_YAWS),
        "primitive_phase_steps": [list(item) for item in PRIMITIVE_PHASE_STEPS],
        "primitive_candidate": list(PRIMITIVE_CANDIDATE.canonical_record()),
        "primitive_tool_positions": [list(value) for value in _fixture_tools()],
    }
    configuration_bytes = json.dumps(
        configuration, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": status,
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "command": list(command),
        "decision": (
            "invalid"
            if evaluation is None
            else (
                "accepted as Cartesian primitive correctness evidence"
                if evaluation["passed"]
                else "rejected"
            )
        ),
        "checks": None if evaluation is None else dict(evaluation["checks"]),
        "primitive_integration_checks": (
            None
            if evaluation is None
            else dict(evaluation["primitive_integration"]["checks"])
        ),
        "configuration": {
            **configuration,
            "sha256": hashlib.sha256(configuration_bytes).hexdigest(),
            "bytes": len(configuration_bytes),
        },
        "model": {"executed": False, "identity": None},
        "source_files": dict(source_identities),
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for name, content in artifacts.items()
        },
    }


def _source_identities(root: Path) -> dict[str, object]:
    return {
        path.as_posix(): _file_identity(root / path)
        for path in SOURCE_PATHS
    }


def _file_identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(Path(__file__).resolve().parents[3]).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _require_clean_source(root: Path) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P51 runner requires clean committed source")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("P51 frozen document commit is not an ancestor")
    history = tuple(
        f"docs/research-loop/{index:04d}" for index in range(1, 9)
    )
    unchanged = subprocess.run(
        ("git", "diff", "--quiet", FROZEN_DOCUMENT_COMMIT, "HEAD", "--", *history),
        cwd=root,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("P51 historical research-loop documents drifted")


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P51 runner requires a full Git source commit")
    return commit


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir()
    try:
        for name, content in artifacts.items():
            _atomic_write(staging / name, content)
        os.replace(staging, output)
    except BaseException:
        for path in staging.glob("*"):
            path.unlink()
        if staging.exists():
            staging.rmdir()
        raise


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
