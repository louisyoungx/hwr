"""Continuation and bank provenance helpers for R0001-P51-E1."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np

from hwr.adapters.mujoco.entity_contact_graph import p40_conservation_differences
from hwr.adapters.mujoco.target_selection_diagnostic import INVALID_GRAPH_FIELDS
from hwr.core.embodied import DualArmObservation
from hwr.eval.cartesian_convergence import (
    canonical_sha256,
    identity,
    observation_identity_record,
    runtime_counter_record,
    wrap_angle,
)
from hwr.eval.target_selection import ACQUISITION_STEPS


CONTINUATION_SCHEMA = "hwr.p51-cartesian-continuation-identity/v1"
PROTECTED_GROUP_PATHS = {
    "selector": (
        "src/hwr/eval/target_selection.py",
        "src/hwr/eval/target_selection_safety.py",
    ),
    "existing_p41_p51_p52": (
        "src/hwr/apps/evaluate_target_selection.py",
        "src/hwr/adapters/mujoco/target_selection_diagnostic.py",
        "src/hwr/apps/evaluate_cartesian_frame_contract.py",
        "src/hwr/apps/evaluate_tool_kinematics.py",
        "src/hwr/eval/tool_kinematics.py",
    ),
    "task_binding": ("configs",),
    "mujoco_xml": ("assets",),
    "backend": (
        "src/hwr/adapters/mujoco",
        "src/hwr/core",
        "src/hwr/eval/seed_contract.py",
        "src/hwr/eval/stability.py",
        "src/hwr/scenarios/formal3d.py",
    ),
    "safety": ("src/hwr/safety",),
    "frozen_document": ("docs/research-loop/0010/03-experiment.md",),
}
UNCHANGED_FLAG_GROUPS = {
    "candidate_generator_changed": ("selector",),
    "candidate_bytes_changed": ("selector",),
    "selector_changed": ("selector",),
    "acquisition_changed": ("selector", "existing_p41_p51_p52"),
    "b0_b1_prefix_changed": ("selector", "existing_p41_p51_p52"),
    "phase_changed": ("selector",),
    "target_formula_changed": ("selector",),
    "velocity_cap_changed": ("selector",),
    "gripper_changed": ("selector", "backend"),
    "fk_changed": ("selector", "existing_p41_p51_p52", "backend", "mujoco_xml"),
    "backend_changed": ("backend", "task_binding", "mujoco_xml"),
    "safety_changed": ("safety", "backend"),
}


def continuation_identity(
    backend,
    observation: DualArmObservation,
    history: Sequence[Sequence[float]],
    history_available: Sequence[bool],
    graph,
) -> dict[str, object]:
    specification = mujoco.mjtState(
        int(mujoco.mjtState.mjSTATE_INTEGRATION)
        | int(mujoco.mjtState.mjSTATE_CTRL)
    )
    runtime = np.empty(
        mujoco.mj_stateSize(backend.model, specification), np.float64
    )
    mujoco.mj_getState(backend.model, backend.data, runtime, specification)
    components = {
        "mujoco_model_state": _array_bundle_identity(
            (
                backend.model.body_mass,
                backend.model.body_inertia,
                backend.model.geom_friction,
                backend.model.light_diffuse,
                backend.model.mat_rgba,
                backend.model.cam_pos,
                backend.model.cam_quat,
                backend.model.cam_fovy,
            )
        ),
        "mujoco_data_state": _array_bundle_identity(
            (runtime, backend.data.ctrl)
        ),
        "actuator_servo_targets": identity(
            {
                "ctrl": backend.data.ctrl.tolist(),
                "left_targets": backend._left_targets.tolist(),
                "right_targets": backend._right_targets.tolist(),
            }
        ),
        "action_latency_queue": identity(
            [list(action.vector()) for action in backend._action_queue]
        ),
        "observation_latency_queue": identity(
            [
                observation_identity_record(value)
                for value in backend._observation_queue
            ]
        ),
        "policy_history_availability": identity(
            {
                "history": [list(value) for value in history],
                "available": list(history_available),
            }
        ),
        "current_observation": identity(
            observation_identity_record(observation)
        ),
        "timestamp_sequence_runtime_safety_counters": identity(
            runtime_counter_record(backend, observation, graph)
        ),
    }
    payload = {
        "schema_version": CONTINUATION_SCHEMA,
        "components": components,
    }
    return {**payload, "identity": identity(payload)}


def bank_prefix_record(run) -> dict[str, object]:
    candidate_bytes = (
        b""
        if run.candidate_set is None
        else run.candidate_set.canonical_bytes
    )
    graph_report = run.graph.report()
    audit = run.backend.task_audit()
    conservation = p40_conservation_differences(
        graph_report, run.backend.contact_ledger.report()
    )
    targets = {
        name: list(value) for name, value in run.preposition_targets.items()
    }
    return {
        "eligible": run.failure is None,
        "eligibility_reason": (
            "eligible" if run.failure is None else run.failure
        ),
        "candidate_count": (
            0
            if run.candidate_set is None
            else len(run.candidate_set.candidates)
        ),
        "candidate_set_sha256": hashlib.sha256(
            candidate_bytes
        ).hexdigest(),
        "candidate_bytes_hex": candidate_bytes.hex(),
        "selected_index": run.selected_index,
        "selected_record": (
            None if run.candidate is None else asdict(run.candidate)
        ),
        "prefix_failure_reason": run.failure,
        "input_failure_reason": run.input_failure,
        "prefix_step_count": len(run.trace),
        "prefix_complete": (
            len(run.trace) == ACQUISITION_STEPS + 400
            and not any(row["terminal"] for row in run.trace)
        ),
        "prefix_terminal_observed": any(row["terminal"] for row in run.trace),
        "prefix_safety_intervention_count": sum(
            bool(row["safety_intervened"]) for row in run.trace
        ),
        "prefix_action_bounds_valid": all(
            row["action_bounds_valid"] for row in run.trace
        ),
        "prefix_stale_action_applied_count": sum(
            bool(row["outside_validity_window"])
            and row["applied_action"] != row["hold_action"]
            for row in run.trace
        ),
        "prefix_severe_collision_count": int(audit["severe_collision_count"]),
        "prefix_invalid_force_count": sum(
            int(graph_report[name]) for name in INVALID_GRAPH_FIELDS
        ),
        "prefix_p40_conservation_maximum_absolute_difference": float(
            conservation["maximum_absolute_difference"]
        ),
        "acquisition_main_event": run.acquisition_main_event,
        "acquisition_input_hashes": run.acquisition_input_hashes,
        "acquisition_input_sequence_sha256": canonical_sha256(
            run.acquisition_input_hashes
        ),
        "prefix_trace_sha256": canonical_sha256(run.trace),
        "b0_b1_proposed_action_sha256": canonical_sha256(
            [
                row["proposed_action"]
                for row in run.trace[ACQUISITION_STEPS:]
            ]
        ),
        "b0_b1_applied_action_sha256": canonical_sha256(
            [
                row["applied_action"]
                for row in run.trace[ACQUISITION_STEPS:]
            ]
        ),
        "relative_yaw_at_b2": wrap_angle(
            run.backend._base_state()[0][2] - run.acquisition_pose[2]
        ),
        "b2_policy_base_pose": list(run.observation.proprioception.base_pose),
        "acquisition_base_pose": list(run.acquisition_pose),
        "acquisition_world_origin": list(run.acquisition_world_origin),
        "continuation_identity": run.continuation_identity,
        "first_treatment_actions": {
            role: list(action)
            for role, action in run.first_actions.items()
        },
        "first_treatment_guard": run.first_guard,
        "preposition_targets": targets,
        "preposition_target_identity": identity(targets),
        "preposition_target_identities": {
            name: identity(value) for name, value in targets.items()
        },
        "primitive_target_crosscheck": run.primitive_target_crosscheck,
    }


def raw_runtime_step_evidence(
    b2_step: int,
    row,
    backend,
    graph,
    tool_distance,
    hard_failure_reason,
) -> dict[str, object]:
    result = backend.result()
    graph_report = graph.report()
    invalid_force_count = sum(
        int(graph_report[name]) for name in INVALID_GRAPH_FIELDS
    )
    conservation = float(
        p40_conservation_differences(
            graph_report, backend.contact_ledger.report()
        )["maximum_absolute_difference"]
    )
    evidence = {
        "b2_step": b2_step,
        "runtime_step": int(row["step"]),
        "executed": bool(row["executed"]),
        "terminated": bool(row["terminated"]),
        "truncated": bool(row["truncated"]),
        "terminal": bool(row["terminal"]),
        "observation_timestamp_ns": int(backend._timestamp_ns()),
        "events": list(row["events"]),
        "events_sha256": canonical_sha256(row["events"]),
        "episode_result": None if result is None else asdict(result),
        "hard_failure_reason": hard_failure_reason,
        "action_bounds_valid": bool(row["action_bounds_valid"]),
        "outside_validity_window": bool(row["outside_validity_window"]),
        "safety_intervened": bool(row["safety_intervened"]),
        "severe_collision_count": int(
            backend.task_audit()["severe_collision_count"]
        ),
        "invalid_force_count": invalid_force_count,
        "p40_conservation_maximum_absolute_difference": conservation,
        "nonfinite_value_count": _nonfinite_runtime_value_count(
            row, tool_distance
        ),
        "hold_action": list(row["hold_action"]),
        "proposed_action": list(row["proposed_action"]),
        "applied_action": list(row["applied_action"]),
        "tool_distance": dict(tool_distance),
    }
    return {**evidence, "trace_sha256": canonical_sha256(evidence)}


def protected_source_status(root: Path, frozen_commit: str) -> dict[str, object]:
    pathspecs = tuple(dict.fromkeys(
        path for paths in PROTECTED_GROUP_PATHS.values() for path in paths
    ))
    frozen = _tree_entries(root, frozen_commit, pathspecs)
    checked_paths = tuple(sorted(frozen))
    current = _tree_entries(root, "HEAD", checked_paths)
    changed = _git_lines(
        root, (
            "diff", "--name-only", f"{frozen_commit}..HEAD", "--", *checked_paths
        )
    )
    blobs = {
        path: {
            "frozen_blob": frozen.get(path),
            "current_blob": current.get(path),
            "matches": frozen.get(path) == current.get(path),
        }
        for path in sorted(set(frozen) | set(current))
    }
    groups = {
        name: {
            "pathspecs": list(paths),
            "changed_paths": [
                path for path in changed if any(_under(path, item) for item in paths)
            ],
            "matches": all(
                value["matches"]
                for path, value in blobs.items()
                if any(_under(path, item) for item in paths)
            ),
        }
        for name, paths in PROTECTED_GROUP_PATHS.items()
    }
    return {
        "base_commit": frozen_commit,
        "checked_pathspecs": list(pathspecs),
        "checked_paths": list(checked_paths),
        "changed_paths": changed,
        "blob_identities": blobs,
        "groups": groups,
        "passed": not changed and all(value["matches"] for value in blobs.values()),
    }


def require_protected_source(status: Mapping[str, object]) -> None:
    blobs = status.get("blob_identities")
    groups = status.get("groups")
    if (
        status.get("passed") is not True
        or status.get("changed_paths")
        or not isinstance(blobs, Mapping)
        or not blobs
        or any(value.get("matches") is not True for value in blobs.values())
        or not isinstance(groups, Mapping)
        or set(groups) != set(PROTECTED_GROUP_PATHS)
        or any(value.get("matches") is not True for value in groups.values())
    ):
        raise RuntimeError("P51-E1 protected frozen source drifted")


def unchanged_flags(identities: Mapping[str, object]) -> dict[str, object]:
    protected = identities.get("protected_frozen_source")
    if not isinstance(protected, Mapping):
        return dict.fromkeys(UNCHANGED_FLAG_GROUPS, None)
    groups = protected.get("groups")
    if not isinstance(groups, Mapping):
        return dict.fromkeys(UNCHANGED_FLAG_GROUPS, None)
    return {
        flag: not all(
            isinstance(groups.get(group), Mapping)
            and groups[group].get("matches") is True
            for group in required
        )
        for flag, required in UNCHANGED_FLAG_GROUPS.items()
    }


def _tree_entries(root: Path, commit: str, pathspecs) -> dict[str, str]:
    lines = _git_lines(root, ("ls-tree", "-r", commit, "--", *pathspecs))
    result = {}
    for line in lines:
        metadata, path = line.split("\t", 1)
        _, kind, blob = metadata.split()
        if kind == "blob":
            result[path] = blob
    return result


def _git_lines(root: Path, arguments) -> list[str]:
    output = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


def _under(path: str, pathspec: str) -> bool:
    return path == pathspec or path.startswith(pathspec.rstrip("/") + "/")


def _nonfinite_runtime_value_count(row, tool_distance) -> int:
    values = (
        row["proposed_action"],
        row["applied_action"],
        row["hold_action"],
        tuple(tool_distance.values()),
    )
    return sum(
        int(np.size(value) - np.count_nonzero(np.isfinite(np.asarray(value, np.float64))))
        for value in values
    )


def _array_bundle_identity(
    values: Sequence[np.ndarray],
) -> dict[str, object]:
    chunks = []
    for value in values:
        array = np.ascontiguousarray(value)
        chunks.extend(
            (
                array.dtype.str.encode("ascii"),
                str(tuple(array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return identity(b"".join(chunks))
