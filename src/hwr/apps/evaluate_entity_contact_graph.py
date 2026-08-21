"""Run the frozen R0001-P40-E2 entity-contact measurement contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from hwr.adapters.mujoco import (
    CONTACT_CATEGORIES, MEASUREMENT_SCHEMA, ROBOT_BODY_ROOT_NAMES,
    ContactLedger, ContactPointObservation, EntityContactGraph,
    EntityContactGraphError, EntityContactPointObservation, EntityMotionSource,
    MujocoFormalHouseholdDualArmBackend, load_default_formal_household_catalogs,
    p40_conservation_differences, resolve_allowed_contact_role_ids,
    resolve_robot_part_by_geom,
)
from hwr.core.embodied import DualArmAction, DualArmActionFrame


MODULE_NAME = "hwr.apps.evaluate_entity_contact_graph"
PROPOSAL_ID = "R0001-P40-E2"
REPORT_SCHEMA = "hwr.entity-contact-graph-contract-report/v1"
MANIFEST_SCHEMA = "hwr.entity-contact-graph-contract-artifacts/v1"
FAILURE_SCHEMA = "hwr.entity-contact-graph-contract-failure/v1"
FIXTURE_SCHEMA = "hwr.mujoco-entity-contact-graph-fixture/v1"
TASK_IDS = (
    "tidy_living_room_3d/v1", "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
)
BASE_SEED, SEED_STRIDE, CONTROL_STEP_LIMIT = 20_264_002, 104_729, 32
FROZEN_PARENT_COMMIT = "4c4efda16759577cb05098a7628f29d3bfbef890"
FROZEN_DOCUMENT_COMMIT = "b56ee96953652e2e80d644b8167181e8449c0a8b"
FROZEN_BINDING_SHA256 = "7984ef2544bb618269681d274257a598b02621371a26de002bfdd8bbf7decab6"
FROZEN_BINDING_BYTES = 3051
INVALID_COUNT_FIELDS = (
    "missing_normal_force_count", "nonfinite_normal_force_count",
    "invalid_negative_normal_force_count", "unknown_mapping_count",
    "invalid_motion_state_count",
)
CLAIM_FLAGS = {
    "measurement_only": True,
    "policy_inference_executed": False,
    "closed_loop_capability_episode_executed": False,
    "capability_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
    "action_causality_claim_allowed": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    binding_path = root / "configs/adapters/mujoco/formal_3d_v1.json"
    binding_identity = _file_identity(root, binding_path)
    command = [
        ".venv/bin/python", "-m", MODULE_NAME, "--output", str(arguments.output)
    ]
    try:
        _require_clean_source(root, binding_identity)
        evaluation = _evaluate_contract(root)
        report = _build_report(source_commit, command, evaluation)
        artifacts = {"report.json": _json_bytes(report)}
        manifest = _manifest(
            source_commit, command, binding_identity, evaluation, artifacts,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "source_commit": source_commit,
            "error_type": type(error).__name__,
            "error": str(error),
            **CLAIM_FLAGS,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _manifest(
            source_commit, command, binding_identity, None, artifacts,
            status="failed",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
        raise
    return {
        "output": str(output),
        "decision": report["decision"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
    }


def _evaluate_contract(root: Path) -> dict[str, object]:
    tasks, bindings = load_default_formal_household_catalogs(root)
    if set(tasks) != set(TASK_IDS):
        raise RuntimeError("P40-E2 task catalog differs from the frozen contract")
    fixture = _run_fixture()
    reports = []
    for task_index, task_id in enumerate(TASK_IDS):
        seed = BASE_SEED + task_index * SEED_STRIDE
        disabled = _run_trace(
            tasks[task_id], bindings[task_id], seed=seed, enabled=False
        )
        enabled = _run_trace(
            tasks[task_id], bindings[task_id], seed=seed, enabled=True
        )
        conservation = p40_conservation_differences(
            enabled["entity_contact_graph"], enabled["contact_ledger"]
        )
        reports.append(
            {
                "task_id": task_id,
                "seed": seed,
                "fixed_hold_action": enabled["fixed_hold_action"],
                "camera_rendering_enabled": False,
                "control_step_limit": CONTROL_STEP_LIMIT,
                "disabled_legacy_trace": disabled["trace"],
                "enabled_legacy_trace": enabled["trace"],
                "disabled_trace_sha256": _canonical_sha256(disabled["trace"]),
                "enabled_trace_sha256": _canonical_sha256(enabled["trace"]),
                "legacy_trace_bit_identical": disabled["trace"] == enabled["trace"],
                "contact_ledger": enabled["contact_ledger"],
                "entity_contact_graph": enabled["entity_contact_graph"],
                "p40_conservation": conservation,
                "physics": enabled["physics"],
            }
        )
    checks = _contract_checks(fixture, reports)
    return {
        "fixture": fixture,
        "tasks": reports,
        "checks": checks,
        "passed": all(checks.values()),
        "physics": {value["task_id"]: value["physics"] for value in reports},
        "robot_body_roots": {
            value["task_id"]: value["entity_contact_graph"]["mapping"][
                "robot_body_roots"
            ]
            for value in reports
        },
    }


def _run_trace(task, binding, *, seed: int, enabled: bool) -> dict[str, object]:
    backend = MujocoFormalHouseholdDualArmBackend(
        task, binding, camera_width=16, camera_height=12, evaluation_profile=True
    )
    graph = _graph_from_backend(backend, enabled=enabled)
    original_substep = backend._after_physics_substep
    action = None
    ledger = None
    graph_report = None
    physics = None
    def observe_substep() -> None:
        original_substep()
        graph.sample_mujoco_substep(backend.model, backend.data)

    backend._after_physics_substep = observe_substep
    trace: list[dict[str, object]] = []
    try:
        backend.contact_ledger.set_enabled(enabled)
        observation = backend.reset(seed=seed, task_id=task.task_id)
        graph.reset()
        backend.set_camera_rendering(False)
        action = DualArmAction(
            0.0, 0.0, (0.0,) * 6, (0.0,) * 6,
            observation.proprioception.left_gripper_position,
            observation.proprioception.right_gripper_position,
        )
        for step in range(CONTROL_STEP_LIMIT):
            graph.begin_control_period(
                graph.capture_motion_state(backend.model, backend.data)
            )
            timestamp = observation.timestamp_ns
            outcome = backend.apply(
                DualArmActionFrame(
                    timestamp, timestamp, timestamp + 250_000_000,
                    "R0001-P40-E2-fixed-hold", action,
                )
            )
            graph.end_control_period(
                graph.capture_motion_state(backend.model, backend.data)
            )
            observation = outcome.observation
            trace.append(_legacy_trace_step(backend, outcome, observation, step))
            if outcome.terminated or outcome.truncated:
                break
        ledger = backend.contact_ledger.report()
        graph_report = graph.report()
        physics = {
            "mujoco_version": importlib.metadata.version("mujoco"),
            "timestep": float(backend.model.opt.timestep),
            "solver": int(backend.model.opt.solver),
            "iterations": int(backend.model.opt.iterations),
            "tolerance": float(backend.model.opt.tolerance),
            "control_hz": float(task.control_hz),
            "substeps_per_control_period": int(backend._substeps),
        }
    finally:
        backend.close()
    if action is None or ledger is None or graph_report is None or physics is None:
        raise RuntimeError("P40-E2 trace did not produce complete evidence")
    return {
        "trace": trace,
        "contact_ledger": ledger,
        "entity_contact_graph": graph_report,
        "fixed_hold_action": list(action.vector()),
        "physics": physics,
    }


def _legacy_trace_step(backend, outcome, observation, step: int) -> dict[str, object]:
    audit = backend.task_audit()
    result = backend.result()
    applied = outcome.info["applied_action"]
    return {
        "step": step,
        "applied_action_vector": list(applied.action.vector()),
        "proprioception": list(observation.proprioception.vector()),
        "reward": outcome.reward,
        "terminated": outcome.terminated,
        "truncated": outcome.truncated,
        "success": None if result is None else result.success,
        "reason": None if result is None else result.reason,
        "severe_collision_count": audit["severe_collision_count"],
        "maximum_forbidden_force": audit["maximum_forbidden_force"],
        "maximum_forbidden_pair": audit["maximum_forbidden_pair"],
        "safety_intervened": outcome.info["safety_intervened"],
    }


def _graph_from_backend(backend, *, enabled: bool) -> EntityContactGraph:
    model, ids, binding = backend.model, backend.household_ids, backend.binding
    robot_geoms = frozenset(int(value) for value in ids.robot_geoms)
    robot_parts, root_identity = resolve_robot_part_by_geom(model, robot_geoms)
    _, role_by_geom = resolve_allowed_contact_role_ids(
        model, binding.allowed_robot_contact_geoms,
        binding.allowed_robot_contact_roles,
    )
    object_by_geom = {
        int(model.geom(value.collision_geom).id): object_id
        for object_id, value in binding.objects.items()
    }
    if object_by_geom != {
        int(geom): object_id for object_id, geom in ids.object_geoms.items()
    }:
        raise ValueError("backend object geometry identity differs from binding")
    object_role_geoms = {
        geom for geom, role in role_by_geom.items() if role == "manipulated_object"
    }
    if object_role_geoms != set(object_by_geom):
        raise ValueError("manipulated-object roles differ from object bindings")
    articulation = binding.articulation
    articulation_geom = (
        int(model.geom(articulation.handle_geom).id) if articulation else None
    )
    articulation_role_geoms = {
        geom for geom, role in role_by_geom.items() if role == "articulation"
    }
    if articulation_role_geoms != (
        set() if articulation_geom is None else {articulation_geom}
    ):
        raise ValueError("articulation roles differ from articulation binding")
    entity_by_geom = _environment_entities(
        model, robot_geoms, role_by_geom, object_by_geom, articulation
    )
    motion_sources = {
        f"manipulated_object:{object_id}": EntityMotionSource("translation", geom)
        for geom, object_id in object_by_geom.items()
    }
    if articulation:
        joint = int(model.joint(articulation.joint).id)
        if joint != ids.articulation_joint:
            raise ValueError("backend articulation identity differs from binding")
        motion_sources[f"articulation:{articulation.articulation_id}"] = (
            EntityMotionSource("joint", joint)
        )
    pads = {
        part: (
            (int(model.geom(f"{side}_gripper_left_pad").id),),
            (int(model.geom(f"{side}_gripper_right_pad").id),),
        )
        for part, side in (("left_arm", "left"), ("right_arm", "right"))
    }
    return EntityContactGraph(
        all_geom_ids=range(model.ngeom),
        robot_part_by_geom=robot_parts,
        entity_by_geom=entity_by_geom,
        timestep=float(model.opt.timestep),
        enabled=enabled,
        motion_source_by_entity=motion_sources,
        gripper_pad_groups=pads,
        geom_name_by_id={
            geom: model.geom(geom).name or f"geom_{geom}"
            for geom in range(model.ngeom)
        },
        robot_body_roots=root_identity,
    )


def _environment_entities(
    model, robot_geoms, role_by_geom, object_by_geom, articulation
) -> dict[int, str]:
    result = {}
    for geom in range(model.ngeom):
        if geom in robot_geoms:
            continue
        role = role_by_geom.get(geom, "forbidden")
        if role == "manipulated_object":
            identifier = object_by_geom[geom]
        elif role == "articulation":
            identifier = articulation.articulation_id
        else:
            identifier = model.geom(geom).name or f"geom_{geom}"
        result[geom] = f"{role}:{identifier}"
    return result


def _run_fixture() -> dict[str, object]:
    graph = _fixture_graph()
    ledger = ContactLedger(
        robot_geoms=range(1, 8),
        allowed_role_by_geom={
            10: "manipulated_object", 11: "manipulated_object",
            12: "floor_support", 13: "target_container", 14: "articulation",
        },
        timestep=0.01,
        enabled=True,
    )
    start = _fixture_motion_state()
    substeps = _fixture_substeps()
    graph.begin_control_period(start)
    ledger.begin_control_period()
    for observations in substeps:
        graph.record_substep(observations)
        ledger.record_substep(
            ContactPointObservation(item.geom1, item.geom2, item.normal_force)
            for item in observations
        )
    moved = dict(start)
    moved["manipulated_object:a"] = (0.03, 0.04, 0.0)
    moved["articulation:drawer"] = 0.2
    first_period = graph.end_control_period(moved)
    ledger.end_control_period()
    first_report = graph.report()
    conservation = p40_conservation_differences(first_report, ledger.report())
    graph.begin_control_period(moved)
    free_motion = dict(moved)
    free_motion["manipulated_object:b"] = (0.1, 0.0, 0.0)
    graph.record_substep(())
    no_contact_period = graph.end_control_period(free_motion)
    graph.begin_control_period(free_motion)
    graph.record_substep((EntityContactPointObservation(7, 11, 1.0),))
    graph.end_control_period(free_motion)
    graph.begin_control_period(free_motion)
    inertia = dict(free_motion)
    inertia["manipulated_object:b"] = (0.2, 0.0, 0.0)
    graph.record_substep(())
    inertia_period = graph.end_control_period(inertia)
    report = graph.report()
    cases = _fixture_case_results(
        report, first_period, no_contact_period, inertia_period
    )
    invalid = {
        name: _invalid_force_fixture(value, counter)
        for name, value, counter in (
            ("missing", None, "missing_normal_force_count"),
            ("nan", float("nan"), "nonfinite_normal_force_count"),
            ("inf", float("inf"), "nonfinite_normal_force_count"),
            ("negative", -1.0, "invalid_negative_normal_force_count"),
        )
    }
    mappings = _mapping_failure_fixture()
    passed = all(cases.values()) and conservation["passed"]
    passed = passed and all(invalid.values()) and all(mappings.values())
    passed = passed and not any(report[name] for name in INVALID_COUNT_FIELDS)
    return {
        "schema_version": FIXTURE_SCHEMA,
        "cases": cases,
        "classification_precision": 1.0 if all(cases.values()) else 0.0,
        "classification_recall": 1.0 if all(cases.values()) else 0.0,
        "p40_conservation": conservation,
        "fail_closed_cases": invalid,
        "mapping_fail_closed_cases": mappings,
        "valid_fixture_invalid_counts": {
            name: report[name] for name in INVALID_COUNT_FIELDS
        },
        "passed": passed,
    }


def _fixture_graph(**overrides) -> EntityContactGraph:
    arguments = {
        "all_geom_ids": (*range(1, 8), *range(10, 16)),
        "robot_part_by_geom": {
            1: "base", 2: "left_arm", 3: "left_arm", 4: "left_arm",
            5: "right_arm", 6: "right_arm", 7: "right_arm",
        },
        "entity_by_geom": {
            10: "manipulated_object:a", 11: "manipulated_object:b",
            12: "floor_support:floor", 13: "target_container:container",
            14: "articulation:drawer", 15: "forbidden:wall",
        },
        "timestep": 0.01,
        "enabled": True,
        "motion_source_by_entity": {
            "manipulated_object:a": EntityMotionSource("translation", 10),
            "manipulated_object:b": EntityMotionSource("translation", 11),
            "articulation:drawer": EntityMotionSource("joint", 14),
        },
        "gripper_pad_groups": {
            "left_arm": ((2,), (3,)), "right_arm": ((5,), (6,))
        },
    }
    arguments.update(overrides)
    return EntityContactGraph(**arguments)


def _fixture_motion_state() -> dict[str, object]:
    return dict.fromkeys(
        ("manipulated_object:a", "manipulated_object:b"), (0.0, 0.0, 0.0)
    ) | {"articulation:drawer": 0.0}


def _fixture_substeps() -> tuple[tuple[EntityContactPointObservation, ...], ...]:
    point = EntityContactPointObservation
    return (
        (point(2, 10, 2.0), point(10, 2, 3.0), point(3, 10, 4.0),
         point(5, 10, 6.0), point(6, 10, 7.0)),
        (point(4, 10, 2.0), point(7, 11, 3.0)),
        (point(4, 10, 1.0),),
        (point(1, 12, 5.0),),
        (point(4, 14, 4.0),),
        (point(10, 13, 8.0),),
        (point(10, 12, 9.0),),
        (point(10, 11, 6.0),),
        (point(1, 2, 1.0), point(12, 13, 1.0)),
    )


def _fixture_case_results(
    report, first_period, no_contact_period, inertia_period
) -> dict[str, bool]:
    observations = report["substeps"]
    world_edges = report["task_relevant_world_world_edges"]
    return {
        "same_entity_dual_arm": observations[0][
            "same_entity_dual_arm_contacts"
        ] == ["manipulated_object:a"],
        "same_object_dual_arm_grasp": observations[0][
            "same_object_dual_arm_grasps"
        ] == ["manipulated_object:a"],
        "distinct_entity_dual_arm": observations[1][
            "distinct_entity_dual_arm_contacts"
        ] == [["manipulated_object:a", "manipulated_object:b"]],
        "single_arm": observations[2]["left_only_entities"]
        == ["manipulated_object:a"] and not observations[2]["right_only_entities"],
        "base_floor_only": not any(observations[3][name] for name in (
            "same_entity_dual_arm_contacts", "left_only_entities",
            "right_only_entities",
        )),
        "articulation_motion": first_period["entity_motion"][
            "articulation:drawer"
        ]["contact_associated_motion"] == 0.2,
        "no_contact_motion": no_contact_period["entity_motion"][
            "manipulated_object:b"
        ]["contact_associated_motion"] == 0.0,
        "post_contact_inertia": inertia_period["entity_motion"][
            "manipulated_object:b"
        ]["contact_associated_motion"] == 0.0,
        "object_container_edge": _has_world_edge(
            world_edges,
            ("manipulated_object:a", "target_container:container"),
        ),
        "object_support_edge": _has_world_edge(
            world_edges, ("floor_support:floor", "manipulated_object:a")
        ),
        "object_object_edge": _has_world_edge(
            world_edges, ("manipulated_object:a", "manipulated_object:b")
        ),
        "duplicate_pair_summed": report["legacy_p40_categories"][
            "manipulated_object"
        ]["pair_peak_force"] == 7.0,
    }


def _has_world_edge(edges, entities: tuple[str, str]) -> bool:
    return any(edge["entities"] == list(entities) for edge in edges)


def _invalid_force_fixture(value: float | None, counter: str) -> bool:
    graph = _fixture_graph()
    graph.begin_control_period(_fixture_motion_state())
    try:
        graph.record_substep((EntityContactPointObservation(4, 10, value),))
    except EntityContactGraphError:
        report = graph.report()
        return report[counter] == 1 and report["contract_valid"] is False
    return False


def _mapping_failure_fixture() -> dict[str, bool]:
    cases: dict[str, Callable[[], object]] = {
        "missing": lambda: _fixture_graph(
            all_geom_ids=(*range(1, 8), *range(10, 17))
        ),
        "overlap": lambda: _fixture_graph(
            entity_by_geom={
                1: "forbidden:x", 10: "manipulated_object:a",
                11: "manipulated_object:b", 12: "floor_support:floor",
                13: "target_container:container", 14: "articulation:drawer",
                15: "forbidden:wall",
            }
        ),
        "unknown_root": lambda: _fixture_graph(
            robot_body_roots={
                "base": {"body_name": "robot_base", "body_id": 1},
                "left_arm": {"body_name": "unknown", "body_id": 2},
                "right_arm": {
                    "body_name": "right_shoulder_pan_link", "body_id": 3
                },
            },
        ),
        "unknown_entity_role": lambda: _fixture_graph(
            entity_by_geom={
                10: "manipulated_object:a", 11: "manipulated_object:b",
                12: "floor_support:floor", 13: "target_container:container",
                14: "articulation:drawer", 15: "unknown:wall",
            }
        ),
    }
    return {name: _raises_value_error(function) for name, function in cases.items()}


def _raises_value_error(function: Callable[[], object]) -> bool:
    try:
        function()
    except ValueError:
        return True
    return False


def _contract_checks(fixture, reports) -> dict[str, bool]:
    return {
        "fixture_passed": fixture["passed"] is True,
        "fixture_precision_recall_one": fixture["classification_precision"] == 1.0
        and fixture["classification_recall"] == 1.0,
        "all_mappings_complete": all(
            set(value["entity_contact_graph"]["mapping"]["robot_body_roots"])
            == set(ROBOT_BODY_ROOT_NAMES)
            and len(value["entity_contact_graph"]["mapping"]["robot_geoms"]) == 48
            for value in reports
        ),
        "legacy_traces_bit_identical": all(
            value["legacy_trace_bit_identical"] for value in reports
        ),
        "all_measurements_valid": all(
            value["contact_ledger"]["contract_valid"]
            and value["entity_contact_graph"]["contract_valid"]
            for value in reports
        ),
        "p40_conservation_exact": all(
            value["p40_conservation"]["passed"]
            and value["p40_conservation"]["maximum_absolute_difference"] <= 1.0e-12
            for value in reports
        ),
        "world_world_excluded_from_p40_conservation": all(
            "task_relevant_world_world_edges" in value["entity_contact_graph"]
            and value["p40_conservation"]["scope"] == "robot_environment_only"
            and value["p40_conservation"]["world_world_included"] is False
            and set(value["p40_conservation"]["categories"])
            == set(CONTACT_CATEGORIES)
            for value in reports
        ),
        "formal_invalid_counts_zero": all(
            all(value["entity_contact_graph"][name] == 0 for name in INVALID_COUNT_FIELDS)
            for value in reports
        ),
        "all_periods_complete": all(
            value["entity_contact_graph"]["control_period_count"]
            == len(value["enabled_legacy_trace"])
            for value in reports
        ),
        "all_interaction_partitions_published": all(
            set(value["entity_contact_graph"]["interactions"])
            >= {
                "same_entity_dual_arm_substep_count",
                "distinct_entity_dual_arm_substep_count",
                "single_arm_substep_count",
                "same_object_dual_arm_grasp_substep_count",
            }
            for value in reports
        ),
    }


def _build_report(
    source_commit: str,
    command: Sequence[str],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    legacy_unchanged = bool(evaluation["checks"]["legacy_traces_bit_identical"])
    return {
        "schema_version": REPORT_SCHEMA,
        "measurement_schema": MEASUREMENT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "decision": (
            "accepted as entity-contact measurement contract evidence"
            if evaluation["passed"] else "rejected"
        ),
        **CLAIM_FLAGS,
        "legacy_runtime_behavior_unchanged": legacy_unchanged,
        **evaluation,
    }


def _manifest(
    source_commit: str,
    command: Sequence[str],
    binding_identity: Mapping[str, object],
    evaluation: Mapping[str, object] | None,
    artifacts: Mapping[str, bytes],
    *,
    status: str,
) -> dict[str, object]:
    legacy_unchanged = bool(
        evaluation is not None
        and evaluation["checks"]["legacy_traces_bit_identical"]
    )
    return {
        "schema_version": MANIFEST_SCHEMA,
        "measurement_schema": MEASUREMENT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": status,
        "source_commit": source_commit,
        "frozen_parent_commit": FROZEN_PARENT_COMMIT,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "frozen_document_commit_is_ancestor": status == "complete",
        "command": list(command),
        "binding": dict(binding_identity),
        "robot_body_roots": (
            None if evaluation is None else evaluation["robot_body_roots"]
        ),
        "physics": None if evaluation is None else evaluation["physics"],
        "constants": {
            "base_seed": BASE_SEED,
            "seed_stride": SEED_STRIDE,
            "control_step_limit": CONTROL_STEP_LIMIT,
            "task_ids": list(TASK_IDS),
            "robot_body_root_names": dict(ROBOT_BODY_ROOT_NAMES),
        },
        **CLAIM_FLAGS,
        "legacy_runtime_behavior_unchanged": legacy_unchanged,
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for name, content in artifacts.items()
        },
    }


def _require_clean_source(
    root: Path, binding_identity: Mapping[str, object]
) -> None:
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root, check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("P40-E2 runner requires clean committed source")
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
        cwd=root, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("P40-E2 frozen document commit is not an ancestor")
    history = tuple(f"docs/research-loop/{index:04d}" for index in range(1, 8))
    result = subprocess.run(
        ("git", "diff", "--quiet", FROZEN_PARENT_COMMIT, "HEAD", "--", *history),
        cwd=root, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("P40-E2 historical research-loop documents drifted")
    if (
        binding_identity["sha256"] != FROZEN_BINDING_SHA256
        or binding_identity["bytes"] != FROZEN_BINDING_BYTES
    ):
        raise RuntimeError("P40-E2 binding identity differs from the frozen contract")


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root, check=True, capture_output=True, text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P40-E2 runner requires a full Git source commit")
    return commit


def _file_identity(root: Path, path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


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


def _canonical_sha256(value: object) -> str:
    content = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    value = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return value.encode("utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
