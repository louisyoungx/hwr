from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from hwr.adapters.mujoco import (  # noqa: E402
    CONTACT_CATEGORIES,
    ROBOT_BODY_ROOT_NAMES,
    ContactLedger,
    ContactPointObservation,
    EntityContactGraph,
    EntityContactGraphError,
    EntityContactPointObservation,
    EntityMotionSource,
    load_default_formal_household_catalogs,
    p40_conservation_differences,
    resolve_robot_part_by_geom,
)
from hwr.apps import evaluate_entity_contact_graph as app  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TASKS, BINDINGS = load_default_formal_household_catalogs(ROOT)


def _graph(
    *,
    enabled: bool = True,
    excluded_initial_periods: int = 1,
    timestep: float = 0.01,
) -> EntityContactGraph:
    return EntityContactGraph(
        all_geom_ids=(1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15),
        robot_part_by_geom={
            1: "base",
            2: "left_arm",
            3: "left_arm",
            4: "right_arm",
            5: "right_arm",
            6: "right_arm",
        },
        entity_by_geom={
            10: "manipulated_object:a",
            11: "manipulated_object:b",
            12: "floor_support:floor",
            13: "target_container:container",
            14: "articulation:drawer",
            15: "forbidden:wall",
        },
        timestep=timestep,
        enabled=enabled,
        excluded_initial_periods=excluded_initial_periods,
        motion_source_by_entity={
            "manipulated_object:a": EntityMotionSource("translation", 10),
            "manipulated_object:b": EntityMotionSource("translation", 11),
            "articulation:drawer": EntityMotionSource("joint", 14),
        },
        gripper_pad_groups={
            "left_arm": ((2,), (3,)),
            "right_arm": ((4,), (5,)),
        },
    )


def _motion(
    *,
    first: tuple[float, float, float] = (0.0, 0.0, 0.0),
    second: tuple[float, float, float] = (0.0, 0.0, 0.0),
    articulation: float = 0.0,
) -> dict[str, object]:
    return {
        "manipulated_object:a": first,
        "manipulated_object:b": second,
        "articulation:drawer": articulation,
    }


def test_frozen_fixture_has_exact_classification_and_conservation() -> None:
    fixture = app._run_fixture()

    assert fixture["passed"] is True
    assert fixture["classification_precision"] == 1.0
    assert fixture["classification_recall"] == 1.0
    assert all(fixture["cases"].values())
    assert all(fixture["fail_closed_cases"].values())
    assert all(fixture["mapping_fail_closed_cases"].values())
    assert fixture["p40_conservation"]["maximum_absolute_difference"] == 0.0
    assert all(value == 0 for value in fixture["valid_fixture_invalid_counts"].values())


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_formal_models_map_every_robot_geom_to_nearest_frozen_root(task_id) -> None:
    binding = BINDINGS[task_id]
    model = mujoco.MjModel.from_xml_path(str(binding.model_path))
    base = int(model.body("robot_base").id)
    robot_geoms = frozenset(
        geom
        for geom in range(model.ngeom)
        if int(model.body_rootid[int(model.geom_bodyid[geom])])
        == int(model.body_rootid[base])
    )

    mapping, roots = resolve_robot_part_by_geom(model, robot_geoms)

    assert set(mapping) == set(robot_geoms)
    assert set(mapping.values()) == set(ROBOT_BODY_ROOT_NAMES)
    assert {
        part: sum(value == part for value in mapping.values())
        for part in ROBOT_BODY_ROOT_NAMES
    } == {"base": 12, "left_arm": 18, "right_arm": 18}
    assert {
        part: value["body_name"] for part, value in roots.items()
    } == ROBOT_BODY_ROOT_NAMES


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_backend_binding_builds_complete_nonoverlapping_entity_mapping(task_id) -> None:
    binding = BINDINGS[task_id]
    model = mujoco.MjModel.from_xml_path(str(binding.model_path))
    base = int(model.body("robot_base").id)
    robot_geoms = frozenset(
        geom
        for geom in range(model.ngeom)
        if int(model.body_rootid[int(model.geom_bodyid[geom])])
        == int(model.body_rootid[base])
    )
    object_geoms = {
        object_id: int(model.geom(value.collision_geom).id)
        for object_id, value in binding.objects.items()
    }
    articulation_joint = (
        int(model.joint(binding.articulation.joint).id)
        if binding.articulation is not None
        else None
    )
    backend = SimpleNamespace(
        model=model,
        binding=binding,
        household_ids=SimpleNamespace(
            robot_geoms=robot_geoms,
            object_geoms=object_geoms,
            articulation_joint=articulation_joint,
        ),
    )

    graph = app._graph_from_backend(backend, enabled=True)
    mapping = graph.mapping_report()
    entities = {item["entity"] for item in mapping["environment_geoms"]}

    assert len(mapping["robot_geoms"]) == len(robot_geoms)
    assert len(mapping["environment_geoms"]) + len(robot_geoms) == model.ngeom
    assert {
        f"manipulated_object:{object_id}" for object_id in binding.objects
    } <= entities
    assert {
        item["entity"] for item in mapping["environment_geoms"]
        if item["entity"].startswith("articulation:")
    } == (
        set()
        if binding.articulation is None
        else {f"articulation:{binding.articulation.articulation_id}"}
    )


def test_unordered_geom_pairs_are_summed_once_and_conserve_legacy_p40() -> None:
    graph = _graph()
    ledger = ContactLedger(
        robot_geoms=(1, 2, 3, 4, 5, 6),
        allowed_role_by_geom={
            10: "manipulated_object",
            11: "manipulated_object",
            12: "floor_support",
            13: "target_container",
            14: "articulation",
        },
        timestep=0.01,
        enabled=True,
    )
    observations = (
        EntityContactPointObservation(2, 10, 2.0),
        EntityContactPointObservation(10, 2, 3.0),
        EntityContactPointObservation(4, 10, 7.0),
        EntityContactPointObservation(1, 12, 5.0),
        EntityContactPointObservation(10, 13, 11.0),
        EntityContactPointObservation(10, 12, 13.0),
    )
    graph.begin_control_period(_motion())
    ledger.begin_control_period()

    graph.record_substep(observations)
    ledger.record_substep(
        ContactPointObservation(value.geom1, value.geom2, value.normal_force)
        for value in observations
    )
    graph.end_control_period(_motion(first=(0.0, 0.0, 0.01)))
    ledger.end_control_period()
    report = graph.report()
    conservation = p40_conservation_differences(report, ledger.report())

    manipulated = report["legacy_p40_categories"]["manipulated_object"]
    assert manipulated["pair_peak_force"] == 7.0
    assert manipulated["category_peak_force"] == 12.0
    assert manipulated["contact_point_count"] == 3
    assert manipulated["unique_pair_observation_count"] == 2
    assert manipulated["cumulative_impulse"] == pytest.approx(0.12)
    assert conservation["maximum_absolute_difference"] == 0.0
    assert conservation["passed"] is True
    assert conservation["scope"] == "robot_environment_only"
    assert conservation["world_world_included"] is False
    assert set(conservation["categories"]) == set(CONTACT_CATEGORIES)
    assert report["task_relevant_world_world_contact_point_count"] == 2
    assert len(report["task_relevant_world_world_edges"]) == 2


def test_long_trace_preserves_period_level_p40_summation_order() -> None:
    graph = _graph(excluded_initial_periods=0, timestep=0.002)
    ledger = ContactLedger(
        robot_geoms=(1, 2, 3, 4, 5, 6),
        allowed_role_by_geom={10: "manipulated_object"},
        timestep=0.002,
        enabled=True,
    )
    for period_index in range(1655):
        graph.begin_control_period(_motion())
        ledger.begin_control_period()
        for substep in range(25):
            force = 1000.0 / (period_index + substep + 1)
            observations = (
                EntityContactPointObservation(2, 10, force),
                EntityContactPointObservation(10, 2, force / 3.0),
            )
            graph.record_substep(observations)
            ledger.record_substep(
                ContactPointObservation(
                    value.geom1, value.geom2, value.normal_force
                )
                for value in observations
            )
        graph.end_control_period(_motion())
        ledger.end_control_period()

    conservation = p40_conservation_differences(graph.report(), ledger.report())

    assert conservation["maximum_absolute_difference"] == 0.0
    assert conservation["passed"] is True


def test_same_distinct_single_and_grasp_qualified_contacts_do_not_mix() -> None:
    graph = _graph()
    graph.begin_control_period(_motion())
    graph.record_substep(
        (
            EntityContactPointObservation(2, 10, 1.0),
            EntityContactPointObservation(3, 10, 1.0),
            EntityContactPointObservation(4, 10, 1.0),
            EntityContactPointObservation(5, 10, 1.0),
        )
    )
    graph.record_substep(
        (
            EntityContactPointObservation(2, 10, 1.0),
            EntityContactPointObservation(4, 11, 1.0),
        )
    )
    graph.record_substep((EntityContactPointObservation(2, 10, 1.0),))
    graph.end_control_period(_motion())

    observations = graph.report()["substeps"]
    assert observations[0]["same_entity_dual_arm_contacts"] == [
        "manipulated_object:a"
    ]
    assert observations[0]["same_object_dual_arm_grasps"] == [
        "manipulated_object:a"
    ]
    assert observations[1]["same_entity_dual_arm_contacts"] == []
    assert observations[1]["distinct_entity_dual_arm_contacts"] == [
        ["manipulated_object:a", "manipulated_object:b"]
    ]
    assert observations[2]["left_only_entities"] == ["manipulated_object:a"]
    assert observations[2]["right_only_entities"] == []


def test_motion_is_associated_only_with_same_period_entity_contact() -> None:
    graph = _graph()
    graph.begin_control_period(_motion())
    graph.record_substep((EntityContactPointObservation(2, 10, 1.0),))
    settling = graph.end_control_period(_motion(first=(0.1, 0.0, 0.0)))
    graph.begin_control_period(_motion(first=(0.1, 0.0, 0.0)))
    graph.record_substep((EntityContactPointObservation(2, 10, 1.0),))
    contact = graph.end_control_period(_motion(first=(0.2, 0.0, 0.0)))
    graph.begin_control_period(_motion(first=(0.2, 0.0, 0.0)))
    graph.record_substep(())
    inertia = graph.end_control_period(_motion(first=(0.3, 0.0, 0.0)))

    entity = "manipulated_object:a"
    assert settling["entity_motion"][entity] == {
        "motion": 0.1,
        "robot_contact_observed": True,
        "reset_settling_excluded": True,
        "contact_associated_motion": 0.0,
    }
    assert settling["reset_settling_excluded"] is True
    assert contact["reset_settling_excluded"] is False
    assert contact["entity_motion"][entity]["reset_settling_excluded"] is False
    assert contact["entity_motion"][entity]["contact_associated_motion"] == (
        pytest.approx(0.1)
    )
    assert inertia["entity_motion"][entity]["contact_associated_motion"] == 0.0
    assert "controlled" not in " ".join(contact["entity_motion"][entity])
    assert graph.report()["contact_associated_motion_exclusion"] == {
        "reason": "reset_settling",
        "rule": "period_index < excluded_initial_periods",
        "excluded_initial_periods": 1,
    }


@pytest.mark.parametrize(
    ("normal_force", "counter"),
    [
        (None, "missing_normal_force_count"),
        (float("nan"), "nonfinite_normal_force_count"),
        (float("inf"), "nonfinite_normal_force_count"),
        (-1.0, "invalid_negative_normal_force_count"),
    ],
)
def test_invalid_force_evidence_fails_closed(normal_force, counter) -> None:
    graph = _graph()
    graph.begin_control_period(_motion())

    with pytest.raises(EntityContactGraphError, match="invalid force"):
        graph.record_substep(
            (EntityContactPointObservation(2, 10, normal_force),)
        )

    assert graph.report()[counter] == 1
    assert graph.report()["contract_valid"] is False


def test_missing_overlap_unknown_part_and_unknown_entity_fail_closed() -> None:
    base = {
        "all_geom_ids": (1, 2, 3, 4),
        "robot_part_by_geom": {1: "base", 2: "left_arm", 3: "right_arm"},
        "entity_by_geom": {4: "forbidden:wall"},
        "timestep": 0.01,
        "enabled": True,
    }
    invalid = (
        {**base, "all_geom_ids": (1, 2, 3, 4, 5)},
        {**base, "entity_by_geom": {1: "forbidden:x", 4: "forbidden:wall"}},
        {
            **base,
            "robot_part_by_geom": {1: "base", 2: "left_arm", 3: "unknown"},
        },
        {**base, "entity_by_geom": {4: "unknown:wall"}},
    )

    for arguments in invalid:
        with pytest.raises(ValueError):
            EntityContactGraph(**arguments)


def test_robot_body_root_identity_fails_closed() -> None:
    graph = _graph()
    arguments = {
        "all_geom_ids": graph.all_geom_ids,
        "robot_part_by_geom": graph.robot_part_by_geom,
        "entity_by_geom": graph.entity_by_geom,
        "timestep": graph.timestep,
        "enabled": True,
        "robot_body_roots": {
            "base": {"body_name": "robot_base", "body_id": 1},
            "left_arm": {"body_name": "wrong", "body_id": 2},
            "right_arm": {"body_name": "right_shoulder_pan_link", "body_id": 3},
        },
    }

    with pytest.raises(ValueError, match="frozen contract"):
        EntityContactGraph(**arguments)


@pytest.mark.parametrize("value", (-1, 1.0, True))
def test_initial_period_exclusion_requires_nonnegative_integer(value) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        _graph(excluded_initial_periods=value)


def test_disabled_graph_is_a_deterministic_zero_measurement() -> None:
    graph = _graph(enabled=False)

    graph.begin_control_period(_motion())
    graph.record_substep((EntityContactPointObservation(2, 10, 4.0),))
    graph.end_control_period(_motion(first=(1.0, 0.0, 0.0)))
    report = graph.report()

    assert report["enabled"] is False
    assert report["physics_substep_count"] == 0
    assert report["control_period_count"] == 0
    assert report["periods"] == []
    assert report["robot_environment_edges"] == []
    assert all(
        value["cumulative_impulse"] == 0.0
        for value in report["legacy_p40_categories"].values()
    )
