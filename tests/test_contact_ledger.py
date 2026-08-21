from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from hwr.adapters.mujoco import (  # noqa: E402
    ALLOWED_CONTACT_ROLES,
    CONTACT_CATEGORIES,
    ContactLedger,
    ContactLedgerError,
    ContactPointObservation,
    load_mujoco_task_bindings,
    run_timestep_stability_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
BINDING_PATH = ROOT / "configs/adapters/mujoco/formal_3d_v1.json"
EXPECTED_ALLOWED = {
    "tidy_living_room_3d/v1": {
        "floor",
        "toy_duck_collision",
        "toy_football_collision",
        "basket_front_collision",
        "basket_back_collision",
        "basket_left_collision",
        "basket_right_collision",
        "basket_bottom_collision",
    },
    "clear_dining_table_3d/v1": {
        "floor",
        "dining_cup_collision",
        "dining_plate_collision",
        "cup_holder",
        "plate_holder",
    },
    "store_kitchen_items_3d/v1": {
        "floor",
        "cleaner_yellow_collision",
        "cleaner_pink_collision",
        "drawer_handle",
        "drawer_bottom",
        "drawer_front",
        "drawer_back",
        "drawer_left",
        "drawer_right",
        "drawer_divider",
    },
}


def _ledger(*, enabled: bool = True) -> ContactLedger:
    return ContactLedger(
        robot_geoms=(1, 2),
        allowed_role_by_geom={10: "floor_support", 11: "manipulated_object"},
        timestep=0.01,
        enabled=enabled,
    )


def _mutated_binding(tmp_path: Path, mutation) -> Path:
    value = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    mutation(value["bindings"][0])
    path = tmp_path / "bindings.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_formal_roles_are_complete_disjoint_and_preserve_legacy_allow_lists() -> None:
    bindings = load_mujoco_task_bindings(BINDING_PATH, root=ROOT)

    assert set(bindings) == set(EXPECTED_ALLOWED)
    for task_id, binding in bindings.items():
        assert set(binding.allowed_robot_contact_roles) == set(ALLOWED_CONTACT_ROLES)
        role_sets = tuple(binding.allowed_robot_contact_roles.values())
        assert sum(len(value) for value in role_sets) == len(set().union(*role_sets))
        assert set().union(*role_sets) == EXPECTED_ALLOWED[task_id]
        assert set(binding.allowed_robot_contact_geoms) == EXPECTED_ALLOWED[task_id]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: item["allowed_robot_contact_roles"].pop("articulation"),
            "role keys",
        ),
        (
            lambda item: item["allowed_robot_contact_roles"].update({"unknown": []}),
            "role keys",
        ),
        (
            lambda item: item["allowed_robot_contact_roles"]["target_container"].append(
                "floor"
            ),
            "overlap",
        ),
        (
            lambda item: item["allowed_robot_contact_roles"]["floor_support"].clear(),
            "union",
        ),
        (
            lambda item: item["allowed_robot_contact_roles"]["floor_support"].append(
                "floor"
            ),
            "duplicates",
        ),
    ],
)
def test_role_contract_rejects_missing_unknown_overlap_union_and_duplicates(
    tmp_path, mutation, message
) -> None:
    path = _mutated_binding(tmp_path, mutation)

    with pytest.raises(ValueError, match=message):
        load_mujoco_task_bindings(path, root=ROOT)


def test_contact_points_aggregate_through_unordered_pairs_then_categories() -> None:
    ledger = _ledger()
    ledger.begin_control_period()
    ledger.record_substep(
        (
            ContactPointObservation(1, 10, 2.0),
            ContactPointObservation(10, 1, 3.0),
            ContactPointObservation(2, 10, 7.0),
            ContactPointObservation(1, 11, 4.0),
            ContactPointObservation(1, 2),
            ContactPointObservation(10, 11),
            ContactPointObservation(1, 12, 5.0),
        )
    )
    ledger.record_substep(
        (
            ContactPointObservation(10, 1, 1.0),
            ContactPointObservation(2, 10, 2.0),
        )
    )
    period = ledger.end_control_period()
    report = ledger.report()

    floor = period["categories"]["floor_support"]
    assert floor == {
        "pair_peak_force": 7.0,
        "category_peak_force": 12.0,
        "category_impulse": pytest.approx(0.15),
        "contact_duration_seconds": pytest.approx(0.02),
        "contact_point_count": 5,
        "unique_pair_observation_count": 4,
    }
    assert period["categories"]["manipulated_object"]["category_impulse"] == (
        pytest.approx(0.04)
    )
    assert period["categories"]["forbidden"]["category_impulse"] == pytest.approx(0.05)
    assert report["categories"]["floor_support"]["cumulative_impulse"] == (
        pytest.approx(0.15)
    )
    assert report["contact_point_count"] == 9
    assert report["robot_environment_contact_point_count"] == 7
    assert report["ignored_robot_self_contact_point_count"] == 1
    assert report["ignored_world_world_contact_point_count"] == 1
    assert report["contract_valid"] is True
    assert set(report["categories"]) == set(CONTACT_CATEGORIES)


@pytest.mark.parametrize(
    ("normal", "counter"),
    [
        (None, "missing_normal_force_count"),
        (float("nan"), "nonfinite_normal_force_count"),
        (float("inf"), "nonfinite_normal_force_count"),
        (-1.0, "invalid_negative_normal_force_count"),
    ],
)
def test_invalid_force_evidence_is_counted_and_fails_closed(normal, counter) -> None:
    ledger = _ledger()
    ledger.begin_control_period()

    with pytest.raises(ContactLedgerError, match="missing, nonfinite, or negative"):
        ledger.record_substep((ContactPointObservation(1, 10, normal),))

    report = ledger.report()
    assert report[counter] == 1
    assert report["contract_valid"] is False


def test_disabled_ledger_is_a_deterministic_zero_trace() -> None:
    ledger = _ledger(enabled=False)

    report = ledger.report()

    assert report["enabled"] is False
    assert report["physics_substep_count"] == 0
    assert report["periods"] == []
    assert all(
        value["cumulative_impulse"] == 0.0
        for value in report["categories"].values()
    )


def test_timestep_halving_fixture_preserves_nonzero_category_impulse() -> None:
    report = run_timestep_stability_fixture()

    assert report["passed"] is True
    assert set(report["relative_impulse_differences"]) == {"floor_support"}
    assert report["maximum_relative_impulse_difference"] <= 0.10
    assert all(
        value["categories"]["floor_support"]["cumulative_impulse"] > 0.0
        for value in report["reports"]
    )
