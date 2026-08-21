from __future__ import annotations

import math

import numpy as np
import pytest

from hwr.eval.tool_kinematics import (
    ARM_ORDER,
    CENTRAL_RANGE,
    HALTON_SAMPLE_COUNT,
    JOINTS_PER_ARM,
    STATE_SEED,
    aggregate_task_reports,
    audit_action_isolation,
    frame_invariance_report,
    frozen_decision,
    frozen_state_grid,
    measure_task,
    policy_tool_position,
    recursive_xml_input_identity,
    scrambled_halton,
    state_grid_report,
    summarize_terminals,
    task_arm_replay_status,
    world_site_to_base,
)


JOINT_RANGES = (
    (-3.05, 3.05),
    (-1.75, 1.45),
    (-2.55, 2.55),
    (-3.05, 3.05),
    (-2.10, 2.10),
    (-3.05, 3.05),
)


def _domain():
    return (
        {arm: (0.0,) * JOINTS_PER_ARM for arm in ARM_ORDER},
        {arm: JOINT_RANGES for arm in ARM_ORDER},
    )


def _exact_sites(state):
    return {
        arm: policy_tool_position(state.arm_joint_position(arm), arm)
        for arm in ARM_ORDER
    }


def test_frozen_grid_has_qpos0_single_joint_and_scrambled_halton_states() -> None:
    qpos0, ranges = _domain()

    first = frozen_state_grid(qpos0, ranges)
    second = frozen_state_grid(qpos0, ranges)

    assert first == second
    assert len(first) == 1 + len(ARM_ORDER) * JOINTS_PER_ARM * 2 + HALTON_SAMPLE_COUNT
    assert first[0].state_id == "qpos0"
    single = [value for value in first if value.state_kind == "single_joint"]
    assert len(single) == 24
    for state in single:
        changed = [
            index
            for index, value in enumerate(state.joint_position)
            if value != 0.0
        ]
        assert len(changed) == 1
    central = [
        value for value in first if value.state_kind == "scrambled_halton_central"
    ]
    assert len(central) == HALTON_SAMPLE_COUNT
    assert any(
        state.arm_joint_position("left") != state.arm_joint_position("right")
        for state in central
    )
    for state in central:
        for arm in ARM_ORDER:
            for value, (low, high) in zip(
                state.arm_joint_position(arm), JOINT_RANGES, strict=True
            ):
                assert low + CENTRAL_RANGE[0] * (high - low) <= value
                assert value <= low + CENTRAL_RANGE[1] * (high - low)
    report = state_grid_report(first)
    assert report["state_count"] == 153
    assert report["identity"] == state_grid_report(second)["identity"]


def test_scrambled_halton_is_seeded_unique_and_inside_unit_cube() -> None:
    first = scrambled_halton(128, 6, seed=STATE_SEED)
    replay = scrambled_halton(128, 6, seed=STATE_SEED)
    other = scrambled_halton(128, 6, seed=STATE_SEED + 1)

    assert first == replay
    assert first != other
    assert len(set(first)) == 128
    assert all(0.0 <= coordinate < 1.0 for row in first for coordinate in row)


def test_policy_fk_matches_frozen_zero_pose_geometry() -> None:
    assert policy_tool_position((0.0,) * 6, "left") == pytest.approx(
        (1.025, 0.31, 0.905), abs=1.0e-15
    )
    assert policy_tool_position((0.0,) * 6, "right") == pytest.approx(
        (1.025, -0.31, 0.905), abs=1.0e-15
    )


def test_world_site_conversion_is_invariant_to_translation_and_yaw() -> None:
    base_site = np.asarray((0.82, -0.31, 0.905), dtype=np.float64)
    fixtures = []
    for fixture_id, origin, yaw in (
        ("identity", (0.0, 0.0, 0.22), 0.0),
        ("translation", (1.2, -0.7, 0.53), 0.0),
        ("yaw", (-0.4, 0.8, 0.22), 1.3),
    ):
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        site_world = rotation @ base_site + np.asarray(origin)
        recovered = world_site_to_base(site_world, origin, rotation)
        fixtures.append((fixture_id, {arm: recovered for arm in ARM_ORDER}))

    report = frame_invariance_report(fixtures)

    assert report["passed"] is True
    assert report["max_absolute_error_m"] <= 1.0e-12


def test_measurement_reports_unique_finite_dual_arm_terminals_and_statistics() -> None:
    qpos0, ranges = _domain()
    states = frozen_state_grid(qpos0, ranges)[:4]

    report = measure_task("task-a", states, _exact_sites)

    assert report["planned_state_count"] == 4
    assert report["planned_terminal_count"] == 8
    assert report["terminal_count"] == 8
    assert report["unique_finite_terminals"] is True
    assert {value["arm"] for value in report["terminals"]} == set(ARM_ORDER)
    for arm in ARM_ORDER:
        assert report["by_arm"][arm]["count"] == 4
        assert report["by_arm"][arm]["finite"] is True
        assert report["by_arm"][arm]["euclidean_error_m"]["max"] == 0.0


def test_frozen_decision_and_weakest_task_arm_use_all_terminal_errors() -> None:
    def terminal(error, task, arm, index):
        return {
            "terminal_id": f"{task}|{arm}|{index}",
            "task_id": task,
            "arm": arm,
            "euclidean_error_m": error,
            "absolute_error_m": (error, 0.0, 0.0),
        }

    reports = []
    for task_index, task in enumerate(("a", "b", "c")):
        terminals = [
            terminal(
                0.004 + 0.001 * task_index + 0.001 * (arm == "right"),
                task,
                arm,
                index,
            )
            for arm in ARM_ORDER
            for index in range(4)
        ]
        reports.append(
            {
                "task_id": task,
                "terminals": terminals,
                "by_arm": {
                    arm: summarize_terminals(
                        [value for value in terminals if value["arm"] == arm]
                    )
                    for arm in ARM_ORDER
                },
            }
        )
    aggregate = aggregate_task_reports(reports)
    passing = {"complete": True}

    assert aggregate["all_task_arm_states"]["count"] == 24
    assert aggregate["weakest_task_arm"]["task_id"] == "c"
    assert aggregate["weakest_task_arm"]["arm"] == "right"
    assert (
        frozen_decision(passing, aggregate["all_task_arm_states"])
        == "accepted as FK agreement contract evidence"
    )
    mismatch = summarize_terminals(
        [terminal(0.04, "a", "left", index) for index in range(4)]
    )
    assert (
        frozen_decision(passing, mismatch)
        == "accepted as material FK mismatch evidence"
    )
    assert frozen_decision({"complete": False}, mismatch) == "invalid"


@pytest.mark.parametrize(
    "source, expected_kind",
    (
        (
            "from hwr.core.embodied import DualArmAction as DA\n"
            "wrapped = DA\n"
            "wrapped(0, 0, (), (), 0, 0)\n",
            "forbidden_import",
        ),
        (
            "import hwr.eval.target_selection as selection\n"
            "wrapped = selection.select_candidate_index\n"
            "wrapped(None, ())\n",
            "forbidden_call",
        ),
        (
            "from hwr.adapters.mujoco.dual_arm_backend import "
            "MujocoDualArmBackend as Runtime\n",
            "forbidden_import",
        ),
        (
            "import hwr.adapters.mujoco.dual_arm_backend as runtime\n",
            "forbidden_import",
        ),
        ("backend.apply(frame)\n", "forbidden_call"),
        ("getattr(backend, 'step')()\n", "forbidden_dynamic_lookup"),
        ("backend.__getattribute__('apply')()\n", "forbidden_dynamic_lookup"),
        (
            "import hwr.eval.target_selection as selection\n"
            "consume(selection.primitive_action)\n",
            "forbidden_reference",
        ),
    ),
)
def test_action_isolation_audit_rejects_alias_wrappers_and_calls(
    source: str, expected_kind: str
) -> None:
    report = audit_action_isolation({"synthetic.py": source})

    assert report["passed"] is False
    assert expected_kind in {value["kind"] for value in report["violations"]}


def test_action_isolation_audit_accepts_measurement_only_kinematics() -> None:
    report = audit_action_isolation(
        {
            "measurement.py": (
                "from hwr.eval.target_selection import _tool_position as fk\n"
                "result = fk((0.0,) * 6, 0.31)\n"
                "site = data.site_xpos[site_id]\n"
            )
        }
    )

    assert report["passed"] is True
    assert report["violations"] == []


def test_recursive_xml_identity_tracks_every_include(tmp_path) -> None:
    root = tmp_path.resolve()
    scene = root / "scene/main.xml"
    common = root / "common"
    scene.parent.mkdir()
    common.mkdir()
    scene.write_text(
        '<mujoco><include file="../common/robot.xml"/></mujoco>',
        encoding="utf-8",
    )
    (common / "robot.xml").write_text(
        '<mujocoinclude><include file="defaults.xml"/></mujocoinclude>',
        encoding="utf-8",
    )
    defaults = common / "defaults.xml"
    defaults.write_text("<mujocoinclude/>", encoding="utf-8")

    first = recursive_xml_input_identity(root, scene)
    replay = recursive_xml_input_identity(root, scene)
    defaults.write_text("<mujocoinclude><default/></mujocoinclude>", encoding="utf-8")
    changed = recursive_xml_input_identity(root, scene)

    assert first == replay
    assert [value["path"] for value in first["dependencies"]] == [
        "common/defaults.xml",
        "common/robot.xml",
        "scene/main.xml",
    ]
    assert first["identity"] != changed["identity"]


def test_task_arm_replay_status_is_explicit_and_fail_closed() -> None:
    reports = []
    for task_id in (
        "tidy_living_room_3d/v1",
        "clear_dining_table_3d/v1",
        "store_kitchen_items_3d/v1",
    ):
        terminals = [
            {
                "terminal_id": f"{task_id}|state|{arm}",
                "task_id": task_id,
                "arm": arm,
                "euclidean_error_m": 0.0,
                "absolute_error_m": (0.0, 0.0, 0.0),
            }
            for arm in ARM_ORDER
        ]
        reports.append(
            {
                "task_id": task_id,
                "by_arm": {
                    arm: summarize_terminals(
                        [value for value in terminals if value["arm"] == arm]
                    )
                    for arm in ARM_ORDER
                },
                "terminals": terminals,
            }
        )

    replay = task_arm_replay_status(reports, reports)
    missing = task_arm_replay_status(reports, reports[:-1])

    assert replay["all_bit_identical"] is True
    assert replay["task_arm_count"] == 6
    assert all(value["bit_identical"] for value in replay["task_arms"])
    assert missing["all_bit_identical"] is False
