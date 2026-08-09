from __future__ import annotations

from pathlib import Path

from scripts.verify_formal_scenes import verify


ROOT = Path(__file__).resolve().parents[1]


def test_three_formal_scenes_compile_and_meet_visual_physics_contract() -> None:
    result = verify(ROOT / "configs/scenes/formal_3d_v1.json")

    assert result["valid"] is True
    assert result["scene_count"] == 3
    assert all(report["furniture_count"] >= 3 for report in result["reports"])
    assert all(report["manipulated_object_count"] >= 2 for report in result["reports"])
    assert all(report["separate_visual_collision"] for report in result["reports"])


def test_kitchen_drawer_has_no_actuator_or_equality_shortcut() -> None:
    result = verify(ROOT / "configs/scenes/formal_3d_v1.json")
    kitchen = next(report for report in result["reports"] if "kitchen" in report["scene_id"])

    assert kitchen["articulated_furniture_unactuated"] is True
    assert kitchen["compiled"]["equality_constraints"] == 0
