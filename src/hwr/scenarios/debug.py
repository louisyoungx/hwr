"""Small deterministic scenario used to validate the platform runtime."""

from __future__ import annotations

from hwr.sim.specs import Bounds, HouseholdTaskSpec, ObjectSpec, SceneSpec, ZoneSpec


def debug_pick_place_task() -> HouseholdTaskSpec:
    scene = SceneSpec(
        scene_id="debug_pick_place_room/v1",
        bounds=Bounds(0.0, 4.0, 0.0, 3.0),
        robot_start=(0.5, 0.5, 0.0),
        objects=(
            ObjectSpec("sponge-1", "sponge", 1.8, 0.8, 0.05, 0.08, "basket"),
        ),
        zones=(ZoneSpec("basket", 3.2, 2.2, 0.25, ("sponge",)),),
    )
    return HouseholdTaskSpec("debug_pick_place/v1", scene, max_steps=300)

