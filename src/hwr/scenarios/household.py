"""Versioned household task specifications used by training benchmarks."""

from __future__ import annotations

from hwr.sim.specs import (
    Bounds,
    HouseholdTaskSpec,
    ObjectSpec,
    ObstacleSpec,
    SceneSpec,
    ZoneSpec,
)


def tidy_table_task() -> HouseholdTaskSpec:
    """Collect two small tabletop items into one storage basket."""
    scene = SceneSpec(
        scene_id="tidy_table_room/v1",
        bounds=Bounds(0.0, 4.2, 0.0, 3.2),
        robot_start=(0.45, 0.55, 0.05),
        start_jitter=0.04,
        objects=(
            ObjectSpec("sponge", "cleaning_item", 1.45, 0.75, 0.05, 0.08, "storage", 0.07),
            ObjectSpec("toy", "small_item", 1.85, 1.35, 0.06, 0.12, "storage", 0.07),
        ),
        zones=(ZoneSpec("storage", 3.45, 2.45, 0.30, ("cleaning_item", "small_item")),),
        obstacles=(ObstacleSpec("cabinet", 0.15, 0.45, 2.45, 3.0),),
    )
    return HouseholdTaskSpec("tidy_table/v1", scene, max_steps=520)


def sort_laundry_task() -> HouseholdTaskSpec:
    """Move light and dark laundry items into separate hampers."""
    scene = SceneSpec(
        scene_id="sort_laundry_room/v1",
        bounds=Bounds(0.0, 4.4, 0.0, 3.4),
        robot_start=(0.55, 1.65, 0.0),
        start_jitter=0.05,
        objects=(
            ObjectSpec("light_sock", "light_laundry", 1.35, 2.25, 0.06, 0.06, "light_hamper", 0.08),
            ObjectSpec("dark_towel", "dark_laundry", 1.45, 0.75, 0.08, 0.14, "dark_hamper", 0.08),
        ),
        zones=(
            ZoneSpec("light_hamper", 3.55, 2.55, 0.32, ("light_laundry",)),
            ZoneSpec("dark_hamper", 3.55, 0.65, 0.32, ("dark_laundry",)),
        ),
        obstacles=(ObstacleSpec("wardrobe", 0.1, 0.4, 0.1, 0.7),),
    )
    return HouseholdTaskSpec("sort_laundry/v1", scene, max_steps=560)


def clear_dishes_task() -> HouseholdTaskSpec:
    """Transfer a cup and plate from a dining area into a dish rack."""
    scene = SceneSpec(
        scene_id="clear_dishes_room/v1",
        bounds=Bounds(0.0, 4.6, 0.0, 3.6),
        robot_start=(0.55, 0.45, 0.1),
        start_jitter=0.04,
        objects=(
            ObjectSpec("cup", "dish", 1.40, 0.85, 0.055, 0.16, "dish_rack", 0.06),
            ObjectSpec("plate", "dish", 1.95, 1.30, 0.075, 0.22, "dish_rack", 0.06),
        ),
        zones=(ZoneSpec("dish_rack", 3.85, 2.75, 0.34, ("dish",)),),
        obstacles=(ObstacleSpec("counter", 0.15, 0.45, 2.65, 3.35),),
    )
    return HouseholdTaskSpec("clear_dishes/v1", scene, max_steps=600)


def household_task_registry() -> dict[str, HouseholdTaskSpec]:
    tasks = (tidy_table_task(), sort_laundry_task(), clear_dishes_task())
    return {task.task_id: task for task in tasks}

