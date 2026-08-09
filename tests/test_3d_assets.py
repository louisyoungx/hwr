from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_3d_assets import verify


ROOT = Path(__file__).resolve().parents[1]


def test_formal_household_assets_are_locked_and_textured() -> None:
    result = verify(
        ROOT / "assets/manifests/household_v1_sources.json",
        ROOT / "assets/manifests/household_v1.lock.json",
    )

    assert result["valid"] is True
    assert result["asset_count"] == 12
    assert result["textured_meshes"] == 12
    assert result["total_faces"] > 50_000


def test_formal_assets_cover_each_scene_and_manipulated_object() -> None:
    source = json.loads(
        (ROOT / "assets/manifests/household_v1_sources.json").read_text()
    )
    ids = {asset["id"] for asset in source["assets"]}

    assert {"living_sofa", "living_tea_table", "living_storage_basket"} <= ids
    assert {"toy_duck", "toy_football"} <= ids
    assert {"dining_table", "dining_chair", "dining_cup", "dining_plate"} <= ids
    assert {"kitchen_cabinet", "cleaner_yellow", "cleaner_pink"} <= ids
