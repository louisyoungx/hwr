from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PIL import Image

from hwr.apps.render_formal_scenes import run


def test_formal_scene_gallery_contains_three_physical_scene_rows(tmp_path) -> None:
    output = tmp_path / "formal-scenes.png"
    result = run(
        Namespace(
            catalog=Path("configs/scenes/formal_3d_v1.json"),
            output=output,
            width=120,
            height=80,
        )
    )

    assert result["trained_rollout"] is False
    assert len(result["scenes"]) == 3
    with Image.open(output) as image:
        assert image.size == (240, 52 + 3 * 108)
    assert output.with_suffix(".json").is_file()
