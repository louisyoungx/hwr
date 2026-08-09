"""Render an auditable gallery of the three formal household scene models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwr.adapters.mujoco import render_scene_preview


ROOT = Path(__file__).resolve().parents[3]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:  # pragma: no cover - platform font availability
        return ImageFont.load_default(size=size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image(payload: bytes, width: int, height: int) -> Image.Image:
    pixels = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    return Image.fromarray(pixels, mode="RGB")


def _compose(
    rows: list[tuple[str, Image.Image, Image.Image]], width: int, height: int
) -> Image.Image:
    banner = 52
    label_height = 28
    row_height = height + label_height
    canvas = Image.new("RGB", (width * 2, banner + len(rows) * row_height), (18, 21, 25))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width * 2, banner), fill=(25, 70, 78))
    draw.text((14, 8), "FORMAL 3D SCENE ASSET REVIEW", font=_font(18), fill=(245, 250, 248))
    draw.text(
        (14, 31),
        "Physical reset only - not a trained rollout",
        font=_font(13),
        fill=(190, 226, 221),
    )
    for row, (name, third, head) in enumerate(rows):
        top = banner + row * row_height
        canvas.paste(third, (0, top + label_height))
        canvas.paste(head, (width, top + label_height))
        draw.text((10, top + 5), f"{name} / third-person", font=_font(14), fill=(235, 239, 242))
        draw.text((width + 10, top + 5), f"{name} / onboard head RGB", font=_font(14), fill=(235, 239, 242))
        draw.rectangle((0, top, width * 2 - 1, top + row_height - 1), outline=(80, 95, 104))
        draw.line((width, top, width, top + row_height), fill=(80, 95, 104))
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/scenes/formal_3d_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/formal-scenes.png"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    catalog_path = arguments.catalog.resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows: list[tuple[str, Image.Image, Image.Image]] = []
    scene_reports: list[dict[str, object]] = []
    for scene in catalog["scenes"]:
        model_path = (ROOT / scene["model"]).resolve()
        preview = render_scene_preview(
            model_path, width=arguments.width, height=arguments.height
        )
        name = scene["scene_id"].split("/")[0]
        rows.append(
            (
                name,
                _image(preview.third_person_rgb, preview.width, preview.height),
                _image(preview.head_rgb, preview.width, preview.height),
            )
        )
        scene_reports.append(
            {
                "scene_id": scene["scene_id"],
                "task_id": scene["task_id"],
                "model": str(model_path),
                "model_sha256": _sha256(model_path),
                "simulation_time_seconds": preview.simulation_time,
                "contacts_after_settle": preview.contacts,
            }
        )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    _compose(rows, arguments.width, arguments.height).save(temporary, format="PNG")
    os.replace(temporary, output)
    report: dict[str, object] = {
        "schema_version": "hwr.formal-scene-gallery/v1",
        "catalog": str(catalog_path),
        "catalog_sha256": _sha256(catalog_path),
        "asset_lock_sha256": _sha256(ROOT / "assets/manifests/household_v1.lock.json"),
        "scenes": scene_reports,
        "image": {"path": str(output), "sha256": _sha256(output)},
        "trained_rollout": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
