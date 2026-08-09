"""Render an auditable four-view smoke image from the MuJoCo 3D backend."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwr.adapters.mujoco import Mujoco3DBackend, Mujoco3DConfig, inspect_robot_model
from hwr.core.types import ActionFrame, CameraFrame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("assets/mujoco/mobile_manipulator_smoke.xml"),
    )
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/3d-smoke.png"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rgb_image(frame: CameraFrame) -> Image.Image:
    pixels = np.frombuffer(frame.payload, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    return Image.fromarray(pixels, mode="RGB")


def _depth_image(frame: CameraFrame) -> Image.Image:
    depth = np.frombuffer(frame.payload, dtype=np.float32).reshape(frame.height, frame.width)
    normalized = 1.0 - np.clip((depth - 0.2) / 4.8, 0.0, 1.0)
    red = (normalized * 255).astype(np.uint8)
    green = (np.sqrt(normalized) * 210).astype(np.uint8)
    blue = ((1.0 - normalized) * 180).astype(np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=-1), mode="RGB")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:  # pragma: no cover - platform font availability
        return ImageFont.load_default(size=size)


def _compose(
    images: tuple[tuple[str, Image.Image], ...],
    *,
    width: int,
    height: int,
) -> Image.Image:
    banner_height = 40
    title_height = 30
    canvas = Image.new(
        "RGB",
        (width * 2, banner_height + (height + title_height) * 2),
        (20, 24, 30),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width * 2, banner_height), fill=(142, 75, 20))
    draw.text(
        (12, 9),
        "3D PHYSICS SMOKE ONLY - not a formal household scene",
        font=_font(16),
        fill=(255, 250, 244),
    )
    font = _font(15)
    for index, (label, image) in enumerate(images):
        column = index % 2
        row = index // 2
        left = column * width
        top = banner_height + row * (height + title_height)
        canvas.paste(image, (left, top + title_height))
        draw.text((left + 10, top + 6), label, font=font, fill=(238, 242, 247))
        draw.rectangle(
            (left, top, left + width - 1, top + height + title_height - 1),
            outline=(96, 107, 120),
        )
    return canvas


def run_render(arguments: argparse.Namespace) -> dict[str, object]:
    model_path = arguments.model.resolve()
    config = Mujoco3DConfig(
        model_path=model_path,
        camera_width=arguments.width,
        camera_height=arguments.height,
    )
    backend = Mujoco3DBackend(config)
    try:
        observation = backend.reset(seed=arguments.seed, task_id=config.task_id)
        for _ in range(8):
            now = observation.timestamp_ns
            action = ActionFrame(
                now,
                now,
                now + 100_000_000,
                "smoke-control",
                base_linear=0.10,
                arm_command=(0.0,) * 6,
            )
            observation = backend.apply(action).observation
        frame_by_id = {frame.camera_id: frame for frame in observation.cameras}
        evidence = CameraFrame(
            camera_id="third_person",
            timestamp_ns=observation.timestamp_ns,
            frame_index=observation.sequence_id,
            width=arguments.width,
            height=arguments.height,
            payload=backend.render_evidence_rgb(),
        )
        report = inspect_robot_model(backend.model)
    finally:
        backend.close()
    images = (
        ("Third-person (not a policy input)", _rgb_image(evidence)),
        ("Head RGB", _rgb_image(frame_by_id["head_rgb"])),
        ("Head depth (metres)", _depth_image(frame_by_id["head_depth"])),
        ("Wrist RGB", _rgb_image(frame_by_id["wrist_rgb"])),
    )
    output_path = arguments.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    _compose(images, width=arguments.width, height=arguments.height).save(
        temporary_path,
        format="PNG",
    )
    os.replace(temporary_path, output_path)
    metadata: dict[str, object] = {
        "schema_version": "hwr.3d-smoke-evidence/v1",
        "engine": {"name": "mujoco", "version": importlib.metadata.version("mujoco")},
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "seed": arguments.seed,
        "simulation_time_seconds": observation.timestamp_ns / 1_000_000_000,
        "policy_camera_ids": [frame.camera_id for frame in observation.cameras],
        "policy_features": dict(observation.features),
        "robot_model": report.to_dict(),
        "image": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "width": arguments.width * 2,
            "height": 40 + (arguments.height + 30) * 2,
        },
    }
    _write_json_atomic(output_path.with_suffix(".json"), metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    result = run_render(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
