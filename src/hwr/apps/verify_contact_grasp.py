"""Record a no-weld, bilateral-contact grasp and lift physics smoke trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwr.adapters.mujoco import Mujoco3DBackend, Mujoco3DConfig, run_contact_grasp_trial
from hwr.adapters.mujoco.smoke_trial import ContactTrialFrame
from hwr.render.video import FFmpegWriter, VideoConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("assets/mujoco/mobile_manipulator_smoke.xml"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/contact-grasp-smoke.mp4"),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:  # pragma: no cover - platform font availability
        return ImageFont.load_default(size=size)


def _evidence_image(backend: Mujoco3DBackend, frame: ContactTrialFrame) -> Image.Image:
    width = backend.config.camera_width
    height = backend.config.camera_height
    pixels = np.frombuffer(backend.render_evidence_rgb(), dtype=np.uint8).reshape(height, width, 3)
    view = Image.fromarray(pixels)
    canvas = Image.new("RGB", (width, height + 60), (22, 27, 33))
    canvas.paste(view, (0, 60))
    draw = ImageDraw.Draw(canvas)
    status = "BILATERAL CONTACT" if frame.contact.bilateral else "no bilateral grasp"
    status_color = (38, 166, 105) if frame.contact.bilateral else (169, 86, 25)
    draw.rectangle((0, 0, width, 28), fill=status_color)
    draw.text(
        (10, 5),
        "CONTACT PHYSICS SMOKE - not a formal household task",
        font=_font(15),
        fill=(255, 255, 255),
    )
    detail = (
        f"{frame.stage}  step {frame.observation.sequence_id:03d}  "
        f"object z {frame.object_position[2]:.3f} m  {status}  "
        f"L/R {frame.contact.left_normal_force:.1f}/{frame.contact.right_normal_force:.1f} N"
    )
    draw.text((10, 36), detail, font=_font(13), fill=(232, 237, 242))
    return canvas


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_trial(arguments: argparse.Namespace) -> dict[str, object]:
    model_path = arguments.model.resolve()
    output_path = arguments.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    video_config = VideoConfig(frames_per_second=10, playback_speed=1.0, crf=19)
    writer = FFmpegWriter(
        temporary_video,
        width=arguments.width,
        height=arguments.height + 60,
        config=video_config,
    )
    backend = Mujoco3DBackend(
        Mujoco3DConfig(
            model_path=model_path,
            max_steps=400,
            camera_width=arguments.width,
            camera_height=arguments.height,
        )
    )
    timeline: list[dict[str, object]] = []
    try:
        def capture(frame: ContactTrialFrame) -> None:
            if frame.observation.sequence_id % 2:
                return
            writer.append(_evidence_image(backend, frame))
            timeline.append(
                {
                    "video_frame": len(timeline),
                    "episode_step": frame.observation.sequence_id,
                    "stage": frame.stage,
                    "object_position": list(frame.object_position),
                    "contact": frame.contact.to_dict(),
                    "action_source": "privileged_3d_expert",
                }
            )

        report = run_contact_grasp_trial(
            backend,
            seed=arguments.seed,
            frame_callback=capture,
        )
        writer.close()
        os.replace(temporary_video, output_path)
    except BaseException:
        writer.abort()
        temporary_video.unlink(missing_ok=True)
        raise
    finally:
        backend.close()
    value: dict[str, object] = {
        "schema_version": "hwr.contact-grasp-smoke/v1",
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "physics": report.to_dict(),
        "video": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "frames": len(timeline),
            "frames_per_second": video_config.frames_per_second,
            "width": arguments.width,
            "height": arguments.height + 60,
        },
        "timeline": timeline,
    }
    _write_json_atomic(output_path.with_suffix(".json"), value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    value = run_trial(build_parser().parse_args(argv))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["physics"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
