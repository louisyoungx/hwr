"""Compile and verify the current MuJoCo mobile manipulator model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from hwr.adapters.mujoco.model import MujocoModelBundle
from hwr.adapters.mujoco.inspection import inspect_robot_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("assets/mujoco/mobile_manipulator_smoke.xml"),
    )
    parser.add_argument("--report-path", type=Path)
    return parser


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    bundle = MujocoModelBundle.load(arguments.model)
    report = inspect_robot_model(bundle.model)
    value = report.to_dict()
    value["schema_version"] = "hwr.robot-model-report/v1"
    value["model_path"] = str(bundle.model_path)
    if arguments.report_path is not None:
        _write_json_atomic(arguments.report_path, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
