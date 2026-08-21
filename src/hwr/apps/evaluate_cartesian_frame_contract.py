"""Evaluate the frozen R0001-P51 Cartesian acquisition-to-base contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.eval.target_selection import acquisition_error_to_base_velocity


MODULE_NAME = "hwr.apps.evaluate_cartesian_frame_contract"
PROPOSAL_ID = "R0001-P51"
REPORT_SCHEMA = "hwr.p51-cartesian-frame-contract-report/v1"
MANIFEST_SCHEMA = "hwr.p51-cartesian-frame-contract-artifacts/v1"
FAILURE_SCHEMA = "hwr.p51-cartesian-frame-contract-failure/v1"
FROZEN_DOCUMENT_COMMIT = "4385ceee2fffcbd23788b498d258747dc273465c"
VELOCITY_MAX = 0.08
ACQUISITION_YAWS = (0.0, math.pi / 3.0, -math.pi / 2.0)
RELATIVE_YAWS = (
    0.0,
    math.pi / 6.0,
    -math.pi / 6.0,
    math.pi / 2.0,
    -math.pi / 2.0,
    math.pi,
)
ACQUISITION_ERRORS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.3, -0.4, 0.5),
)
ARMS = ("left", "right")
TOLERANCE = 1e-12
CLAIM_FLAGS = {
    "training_executed": False,
    "policy_inference_executed": False,
    "closed_loop_physics_executed": False,
    "capability_claim_allowed": False,
    "task_success_claim_allowed": False,
    "generalization_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
}
UNCHANGED_FLAGS = {
    "candidate_generator_changed": False,
    "candidate_bytes_changed": False,
    "selector_changed": False,
    "acquisition_changed": False,
    "phase_changed": False,
    "target_changed": False,
    "velocity_cap_changed": False,
    "gripper_changed": False,
    "backend_changed": False,
    "safety_changed": False,
}
SOURCE_PATHS = (
    Path("src/hwr/eval/target_selection.py"),
    Path("src/hwr/apps/evaluate_cartesian_frame_contract.py"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--output",
        str(arguments.output),
    ]
    identities = _source_identities(root)
    try:
        _require_clean_source(root)
        evaluation = evaluate_contract()
        report = _build_report(source_commit, command, evaluation)
        artifacts = {"report.json": _json_bytes(report)}
        manifest = _manifest(
            source_commit,
            command,
            identities,
            artifacts,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "source_commit": source_commit,
            "decision": "invalid",
            "error_type": type(error).__name__,
            "error": str(error),
            **CLAIM_FLAGS,
            **UNCHANGED_FLAGS,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _manifest(
            source_commit,
            command,
            identities,
            artifacts,
            status="failed",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
        raise
    return {
        "output": str(output),
        "decision": report["decision"],
        "cell_count": evaluation["cell_count"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest(),
    }


def evaluate_contract() -> dict[str, object]:
    cells = []
    for acquisition_yaw in ACQUISITION_YAWS:
        for relative_yaw in RELATIVE_YAWS:
            current_base_yaw = acquisition_yaw + relative_yaw
            for acquisition_error in ACQUISITION_ERRORS:
                expected = _clip_norm(
                    2.0 * np.asarray(acquisition_error, np.float64),
                    VELOCITY_MAX,
                )
                legacy = expected.copy()
                candidate = acquisition_error_to_base_velocity(
                    acquisition_error,
                    VELOCITY_MAX,
                    acquisition_yaw=acquisition_yaw,
                    current_base_yaw=current_base_yaw,
                )
                for arm in ARMS:
                    cells.append(
                        _cell(
                            arm,
                            acquisition_yaw,
                            relative_yaw,
                            acquisition_error,
                            expected,
                            legacy,
                            candidate,
                        )
                    )
    candidate_errors = [cell["candidate"]["errors"] for cell in cells]
    counterexamples = [
        cell
        for cell in cells
        if abs(abs(cell["relative_yaw"]) - math.pi / 2.0) <= TOLERANCE
        and np.linalg.norm(cell["acquisition_error"][:2]) > 0.0
    ]
    expected_count = (
        len(ACQUISITION_YAWS)
        * len(RELATIVE_YAWS)
        * len(ACQUISITION_ERRORS)
        * len(ARMS)
    )
    maxima = {
        name: max(float(errors[name]) for errors in candidate_errors)
        for name in ("maximum_absolute", "angular", "norm", "z")
    }
    checks = {
        "matrix_complete": len(cells) == expected_count,
        "both_arms_covered": {cell["arm"] for cell in cells} == set(ARMS),
        "all_values_finite": all(_cell_is_finite(cell) for cell in cells),
        "zero_relative_yaw_float64_bytes_identical": all(
            cell["candidate_legacy_float64_bytes_identical"]
            for cell in cells
            if cell["relative_yaw"] == 0.0
        ),
        "candidate_maximum_absolute_error_within_tolerance": (
            maxima["maximum_absolute"] <= TOLERANCE
        ),
        "candidate_angular_error_within_tolerance": (
            maxima["angular"] <= TOLERANCE
        ),
        "candidate_norm_error_within_tolerance": maxima["norm"] <= TOLERANCE,
        "candidate_z_error_within_tolerance": maxima["z"] <= TOLERANCE,
        "legacy_quarter_turn_counterexamples_rejected": bool(counterexamples)
        and all(
            cell["legacy"]["errors"]["angular"] >= math.pi / 2.0 - TOLERANCE
            for cell in counterexamples
        ),
    }
    return {
        "formula": {
            "error": "e_A = target_A - tool_A",
            "rotation": "R_B_from_A = Rz(theta_A - theta_B)",
            "candidate": "v_B = R_B_from_A * clip_norm(2 * e_A, velocity_max)",
            "legacy": "v_B_legacy = clip_norm(2 * e_A, velocity_max)",
            "relative_yaw_definition": "theta_B - theta_A",
            "xy_only_rotation": True,
            "rotation_after_norm_clipping": True,
        },
        "matrix": {
            "acquisition_yaws": list(ACQUISITION_YAWS),
            "relative_yaws": list(RELATIVE_YAWS),
            "acquisition_errors": [list(value) for value in ACQUISITION_ERRORS],
            "arms": list(ARMS),
            "velocity_max": VELOCITY_MAX,
            "tolerance": TOLERANCE,
        },
        "cell_count": len(cells),
        "expected_cell_count": expected_count,
        "legacy_counterexample_count": len(counterexamples),
        "candidate_error_maxima": maxima,
        "checks": checks,
        "passed": all(checks.values()),
        "cells": cells,
    }


def _cell(
    arm: str,
    acquisition_yaw: float,
    relative_yaw: float,
    acquisition_error: Sequence[float],
    expected: np.ndarray,
    legacy: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    legacy_realized = _base_to_acquisition(legacy, relative_yaw)
    candidate_realized = _base_to_acquisition(candidate, relative_yaw)
    return {
        "arm": arm,
        "acquisition_yaw": acquisition_yaw,
        "current_base_yaw": acquisition_yaw + relative_yaw,
        "relative_yaw": relative_yaw,
        "acquisition_error": list(acquisition_error),
        "expected_acquisition_velocity": expected.tolist(),
        "legacy": _command_record(legacy, legacy_realized, expected),
        "candidate": _command_record(candidate, candidate_realized, expected),
        "candidate_legacy_float64_bytes_identical": (
            candidate.astype("<f8").tobytes() == legacy.astype("<f8").tobytes()
        ),
    }


def _command_record(
    command: np.ndarray,
    realized: np.ndarray,
    expected: np.ndarray,
) -> dict[str, object]:
    return {
        "base_frame_command": command.tolist(),
        "base_frame_command_float64_sha256": hashlib.sha256(
            command.astype("<f8").tobytes()
        ).hexdigest(),
        "realized_acquisition_frame_vector": realized.tolist(),
        "errors": {
            "maximum_absolute": float(np.max(np.abs(realized - expected))),
            "angular": _horizontal_angular_error(realized, expected),
            "norm": abs(float(np.linalg.norm(realized) - np.linalg.norm(expected))),
            "z": abs(float(realized[2] - expected[2])),
        },
    }


def _base_to_acquisition(vector: np.ndarray, relative_yaw: float) -> np.ndarray:
    if relative_yaw == 0.0:
        return vector.copy()
    cosine, sine = math.cos(relative_yaw), math.sin(relative_yaw)
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            vector[2],
        ),
        np.float64,
    )


def _horizontal_angular_error(actual: np.ndarray, expected: np.ndarray) -> float:
    cross = float(actual[0] * expected[1] - actual[1] * expected[0])
    dot = float(np.dot(actual[:2], expected[:2]))
    return abs(math.atan2(cross, dot))


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum or norm == 0.0 else vector * maximum / norm


def _cell_is_finite(cell: Mapping[str, object]) -> bool:
    values = (
        *cell["legacy"]["base_frame_command"],
        *cell["legacy"]["realized_acquisition_frame_vector"],
        *cell["legacy"]["errors"].values(),
        *cell["candidate"]["base_frame_command"],
        *cell["candidate"]["realized_acquisition_frame_vector"],
        *cell["candidate"]["errors"].values(),
    )
    return all(math.isfinite(float(value)) for value in values)


def _build_report(
    source_commit: str,
    command: Sequence[str],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "decision": (
            "accepted as Cartesian primitive correctness evidence"
            if evaluation["passed"]
            else "rejected"
        ),
        "evidence_scope": "deterministic analytic coordinate contract only",
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
        **evaluation,
    }


def _manifest(
    source_commit: str,
    command: Sequence[str],
    source_identities: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    *,
    status: str,
) -> dict[str, object]:
    configuration = {
        "acquisition_yaws": list(ACQUISITION_YAWS),
        "relative_yaws": list(RELATIVE_YAWS),
        "acquisition_errors": [list(value) for value in ACQUISITION_ERRORS],
        "arms": list(ARMS),
        "velocity_max": VELOCITY_MAX,
        "tolerance": TOLERANCE,
    }
    configuration_bytes = json.dumps(
        configuration, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": status,
        "source_commit": source_commit,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "command": list(command),
        "configuration": {
            **configuration,
            "sha256": hashlib.sha256(configuration_bytes).hexdigest(),
            "bytes": len(configuration_bytes),
        },
        "model": {"executed": False, "identity": None},
        "source_files": dict(source_identities),
        **CLAIM_FLAGS,
        **UNCHANGED_FLAGS,
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for name, content in artifacts.items()
        },
    }


def _source_identities(root: Path) -> dict[str, object]:
    return {
        path.as_posix(): _file_identity(root / path)
        for path in SOURCE_PATHS
    }


def _file_identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(Path(__file__).resolve().parents[3]).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _require_clean_source(root: Path) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P51 runner requires clean committed source")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("P51 frozen document commit is not an ancestor")
    history = tuple(
        f"docs/research-loop/{index:04d}" for index in range(1, 9)
    )
    unchanged = subprocess.run(
        ("git", "diff", "--quiet", FROZEN_DOCUMENT_COMMIT, "HEAD", "--", *history),
        cwd=root,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("P51 historical research-loop documents drifted")


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P51 runner requires a full Git source commit")
    return commit


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir()
    try:
        for name, content in artifacts.items():
            _atomic_write(staging / name, content)
        os.replace(staging, output)
    except BaseException:
        for path in staging.glob("*"):
            path.unlink()
        if staging.exists():
            staging.rmdir()
        raise


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
