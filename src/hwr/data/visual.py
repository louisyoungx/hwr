"""Whitelisted visual-policy tensors and immutable sharded datasets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hwr.core.types import ActionFrame, CameraFrame, ObservationFrame


VISUAL_DATASET_SCHEMA = "hwr.visual-behavior-dataset/v2"
POLICY_INPUT_SCHEMA = "hwr.formal-policy-input/v1"
POLICY_INPUT_FIELDS = frozenset(
    {
        "head_rgb",
        "head_depth",
        "wrist_rgb",
        "proprioception",
        "instruction_id",
        "action_history",
    }
)
SHARD_ARRAYS = frozenset(
    {
        *(f"input__{name}" for name in POLICY_INPUT_FIELDS),
        "label__action",
        "label__phase",
        "step_index",
    }
)
CAMERA_LAYOUT = {
    "head_rgb": "rgb8",
    "head_depth": "depth32f",
    "wrist_rgb": "rgb8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resize_nearest(array: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = array.shape[:2]
    y = np.linspace(0, source_height - 1, height).round().astype(np.int64)
    x = np.linspace(0, source_width - 1, width).round().astype(np.int64)
    return np.ascontiguousarray(array[y][:, x])


def _camera_array(frame: CameraFrame, width: int, height: int) -> np.ndarray:
    if frame.payload is None:
        raise ValueError(f"camera {frame.camera_id} has no online payload")
    if frame.encoding == "rgb8":
        value = np.frombuffer(frame.payload, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    elif frame.encoding == "depth32f":
        value = np.frombuffer(frame.payload, dtype=np.float32).reshape(frame.height, frame.width)
    else:
        raise ValueError(f"unsupported camera encoding: {frame.encoding}")
    return _resize_nearest(value, width, height)


def formal_action_vector(action: ActionFrame) -> np.ndarray:
    values = (
        action.base_linear,
        action.base_angular,
        *action.arm_command,
        action.gripper_target,
    )
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (9,) or not np.isfinite(vector).all():
        raise ValueError("formal action must contain base, six arm, and gripper controls")
    return vector


@dataclass(frozen=True)
class FormalPolicyInput:
    head_rgb: np.ndarray
    head_depth: np.ndarray
    wrist_rgb: np.ndarray
    proprioception: np.ndarray
    instruction_id: np.ndarray
    action_history: np.ndarray

    def named_arrays(self) -> dict[str, np.ndarray]:
        values = {
            "head_rgb": self.head_rgb,
            "head_depth": self.head_depth,
            "wrist_rgb": self.wrist_rgb,
            "proprioception": self.proprioception,
            "instruction_id": self.instruction_id,
            "action_history": self.action_history,
        }
        if frozenset(values) != POLICY_INPUT_FIELDS:
            raise ValueError("formal policy input whitelist changed unexpectedly")
        return values


@dataclass(frozen=True)
class VisualBehaviorSample:
    step_index: int
    policy_input: FormalPolicyInput
    action: np.ndarray
    phase: str = "default"


def extract_formal_policy_input(
    observation: ObservationFrame,
    *,
    instruction_id: int,
    action_history: Sequence[np.ndarray],
    image_width: int,
    image_height: int,
) -> FormalPolicyInput:
    """Project an online observation through the deployment-field whitelist."""
    if observation.task_stage != "instruction_following" or observation.features:
        raise ValueError("privileged task stage or feature entered formal policy input")
    if len(observation.joint_position) != 6 or len(observation.joint_velocity) != 6:
        raise ValueError("formal policy requires six-axis joint proprioception")
    cameras = {frame.camera_id: frame for frame in observation.cameras}
    if set(cameras) != set(CAMERA_LAYOUT):
        raise ValueError("formal policy camera set differs from the frozen whitelist")
    for camera_id, encoding in CAMERA_LAYOUT.items():
        if cameras[camera_id].encoding != encoding:
            raise ValueError(f"camera {camera_id} encoding must be {encoding}")
    history = np.asarray(action_history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 9 or not np.isfinite(history).all():
        raise ValueError("action history must have shape (history, 9)")
    proprioception = np.asarray(
        (
            *observation.joint_position,
            *observation.joint_velocity,
            observation.gripper_position,
            *observation.base_pose,
            *observation.base_twist,
            *observation.imu,
        ),
        dtype=np.float32,
    )
    if proprioception.shape != (24,) or not np.isfinite(proprioception).all():
        raise ValueError("formal proprioception must contain 24 finite values")
    return FormalPolicyInput(
        head_rgb=_camera_array(cameras["head_rgb"], image_width, image_height),
        head_depth=_camera_array(cameras["head_depth"], image_width, image_height),
        wrist_rgb=_camera_array(cameras["wrist_rgb"], image_width, image_height),
        proprioception=proprioception,
        instruction_id=np.asarray([instruction_id], dtype=np.int16),
        action_history=np.ascontiguousarray(history),
    )


class VisualDatasetBuilder:
    """Write one compressed shard per episode, then seal it with hashes."""

    def __init__(
        self,
        root: Path,
        dataset_id: str,
        *,
        task_id: str,
        instruction: str,
        image_size: tuple[int, int],
        action_history: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not dataset_id or not task_id or not instruction:
            raise ValueError("dataset, task, and instruction identities are required")
        if min(*image_size, action_history) <= 0:
            raise ValueError("image size and action history must be positive")
        self.path = root / dataset_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.task_id = task_id
        self.instruction = instruction
        self.image_size = image_size
        self.action_history = action_history
        self.metadata = dict(metadata or {})
        self.shards: list[dict[str, Any]] = []
        self.phase_names: list[str] = []
        self._declared_phase_order: tuple[str, ...] = ()
        self._observed_phases: set[str] = set()
        self._sealed = False

    def declare_phase_order(self, phase_names: Sequence[str]) -> None:
        values = tuple(phase_names)
        if not values or any(not name for name in values) or len(set(values)) != len(values):
            raise ValueError("declared phase order must contain unique non-empty names")
        if self._declared_phase_order and self._declared_phase_order != values:
            raise ValueError("expert phase order changed between episodes")
        self._declared_phase_order = values

    def write_episode(
        self,
        episode_id: str,
        seed: int,
        samples: Sequence[VisualBehaviorSample],
    ) -> Path:
        if self._sealed or not episode_id or not samples:
            raise ValueError("builder must be open and episode samples must be non-empty")
        arrays: dict[str, np.ndarray] = {
            f"input__{name}": np.stack(
                [sample.policy_input.named_arrays()[name] for sample in samples]
            )
            for name in POLICY_INPUT_FIELDS
        }
        arrays["label__action"] = np.stack([sample.action for sample in samples])
        phases = [sample.phase for sample in samples]
        if any(not phase for phase in phases):
            raise ValueError("visual behavior phase labels must be non-empty")
        for phase in phases:
            if self._declared_phase_order and phase not in self._declared_phase_order:
                raise ValueError(f"sample uses undeclared phase: {phase}")
            self._observed_phases.add(phase)
            if not self._declared_phase_order and phase not in self.phase_names:
                self.phase_names.append(phase)
        arrays["label__phase"] = np.asarray(phases, dtype=np.str_)
        arrays["step_index"] = np.asarray([sample.step_index for sample in samples], dtype=np.int32)
        _validate_shard_arrays(arrays, self.image_size, self.action_history)
        filename = f"{episode_id}.npz"
        path = self.path / filename
        np.savez_compressed(path, **arrays)
        self.shards.append(
            {
                "episode_id": episode_id,
                "seed": int(seed),
                "path": filename,
                "sample_count": len(samples),
                "sha256": _sha256(path),
            }
        )
        return path

    def seal(self) -> Path:
        if self._sealed or not self.shards:
            raise ValueError("dataset must contain unsealed episode shards")
        if self._declared_phase_order:
            self.phase_names = [
                name for name in self._declared_phase_order if name in self._observed_phases
            ]
        manifest = {
            "schema_version": VISUAL_DATASET_SCHEMA,
            "policy_input_schema": POLICY_INPUT_SCHEMA,
            "dataset_id": self.path.name,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "policy_input_fields": sorted(POLICY_INPUT_FIELDS),
            "label_fields": ["action", "phase"],
            "phase_names": self.phase_names,
            "image_size": list(self.image_size),
            "action_history": self.action_history,
            "episode_count": len(self.shards),
            "sample_count": sum(item["sample_count"] for item in self.shards),
            "seeds": [item["seed"] for item in self.shards],
            "shards": self.shards,
            "metadata": self.metadata,
        }
        _write_json_atomic(self.path / "manifest.json", manifest)
        self._sealed = True
        return self.path


def _validate_shard_arrays(
    arrays: Mapping[str, np.ndarray],
    image_size: tuple[int, int],
    action_history: int,
) -> None:
    if frozenset(arrays) != SHARD_ARRAYS:
        extras = sorted(set(arrays) - SHARD_ARRAYS)
        missing = sorted(SHARD_ARRAYS - set(arrays))
        raise ValueError(f"visual shard violates whitelist: extras={extras}, missing={missing}")
    count = arrays["label__action"].shape[0]
    width, height = image_size
    expected = {
        "input__head_rgb": (count, height, width, 3),
        "input__head_depth": (count, height, width),
        "input__wrist_rgb": (count, height, width, 3),
        "input__proprioception": (count, 24),
        "input__instruction_id": (count, 1),
        "input__action_history": (count, action_history, 9),
        "label__action": (count, 9),
        "label__phase": (count,),
        "step_index": (count,),
    }
    mismatches = {
        name: (arrays[name].shape, shape)
        for name, shape in expected.items()
        if arrays[name].shape != shape
    }
    if mismatches:
        raise ValueError(f"visual shard tensor shapes are invalid: {mismatches}")
    phases = arrays["label__phase"]
    if phases.dtype.kind != "U" or any(not str(value) for value in phases):
        raise ValueError("visual shard phase labels must be non-empty unicode strings")


def verify_visual_dataset(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VISUAL_DATASET_SCHEMA:
        raise ValueError("visual dataset schema mismatch")
    if frozenset(manifest.get("policy_input_fields", ())) != POLICY_INPUT_FIELDS:
        raise ValueError("visual dataset manifest violates the policy input whitelist")
    if set(manifest.get("label_fields", ())) != {"action", "phase"}:
        raise ValueError("visual dataset label fields are invalid")
    phase_names = tuple(manifest.get("phase_names", ()))
    if not phase_names or len(set(phase_names)) != len(phase_names):
        raise ValueError("visual dataset phase vocabulary is invalid")
    image_size = tuple(int(value) for value in manifest["image_size"])
    history = int(manifest["action_history"])
    for shard in manifest["shards"]:
        shard_path = path / shard["path"]
        if _sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"visual dataset shard checksum mismatch: {shard['path']}")
        with np.load(shard_path, allow_pickle=False) as arrays:
            _validate_shard_arrays(arrays, image_size, history)
            if not set(arrays["label__phase"].tolist()).issubset(phase_names):
                raise ValueError("visual shard uses an undeclared phase label")
    return manifest
