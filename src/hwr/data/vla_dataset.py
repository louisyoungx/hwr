"""Immutable end-to-end VLA behavior dataset without symbolic phase labels."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hwr.core.embodied import ActionChunk, DUAL_ARM_ACTION_DIM
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS, VLAActorInput


VLA_DATASET_SCHEMA = "hwr.vla-behavior-dataset/v1"
VLA_LABEL_FIELDS = frozenset({"action_chunk", "valid_steps"})
FORBIDDEN_SYMBOLIC_FIELDS = frozenset(
    {"phase", "skill_plan", "object_token", "target_token", "instruction_id"}
)


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


def _contains_symbolic_field(value: object) -> bool:
    if isinstance(value, Mapping):
        normalized_keys = (str(key).lower() for key in value)
        if any(
            key == forbidden
            or key.startswith(forbidden + "_")
            or key.endswith("_" + forbidden)
            for key in normalized_keys
            for forbidden in FORBIDDEN_SYMBOLIC_FIELDS
        ):
            return True
        return any(_contains_symbolic_field(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_symbolic_field(item) for item in value)
    return False


def _validate_arrays(arrays: Mapping[str, np.ndarray], action_chunk_size: int) -> None:
    expected_names = {
        *(f"input__{name}" for name in VLA_POLICY_INPUT_FIELDS),
        "label__action_chunk",
        "label__valid_steps",
        "step_index",
    }
    if set(arrays) != expected_names:
        raise ValueError("VLA shard violates the field whitelist")
    count = arrays["label__action_chunk"].shape[0]
    expected_action_shape = (count, action_chunk_size, DUAL_ARM_ACTION_DIM)
    if arrays["label__action_chunk"].shape != expected_action_shape:
        raise ValueError("VLA action chunk tensor shape is invalid")
    valid_steps = arrays["label__valid_steps"]
    if valid_steps.shape != (count,) or np.any(valid_steps < 1) or np.any(
        valid_steps > action_chunk_size
    ):
        raise ValueError("VLA valid action steps are invalid")
    if any(arrays[f"input__{name}"].shape[0] != count for name in VLA_POLICY_INPUT_FIELDS):
        raise ValueError("VLA shard sample counts differ")


@dataclass(frozen=True)
class VLABehaviorSample:
    step_index: int
    actor_input: VLAActorInput
    action_chunk: ActionChunk

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("VLA sample step cannot be negative")


class VLABehaviorDatasetBuilder:
    def __init__(
        self,
        root: Path,
        dataset_id: str,
        *,
        instruction: str,
        action_chunk_size: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not dataset_id or not instruction or action_chunk_size <= 0:
            raise ValueError("dataset identity, instruction, and action chunk are required")
        self.path = root / dataset_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.instruction = " ".join(instruction.split())
        self.action_chunk_size = int(action_chunk_size)
        self.metadata = dict(metadata or {})
        self.shards: list[dict[str, Any]] = []
        self.input_shapes: dict[str, list[int]] | None = None
        self.preprocess_fingerprint: str | None = None
        self.language_encoder_id: str | None = None
        self.language_weights_sha256: str | None = None
        self._sealed = False

    def write_episode(
        self,
        episode_id: str,
        seed: int,
        samples: Sequence[VLABehaviorSample],
    ) -> Path:
        if self._sealed or not episode_id or not samples:
            raise ValueError("builder must be open and episode samples must be non-empty")
        self._check_sample_metadata(samples)
        arrays = {
            f"input__{name}": np.stack(
                [sample.actor_input.named_arrays()[name] for sample in samples]
            )
            for name in VLA_POLICY_INPUT_FIELDS
        }
        arrays["label__action_chunk"] = np.asarray(
            [sample.action_chunk.vectors() for sample in samples], dtype=np.float32
        )
        arrays["label__valid_steps"] = np.asarray(
            [sample.action_chunk.valid_steps for sample in samples], dtype=np.int16
        )
        arrays["step_index"] = np.asarray(
            [sample.step_index for sample in samples], dtype=np.int32
        )
        self._set_or_check_shapes(arrays)
        _validate_arrays(arrays, self.action_chunk_size)
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
        if self._sealed or not self.shards or self.input_shapes is None:
            raise ValueError("dataset must contain unsealed episode shards")
        manifest = {
            "schema_version": VLA_DATASET_SCHEMA,
            "dataset_id": self.path.name,
            "instruction": self.instruction,
            "policy_input_fields": sorted(VLA_POLICY_INPUT_FIELDS),
            "label_fields": sorted(VLA_LABEL_FIELDS),
            "input_shapes": self.input_shapes,
            "action_chunk_size": self.action_chunk_size,
            "action_dim": DUAL_ARM_ACTION_DIM,
            "preprocess_fingerprint": self.preprocess_fingerprint,
            "language_encoder_id": self.language_encoder_id,
            "language_weights_sha256": self.language_weights_sha256,
            "episode_count": len(self.shards),
            "sample_count": sum(item["sample_count"] for item in self.shards),
            "seeds": [item["seed"] for item in self.shards],
            "shards": self.shards,
            "metadata": self.metadata,
        }
        _write_json_atomic(self.path / "manifest.json", manifest)
        self._sealed = True
        return self.path

    def _check_sample_metadata(self, samples: Sequence[VLABehaviorSample]) -> None:
        first = samples[0].actor_input
        expected = (
            first.preprocess_fingerprint,
            first.language_encoder_id,
            first.language_weights_sha256,
        )
        if any(
            (
                sample.actor_input.preprocess_fingerprint,
                sample.actor_input.language_encoder_id,
                sample.actor_input.language_weights_sha256,
            )
            != expected
            for sample in samples
        ):
            raise ValueError("VLA sample preprocessing or language identity changed")
        current = (
            self.preprocess_fingerprint,
            self.language_encoder_id,
            self.language_weights_sha256,
        )
        if self.preprocess_fingerprint is not None and current != expected:
            raise ValueError("VLA episode metadata differs from the dataset")
        (
            self.preprocess_fingerprint,
            self.language_encoder_id,
            self.language_weights_sha256,
        ) = expected
        if any(len(sample.action_chunk.actions) != self.action_chunk_size for sample in samples):
            raise ValueError("VLA action chunk size changed inside the dataset")

    def _set_or_check_shapes(self, arrays: Mapping[str, np.ndarray]) -> None:
        shapes = {
            name: list(arrays[f"input__{name}"].shape[1:])
            for name in VLA_POLICY_INPUT_FIELDS
        }
        if self.input_shapes is None:
            self.input_shapes = shapes
        elif self.input_shapes != shapes:
            raise ValueError("VLA policy input shapes changed between episodes")

def verify_vla_dataset(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VLA_DATASET_SCHEMA:
        raise ValueError("VLA dataset schema mismatch")
    if frozenset(manifest.get("policy_input_fields", ())) != VLA_POLICY_INPUT_FIELDS:
        raise ValueError("VLA dataset violates the Actor input whitelist")
    if frozenset(manifest.get("label_fields", ())) != VLA_LABEL_FIELDS:
        raise ValueError("VLA dataset label fields are invalid")
    if _contains_symbolic_field(manifest):
        raise ValueError("symbolic intermediate metadata entered the VLA dataset")
    action_chunk_size = int(manifest["action_chunk_size"])
    for shard in manifest["shards"]:
        shard_path = path / shard["path"]
        if _sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"VLA dataset checksum mismatch: {shard['path']}")
        with np.load(shard_path, allow_pickle=False) as arrays:
            _validate_arrays(arrays, action_chunk_size)
            for name, shape in manifest["input_shapes"].items():
                if list(arrays[f"input__{name}"].shape[1:]) != shape:
                    raise ValueError(f"VLA input shape changed for {name}")
    return manifest
