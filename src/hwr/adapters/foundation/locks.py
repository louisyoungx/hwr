"""Load local foundation model locks without importing their runtimes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hwr.perception.foundation import FoundationModelLock, WeightArtifact


LOCK_SET_SCHEMA = "hwr.foundation-model-lock-set/v1"


@dataclass(frozen=True)
class LockedFoundationModel:
    adapter: str
    local_name: str
    model_root: Path
    model_lock: FoundationModelLock

    def __post_init__(self) -> None:
        if not self.adapter or not self.local_name:
            raise ValueError("foundation adapter and local name are required")
        if Path(self.local_name).name != self.local_name:
            raise ValueError("foundation local name must be one path component")

    @property
    def local_path(self) -> Path:
        return self.model_root / self.local_name

    def verify(self) -> None:
        errors = self.model_lock.verify_local_files(self.model_root)
        if errors:
            raise ValueError(
                f"foundation model {self.model_lock.model_id} failed verification: "
                + ", ".join(errors)
            )


def load_foundation_model_locks(
    lock_path: Path, model_root: Path
) -> dict[str, LockedFoundationModel]:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LOCK_SET_SCHEMA:
        raise ValueError("foundation model lock set schema mismatch")
    models: dict[str, LockedFoundationModel] = {}
    for value in payload.get("models", ()):
        name = str(value["local_name"])
        if name in models:
            raise ValueError("foundation model lock names must be unique")
        lock = FoundationModelLock(
            model_id=str(value["model_id"]),
            revision=str(value["revision"]),
            role=str(value["role"]),
            license_id=str(value["license_id"]),
            output_dimension=int(value["output_dimension"]),
            representation_id=str(value["representation_id"]),
            artifacts=tuple(WeightArtifact(**artifact) for artifact in value["artifacts"]),
        )
        models[name] = LockedFoundationModel(
            adapter=str(value["adapter"]),
            local_name=name,
            model_root=model_root,
            model_lock=lock,
        )
    if not models:
        raise ValueError("foundation model lock set cannot be empty")
    return models
