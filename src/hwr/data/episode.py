"""Immutable Episode recorder with deterministic replay and checksums."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

from hwr.core.types import (
    EpisodeEvent,
    EpisodeMetadata,
    EpisodeResult,
    StepRecord,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class EpisodeRecorder:
    """Append-only writer for a single Episode directory."""

    def __init__(self, root: Path, metadata: EpisodeMetadata) -> None:
        self.path = root / metadata.episode_id
        self.path.mkdir(parents=True, exist_ok=False)
        self._metadata = metadata
        self._steps_path = self.path / "steps.jsonl"
        self._events_path = self.path / "events.jsonl"
        self._steps = self._steps_path.open("x", encoding="utf-8")
        self._events = self._events_path.open("x", encoding="utf-8")
        self._step_count = 0
        self._event_count = 0
        self._last_sequence = -1
        self._closed = False
        _write_atomic(
            self.path / "manifest.json",
            {"metadata": metadata.to_dict(), "status": "recording"},
        )

    def append_step(self, step: StepRecord) -> None:
        self._ensure_open()
        sequence = step.observation.sequence_id
        if sequence <= self._last_sequence:
            raise ValueError("observation sequence must be strictly increasing")
        self._steps.write(_canonical_json(step.to_dict()) + "\n")
        self._steps.flush()
        self._last_sequence = sequence
        self._step_count += 1

    def append_event(self, event: EpisodeEvent) -> None:
        self._ensure_open()
        self._events.write(_canonical_json(event.to_dict()) + "\n")
        self._events.flush()
        self._event_count += 1

    def close(self, result: EpisodeResult) -> Path:
        self._ensure_open()
        self._steps.close()
        self._events.close()
        manifest = {
            "metadata": self._metadata.to_dict(),
            "result": result.to_dict(),
            "status": "complete",
            "step_count": self._step_count,
            "event_count": self._event_count,
            "checksums": {
                "steps.jsonl": _sha256(self._steps_path),
                "events.jsonl": _sha256(self._events_path),
            },
        }
        _write_atomic(self.path / "manifest.json", manifest)
        self._closed = True
        return self.path

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("episode recorder is closed")

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._closed:
            self._steps.close()
            self._events.close()


class EpisodeReader:
    """Read and validate an immutable Episode directory."""

    def __init__(self, path: Path, *, verify_checksums: bool = True) -> None:
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError("episode is not complete")
        if verify_checksums:
            for filename, expected in self.manifest["checksums"].items():
                actual = _sha256(path / filename)
                if actual != expected:
                    raise ValueError(f"checksum mismatch for {filename}")

    @property
    def metadata(self) -> EpisodeMetadata:
        return EpisodeMetadata(**self.manifest["metadata"])

    @property
    def result(self) -> EpisodeResult:
        return EpisodeResult(**self.manifest["result"])

    def steps(self) -> Iterator[StepRecord]:
        with (self.path / "steps.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                yield StepRecord.from_dict(json.loads(line))

    def events(self) -> Iterator[EpisodeEvent]:
        with (self.path / "events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                yield EpisodeEvent.from_dict(json.loads(line))

