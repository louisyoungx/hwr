"""Atomic runner snapshots and replay rollback for interrupted formal training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from hwr.data.autonomous_trajectory import (
    ALLOWED_ACTION_SOURCES,
    AUTONOMOUS_TRAJECTORY_SCHEMA,
    TRAJECTORY_ARRAY_FIELDS,
    AppendableAutonomousTrajectoryStore,
)


RECOVERY_SCHEMA = "hwr.foundation-runner-recovery/v2"


@dataclass(frozen=True)
class RestoredRunnerState:
    cycle: int
    rng_state: Mapping[str, Any]
    torch_rng_state: Mapping[str, Any]
    task_sampler: Mapping[str, Any]
    records: tuple[dict[str, Any], ...]
    discarded_observation_sources: tuple[str, ...]


def publish_runner_progress(
    run_path: Path,
    checkpoint_path: Path,
    deployment_path: Path | None,
    causality_report: Path,
    *,
    cycle: int,
    update_count: int,
    rng_state: Mapping[str, Any],
    torch_rng_state: Mapping[str, Any],
    task_sampler: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    replay_manifest: Mapping[str, Any],
    causality_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one recovery-complete checkpoint before advancing latest.json."""
    if cycle <= 0 or update_count <= 0 or not checkpoint_path.is_dir():
        raise ValueError("foundation recovery snapshot identity is invalid")
    state = {
        "cycle": cycle,
        "rng_state": dict(rng_state),
        "torch_rng_state": dict(torch_rng_state),
        "task_sampler": dict(task_sampler),
        "records": [dict(value) for value in records],
    }
    recovery = checkpoint_path / "recovery"
    if recovery.exists():
        raise FileExistsError(recovery)
    temporary = Path(
        tempfile.mkdtemp(prefix=".recovery-", dir=checkpoint_path)
    )
    try:
        torch.save(state, temporary / "runner-state.pt")
        _write_json(temporary / "replay-manifest.json", replay_manifest)
        _write_json(temporary / "causality-manifest.json", causality_manifest)
        _write_records(temporary / "episodes.jsonl", records)
        artifacts = {
            name: {"sha256": _sha256(temporary / name)}
            for name in (
                "runner-state.pt",
                "replay-manifest.json",
                "causality-manifest.json",
                "episodes.jsonl",
            )
        }
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": RECOVERY_SCHEMA,
                "cycle": cycle,
                "update_count": update_count,
                "episode_count": len(records),
                "artifacts": artifacts,
            },
        )
        os.replace(temporary, recovery)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _replace_bytes(recovery / "runner-state.pt", run_path / "runner-state.pt")
    _replace_bytes(recovery / "episodes.jsonl", run_path / "episodes.jsonl")
    latest = {
        "schema_version": "hwr.foundation-online-latest/v1",
        "training_checkpoint": str(checkpoint_path.relative_to(run_path)),
        "action_causality_report": str(causality_report.relative_to(run_path)),
        "action_causality_sha256": _sha256(causality_report),
        "recovery_snapshot": str((recovery / "manifest.json").relative_to(run_path)),
        "recovery_snapshot_sha256": _sha256(recovery / "manifest.json"),
        "episode_count": len(records),
        "update_count": update_count,
    }
    if deployment_path is not None:
        latest["deployment"] = str(deployment_path.relative_to(run_path))
    _write_json_atomic(run_path / "latest.json", latest)
    return latest


def restore_runner_progress(
    run_path: Path,
    checkpoint_path: Path,
    latest: Mapping[str, Any],
    replay_store: AppendableAutonomousTrajectoryStore,
    causality_store: AppendableAutonomousTrajectoryStore,
    *,
    replay_archive: Path,
) -> RestoredRunnerState:
    recovery_manifest = _run_member(run_path, latest.get("recovery_snapshot"))
    expected = checkpoint_path / "recovery/manifest.json"
    if recovery_manifest != expected.resolve():
        raise ValueError("latest recovery snapshot does not belong to its checkpoint")
    if _sha256(recovery_manifest) != latest.get("recovery_snapshot_sha256"):
        raise ValueError("foundation recovery snapshot hash differs")
    manifest = _read_json(recovery_manifest)
    if manifest.get("schema_version") != RECOVERY_SCHEMA:
        raise ValueError("foundation recovery snapshot schema differs")
    recovery = recovery_manifest.parent
    paths = {
        name: recovery / name for name in manifest.get("artifacts", {})
    }
    if set(paths) != {
        "runner-state.pt",
        "replay-manifest.json",
        "causality-manifest.json",
        "episodes.jsonl",
    }:
        raise ValueError("foundation recovery artifact set differs")
    for name, path in paths.items():
        if _sha256(path) != manifest["artifacts"][name].get("sha256"):
            raise ValueError(f"foundation recovery artifact hash differs: {name}")
    replay_snapshot = _read_json(paths["replay-manifest.json"])
    causality_snapshot = _read_json(paths["causality-manifest.json"])
    if causality_store.manifest != causality_snapshot:
        raise ValueError("causality holdout changed after the checkpoint")
    discarded, restored_count = _restore_replay(
        replay_store, replay_snapshot, replay_archive
    )
    state = torch.load(paths["runner-state.pt"], map_location="cpu", weights_only=True)
    records = tuple(_read_records(paths["episodes.jsonl"]))
    if state.get("records") != list(records):
        raise ValueError("runner state and Episode evidence differ")
    if (
        int(state.get("cycle", -1)) != int(manifest.get("cycle", -2))
        or len(records) != int(manifest.get("episode_count", -1))
        or len(records) != int(latest.get("episode_count", -2))
        or int(manifest.get("update_count", -1))
        != int(latest.get("update_count", -2))
    ):
        raise ValueError("foundation recovery progress counters differ")
    _replace_bytes(paths["runner-state.pt"], run_path / "runner-state.pt")
    _replace_bytes(paths["episodes.jsonl"], run_path / "episodes.jsonl")
    _write_json_atomic(
        run_path / "recovery/last-resume.json",
        {
            "schema_version": "hwr.foundation-recovery-event/v1",
            "checkpoint": str(checkpoint_path.relative_to(run_path)),
            "restored_archived_shards": restored_count,
            "discarded_uncheckpointed_shards": discarded,
        },
    )
    _clear_archive(replay_archive)
    return RestoredRunnerState(
        int(state["cycle"]),
        state["rng_state"],
        state["torch_rng_state"],
        state["task_sampler"],
        records,
        tuple(
            source
            for shard in discarded
            for source in shard["observation_source_sha256"]
        ),
    )


def capture_torch_rng_state(device: str | torch.device) -> dict[str, Any]:
    """Capture every global Torch generator used by one training device."""
    kind = torch.device(device).type
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if kind == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("cannot capture unavailable MPS random state")
        state["mps"] = torch.mps.get_rng_state()
    elif kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cannot capture unavailable CUDA random state")
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif kind != "cpu":
        raise ValueError(f"unsupported foundation training device: {kind}")
    return state


def restore_torch_rng_state(
    state: Mapping[str, Any], device: str | torch.device
) -> None:
    """Restore exactly the Torch generators required by one training device."""
    kind = torch.device(device).type
    expected = {"cpu", kind} if kind != "cpu" else {"cpu"}
    if set(state) != expected:
        raise ValueError("foundation Torch RNG state device set differs")
    cpu = state["cpu"]
    if not isinstance(cpu, torch.Tensor):
        raise ValueError("foundation CPU RNG state is invalid")
    torch.set_rng_state(cpu)
    if kind == "mps":
        torch.mps.set_rng_state(state["mps"])
    elif kind == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def clear_replay_archive(path: Path) -> None:
    _clear_archive(path)


def _restore_replay(
    store: AppendableAutonomousTrajectoryStore,
    snapshot: Mapping[str, Any],
    archive: Path,
) -> tuple[list[dict[str, Any]], int]:
    _validate_trajectory_snapshot(store, snapshot)
    expected = {str(shard["path"]): shard for shard in snapshot["shards"]}
    restored_count = 0
    for name, shard in expected.items():
        active = store.path / name
        archived = archive / name
        if not active.is_file() and archived.is_file():
            active.parent.mkdir(parents=True, exist_ok=True)
            os.replace(archived, active)
            restored_count += 1
        if not active.is_file() or _sha256(active) != shard["sha256"]:
            raise ValueError(f"checkpoint replay shard is unavailable: {name}")
    discarded = []
    for shard in store.manifest["shards"]:
        name = str(shard["path"])
        if name in expected:
            continue
        path = store.path / name
        with np.load(path, allow_pickle=False) as arrays:
            sources = [str(value) for value in arrays["observation_source_sha256"]]
        discarded.append(
            {
                "episode_id": str(shard["episode_id"]),
                "path": name,
                "sha256": str(shard["sha256"]),
                "observation_source_sha256": sources,
            }
        )
    _write_json_atomic(store.path / "manifest.json", snapshot)
    store.manifest = dict(snapshot)
    root = store.path.resolve()
    for shard in discarded:
        path = (store.path / shard["path"]).resolve()
        if path.parent != root:
            raise ValueError("discarded replay shard escaped its dataset")
        path.unlink()
    return discarded, restored_count


def _validate_trajectory_snapshot(
    store: AppendableAutonomousTrajectoryStore, snapshot: Mapping[str, Any]
) -> None:
    if (
        snapshot.get("schema_version") != AUTONOMOUS_TRAJECTORY_SCHEMA
        or snapshot.get("dataset_id") != store.manifest.get("dataset_id")
        or frozenset(snapshot.get("array_fields", ())) != TRAJECTORY_ARRAY_FIELDS
        or frozenset(snapshot.get("allowed_action_sources", ()))
        != ALLOWED_ACTION_SOURCES
    ):
        raise ValueError("checkpoint replay manifest contract differs")
    shards = tuple(snapshot.get("shards", ()))
    if (
        int(snapshot.get("episode_count", -1)) != len(shards)
        or int(snapshot.get("transition_count", -1))
        != sum(int(shard["transition_count"]) for shard in shards)
        or len({str(shard["path"]) for shard in shards}) != len(shards)
    ):
        raise ValueError("checkpoint replay manifest counters differ")


def _run_member(run_path: Path, relative: object) -> Path:
    value = Path(str(relative))
    if value.is_absolute():
        raise ValueError("foundation recovery path must be run-relative")
    root = run_path.resolve()
    result = (root / value).resolve()
    if result == root or root not in result.parents:
        raise ValueError("foundation recovery path escaped its run")
    return result


def _clear_archive(path: Path) -> None:
    if not path.exists():
        return
    for value in path.iterdir():
        if not value.is_file() or value.is_symlink():
            raise ValueError("replay recovery archive contains an unexpected entry")
        value.unlink()
    path.rmdir()


def _replace_bytes(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n"
            for value in records
        ),
        encoding="utf-8",
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
