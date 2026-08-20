from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hwr.apps import evaluate_replay_conditional_information as app
from hwr.data.autonomous_trajectory import (
    ALLOWED_ACTION_SOURCES,
    AUTONOMOUS_TRAJECTORY_SCHEMA,
    TRAJECTORY_ARRAY_FIELDS,
)
from hwr.eval.replay_conditional_information import (
    POWER_BASE_SEED,
    ReplayDataset,
    build_fold_manifest,
    evaluate_replay_conditional_information,
)


def _dataset(*, rows_per_source: int) -> ReplayDataset:
    rng = np.random.default_rng(29)
    task_ids = []
    source_ids = []
    shard_ids = []
    offsets = []
    states = []
    actions = []
    rates = []
    configurations = []
    state_weight = rng.standard_normal((37, 17)) * 0.05
    action_weight = rng.standard_normal((16, 17)) * 0.4
    for task_id in ("task-a/v1", "task-b/v1", "task-c/v1"):
        for source_index in range(6):
            source = f"{task_id}-source-{source_index}"
            state = rng.standard_normal((rows_per_source, 37))
            action = rng.standard_normal((rows_per_source, 16))
            target = (
                state @ state_weight
                + action @ action_weight
                + rng.standard_normal((rows_per_source, 17)) * 0.02
            )
            task_ids.append(np.full(rows_per_source, task_id))
            source_ids.append(np.full(rows_per_source, source))
            shard_ids.append(np.full(rows_per_source, f"{source}-shard"))
            offsets.append(np.arange(rows_per_source))
            states.append(state)
            actions.append(action)
            rates.append(target[:, :16])
            configurations.append(target)
    action = np.concatenate(actions)
    count = len(action)
    return ReplayDataset(
        task_ids=np.concatenate(task_ids),
        source_ids=np.concatenate(source_ids),
        shard_ids=np.concatenate(shard_ids),
        shard_offsets=np.concatenate(offsets),
        state=np.concatenate(states),
        action=action,
        actor_proposal=np.zeros_like(action),
        rate_target=np.concatenate(rates),
        configuration_target=np.concatenate(configurations),
        safety_rewrite=np.zeros(count, bool),
    )


def _arrays(value: float) -> dict[str, np.ndarray]:
    observations = 17
    transitions = 16
    return {
        "rgb_uint8": np.zeros((observations, 3, 1, 1, 3), np.uint8),
        "raw_head_depth_m": np.zeros((observations, 1, 1), np.float32),
        "head_depth_valid": np.zeros((observations, 1, 1), bool),
        "camera_validity": np.ones((observations, 4), bool),
        "frame_timestamps_ns": np.zeros((observations, 4), np.int64),
        "proprioception": np.full((observations, 37), value, np.float32),
        "observation_source_sha256": np.full(observations, "a" * 64),
        "actor_proposal": np.full((transitions, 16), value, np.float32),
        "executed_action": np.full((transitions, 16), value + 0.1, np.float32),
        "reward": np.zeros(transitions, np.float32),
        "terminated": np.zeros(transitions, bool),
        "truncated": np.zeros(transitions, bool),
        "safety_intervention": np.zeros(transitions, np.float32),
        "action_source": np.full(transitions, "random_rl_exploration"),
        "intrinsics": np.zeros((observations, 4, 4), np.float32),
        "robot_from_camera": np.zeros((observations, 4, 4, 4), np.float32),
    }


def _write_replay(
    root: Path, *, corrupt_hash: bool = False, overlapping: bool = False
) -> Path:
    replay = root / "replay/autonomous"
    replay.mkdir(parents=True)
    shards = []
    for slot in range(7):
        arrays = _arrays(float(slot))
        name = f"source-a--sequence-{slot:02d}.npz"
        path = replay / name
        np.savez_compressed(path, **arrays)
        start = slot * 16
        if overlapping and slot == 1:
            start = 8
        shards.append(
            {
                "episode_id": name.removesuffix(".npz"),
                "task_id": "task-a/v1",
                "seed": 1,
                "instruction": "fixture",
                "locale": "zh-CN",
                "environment_version": "fixture/v1",
                "source_commit": "b" * 40,
                "preprocess_fingerprint": "c" * 64,
                "legal_transform_ids": [],
                "metadata": {
                    "sequence_reservoir": {
                        "schema_version": "fixture/v1",
                        "source_episode_id": "source-a",
                        "source_transition_count": 112,
                        "transition_start": start,
                        "transition_stop": start + 16,
                        "slot": slot,
                        "slot_count": 7,
                    }
                },
                "path": name,
                "observation_count": 17,
                "transition_count": 16,
                "sha256": (
                    "0" * 64
                    if corrupt_hash and slot == 0
                    else hashlib.sha256(path.read_bytes()).hexdigest()
                ),
            }
        )
    manifest = {
        "schema_version": AUTONOMOUS_TRAJECTORY_SCHEMA,
        "dataset_id": "autonomous",
        "array_fields": sorted(TRAJECTORY_ARRAY_FIELDS),
        "allowed_action_sources": sorted(ALLOWED_ACTION_SOURCES),
        "episode_count": 7,
        "transition_count": 112,
        "shards": shards,
    }
    (replay / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root


def _arguments(output: Path, *, mode: str = "smoke") -> argparse.Namespace:
    return argparse.Namespace(
        input_run=Path("unused"),
        output=output,
        mode=mode,
        bootstrap_samples=5,
        power_trials=1,
        power_bootstrap_samples=5,
    )


def test_current_frozen_manifest_matches_hash_and_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / app.DEFAULT_INPUT_RUN / app.REPLAY_RELATIVE / "manifest.json"

    assert manifest.stat().st_size == app.EXPECTED_MANIFEST_BYTES
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        app.EXPECTED_MANIFEST_SHA256
    )


def test_loader_verifies_shards_lineage_and_logical_contract_schema(
    tmp_path: Path,
) -> None:
    input_run = _write_replay(tmp_path / "input")

    dataset, identity = app.load_replay_dataset(input_run, formal=False)

    assert isinstance(dataset, ReplayDataset)
    assert len(dataset.state) == 112
    assert identity["contract_schema"] == app.INPUT_CONTRACT_SCHEMA
    assert identity["storage_schema"] == AUTONOMOUS_TRAJECTORY_SCHEMA
    assert identity["source_episode_count"] == 1
    assert identity["shard_count"] == 7


@pytest.mark.parametrize(
    ("corrupt_hash", "overlapping", "message"),
    (
        (True, False, "shard hash differs"),
        (False, True, "ranges overlap"),
    ),
)
def test_loader_fails_closed_on_hash_or_absolute_range_drift(
    tmp_path: Path,
    corrupt_hash: bool,
    overlapping: bool,
    message: str,
) -> None:
    input_run = _write_replay(
        tmp_path / "input",
        corrupt_hash=corrupt_hash,
        overlapping=overlapping,
    )

    with pytest.raises(ValueError, match=message):
        app.load_replay_dataset(input_run, formal=False)


def test_formal_mode_rejects_all_trial_and_bootstrap_overrides(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path / "output", mode="formal")

    with pytest.raises(ValueError, match="rejects trial or bootstrap overrides"):
        app.run(arguments)

    assert not arguments.output.exists()


def test_mechanism_ratio_one_is_not_a_strict_guard_pass() -> None:
    summary = {
        "aggregate": {
            "ratio": 1.0,
            "bootstrap": {"mean_log_ratio_p05": 0.1},
        }
    }

    assert app._guard_pass(summary, 1.0) is False
    assert app._guard_pass({**summary, "aggregate": {**summary["aggregate"], "ratio": 1.02}}, 1.02)


def test_formal_source_failure_happens_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    arguments = argparse.Namespace(
        input_run=Path("unused"),
        output=output,
        mode="formal",
        bootstrap_samples=app.BOOTSTRAP_SAMPLES,
        power_trials=app.POWER_TRIALS,
        power_bootstrap_samples=app.POWER_BOOTSTRAP_SAMPLES,
    )
    monkeypatch.setattr(
        app,
        "_require_formal_source",
        lambda root: (_ for _ in ()).throw(RuntimeError("dirty source")),
    )

    with pytest.raises(RuntimeError, match="dirty source"):
        app.run(arguments)

    assert not output.exists()


def test_exact_pipeline_power_uses_training_only_oracle_calibration() -> None:
    dataset = _dataset(rows_per_source=12)
    folds = build_fold_manifest(dataset)
    _, design = evaluate_replay_conditional_information(
        dataset, folds, bootstrap_samples=5, bootstrap_seed=3
    )

    report = app.run_exact_pipeline_power(
        dataset,
        design,
        trials=1,
        bootstrap_samples=5,
        base_seed=POWER_BASE_SEED,
    )

    assert report["pipeline"] == "exact_nested_source_task_stratified_pipeline"
    assert report["outer_test_used_for_calibration"] is False
    assert set(report["conditions"]) == {
        "zero_action_residual",
        "random_target",
        "planted",
    }
    assert all(
        value["ratio"] == pytest.approx(1.10)
        for value in report["trials"][0]["planted_training_oracle_ratios"]
    )


def test_run_failure_removes_report_and_hashes_failure_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(rows_per_source=8)
    identity = {
        "manifest_sha256": "a" * 64,
        "manifest_bytes": 10,
        "contract_schema": app.INPUT_CONTRACT_SCHEMA,
    }
    output = tmp_path / "output"
    arguments = _arguments(output)
    monkeypatch.setattr(
        app, "load_replay_dataset", lambda input_run, formal: (dataset, identity)
    )
    monkeypatch.setattr(
        app,
        "evaluate_replay_conditional_information",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("probe failed")),
    )

    with pytest.raises(ValueError, match="probe failed"):
        app.run(arguments)

    assert not (output / "report.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert failure["error"] == "probe failed"
    assert set(manifest["artifacts"]) == {"failure.json", "folds.json"}
    for name, identity_value in manifest["artifacts"].items():
        path = output / name
        assert identity_value == {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    assert manifest["claim_boundaries"] == {
        "capability_claim_allowed": False,
        "plant_causality_claim_allowed": False,
        "production_utilization_claim_allowed": False,
        "closed_loop_success_available": False,
    }
