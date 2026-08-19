from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hwr.apps.evaluate_action_input_contribution as contribution_app
import hwr.apps.evaluate_decoder_gain as decoder_app


def test_nonfrozen_invocation_is_rejected_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(decoder_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(decoder_app, "_source_commit", lambda root: "0" * 40)

    with pytest.raises(ValueError, match="invocation differs"):
        decoder_app.run(
            Namespace(
                input_run=tmp_path / "input",
                checkpoint=tmp_path / "checkpoint",
                output=output,
                device="cpu",
            )
        )

    assert not output.exists()


def test_replay_hash_failure_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_run = tmp_path / "input"
    replay_manifest = input_run / "replay/autonomous/manifest.json"
    replay_manifest.parent.mkdir(parents=True)
    replay_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(decoder_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(decoder_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        decoder_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(decoder_app, "_require_checkpoint", lambda path: None)

    with pytest.raises(ValueError, match="Replay manifest identity differs"):
        decoder_app.run(
            Namespace(
                input_run=input_run,
                checkpoint=tmp_path / "checkpoint",
                output=output,
                device="cpu",
            )
        )

    assert not output.exists()


@pytest.mark.parametrize("mismatch", ["manifest", "artifact"])
def test_checkpoint_hash_mismatch_does_not_create_output(
    mismatch: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = checkpoint / "manifest.json"
    artifact = checkpoint / "training-state.pt"
    manifest.write_text("{}\n", encoding="utf-8")
    artifact.write_bytes(b"checkpoint")
    output = tmp_path / "output"
    monkeypatch.setattr(decoder_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(decoder_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        decoder_app, "_require_frozen_invocation", lambda *arguments: None
    )
    expected_manifest = decoder_app._sha256(manifest)
    expected_artifact = decoder_app._sha256(artifact)
    monkeypatch.setattr(
        contribution_app,
        "EXPECTED_CHECKPOINT_MANIFEST",
        "0" * 64 if mismatch == "manifest" else expected_manifest,
    )
    monkeypatch.setattr(
        contribution_app,
        "EXPECTED_CHECKPOINT_ARTIFACT",
        "0" * 64 if mismatch == "artifact" else expected_artifact,
    )

    with pytest.raises(ValueError, match="checkpoint identity differs"):
        decoder_app.run(
            Namespace(
                input_run=tmp_path / "input",
                checkpoint=checkpoint,
                output=output,
                device="cpu",
            )
        )

    assert not output.exists()


def test_window_hash_failure_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Loader:
        def window_metadata(self, index: int) -> dict[str, object]:
            return {
                "episode_id": f"window-{index}",
                "task_id": "task/v1",
                "seed": 1,
                "transition_start": 0,
                "transition_stop": 16,
            }

    output = tmp_path / "output"
    monkeypatch.setattr(decoder_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(decoder_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        decoder_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(decoder_app, "_require_checkpoint", lambda path: None)
    monkeypatch.setattr(
        decoder_app, "_require_replay_manifest", lambda path: None
    )
    monkeypatch.setattr(
        decoder_app,
        "load_frozen_batch_replay_inputs",
        lambda root, input_run, device: SimpleNamespace(training_loader=Loader()),
    )
    monkeypatch.setattr(
        decoder_app,
        "select_source_episode_windows",
        lambda loader, seed: {"source": 0},
    )

    with pytest.raises(ValueError, match="window selection identity differs"):
        decoder_app.run(
            Namespace(
                input_run=tmp_path / "input",
                checkpoint=tmp_path / "checkpoint",
                output=output,
                device="cpu",
            )
        )

    assert not output.exists()


def _patch_lightweight_app_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Namespace, list[str]]:
    input_run = tmp_path / "input"
    replay_manifest = input_run / "replay/autonomous/manifest.json"
    replay_manifest.parent.mkdir(parents=True)
    replay_manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "training-state.pt").write_bytes(b"checkpoint")
    output = tmp_path / "output"
    batch = SimpleNamespace(executed_actions=torch.zeros(1, 16, 1))

    class Loader:
        def window_metadata(self, index: int) -> dict[str, object]:
            return {
                "episode_id": "window-0",
                "task_id": "task/v1",
                "seed": 1,
                "transition_start": 0,
                "transition_stop": 16,
            }

    trainer = SimpleNamespace(
        visual_student=SimpleNamespace(eval=lambda: None),
        world_model=SimpleNamespace(eval=lambda: None),
    )
    events: list[str] = []
    monkeypatch.setattr(decoder_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(decoder_app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        decoder_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(decoder_app, "_require_checkpoint", lambda path: None)
    monkeypatch.setattr(
        decoder_app, "_require_replay_manifest", lambda path: None
    )
    monkeypatch.setattr(
        decoder_app,
        "load_frozen_batch_replay_inputs",
        lambda root, path, device: SimpleNamespace(training_loader=Loader()),
    )
    monkeypatch.setattr(
        decoder_app,
        "select_source_episode_windows",
        lambda loader, seed: {"source-0": 0},
    )
    monkeypatch.setattr(
        decoder_app,
        "_selection_sha256",
        lambda windows: decoder_app.EXPECTED_WINDOW_SELECTION,
    )
    monkeypatch.setattr(
        decoder_app,
        "build_foundation_learning_stack",
        lambda *arguments, **keywords: SimpleNamespace(trainer=trainer),
    )
    monkeypatch.setattr(
        decoder_app, "load_foundation_training_checkpoint", lambda *arguments: None
    )
    monkeypatch.setattr(
        decoder_app,
        "_load_sequence",
        lambda trainer, loader, index: (batch, object()),
    )
    monkeypatch.setattr(
        decoder_app,
        "build_true_decoder_feature",
        lambda *arguments: (
            events.append("true-pass") or torch.zeros(1, 16, 2)
        ),
    )
    monkeypatch.setattr(
        decoder_app,
        "build_decoder_calibration",
        lambda model, features: (
            events.append("calibrate")
            or {
                "schema_version": "hwr.decoder-gain-calibration/v1",
                "transition_count": 384,
                "heads": {},
            }
        ),
    )
    monkeypatch.setattr(
        decoder_app,
        "serialize_decoder_calibration",
        lambda calibration: calibration,
    )
    monkeypatch.setattr(
        decoder_app,
        "deserialize_decoder_calibration",
        lambda value, device: (
            events.append("reload") or value
        ),
    )
    monkeypatch.setattr(
        decoder_app,
        "build_decoder_branches",
        lambda *arguments: (
            events.append("shift-pass")
            or SimpleNamespace(true_feature=torch.zeros(1, 16, 2))
        ),
    )
    monkeypatch.setattr(
        decoder_app,
        "evaluate_decoder_gain",
        lambda *arguments: {"assessment": {"valid": True}},
    )
    monkeypatch.setattr(
        decoder_app,
        "aggregate_decoder_gain",
        lambda reports: {
            "criteria": {"path_segments": 16},
            "assessment": {"decision": "diagnostic_complete"},
        },
    )
    return (
        Namespace(
            input_run=input_run,
            checkpoint=checkpoint,
            output=output,
            device="mps",
        ),
        events,
    )


def test_app_freezes_and_reloads_calibration_before_shift_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, events = _patch_lightweight_app_run(tmp_path, monkeypatch)

    result = decoder_app.run(arguments)

    report = json.loads((arguments.output / "report.json").read_text())
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    assert result["decision"] == "diagnostic_complete"
    assert events == ["true-pass", "calibrate", "reload", "shift-pass"]
    assert report["calibration_sha256"] == decoder_app._sha256(
        arguments.output / "calibration.json"
    )
    assert set(manifest["artifacts"]) == {
        "calibration.json",
        "episodes/source-0.json",
        "report.json",
    }


def test_calibration_first_pass_failure_writes_failure_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, _ = _patch_lightweight_app_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        decoder_app,
        "build_true_decoder_feature",
        lambda *arguments: (_ for _ in ()).throw(
            RuntimeError("synthetic calibration failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic calibration failure"):
        decoder_app.run(arguments)

    failure = json.loads((arguments.output / "failure.json").read_text())
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    assert failure["failure_stage"] == "calibration_true_pass"
    assert failure["completed_episode_count"] == 0
    assert failure["current_source_episode_id"] == "source-0"
    assert failure["current_window"]["episode_id"] == "window-0"
    assert set(manifest["artifacts"]) == {"failure.json"}
