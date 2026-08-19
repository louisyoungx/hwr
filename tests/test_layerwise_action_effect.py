from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hwr.apps.evaluate_action_input_contribution as contribution_app
import hwr.apps.evaluate_layerwise_action_effect as layerwise_app
from hwr.apps.evaluate_layerwise_action_effect import build_parser
from hwr.eval.layerwise_action_effect import (
    ACTION_SHIFTS,
    EFFECT_DENOMINATOR_MINIMUM,
    STAGE_NAMES,
    _assess_shift,
    _effect_report,
    _gain,
    _gru_stages,
    aggregate_layerwise_action_effect,
    assess_layerwise_action_effect_episode,
    evaluate_layerwise_action_effect,
)
from hwr.world_model import ActionConditionedWorldModel, WorldModelConfig


def _fixture(device: str = "cpu"):
    config = WorldModelConfig(
        visual_dimension=8,
        language_dimension=6,
        proprioception_dimension=5,
        action_dimension=3,
        observation_embedding_dimension=16,
        deterministic_dimension=16,
        stochastic_variables=4,
        stochastic_classes=4,
        hidden_dimension=32,
        prior_ensemble=3,
        reward_bins=21,
        action_minimum=(-0.1, -0.5, 0.0),
        action_maximum=(0.1, 0.5, 1.0),
        formal=False,
    )
    model = ActionConditionedWorldModel(config).to(device)
    model.eval()
    transitions = 16
    visual = torch.randn(1, transitions + 1, 8, device=device)
    language = torch.randn(1, 6, device=device)
    proprioception = torch.randn(1, transitions + 1, 5, device=device)
    proposals = torch.zeros(1, transitions, 3, device=device)
    actions = torch.empty(1, transitions, 3, device=device).uniform_(-0.1, 0.1)
    actions[..., 1] *= 5.0
    actions[..., 2] = torch.rand(1, transitions, device=device)
    observed = model.observe(
        visual, language, proprioception, proposals, actions
    )
    return model, observed.sequence, actions


def test_gru_gate_reconstruction_matches_torch_cell() -> None:
    cell = torch.nn.GRUCell(7, 5)
    inputs = torch.randn(2, 11, 7)
    hidden = torch.randn(2, 11, 5)

    stages = _gru_stages(cell, inputs, hidden)
    reference = cell(
        inputs.flatten(0, 1), hidden.flatten(0, 1)
    ).reshape(2, 11, 5)

    torch.testing.assert_close(stages["output"], reference)
    assert stages["maximum_absolute_error"] <= 1.0e-6
    assert set(stages) == {
        "reset",
        "update",
        "new",
        "output",
        "maximum_absolute_error",
    }


def test_action_shifts_are_fixed_and_have_no_position_fixed_points() -> None:
    indices = torch.arange(16)

    assert ACTION_SHIFTS == (1, 5, 9)
    for shift in ACTION_SHIFTS:
        shifted = torch.roll(indices, shifts=shift)
        assert not torch.eq(indices, shifted).any()
        assert set(shifted.tolist()) == set(indices.tolist())


def test_default_device_is_frozen_to_mps() -> None:
    arguments = build_parser().parse_args(())

    assert arguments.device == "mps"


def test_effect_report_rejects_inactive_standardization() -> None:
    constant = torch.ones(1, 16, 8)
    changed = constant + 1.0

    report = _effect_report(constant, changed)

    assert report["raw_paired_rms"] == 1.0
    assert report["active_dimension_count"] == 0
    assert report["standardized_effect"] is None


def test_episode_pass_does_not_require_location_consensus() -> None:
    report = {
        "shifts": {
            "1": {
                "assessment": {
                    "passed": True,
                    "first_low_retention": "activation_to_h",
                }
            },
            "5": {
                "assessment": {
                    "passed": True,
                    "first_low_retention": "h_to_prior",
                }
            },
            "9": {
                "assessment": {
                    "passed": False,
                    "first_low_retention": None,
                }
            },
        }
    }

    assessment = assess_layerwise_action_effect_episode(report)

    assert assessment["passed"]
    assert assessment["shift_pass_count"] == 2
    assert assessment["consensus_first_low_retention"] is None


def test_layerwise_report_is_finite_and_gru_consistent() -> None:
    model, sequence, actions = _fixture()

    report = evaluate_layerwise_action_effect(model, sequence, actions)

    assert report["transition_count"] == 16
    assert set(report["shifts"]) == {"1", "5", "9"}
    assert set(report["gru_gate_distributions"]) == {"reset", "update", "new"}
    assert report["gru_gate_distributions"]["update"]["finite"]
    assert 0.0 <= report["gru_gate_distributions"]["update"]["minimum"]
    assert report["gru_gate_distributions"]["update"]["maximum"] <= 1.0
    assert report["criteria"]["expected_transition_count"] == 16
    assert report["criteria"]["effect_denominator_minimum"] == 1.0e-6
    assert report["criteria"]["aggregate_episode_pass_minimum"] == 20
    for shift in report["shifts"].values():
        assert shift["gru_maximum_absolute_error"] <= 1.0e-5
        assert set(shift["stage_effects"]) == {
            "transition_preactivation",
            "transition_normalized",
            "transition_activation",
            "gru_reset_gate",
            "gru_update_gate",
            "gru_new_gate",
            "next_deterministic",
            "prior_hidden",
            "prior_logits",
            "prior_probability",
        }
        assert shift["stage_effects"]["transition_activation"]["finite"]
        assert shift["local_sensitivity"]["epsilon"] == 0.05


def test_non_sixteen_transition_input_is_rejected() -> None:
    model, sequence, actions = _fixture()

    with pytest.raises(ValueError, match="input shapes are invalid"):
        evaluate_layerwise_action_effect(
            model,
            type(sequence)(
                sequence.deterministic[:, :-1],
                sequence.stochastic[:, :-1],
                sequence.prior_logits[:, :-1],
                sequence.posterior_logits[:, :-1],
                sequence.ensemble_prior_logits[:, :-1],
            ),
            actions[:, :-1],
        )


def test_model_structure_drift_is_rejected() -> None:
    model, sequence, actions = _fixture()
    model.rssm.transition_input[2] = torch.nn.ReLU()

    with pytest.raises(ValueError, match="RSSM structure differs"):
        evaluate_layerwise_action_effect(model, sequence, actions)


def test_constant_actions_fail_before_transition_effect() -> None:
    model, sequence, actions = _fixture()
    constant = torch.zeros_like(actions)

    report = evaluate_layerwise_action_effect(model, sequence, constant)

    assert not report["assessment"]["passed"]
    for shift in report["shifts"].values():
        assert (
            shift["stage_effects"]["transition_activation"]["raw_paired_rms"]
            == 0.0
        )
        assert not shift["assessment"]["checks"][
            "transition_effect_at_least_0_05"
        ]
        assert not shift["assessment"]["checks"]["local_sensitivity_valid"]


def _shift_assessment_fixture() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    stage_effects = {
        name: {
            "finite": True,
            "active_fraction": 1.0,
            "standardized_effect": 1.0,
        }
        for name in STAGE_NAMES
    }
    local = {
        "action_input": {"active_fraction": 1.0},
        "deterministic_input": {"active_fraction": 1.0},
        "next_deterministic": {
            "action_to_deterministic_gain_ratio": 0.1,
        },
        "prior_probability": {
            "action_to_deterministic_gain_ratio": 0.1,
        },
        "valid": True,
    }
    return stage_effects, local


def test_invalid_h_to_prior_denominator_invalidates_shift() -> None:
    stage_effects, local = _shift_assessment_fixture()
    stage_effects["next_deterministic"]["standardized_effect"] = 1.0e-8

    _, _, assessment = _assess_shift(
        stage_effects, local, (0.0, 0.0, 0.0, 0.0)
    )

    assert not assessment["passed"]
    assert not assessment["checks"]["retention_denominators_valid"]
    assert assessment["first_low_retention"] is None


def test_nonfinite_gru_error_invalidates_shift() -> None:
    stage_effects, local = _shift_assessment_fixture()
    stage_effects["next_deterministic"]["standardized_effect"] = 0.25

    _, maximum_error, assessment = _assess_shift(
        stage_effects, local, (0.0, float("nan"), 0.0, 0.0)
    )

    assert maximum_error is None
    assert not assessment["passed"]
    assert not assessment["checks"]["gru_errors_finite"]
    assert not assessment["checks"]["gru_matches_torch"]


def test_subminimum_gain_denominator_is_rejected_without_epsilon() -> None:
    denominator = EFFECT_DENOMINATOR_MINIMUM * 0.5

    assert denominator > 0.0
    assert _gain(1.0, denominator) is None


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_layerwise_report_supports_mps() -> None:
    model, sequence, actions = _fixture("mps")

    report = evaluate_layerwise_action_effect(model, sequence, actions)

    assert report["transition_count"] == 16
    assert all(
        shift["gru_maximum_absolute_error"] <= 1.0e-5
        for shift in report["shifts"].values()
    )


def test_replay_hash_failure_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_run = tmp_path / "input"
    manifest = input_run / "replay/autonomous/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(layerwise_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(layerwise_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        layerwise_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(layerwise_app, "_require_checkpoint", lambda path: None)

    with pytest.raises(ValueError, match="Replay manifest identity differs"):
        layerwise_app.run(
            Namespace(
                input_run=input_run,
                checkpoint=tmp_path / "checkpoint",
                output=output,
                device="cpu",
            )
        )

    assert not output.exists()


def test_nonfrozen_invocation_is_rejected_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(layerwise_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(layerwise_app, "_source_commit", lambda root: "0" * 40)

    with pytest.raises(ValueError, match="invocation differs"):
        layerwise_app.run(
            Namespace(
                input_run=tmp_path / "input",
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
    monkeypatch.setattr(layerwise_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(layerwise_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        layerwise_app, "_require_frozen_invocation", lambda *arguments: None
    )
    expected_manifest = layerwise_app._sha256(manifest)
    expected_artifact = layerwise_app._sha256(artifact)
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
        layerwise_app.run(
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
    monkeypatch.setattr(layerwise_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(layerwise_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        layerwise_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(layerwise_app, "_require_checkpoint", lambda path: None)
    monkeypatch.setattr(
        layerwise_app, "_require_replay_manifest", lambda path: None
    )
    monkeypatch.setattr(
        layerwise_app,
        "load_frozen_batch_replay_inputs",
        lambda root, input_run, device: SimpleNamespace(training_loader=Loader()),
    )
    monkeypatch.setattr(
        layerwise_app,
        "select_source_episode_windows",
        lambda loader, seed: {"source": 0},
    )

    with pytest.raises(ValueError, match="window selection identity differs"):
        layerwise_app.run(
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
) -> tuple[Namespace, dict[str, str]]:
    input_run = tmp_path / "input"
    replay_manifest = input_run / "replay/autonomous/manifest.json"
    replay_manifest.parent.mkdir(parents=True)
    replay_manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_manifest = checkpoint / "manifest.json"
    checkpoint_artifact = checkpoint / "training-state.pt"
    checkpoint_manifest.write_text("{}\n", encoding="utf-8")
    checkpoint_artifact.write_bytes(b"checkpoint")
    output = tmp_path / "output"
    batch = SimpleNamespace(
        observation_count=17,
        student_inputs=object(),
        language_features=torch.zeros(1, 1),
        proprioception=torch.zeros(1, 17, 1),
        actor_proposals=torch.zeros(1, 16, 1),
        executed_actions=torch.zeros(1, 16, 1),
    )

    class Loader:
        def window_metadata(self, index: int) -> dict[str, object]:
            return {
                "episode_id": "window-0",
                "task_id": "task/v1",
                "seed": 1,
                "transition_start": 0,
                "transition_stop": 16,
            }

        def build(self, indices, include_visual_targets: bool):
            return batch

    class Module:
        def eval(self):
            return self

    world_model = Module()
    world_model.config = SimpleNamespace(visual_dimension=2)
    world_model.observe = lambda *arguments: SimpleNamespace(sequence=object())
    trainer = SimpleNamespace(
        visual_student=Module(),
        world_model=world_model,
        config=SimpleNamespace(visual_inference_microbatch_observations=8),
    )
    monkeypatch.setattr(layerwise_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(layerwise_app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        layerwise_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(layerwise_app, "_require_checkpoint", lambda path: None)
    monkeypatch.setattr(
        layerwise_app, "_require_replay_manifest", lambda path: None
    )
    monkeypatch.setattr(
        layerwise_app,
        "load_frozen_batch_replay_inputs",
        lambda root, path, device: SimpleNamespace(training_loader=Loader()),
    )
    monkeypatch.setattr(
        layerwise_app,
        "select_source_episode_windows",
        lambda loader, seed: {"source-0": 0},
    )
    monkeypatch.setattr(
        layerwise_app,
        "_selection_sha256",
        lambda windows: layerwise_app.EXPECTED_WINDOW_SELECTION,
    )
    monkeypatch.setattr(
        layerwise_app,
        "build_foundation_learning_stack",
        lambda *arguments, **keywords: SimpleNamespace(trainer=trainer),
    )
    monkeypatch.setattr(
        layerwise_app, "load_foundation_training_checkpoint", lambda *arguments: None
    )
    monkeypatch.setattr(
        layerwise_app,
        "encode_visual_student_bounded",
        lambda *arguments, **keywords: SimpleNamespace(
            pooled_state=torch.zeros(17, 2)
        ),
    )
    hashes = {
        "replay": layerwise_app._sha256(replay_manifest),
        "checkpoint_manifest": layerwise_app._sha256(checkpoint_manifest),
        "checkpoint_artifact": layerwise_app._sha256(checkpoint_artifact),
    }
    return (
        Namespace(
            input_run=input_run,
            checkpoint=checkpoint,
            output=output,
            device="mps",
        ),
        hashes,
    )


def test_success_report_records_criteria_device_invocation_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, _ = _patch_lightweight_app_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        layerwise_app,
        "evaluate_layerwise_action_effect",
        lambda *arguments: {"assessment": {"passed": False}},
    )
    monkeypatch.setattr(
        layerwise_app,
        "aggregate_layerwise_action_effect",
        lambda reports: {
            "criteria": {"expected_transition_count": 16},
            "assessment": {"decision": "diagnostic_failed"},
        },
    )

    result = layerwise_app.run(arguments)

    report = json.loads((arguments.output / "report.json").read_text())
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    assert result["decision"] == "diagnostic_failed"
    assert report["criteria"]["expected_transition_count"] == 16
    assert report["device"] == "mps"
    assert report["invocation"]["device"] == "mps"
    assert report["invocation"]["output"] == str(arguments.output)
    assert set(manifest["artifacts"]) == {
        "episodes/source-0.json",
        "report.json",
    }


def test_failure_report_records_stage_exception_hashes_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, hashes = _patch_lightweight_app_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        layerwise_app,
        "evaluate_layerwise_action_effect",
        lambda *arguments: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        layerwise_app.run(arguments)

    failure = json.loads((arguments.output / "failure.json").read_text())
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    assert failure["failure_stage"] == "episode_evaluation"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_message"] == "synthetic failure"
    assert failure["device"] == "mps"
    assert failure["input_replay_manifest_sha256"] == hashes["replay"]
    assert failure["checkpoint_manifest_sha256"] == hashes["checkpoint_manifest"]
    assert failure["checkpoint_artifact_sha256"] == hashes["checkpoint_artifact"]
    assert failure["window_selection_sha256"] == (
        layerwise_app.EXPECTED_WINDOW_SELECTION
    )
    assert set(manifest["artifacts"]) == {"failure.json"}


def _aggregate_reports() -> list[dict[str, object]]:
    tasks = (
        ["clear_dining_table_3d/v1"] * 6
        + ["store_kitchen_items_3d/v1"] * 6
        + ["tidy_living_room_3d/v1"] * 12
    )
    reports = []
    for index, task in enumerate(tasks):
        reports.append(
            {
                "transition_count": 16,
                "source_episode_id": f"source-{index:02d}",
                "window": {
                    "episode_id": f"window-{index:02d}",
                    "task_id": task,
                    "transition_start": 0,
                    "transition_stop": 16,
                },
                "shifts": {
                    str(shift): {
                        "assessment": {
                            "passed": True,
                            "first_low_retention": "activation_to_h",
                        }
                    }
                    for shift in ACTION_SHIFTS
                },
                "assessment": {
                    "passed": True,
                    "consensus_first_low_retention": "activation_to_h",
                },
                "gru_gate_distributions": {
                    gate: {
                        "finite": True,
                        "minimum": 0.1,
                        "p05": 0.2,
                        "median": 0.5,
                        "p95": 0.8,
                        "maximum": 0.9,
                    }
                    for gate in ("reset", "update", "new")
                },
            }
        )
    return reports


def test_aggregate_passes_with_task_shift_and_location_consensus() -> None:
    aggregate = aggregate_layerwise_action_effect(_aggregate_reports())

    assert aggregate["assessment"]["decision"] == "diagnostic_passed"
    assert aggregate["assessment"]["concentration_location"] == "activation_to_h"
    assert aggregate["episode_pass_count"] == 24
    assert aggregate["shift_pass_counts"] == {"1": 24, "5": 24, "9": 24}
    assert aggregate["gru_gate_distributions"]["update"]["finite"]
    assert aggregate["gru_gate_distributions"]["update"]["median"]["mean"] == 0.5


def test_aggregate_is_inconclusive_without_location_concentration() -> None:
    reports = _aggregate_reports()
    for report in reports[12:]:
        report["assessment"]["consensus_first_low_retention"] = "h_to_prior"

    aggregate = aggregate_layerwise_action_effect(reports)

    assert aggregate["assessment"]["core_passed"]
    assert not aggregate["assessment"]["concentration_passed"]
    assert aggregate["assessment"]["decision"] == "diagnostic_inconclusive"


def test_aggregate_fails_task_quota_even_with_twenty_episode_passes() -> None:
    reports = _aggregate_reports()
    for report in reports[:4]:
        report["assessment"]["passed"] = False
        report["assessment"]["consensus_first_low_retention"] = None
        for shift in report["shifts"].values():
            shift["assessment"]["passed"] = False
            shift["assessment"]["first_low_retention"] = None

    aggregate = aggregate_layerwise_action_effect(reports)

    assert aggregate["episode_pass_count"] == 20
    assert aggregate["shift_pass_counts"] == {"1": 20, "5": 20, "9": 20}
    assert not aggregate["assessment"]["checks"]["task_pass_quotas_met"]
    assert aggregate["assessment"]["decision"] == "diagnostic_failed"


def test_aggregate_rejects_duplicate_episode_identity() -> None:
    reports = _aggregate_reports()
    reports[1]["source_episode_id"] = reports[0]["source_episode_id"]

    with pytest.raises(ValueError, match="identities are not unique"):
        aggregate_layerwise_action_effect(reports)


def test_aggregate_rejects_duplicate_window_identity() -> None:
    reports = _aggregate_reports()
    reports[1]["window"] = copy.deepcopy(reports[0]["window"])

    with pytest.raises(ValueError, match="identities are not unique"):
        aggregate_layerwise_action_effect(reports)


def test_aggregate_rejects_non_sixteen_transition_report() -> None:
    reports = _aggregate_reports()
    reports[0]["transition_count"] = 15

    with pytest.raises(ValueError, match="transition count differs"):
        aggregate_layerwise_action_effect(reports)


def test_aggregate_rejects_wrong_task_coverage() -> None:
    reports = _aggregate_reports()
    reports[0] = copy.deepcopy(reports[0])
    reports[0]["window"]["task_id"] = "unknown/v1"

    with pytest.raises(ValueError, match="task identity differs"):
        aggregate_layerwise_action_effect(reports)
