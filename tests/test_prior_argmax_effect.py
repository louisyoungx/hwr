from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hwr.apps.evaluate_action_input_contribution as contribution_app
import hwr.apps.evaluate_prior_argmax_effect as argmax_app
from hwr.apps.evaluate_prior_argmax_effect import build_parser
from hwr.eval.prior_argmax_effect import (
    ACTION_SHIFTS,
    _assess_shift,
    _active_scale,
    _common_scale_effect,
    _hard_code,
    _margin_report,
    _ratio,
    aggregate_prior_argmax_effect,
    assess_prior_argmax_episode,
    evaluate_prior_argmax_effect,
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
        stochastic_variables=32,
        stochastic_classes=32,
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


def test_default_device_is_frozen_to_mps() -> None:
    assert build_parser().parse_args(()).device == "mps"


def test_hard_code_is_independent_argmax_one_hot_oracle() -> None:
    probability = torch.tensor(
        [[[[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]]]]
    )

    code = _hard_code(probability)

    torch.testing.assert_close(
        code,
        torch.tensor([[[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]]),
    )


def test_common_scale_effect_uses_true_probability_mask_and_scale() -> None:
    time = torch.linspace(0.0, 1.0, 16)
    true = torch.stack((time, time * 0.0, time * 2.0), dim=-1)[None]
    shifted = true.clone()
    shifted[..., 0] += 0.1
    shifted[..., 1] += 100.0
    active = _active_scale(true)

    report = _common_scale_effect(true, shifted, active)

    assert active["active_dimension_count"] == 2
    assert active["active_mask"].tolist() == [True, False, True]
    assert report["finite"]
    assert report["standardized_effect"] > 0.0


def test_margin_crossing_matches_argmax_flip() -> None:
    true = torch.tensor([[[[0.6, 0.3, 0.1], [0.7, 0.2, 0.1]]]])
    shifted = torch.tensor([[[[0.2, 0.7, 0.1], [0.5, 0.4, 0.1]]]])

    report = _margin_report(true, shifted)
    flip = _hard_code(true).argmax(dim=-1) != _hard_code(shifted).argmax(dim=-1)

    assert report["near_tie_count"] == 0
    assert report["crossing_fraction"] == 0.5
    assert torch.equal(report["crossing_mask"], flip)
    assert report["margin_consumption"]["maximum"] > 0.0


def test_near_tie_is_detected_without_replacement() -> None:
    true = torch.tensor([[[[0.5, 0.5, 0.0]]]])
    shifted = torch.tensor([[[[0.4, 0.6, 0.0]]]])

    report = _margin_report(true, shifted)

    assert report["near_tie_count"] == 1


def _passing_shift_inputs():
    probability_effect = {"finite": True, "standardized_effect": 1.0}
    hard_effect = {"finite": True, "standardized_effect": 0.1}
    margin = {"finite": True, "near_tie_count": 0}
    return probability_effect, hard_effect, margin


def test_near_tie_invalidates_shift_without_replacement() -> None:
    probability_effect, hard_effect, margin = _passing_shift_inputs()
    margin["near_tie_count"] = 1

    assessment = _assess_shift(
        probability_effect,
        hard_effect,
        0.1,
        0.05,
        margin,
        0.5,
        True,
        True,
    )

    assert not assessment["passed"]
    assert assessment["decision"] == "shift_failed"
    assert not assessment["checks"]["near_tie_count_is_zero"]


def test_implementation_mismatch_invalidates_shift() -> None:
    probability_effect, hard_effect, margin = _passing_shift_inputs()

    assessment = _assess_shift(
        probability_effect,
        hard_effect,
        0.1,
        0.05,
        margin,
        0.5,
        False,
        True,
    )

    assert not assessment["passed"]
    assert assessment["decision"] == "shift_invalid"
    assert not assessment["implementation_valid"]


def test_retention_rejects_subminimum_probability_denominator() -> None:
    assert _ratio(1.0, 0.5e-6) is None


def test_prior_argmax_report_has_frozen_stages_and_criteria() -> None:
    model, sequence, actions = _fixture()

    report = evaluate_prior_argmax_effect(model, sequence, actions)

    assert report["transition_count"] == 16
    assert report["criteria"]["expected_transition_count"] == 16
    assert report["criteria"]["active_fraction_minimum"] == 0.25
    assert report["criteria"]["stochastic_variables"] == 32
    assert report["criteria"]["stochastic_classes"] == 32
    assert report["criteria"]["probability_dimension"] == 1024
    assert set(report["shifts"]) == {"1", "5", "9"}
    for shift in report["shifts"].values():
        assert shift["probability_effect"]["finite"]
        assert shift["hard_code_effect"]["finite"]
        assert shift["implementation"]["flip_matches_margin_crossing"]
        assert shift["implementation"]["hard_code_matches_sample_false"]
        assert shift["hard_feature"]["finite"]


def test_non_sixteen_transition_input_is_rejected() -> None:
    model, sequence, actions = _fixture()

    with pytest.raises(ValueError, match="input shapes are invalid"):
        evaluate_prior_argmax_effect(
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
        evaluate_prior_argmax_effect(model, sequence, actions)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_prior_argmax_report_supports_mps() -> None:
    model, sequence, actions = _fixture("mps")

    report = evaluate_prior_argmax_effect(model, sequence, actions)

    assert report["transition_count"] == 16
    assert all(
        shift["implementation"]["hard_code_matches_sample_false"]
        for shift in report["shifts"].values()
    )


def test_episode_pass_and_hard_feature_guard_are_independent() -> None:
    report = {
        "shifts": {
            "1": {
                "assessment": {"passed": True, "implementation_valid": True},
                "hard_feature": {"guard_passed": False},
            },
            "5": {
                "assessment": {"passed": True, "implementation_valid": True},
                "hard_feature": {"guard_passed": True},
            },
            "9": {
                "assessment": {"passed": False, "implementation_valid": True},
                "hard_feature": {"guard_passed": True},
            },
        }
    }

    assessment = assess_prior_argmax_episode(report)

    assert assessment["passed"]
    assert assessment["valid"]
    assert assessment["shift_pass_count"] == 2
    assert assessment["hard_feature_guard_passed"]


def test_hard_feature_failure_does_not_change_p23_episode_pass() -> None:
    report = {
        "shifts": {
            str(shift): {
                "assessment": {"passed": True, "implementation_valid": True},
                "hard_feature": {"guard_passed": False, "finite": False},
            }
            for shift in ACTION_SHIFTS
        }
    }

    assessment = assess_prior_argmax_episode(report)

    assert assessment["passed"]
    assert assessment["valid"]
    assert not assessment["hard_feature_guard_passed"]


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
                            "implementation_valid": True,
                        },
                        "hard_feature": {"guard_passed": True},
                    }
                    for shift in ACTION_SHIFTS
                },
                "assessment": {
                    "passed": True,
                    "valid": True,
                    "hard_feature_guard_passed": True,
                },
            }
        )
    return reports


def test_aggregate_passes_mechanism_and_hard_feature_guard() -> None:
    aggregate = aggregate_prior_argmax_effect(_aggregate_reports())

    assert aggregate["assessment"]["decision"] == "diagnostic_passed"
    assert aggregate["hard_feature_guard_assessment"]["passed"]
    assert aggregate["episode_pass_count"] == 24
    assert aggregate["shift_pass_counts"] == {"1": 24, "5": 24, "9": 24}


def test_aggregate_can_fail_mechanism_but_pass_hard_feature_guard() -> None:
    reports = _aggregate_reports()
    for report in reports[:4]:
        report["assessment"]["passed"] = False
        for shift in report["shifts"].values():
            shift["assessment"]["passed"] = False

    aggregate = aggregate_prior_argmax_effect(reports)

    assert aggregate["episode_pass_count"] == 20
    assert not aggregate["assessment"]["checks"]["task_pass_quotas_met"]
    assert aggregate["assessment"]["decision"] == "diagnostic_failed"
    assert aggregate["hard_feature_guard_assessment"]["passed"]


def test_aggregate_is_invalid_on_implementation_mismatch() -> None:
    reports = _aggregate_reports()
    reports[0]["assessment"]["valid"] = False
    reports[0]["assessment"]["passed"] = False
    reports[0]["shifts"]["1"]["assessment"]["implementation_valid"] = False

    aggregate = aggregate_prior_argmax_effect(reports)

    assert aggregate["assessment"]["decision"] == "diagnostic_invalid"
    assert not aggregate["assessment"]["valid"]
    assert not aggregate["hard_feature_guard_assessment"]["passed"]


def test_aggregate_rejects_duplicate_window_identity() -> None:
    reports = _aggregate_reports()
    reports[1]["window"] = copy.deepcopy(reports[0]["window"])

    with pytest.raises(ValueError, match="identities are not unique"):
        aggregate_prior_argmax_effect(reports)


def test_nonfrozen_invocation_is_rejected_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(argmax_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(argmax_app, "_source_commit", lambda root: "0" * 40)

    with pytest.raises(ValueError, match="invocation differs"):
        argmax_app.run(
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
    monkeypatch.setattr(argmax_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(argmax_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        argmax_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(argmax_app, "_require_checkpoint", lambda path: None)

    with pytest.raises(ValueError, match="Replay manifest identity differs"):
        argmax_app.run(
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
    monkeypatch.setattr(argmax_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(argmax_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        argmax_app, "_require_frozen_invocation", lambda *arguments: None
    )
    expected_manifest = argmax_app._sha256(manifest)
    expected_artifact = argmax_app._sha256(artifact)
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
        argmax_app.run(
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
    monkeypatch.setattr(argmax_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(argmax_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(
        argmax_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(argmax_app, "_require_checkpoint", lambda path: None)
    monkeypatch.setattr(
        argmax_app, "_require_replay_manifest", lambda path: None
    )
    monkeypatch.setattr(
        argmax_app,
        "load_frozen_batch_replay_inputs",
        lambda root, input_run, device: SimpleNamespace(training_loader=Loader()),
    )
    monkeypatch.setattr(
        argmax_app,
        "select_source_episode_windows",
        lambda loader, seed: {"source": 0},
    )

    with pytest.raises(ValueError, match="window selection identity differs"):
        argmax_app.run(
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
    forbidden = lambda *arguments, **keywords: (_ for _ in ()).throw(
        AssertionError("decoder or auxiliary head was called")
    )
    world_model.observe = forbidden
    world_model.decode_features = forbidden
    world_model.predict_safety_intervention = forbidden
    world_model.predict_executed_action = forbidden
    world_model.predict_severe_collision = forbidden
    world_model.encode_observations = lambda *arguments: torch.zeros(1, 17, 2)
    world_model.rssm = SimpleNamespace(
        observe=lambda embeddings, actions: object()
    )
    trainer = SimpleNamespace(
        visual_student=Module(),
        world_model=world_model,
        config=SimpleNamespace(visual_inference_microbatch_observations=8),
    )
    monkeypatch.setattr(argmax_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(argmax_app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        argmax_app, "_require_frozen_invocation", lambda *arguments: None
    )
    monkeypatch.setattr(argmax_app, "_require_checkpoint", lambda path: None)
    monkeypatch.setattr(
        argmax_app, "_require_replay_manifest", lambda path: None
    )
    monkeypatch.setattr(
        argmax_app,
        "load_frozen_batch_replay_inputs",
        lambda root, path, device: SimpleNamespace(training_loader=Loader()),
    )
    monkeypatch.setattr(
        argmax_app,
        "select_source_episode_windows",
        lambda loader, seed: {"source-0": 0},
    )
    monkeypatch.setattr(
        argmax_app,
        "_selection_sha256",
        lambda windows: argmax_app.EXPECTED_WINDOW_SELECTION,
    )
    monkeypatch.setattr(
        argmax_app,
        "build_foundation_learning_stack",
        lambda *arguments, **keywords: SimpleNamespace(trainer=trainer),
    )
    monkeypatch.setattr(
        argmax_app, "load_foundation_training_checkpoint", lambda *arguments: None
    )
    monkeypatch.setattr(
        argmax_app,
        "encode_visual_student_bounded",
        lambda *arguments, **keywords: SimpleNamespace(
            pooled_state=torch.zeros(17, 2)
        ),
    )
    hashes = {
        "replay": argmax_app._sha256(replay_manifest),
        "checkpoint_manifest": argmax_app._sha256(checkpoint_manifest),
        "checkpoint_artifact": argmax_app._sha256(checkpoint_artifact),
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
        argmax_app,
        "evaluate_prior_argmax_effect",
        lambda *arguments: {"assessment": {"passed": False}},
    )
    monkeypatch.setattr(
        argmax_app,
        "aggregate_prior_argmax_effect",
        lambda reports: {
            "criteria": {"expected_transition_count": 16},
            "assessment": {"decision": "diagnostic_failed"},
        },
    )

    result = argmax_app.run(arguments)

    report = json.loads((arguments.output / "report.json").read_text())
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    assert result["decision"] == "diagnostic_failed"
    assert report["criteria"]["expected_transition_count"] == 16
    assert report["device"] == "mps"
    assert report["invocation"]["device"] == "mps"
    assert set(manifest["artifacts"]) == {
        "episodes/source-0.json",
        "report.json",
    }


def test_failure_report_records_stage_exception_hashes_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, hashes = _patch_lightweight_app_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        argmax_app,
        "evaluate_prior_argmax_effect",
        lambda *arguments: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        argmax_app.run(arguments)

    failure = json.loads((arguments.output / "failure.json").read_text())
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    assert failure["failure_stage"] == "episode_evaluation"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["device"] == "mps"
    assert failure["completed_episode_count"] == 0
    assert failure["current_source_episode_id"] == "source-0"
    assert failure["current_window"]["episode_id"] == "window-0"
    assert failure["input_replay_manifest_sha256"] == hashes["replay"]
    assert failure["checkpoint_manifest_sha256"] == hashes["checkpoint_manifest"]
    assert failure["checkpoint_artifact_sha256"] == hashes["checkpoint_artifact"]
    assert failure["window_selection_sha256"] == argmax_app.EXPECTED_WINDOW_SELECTION
    assert failure["criteria"]["expected_transition_count"] == 16
    assert failure["invocation"]["device"] == "mps"
    assert set(manifest["artifacts"]) == {"failure.json"}
