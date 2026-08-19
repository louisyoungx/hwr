from __future__ import annotations

import copy

import pytest
import torch

from hwr.apps.evaluate_decoder_gain import build_parser
from hwr.eval.decoder_gain import (
    ACTION_SHIFTS,
    EDGE_NAMES,
    HEAD_NAMES,
    _assess_branch,
    _calibrated_effect,
    _decoder_stages,
    _path_report,
    _scan_retentions,
    aggregate_decoder_gain,
    assess_decoder_head_episode,
    build_decoder_branches,
    build_decoder_calibration,
    deserialize_decoder_calibration,
    evaluate_decoder_gain,
    serialize_decoder_calibration,
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
    embeddings = model.encode_observations(visual, language, proprioception)
    sequence = model.rssm.observe(embeddings, actions)
    return model, sequence, actions


def test_default_device_is_frozen_to_mps() -> None:
    assert build_parser().parse_args(()).device == "mps"


def test_decoder_stages_reproduce_head_and_layer_norm() -> None:
    model, _, _ = _fixture()
    features = torch.randn(1, 16, model.config.feature_dimension)

    stages, layer_norm = _decoder_stages(model.visual_head, features)

    torch.testing.assert_close(stages["output"], model.visual_head(features))
    torch.testing.assert_close(
        stages["layer_norm_affine"],
        model.visual_head[1](stages["linear_preactivation"]),
    )
    assert layer_norm["eps"] == model.visual_head[1].eps
    assert set(stages) == {
        "feature",
        "linear_preactivation",
        "layer_norm_normalized",
        "layer_norm_affine",
        "hidden",
        "output",
    }


def test_calibration_uses_all_384_true_transitions() -> None:
    model, sequence, actions = _fixture()
    branch = build_decoder_branches(model, sequence, actions)

    calibration = build_decoder_calibration(
        model, [branch.true_feature for _ in range(24)]
    )
    serialized = serialize_decoder_calibration(calibration)

    assert calibration["transition_count"] == 384
    assert set(calibration["heads"]) == {"visual", "proprioception"}
    for head in calibration["heads"].values():
        assert set(head["stages"]) == {
            "feature",
            "linear_preactivation",
            "layer_norm_normalized",
            "layer_norm_affine",
            "hidden",
            "output",
        }
        assert all(stage["finite"] for stage in head["stages"].values())
    assert isinstance(
        serialized["heads"]["visual"]["stages"]["feature"]["scale"], list
    )
    restored = deserialize_decoder_calibration(serialized, device="cpu")
    assert restored["heads"]["visual"]["stages"]["feature"]["scale"].dtype == (
        torch.float64
    )
    assert restored["heads"]["visual"]["stages"]["feature"]["active_mask"].dtype == (
        torch.bool
    )


def test_path_integrated_jvp_reconstructs_linear_jump() -> None:
    layer = torch.nn.Linear(7, 5)
    true_input = torch.randn(1, 16, 7)
    shifted_input = true_input + torch.randn_like(true_input) * 0.1
    true_output = layer(true_input)
    shifted_output = layer(shifted_input)
    input_calibration = _calibration_for(true_input)
    output_calibration = _calibration_for(true_output)

    report = _path_report(
        layer,
        true_input,
        shifted_input,
        true_output,
        shifted_output,
        input_calibration,
        output_calibration,
    )

    assert report["valid"]
    assert report["reconstruction_cosine"] == pytest.approx(1.0, abs=1.0e-6)
    assert report["relative_error"] <= 1.0e-5


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_path_integrated_jvp_executes_on_mps() -> None:
    layer = torch.nn.Linear(7, 5).to("mps")
    true_input = torch.randn(1, 16, 7, device="mps")
    shifted_input = true_input + torch.randn_like(true_input) * 0.1
    report = _path_report(
        layer,
        true_input,
        shifted_input,
        layer(true_input),
        layer(shifted_input),
        _calibration_for(true_input),
        _calibration_for(layer(true_input)),
    )

    assert report["valid"]
    assert report["reconstruction_cosine"] >= 0.999


def test_invalid_early_retention_cannot_skip_to_later_edge() -> None:
    effects = {
        name: {"standardized_effect": 1.0}
        for name in (
            "feature",
            "linear_preactivation",
            "layer_norm_normalized",
            "layer_norm_affine",
            "hidden",
            "output",
        )
    }
    effects["feature"]["standardized_effect"] = 0.0
    effects["linear_preactivation"]["standardized_effect"] = 0.1
    effects["layer_norm_normalized"]["standardized_effect"] = 0.01

    retentions, first_low, invalid = _scan_retentions(effects)

    assert retentions["feature_to_linear"] is None
    assert invalid
    assert first_low is None
    assert "linear_to_norm" not in retentions


def test_calibrated_effect_rejects_low_active_coverage() -> None:
    true = torch.randn(1, 16, 8)
    shifted = true + 0.1
    calibration = _calibration_for(true)
    calibration["active_mask"] = torch.tensor(
        [True, False, False, False, False, False, False, False]
    )
    calibration["active_dimension_count"] = 1
    calibration["active_fraction"] = 0.125

    report = _calibrated_effect(true, shifted, calibration)

    assert not report["valid"]


def test_build_branches_reproduces_p23_hard_feature_guard() -> None:
    model, sequence, actions = _fixture()

    branches = build_decoder_branches(model, sequence, actions)

    assert branches.true_feature.shape == (
        1,
        16,
        model.config.feature_dimension,
    )
    assert set(branches.shifted_features) == {1, 5, 9}
    assert set(branches.p23_guard) == {1, 5, 9}
    assert all(value["finite"] for value in branches.p23_guard.values())


def test_decoder_report_has_independent_head_assessments() -> None:
    model, sequence, actions = _fixture()
    branches = build_decoder_branches(model, sequence, actions)
    calibration = build_decoder_calibration(
        model, [branches.true_feature for _ in range(24)]
    )

    report = evaluate_decoder_gain(model, branches, calibration)

    assert set(report["heads"]) == {"visual", "proprioception"}
    assert report["criteria"]["path_segments"] == 16
    for head in report["heads"].values():
        assert set(head["shifts"]) == {"1", "5", "9"}
        for shift in head["shifts"].values():
            assert set(shift["stage_effects"]) == {
                "feature",
                "linear_preactivation",
                "layer_norm_normalized",
                "layer_norm_affine",
                "hidden",
                "output",
            }
            assert len(shift["path_reports"]) <= 1
            assert set(shift["path_reports"]).issubset(set(EDGE_NAMES))


def test_p23_endpoint_mismatch_invalidates_decoder_branch() -> None:
    model, sequence, actions = _fixture()
    branches = build_decoder_branches(model, sequence, actions)
    branches.p23_endpoint_valid[1] = False
    calibration = build_decoder_calibration(
        model, [branches.true_feature for _ in range(24)]
    )

    report = evaluate_decoder_gain(model, branches, calibration)

    for head in report["heads"].values():
        shift = head["shifts"]["1"]
        assert not shift["assessment"]["valid"]
        assert shift["assessment"]["state"] == "jvp_invalid"
        assert not shift["endpoint_validation"][
            "p23_hard_code_matches_sample_false"
        ]


def test_decoder_endpoint_mismatch_invalidates_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, sequence, actions = _fixture()
    branches = build_decoder_branches(model, sequence, actions)
    calibration = build_decoder_calibration(
        model, [branches.true_feature for _ in range(24)]
    )
    original = model.decode_features
    monkeypatch.setattr(
        model,
        "decode_features",
        lambda features: tuple(value + 1.0 for value in original(features)),
    )

    report = evaluate_decoder_gain(model, branches, calibration)

    for head in report["heads"].values():
        assert not head["assessment"]["valid"]
        assert all(
            not shift["endpoint_validation"][
                "true_decoder_matches_decode_features"
            ]
            for shift in head["shifts"].values()
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_decoder_report_supports_mps_path_jvp() -> None:
    model, sequence, actions = _fixture("mps")
    branches = build_decoder_branches(model, sequence, actions)
    calibration = build_decoder_calibration(
        model, [branches.true_feature for _ in range(24)]
    )

    report = evaluate_decoder_gain(model, branches, calibration)

    assert report["transition_count"] == 16
    assert set(report["heads"]) == {"visual", "proprioception"}


def test_episode_requires_two_shifts_on_same_edge() -> None:
    shifts = {
        "1": _branch_assessment(True, "feature_to_linear", True),
        "5": _branch_assessment(True, "linear_to_norm", True),
        "9": _branch_assessment(False, None, True),
    }

    assessment = assess_decoder_head_episode({"shifts": shifts})

    assert not assessment["passed"]
    assert assessment["localized_edge"] is None


def test_episode_allows_two_valid_shifts_and_one_invalid() -> None:
    shifts = {
        "1": _branch_assessment(True, "feature_to_linear", True),
        "5": _branch_assessment(True, "feature_to_linear", True),
        "9": _branch_assessment(False, None, False),
    }

    assessment = assess_decoder_head_episode({"shifts": shifts})

    assert assessment["valid"]
    assert assessment["valid_shift_count"] == 2
    assert assessment["passed"]
    assert assessment["localized_edge"] == "feature_to_linear"


def test_first_low_edge_jvp_failure_cannot_be_not_localized() -> None:
    assessment = _assess_branch(
        "feature_to_linear",
        {
            "valid": False,
            "path_retention": None,
            "reconstruction_cosine": None,
            "relative_error": None,
        },
        True,
    )

    assert assessment["state"] == "jvp_invalid"
    assert not assessment["valid"]
    assert not assessment["passed"]


def test_feature_guard_failure_invalidates_branch() -> None:
    assessment = _assess_branch(None, None, False)

    assert assessment["state"] == "jvp_invalid"
    assert not assessment["valid"]


def _aggregate_reports() -> list[dict[str, object]]:
    tasks = (
        ["clear_dining_table_3d/v1"] * 6
        + ["store_kitchen_items_3d/v1"] * 6
        + ["tidy_living_room_3d/v1"] * 12
    )
    reports = []
    for index, task in enumerate(tasks):
        heads = {}
        for head_name in HEAD_NAMES:
            shifts = {
                str(shift): _branch_assessment(
                    True, "feature_to_linear", True
                )
                for shift in ACTION_SHIFTS
            }
            heads[head_name] = {
                "shifts": shifts,
                "assessment": {
                    "valid": True,
                    "passed": True,
                    "localized_edge": "feature_to_linear",
                    "output_guard_passed": True,
                },
            }
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
                "heads": heads,
            }
        )
    return reports


def test_aggregate_passes_each_head_independently() -> None:
    aggregate = aggregate_decoder_gain(_aggregate_reports())

    assert aggregate["assessment"]["decision"] == "diagnostic_complete"
    for head in aggregate["heads"].values():
        assert head["state"] == "passed(feature_to_linear)"
        assert head["episode_pass_count"] == 24


def test_aggregate_routes_not_localized_head_to_p25() -> None:
    reports = _aggregate_reports()
    for report in reports:
        head = report["heads"]["visual"]
        head["assessment"]["passed"] = False
        head["assessment"]["localized_edge"] = None
        for shift in head["shifts"].values():
            shift["assessment"] = _branch_assessment(
                False, None, True
            )["assessment"]

    aggregate = aggregate_decoder_gain(reports)

    assert aggregate["heads"]["visual"]["state"] == "not_localized"
    assert aggregate["heads"]["proprioception"]["state"] == (
        "passed(feature_to_linear)"
    )


def test_aggregate_routes_invalid_head_to_blocked_state() -> None:
    reports = _aggregate_reports()
    for report in reports[:5]:
        visual = report["heads"]["visual"]
        visual["assessment"]["valid"] = False
        visual["assessment"]["passed"] = False
        visual["assessment"]["localized_edge"] = None
        for shift in visual["shifts"].values():
            shift["assessment"] = _branch_assessment(
                False, None, False
            )["assessment"]

    aggregate = aggregate_decoder_gain(reports)

    assert aggregate["heads"]["visual"]["state"] == "jvp_invalid"
    assert aggregate["assessment"]["decision"] == "diagnostic_invalid"


def test_not_localized_requires_all_branches_valid() -> None:
    reports = _aggregate_reports()
    for report in reports:
        visual = report["heads"]["visual"]
        visual["assessment"]["passed"] = False
        visual["assessment"]["localized_edge"] = None
        visual["shifts"]["9"]["assessment"] = _branch_assessment(
            False, None, False
        )["assessment"]
        for shift in ("1", "5"):
            visual["shifts"][shift]["assessment"] = _branch_assessment(
                False, None, True
            )["assessment"]
        visual["assessment"]["valid"] = True
        visual["assessment"]["output_guard_passed"] = True

    aggregate = aggregate_decoder_gain(reports)

    assert aggregate["heads"]["visual"]["state"] == "jvp_invalid"
    assert not aggregate["heads"]["visual"]["all_branches_valid"]
    assert aggregate["assessment"]["decision"] == "diagnostic_invalid"


def _calibration_for(value: torch.Tensor) -> dict[str, object]:
    cpu = value.detach().cpu().double()
    mean = cpu.mean(dim=(0, 1))
    scale = (cpu - mean).square().mean(dim=(0, 1)).sqrt()
    active = scale >= 1.0e-4
    return {
        "finite": True,
        "mean": mean,
        "scale": scale,
        "active_mask": active,
        "active_dimension_count": int(active.sum()),
        "active_fraction": float(active.double().mean()),
    }


def _branch_assessment(
    passed: bool,
    edge: str | None,
    valid: bool,
) -> dict[str, object]:
    return {
        "stage_effects": {
            "output": {"standardized_effect": 0.1},
        },
        "assessment": {
            "state": (
                "localized"
                if passed
                else "not_localized"
                if valid
                else "jvp_invalid"
            ),
            "valid": valid,
            "passed": passed,
            "localized_edge": edge,
        },
    }
