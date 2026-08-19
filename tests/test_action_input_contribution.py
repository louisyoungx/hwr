from __future__ import annotations

import copy
import math
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hwr.apps.evaluate_action_input_contribution as contribution_app
from hwr.apps.evaluate_action_input_contribution import (
    DEFAULT_INPUT_RUN,
    EXPECTED_REPLAY_MANIFEST,
    EXPECTED_WINDOW_SELECTION,
    _require_replay_manifest,
    _selected_windows,
    _selection_sha256,
)
from hwr.apps.evaluate_posterior_overshooting import (
    SELECTION_SEED,
    select_source_episode_windows,
)
from hwr.eval.action_input_contribution import (
    _column_norms,
    _dimension_rms,
    _norm,
    _rms,
    aggregate_action_input_contribution,
    assess_action_input_contribution,
    canonical_normalize_actions,
    evaluate_action_input_contribution,
)
from hwr.train.foundation_batch_replay import load_frozen_batch_replay_inputs
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
    batch, transitions = 2, 8
    visual = torch.randn(batch, transitions + 1, 8, device=device)
    language = torch.randn(batch, 6, device=device)
    proprioception = torch.randn(batch, transitions + 1, 5, device=device)
    proposals = torch.zeros(batch, transitions, 3, device=device)
    actions = torch.empty(batch, transitions, 3, device=device).uniform_(-0.1, 0.1)
    actions[..., 1] *= 5.0
    actions[..., 2] = torch.rand(batch, transitions, device=device)
    observed = model.observe(
        visual, language, proprioception, proposals, actions
    )
    return model, observed.sequence, actions


def test_canonical_action_normalization_uses_runtime_bounds() -> None:
    model, _, _ = _fixture()
    actions = torch.tensor([[[-0.1, 0.0, 1.0], [0.1, -0.5, 0.0]]])

    normalized = canonical_normalize_actions(actions, model.config)

    torch.testing.assert_close(
        normalized,
        torch.tensor([[[-1.0, 0.0, 1.0], [1.0, -1.0, -1.0]]]),
    )


def test_action_contribution_report_is_finite_and_bounded() -> None:
    model, sequence, actions = _fixture()

    report = evaluate_action_input_contribution(model, sequence, actions)

    assert report["canonical_actions_finite"]
    assert report["canonical_actions_in_bounds"]
    assert report["canonical_action_nonfinite_count"] == 0
    assert report["canonical_action_out_of_bounds_count"] == 0
    assert report["stochastic_variation_contribution_rms"] > 0.0
    assert report["raw_action_variation_contribution_rms"] > 0.0
    assert report["canonical_action_variation_contribution_rms"] > 0.0
    assert len(report["weights"]["stochastic_column_norms"]) == 16
    assert len(report["weights"]["action_column_norms"]) == 3
    assert report["bias"]["norm"] > 0.0


def test_action_contribution_statistics_use_cpu_float64_oracle() -> None:
    value = torch.tensor(
        [[[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]]], dtype=torch.float32
    )
    oracle = value.cpu().double()

    assert _rms(value) == float(oracle.square().mean().sqrt())
    assert _dimension_rms(value) == (
        oracle.square().mean(dim=(0, 1)).sqrt().tolist()
    )
    assert _norm(value) == float(oracle.norm())
    matrix = value.reshape(-1, value.shape[-1])
    assert _column_norms(matrix) == matrix.double().norm(dim=0).tolist()


def test_window_selection_hash_is_order_sensitive_and_stable() -> None:
    windows = [
        {"source_episode_id": "source-a", "window_index": 1},
        {"source_episode_id": "source-b", "window_index": 2},
    ]

    assert _selection_sha256(windows) == (
        "7428788cbc944725c3010062f87cc682de5df3698122e05507f9213d100c53e3"
    )
    assert _selection_sha256(list(reversed(windows))) != _selection_sha256(windows)


def test_frozen_replay_and_window_selection_hashes_match() -> None:
    root = Path(__file__).resolve().parents[1]
    input_run = root / DEFAULT_INPUT_RUN
    manifest = input_run / "replay/autonomous/manifest.json"
    if not manifest.exists():
        pytest.skip("frozen P20 Replay is unavailable")
    _require_replay_manifest(manifest)
    inputs = load_frozen_batch_replay_inputs(root, input_run, device="cpu")
    selected = select_source_episode_windows(
        inputs.training_loader, seed=SELECTION_SEED
    )
    windows = _selected_windows(inputs.training_loader, selected)

    assert EXPECTED_REPLAY_MANIFEST == (
        "c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985"
    )
    assert _selection_sha256(windows) == EXPECTED_WINDOW_SELECTION
    assert len(windows) == 24


def test_replay_hash_failure_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_run = tmp_path / "input"
    manifest = input_run / "replay/autonomous/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(contribution_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(contribution_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(contribution_app, "_require_checkpoint", lambda path: None)

    with pytest.raises(ValueError, match="Replay manifest identity differs"):
        contribution_app.run(
            Namespace(
                input_run=input_run,
                checkpoint=tmp_path / "checkpoint",
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
    monkeypatch.setattr(contribution_app, "_require_clean_source", lambda root: None)
    monkeypatch.setattr(contribution_app, "_source_commit", lambda root: "0" * 40)
    monkeypatch.setattr(contribution_app, "_require_checkpoint", lambda path: None)
    monkeypatch.setattr(
        contribution_app, "_require_replay_manifest", lambda path: None
    )
    monkeypatch.setattr(
        contribution_app,
        "load_frozen_batch_replay_inputs",
        lambda root, input_run, device: SimpleNamespace(training_loader=Loader()),
    )
    monkeypatch.setattr(
        contribution_app,
        "select_source_episode_windows",
        lambda loader, seed: {"source": 0},
    )

    with pytest.raises(ValueError, match="window selection identity differs"):
        contribution_app.run(
            Namespace(
                input_run=tmp_path / "input",
                checkpoint=tmp_path / "checkpoint",
                output=output,
                device="cpu",
            )
        )

    assert not output.exists()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_action_contribution_report_supports_mps_statistics() -> None:
    model, sequence, actions = _fixture("mps")

    report = evaluate_action_input_contribution(model, sequence, actions)

    assert report["canonical_actions_finite"]
    assert report["canonical_actions_in_bounds"]
    assert report["stochastic_variation_contribution_rms"] > 0.0
    assert len(report["canonical_action_dimension_rms"]) == 3


def test_canonical_constant_offset_is_not_action_variation() -> None:
    model, sequence, actions = _fixture()
    constant_actions = torch.zeros_like(actions)

    report = evaluate_action_input_contribution(
        model, sequence, constant_actions
    )

    assert report["absolute_raw_action_contribution_rms"] == 0.0
    assert report["absolute_canonical_action_contribution_rms"] > 0.0
    assert report["canonical_action_dc_contribution_rms"] > 0.0
    assert report["raw_action_variation_contribution_rms"] == 0.0
    assert report["canonical_action_variation_contribution_rms"] == 0.0
    assert report["canonical_to_raw_variation_contribution_gain"] == 0.0


def test_nonfinite_actions_produce_diagnostic_failure() -> None:
    model, sequence, actions = _fixture()
    actions[0, 0, 0] = torch.nan

    report = evaluate_action_input_contribution(model, sequence, actions)

    assert not report["canonical_actions_finite"]
    assert not report["canonical_actions_in_bounds"]
    assert report["canonical_action_nonfinite_count"] == 1
    assert report["raw_action_variation_contribution_rms"] is None
    assert report["canonical_action_variation_contribution_rms"] is None
    assert report["assessment"]["decision"] == "diagnostic_failed"


def test_out_of_bounds_actions_produce_diagnostic_failure() -> None:
    model, sequence, actions = _fixture()
    actions[0, 0, 0] = 0.2

    report = evaluate_action_input_contribution(model, sequence, actions)

    assert report["canonical_actions_finite"]
    assert not report["canonical_actions_in_bounds"]
    assert report["canonical_action_out_of_bounds_count"] == 1
    assert report["assessment"]["decision"] == "diagnostic_failed"


def _aggregate_reports() -> list[dict[str, object]]:
    model, sequence, actions = _fixture()
    base = evaluate_action_input_contribution(model, sequence, actions)
    reports = []
    for index in range(24):
        report = copy.deepcopy(base)
        transitions = 4 if index == 0 else 16
        dimensions = len(report["canonical_action_dimension_rms"])
        report.update(
            {
                "source_episode_id": f"episode-{index:02d}",
                "transition_count": transitions,
                "canonical_action_value_count": transitions * dimensions,
                "canonical_action_finite_count": transitions * dimensions,
                "canonical_action_nonfinite_count": 0,
                "canonical_action_in_bounds_count": transitions * dimensions,
                "canonical_action_out_of_bounds_count": 0,
                "canonical_action_dimension_finite_count": (
                    [transitions] * dimensions
                ),
                "canonical_action_dimension_nonfinite_count": [0] * dimensions,
                "canonical_action_dimension_in_bounds_count": (
                    [transitions] * dimensions
                ),
                "canonical_action_dimension_out_of_bounds_count": [0] * dimensions,
                "window": {
                    "episode_id": f"window-{index:02d}",
                    "task_id": "task/v1",
                    "seed": index,
                    "transition_start": 0,
                    "transition_stop": transitions,
                },
                "stochastic_variation_contribution_rms": 10.0,
                "raw_action_variation_contribution_rms": (
                    1.0 if index == 0 else 3.0
                ),
                "canonical_action_variation_contribution_rms": 6.0,
                "raw_action_variation_to_stochastic_ratio": (
                    0.1 if index < 20 else 0.3
                ),
                "canonical_to_raw_variation_contribution_gain": (
                    2.0 if index < 20 else 1.0
                ),
            }
        )
        reports.append(report)
    return reports


def test_action_contribution_aggregate_uses_pooled_rms_and_frozen_checks() -> None:
    reports = _aggregate_reports()

    aggregate = aggregate_action_input_contribution(reports)

    assert aggregate["episode_count"] == 24
    assert aggregate["episodes_passing_contribution_conditions"] == 20
    assert aggregate["raw_action_variation_contribution_rms"] == pytest.approx(
        math.sqrt((4 * 1.0**2 + 23 * 16 * 3.0**2) / (4 + 23 * 16))
    )
    assert aggregate["canonical_action_variation_contribution_rms"] == 6.0
    assert aggregate["canonical_action_nonfinite_count"] == 0
    assert set(aggregate["assessment"]["checks"]) == {
        "raw_action_variation_to_stochastic_ratio_below_0_20",
        "canonical_to_raw_variation_gain_at_least_1_50",
        "canonical_actions_finite",
        "canonical_actions_in_bounds",
        "at_least_20_of_24_episodes_pass",
    }


@pytest.mark.parametrize("identity", ["source", "window"])
def test_action_contribution_aggregate_rejects_duplicate_identity(
    identity: str,
) -> None:
    reports = _aggregate_reports()
    if identity == "source":
        reports[1]["source_episode_id"] = reports[0]["source_episode_id"]
    else:
        reports[1]["window"] = copy.deepcopy(reports[0]["window"])

    with pytest.raises(ValueError, match="identities are not unique"):
        aggregate_action_input_contribution(reports)


def test_action_contribution_aggregate_propagates_nonfinite_failure() -> None:
    reports = _aggregate_reports()
    reports[0]["canonical_action_variation_contribution_rms"] = None
    reports[0]["canonical_to_raw_variation_contribution_gain"] = None
    reports[0]["canonical_actions_finite"] = False
    reports[0]["canonical_actions_in_bounds"] = False

    aggregate = aggregate_action_input_contribution(reports)

    assert aggregate["canonical_action_variation_contribution_rms"] is None
    assert aggregate["canonical_to_raw_variation_contribution_gain"] is None
    assert aggregate["assessment"]["decision"] == "diagnostic_failed"


def test_action_contribution_thresholds_are_strict_and_inclusive() -> None:
    report = {
        "raw_action_variation_to_stochastic_ratio": 0.20,
        "canonical_to_raw_variation_contribution_gain": 1.50,
        "canonical_actions_finite": True,
        "canonical_actions_in_bounds": True,
    }

    failed = assess_action_input_contribution(report)
    report["raw_action_variation_to_stochastic_ratio"] = 0.199
    passed = assess_action_input_contribution(report)

    assert not failed["checks"][
        "raw_action_variation_to_stochastic_ratio_below_0_20"
    ]
    assert failed["checks"]["canonical_to_raw_variation_gain_at_least_1_50"]
    assert passed["decision"] == "diagnostic_passed"
