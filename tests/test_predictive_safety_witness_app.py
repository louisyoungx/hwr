from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hwr.apps import evaluate_predictive_safety_witness as app
from hwr.eval.predictive_safety_witness import PROPOSAL_ID


class _Replay:
    def __init__(self, task, binding) -> None:
        del task, binding

    def run(self, **kwargs):
        return {
            "observer_enabled": kwargs["observer_enabled"],
            "prefix": {},
            "diagnostic": {},
        }


def test_run_writes_atomic_artifacts(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input"
    output = tmp_path / "output"
    input_path.mkdir()
    for name, expected in app.EXPECTED_INPUTS.items():
        del expected
        (input_path / name).write_text("{}\n", encoding="utf-8")
    (input_path / "episodes.json").write_text(
        json.dumps(
            {
                "records": [{
                    "planned_episode_id": app.ANCHOR_ID,
                    "task_id": "task",
                    "environment_seed": 1,
                    "policy_rng_seed": 2,
                }]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "FORMAL_INPUT", input_path)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(
        app,
        "_provenance",
        lambda *args: {"checks": {"passed": True}},
    )
    monkeypatch.setattr(app, "_require_disk", lambda *args: None)
    monkeypatch.setattr(
        app, "load_default_formal_household_catalogs",
        lambda root: ({"task": object()}, {"task": object()}),
    )
    monkeypatch.setattr(app, "PredictiveSafetyAnchorReplay", _Replay)
    monkeypatch.setattr(
        app,
        "analyze_predictive_witness",
        lambda disabled, enabled: {
            "proposal_id": PROPOSAL_ID,
            "decision": "accepted as predictive-safety witness contract",
        },
    )
    monkeypatch.setattr(app, "_peak_rss_bytes", lambda: 1)

    result = app.run(
        SimpleNamespace(p60_input=input_path, output=output)
    )

    assert result["decision"] == "accepted as predictive-safety witness contract"
    assert sorted(path.name for path in output.iterdir()) == [
        "disabled.json",
        "manifest.json",
        "report.json",
        "witness.json",
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["proposal_id"] == PROPOSAL_ID
    assert manifest["training_executed"] is False
    assert not output.with_name(output.name + ".tmp").exists()


def test_run_rejects_non_frozen_paths(tmp_path) -> None:
    arguments = SimpleNamespace(
        p60_input=tmp_path / "wrong-input",
        output=tmp_path / "wrong-output",
    )

    try:
        app.run(arguments)
    except ValueError as error:
        assert "P60 input path differs" in str(error)
    else:
        raise AssertionError("non-frozen input path was accepted")
