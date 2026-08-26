from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hwr.apps import evaluate_initial_candidate_association as app
from hwr.eval.initial_candidate_association import PROPOSAL_ID


def _arguments(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        p50_input=tmp_path / "p50",
        mapping_input=tmp_path / "mapping",
        interaction_input=tmp_path / "interaction",
        p72_input=tmp_path / "p72",
        output=tmp_path / "output",
    )


def test_run_writes_three_atomic_artifacts(tmp_path, monkeypatch) -> None:
    arguments = _arguments(tmp_path)
    for path in (
        arguments.p50_input,
        arguments.mapping_input,
        arguments.interaction_input,
        arguments.p72_input,
    ):
        path.mkdir()
    (arguments.p50_input / "capsules.json").write_text("{}\n")
    (arguments.mapping_input / "tables.json").write_text("{}\n")
    (arguments.interaction_input / "transitions.json").write_text("{}\n")
    (arguments.p72_input / "report.json").write_text(
        json.dumps({"p68_dependency_gate_passed": True})
    )
    monkeypatch.setattr(app, "FORMAL_P50_INPUT", arguments.p50_input)
    monkeypatch.setattr(app, "FORMAL_MAPPING_INPUT", arguments.mapping_input)
    monkeypatch.setattr(
        app, "FORMAL_INTERACTION_INPUT", arguments.interaction_input
    )
    monkeypatch.setattr(app, "FORMAL_P72_INPUT", arguments.p72_input)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", arguments.output)
    monkeypatch.setattr(
        app, "_provenance", lambda *args: {"checks": {"passed": True}}
    )
    monkeypatch.setattr(app, "_require_disk", lambda *args: None)
    monkeypatch.setattr(
        app,
        "execute_cohort",
        lambda *args, **kwargs: [
            {
                "task_id": task,
                "planned_episode_id": f"{task}-{index}",
                "classification": "stage_compatible_selected",
            }
            for task in app.TASK_IDS
            for index in range(8)
        ],
    )
    monkeypatch.setattr(app, "_peak_rss_bytes", lambda: 1)

    result = app.run(arguments)

    assert result["decision"] == (
        "accepted as initial-association stopping-gate evidence"
    )
    assert sorted(path.name for path in arguments.output.iterdir()) == [
        "episodes.json",
        "manifest.json",
        "report.json",
    ]
    manifest = json.loads(
        (arguments.output / "manifest.json").read_text()
    )
    assert manifest["proposal_id"] == PROPOSAL_ID
    assert manifest["training_executed"] is False
    assert not arguments.output.with_name(
        arguments.output.name + ".tmp"
    ).exists()


def test_run_rejects_non_frozen_paths(tmp_path) -> None:
    arguments = _arguments(tmp_path)

    try:
        app.run(arguments)
    except ValueError as error:
        assert "path differs from frozen path" in str(error)
    else:
        raise AssertionError("non-frozen paths were accepted")
