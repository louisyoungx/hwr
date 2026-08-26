from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hwr.apps import audit_interaction_contract_mutations as app
from hwr.eval.interaction_contract_mutation import PROPOSAL_ID


def test_run_writes_atomic_artifacts(tmp_path, monkeypatch) -> None:
    contract = tmp_path / "contract.json"
    input_path = tmp_path / "input"
    output = tmp_path / "output"
    contract.write_text("{}\n", encoding="utf-8")
    input_path.mkdir()
    for name in app.EXPECTED_INPUTS:
        (input_path / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(app, "FORMAL_CONTRACT", contract)
    monkeypatch.setattr(app, "FORMAL_INPUT", input_path)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(
        app, "_provenance", lambda *args: {"checks": {"passed": True}}
    )
    monkeypatch.setattr(app, "_git_json", lambda *args: {})
    monkeypatch.setattr(app, "_producer_sources", lambda root: {})
    monkeypatch.setattr(
        app,
        "audit_interaction_contract_mutations",
        lambda *args, **kwargs: {
            "mutations": {"proposal_id": PROPOSAL_ID},
            "report": {
                "proposal_id": PROPOSAL_ID,
                "decision": "accepted as residual P61 contract gap evidence",
                "p68_dependency_gate_passed": True,
            },
        },
    )
    monkeypatch.setattr(app, "_peak_rss_bytes", lambda: 1)

    result = app.run(
        SimpleNamespace(contract=contract, p61_input=input_path, output=output)
    )

    assert result["p68_dependency_gate_passed"] is True
    assert sorted(path.name for path in output.iterdir()) == [
        "manifest.json",
        "mutations.json",
        "report.json",
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["proposal_id"] == PROPOSAL_ID
    assert manifest["capability_claim_allowed"] is False
    assert not output.with_name(output.name + ".tmp").exists()


def test_run_rejects_non_frozen_paths(tmp_path) -> None:
    arguments = SimpleNamespace(
        contract=tmp_path / "wrong-contract",
        p61_input=tmp_path / "wrong-input",
        output=tmp_path / "wrong-output",
    )

    try:
        app.run(arguments)
    except ValueError as error:
        assert "contract path differs" in str(error)
    else:
        raise AssertionError("non-frozen contract path was accepted")
