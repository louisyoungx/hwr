from __future__ import annotations

import json

import pytest

from hwr.train.foundation_diagnostics import publish_action_causality_report


def test_action_causality_report_is_published_as_one_immutable_directory(
    tmp_path,
) -> None:
    diagnostic = {
        "schema_version": "hwr.foundation-action-causality/v3",
        "action_source": "actual_executed_action",
        "report": {"shuffled_to_true_ratio": 1.2},
        "assessment": {"passed": True},
    }
    target = tmp_path / "update-000000001"

    output = publish_action_causality_report(
        target,
        diagnostic,
        source_commit="abc123",
        update_count=1,
        training_data_manifest_sha256="d" * 64,
        audit_data_manifest_sha256="e" * 64,
    )

    value = json.loads(output.read_text())
    assert output == target / "report.json"
    assert value["source_commit"] == "abc123"
    assert value["training_data_manifest_sha256"] == "d" * 64
    assert value["audit_data_manifest_sha256"] == "e" * 64
    assert not list(tmp_path.glob(".update-000000001-*"))
    with pytest.raises(FileExistsError):
        publish_action_causality_report(
            target,
            diagnostic,
            source_commit="abc123",
            update_count=1,
            training_data_manifest_sha256="d" * 64,
            audit_data_manifest_sha256="e" * 64,
        )
