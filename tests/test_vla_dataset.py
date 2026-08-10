from __future__ import annotations

import json

import pytest

from hwr.data import load_vla_dataset, verify_vla_dataset
from hwr.policy.vla_input import VLA_POLICY_INPUT_FIELDS
from tests.vla_fixtures import build_dataset


def test_vla_dataset_round_trip_has_no_symbolic_intermediate_labels(tmp_path) -> None:
    path = build_dataset(tmp_path)

    manifest = verify_vla_dataset(path)
    dataset = load_vla_dataset(path)

    assert manifest["policy_input_fields"] == sorted(VLA_POLICY_INPUT_FIELDS)
    assert manifest["label_fields"] == ["action_chunk", "valid_steps"]
    assert "phase" not in json.dumps(manifest).lower()
    assert len(dataset) == 18
    assert dataset.action_chunks.shape == (18, 3, 16)
    assert set(dataset.episode_ids) == {"episode-0", "episode-1", "episode-2"}


def test_vla_dataset_verifier_rejects_phase_metadata(tmp_path) -> None:
    path = build_dataset(tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["phase_names"] = ["grasp"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="symbolic"):
        verify_vla_dataset(path)
