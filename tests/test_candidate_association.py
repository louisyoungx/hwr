from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hwr.eval import target_selection
from hwr.adapters.mujoco.candidate_association import (
    CandidateAssociationBackend,
    summary_for_identity,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.eval.initial_candidate_association import (
    CandidateSupport,
    RawSupport,
    associate_candidates,
    reconstruct_candidate_support,
)
from hwr.eval.target_selection import (
    Candidate,
    CandidateSet,
    RawCandidate,
    deserialize_policy_input,
)

ROOT = Path(__file__).resolve().parents[1]
P50 = ROOT / "runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001"


def test_backend_binds_segmentation_to_returned_observation_identity() -> None:
    task_id = "tidy_living_room_3d/v1"
    tasks, bindings = load_default_formal_household_catalogs(ROOT)
    backend = CandidateAssociationBackend(tasks[task_id], bindings[task_id])
    try:
        observation = backend.reset(seed=17, task_id=task_id)
        segmentation = backend.segmentation_for(
            (observation.timestamp_ns, observation.sequence_id)
        )
    finally:
        backend.close()

    assert segmentation.shape == (192, 256, 2)
    assert segmentation.dtype == np.int32


def test_support_reconstruction_matches_all_committed_capsules() -> None:
    import json

    document = json.loads((P50 / "capsules.json").read_text())
    for episode in document["episodes"]:
        payloads = [
            (P50 / capture["policy_input"]["path"]).read_bytes()
            for capture in episode["captures"]
        ]
        candidate_set, supports = reconstruct_candidate_support(
            payloads[:-1],
            acquisition_base_pose=episode["acquisition_base_pose"],
            final_input=payloads[-1],
        )

        assert candidate_set.candidate_set_sha256 == episode["candidate_set"]["sha256"]
        assert len(supports) == episode["candidate_set"]["candidate_count"]
        assert sum(item.candidate.support_count for item in supports) == sum(
            len(raw.rows) for item in supports for raw in item.raw_support
        )
        assert deserialize_policy_input(payloads[-1]).sequence_id == (
            episode["captures"][-1]["sequence_id"]
        )


def test_p68_consumer_fails_closed_on_v2_candidate_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = {
        "schema_version": "hwr.p79-target-candidates/v2",
        "acquisition_input_sha256": [],
        "candidate_count": 0,
        "candidates": [],
    }
    canonical = json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    candidate_set = CandidateSet(
        (), (), canonical, hashlib.sha256(canonical).hexdigest()
    )
    monkeypatch.setattr(
        target_selection,
        "generate_candidate_set_legacy_v1",
        lambda *args, **kwargs: candidate_set,
    )

    with pytest.raises(ValueError, match="legacy-v1"):
        reconstruct_candidate_support(
            (),
            acquisition_base_pose=(0.0, 0.0, 0.0),
            final_input=b"unused",
        )


def test_association_counts_every_support_pixel_and_applies_ratio() -> None:
    candidate = Candidate(
        (1.0, 0.0, 0.5),
        (-1.0, 0.0, 0.0),
        0.1,
        0.1,
        5,
        2,
        0,
        10,
        10,
    )
    raw = RawCandidate(
        candidate.center,
        candidate.normal,
        candidate.width,
        candidate.prominence,
        5,
        0,
        10,
        10,
    )
    segmentation = np.full((192, 256, 2), -1, dtype=np.int32)
    segmentation[10, 10] = (0, 5)
    segmentation[10, 11] = (0, 5)
    segmentation[10, 12] = (0, 5)
    segmentation[10, 13] = (0, 5)
    segmentation[10, 14] = (1, 5)
    table = {
        "background": {"label": "background", "role": "background"},
        "geoms": [
            {"label": "object:duck", "role": "manipulated_object"},
            {"label": "other_furniture", "role": "other_furniture"},
        ],
        "sites": [],
    }

    records = associate_candidates(
        (
            CandidateSupport(
                candidate,
                (RawSupport(raw, (10, 10, 10, 10, 10), (10, 11, 12, 13, 14)),),
            ),
        ),
        (segmentation,),
        table,
        frozenset(("object:duck",)),
    )

    assert records[0]["classification"] == "stage_compatible"
    assert records[0]["compatible_ratio"] == 0.8
    assert records[0]["total_support_count"] == 5


def test_identity_summary_excludes_segmentation_sidecar() -> None:
    names = (
        "environment_seed", "policy_rng_seed",
        "runtime_observation_latency_steps", "runtime_action_latency_steps",
        "latency_override_inactive", "runtime_randomization_sha256",
        "physical_trace_sha256", "policy_input_trace_sha256",
        "observation_identity_trace_sha256", "capture_identity_sequence",
        "capture_payload_sha256", "proposed_action_sha256",
        "applied_action_sha256", "candidate_sha256", "candidate_count",
        "candidate_score_sha256", "selected_index", "trace_step_count",
        "runtime_terminal", "failure", "action_bounds_valid",
        "stale_action_applied_count", "severe_collision_count",
        "invalid_force_count", "p40_conservation_maximum_difference",
        "safety_intervention_count",
    )
    run = {name: name for name in names}
    run["segmentations"] = [np.zeros((1, 1, 2), dtype=np.int32)]

    summary = summary_for_identity(run)

    assert tuple(summary) == names
    assert "segmentations" not in summary
