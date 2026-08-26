from __future__ import annotations

import json
from pathlib import Path

from hwr.apps.audit_interaction_contract import build_source_audit
from hwr.eval.interaction_contract import (
    audit_interaction_contract,
    source_requirement_fields,
)
from hwr.eval.interaction_contract_mutation import (
    MUTATION_NAMES,
    audit_interaction_contract_mutations,
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "6bf0400f51a25bfb6f45e951299c410efd5c2c7a"


def _git_text(path: str) -> str:
    import subprocess

    return subprocess.run(
        ("git", "show", f"{PRODUCER}:{path}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _inputs():
    paths = __import__("subprocess").run(
        ("git", "ls-tree", "-r", "--name-only", PRODUCER, "src/hwr"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    sources = {path: _git_text(path) for path in paths if path.endswith(".py")}
    return (
        json.loads(_git_text("configs/eval/interaction_contract_v1.json")),
        json.loads(_git_text("configs/tasks/formal_3d_v1.json")),
        json.loads(_git_text("configs/adapters/mujoco/formal_3d_v1.json")),
        sources,
        json.loads(
            (
                ROOT
                / "runs/research-loop/0012/"
                "r0012-p61-interaction-contract-s20266101/transitions.json"
            ).read_text()
        ),
    )


def test_all_frozen_mutations_are_valid_and_expose_expected_residuals() -> None:
    result = audit_interaction_contract_mutations(
        *_inputs(),
        build_source_audit=build_source_audit,
        audit_contract=audit_interaction_contract,
        requirement_fields=source_requirement_fields,
    )
    report = result["report"]

    assert report["harness_checks"]["passed"] is True
    assert report["mutation_count"] == len(MUTATION_NAMES) == 14
    assert report["p68_dependency_gate_passed"] is True
    assert report["decision"] == "accepted as residual P61 contract gap evidence"
    assert report["residual_exact_reference_gaps"] == list(MUTATION_NAMES[:5])
    assert report["residual_planner_evidence_gap"] is True
    assert report["residual_verdict_dependency_gaps"] == []
    role_only = result["mutations"]["mutations"][9]["observations"]
    assert role_only["planner_call_state_available"] is True
    assert role_only["independent_planner_state_or_call_evidence"] is False


def test_mutations_are_single_variable_replayed_and_reach_auditor() -> None:
    result = audit_interaction_contract_mutations(
        *_inputs(),
        build_source_audit=build_source_audit,
        audit_contract=audit_interaction_contract,
        requirement_fields=source_requirement_fields,
    )

    for record in result["mutations"]["mutations"]:
        assert record["clean_copy_verified"] is True
        assert record["mutation_reached"] is True
        assert record["single_variable_verified"] is True
        assert record["mutated_sources_compile"] is True
        assert record["canonical_replay_bit_identical"] is True
