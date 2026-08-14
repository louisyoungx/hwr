from __future__ import annotations

from pathlib import Path

from hwr.train.development_semantics import (
    FORMAL_HOUSEHOLD_TASK_IDS,
    verify_training_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_executable_training_semantics_gate_passes_the_formal_stack() -> None:
    report = verify_training_semantics(ROOT)

    assert report["passed"] is True
    checks = report["checks"]
    reachability = checks["visual_deployment_reachability"]
    assert all(value > 0.0 for value in reachability["gradient_norms"].values())
    assert all(reachability["parameters_changed"].values())
    assert checks["formal_household_contract"]["task_ids"] == sorted(
        FORMAL_HOUSEHOLD_TASK_IDS
    )
    assert checks["retained_replay_contract"]["retained_transitions"] >= 10_000
    assert checks["action_bounds_contract"]["dimension"] == 16
