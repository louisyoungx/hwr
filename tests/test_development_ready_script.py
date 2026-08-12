from pathlib import Path

from scripts.verify_development_ready import (
    FOUNDATION_ALGORITHM_PATHS,
    _algorithm_audit,
    parse_args,
)


ROOT = Path(__file__).resolve().parents[1]


def test_development_ready_command_has_no_skip_switches() -> None:
    parser_values = vars(parse_args([]))

    assert set(parser_values) == {"output", "model_root", "foundation_device"}


def test_foundation_algorithm_audit_has_no_task_specific_branch_literals() -> None:
    report = _algorithm_audit(ROOT)

    assert report["passed"] is True
    assert report["task_literals"] is False
    assert report["scene_training_branches"] is False
    assert "src/hwr/train/foundation_online.py" in FOUNDATION_ALGORITHM_PATHS
