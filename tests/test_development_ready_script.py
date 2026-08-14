from pathlib import Path

from scripts.verify_development_ready import (
    EXPECTED_TASK_IDS,
    _algorithm_audit,
    _committed_snapshot,
    _configuration_audit,
    _forbidden_configuration_keys,
    _forbidden_foundation_import,
    _model_selection_audit,
    parse_args,
)
from hwr.train.development_gate import current_commit


ROOT = Path(__file__).resolve().parents[1]


def test_development_ready_command_has_no_skip_switches() -> None:
    parser_values = vars(parse_args([]))

    assert set(parser_values) == {"output", "model_root", "foundation_device"}


def test_foundation_algorithm_audit_has_no_task_specific_branch_literals() -> None:
    report = _algorithm_audit(ROOT)

    assert report["passed"] is True
    assert report["task_literals"] is False
    assert report["scene_training_branches"] is False
    assert "src/hwr/train/foundation_online.py" in report["files"]
    assert "src/hwr/world_model/model.py" in report["files"]
    assert "src/hwr/policy/latent_actor.py" in report["files"]
    assert "src/hwr/perception/student.py" in report["files"]


def test_foundation_algorithm_audit_rejects_any_expert_module_family() -> None:
    assert _forbidden_foundation_import("hwr.adapters.mujoco.formal_expert")
    assert _forbidden_foundation_import("hwr.future.formal_expert_types")
    assert not _forbidden_foundation_import("hwr.world_model.model")


def test_foundation_configuration_audit_rejects_nested_forbidden_lineage(
    tmp_path,
) -> None:
    config = tmp_path / "configs/foundation"
    config.mkdir(parents=True)
    (config / "fixture.json").write_text(
        '{"nested": [{"object_token": "forbidden"}]}'
    )

    assert _forbidden_configuration_keys(tmp_path) == (
        "fixture.json.nested[0]:object_token",
    )


def test_development_checks_use_an_isolated_committed_snapshot() -> None:
    with _committed_snapshot(ROOT) as snapshot:
        assert snapshot.resolve() != ROOT.resolve()
        assert current_commit(snapshot) == current_commit(ROOT)
        assert (snapshot / "scripts/check_python_size.py").is_file()


def test_foundation_model_selection_is_bound_across_source_lock_and_runtime() -> None:
    report = _model_selection_audit(ROOT, ROOT / "models/foundation")

    assert report["passed"] is True
    assert "dinov3-vits16-pretrain-lvd1689m" in report["models"]


def test_formal_causality_audit_uses_task_balanced_optimizer_disjoint_data() -> None:
    report = _configuration_audit(ROOT)

    assert report["task_count"] == len(EXPECTED_TASK_IDS)
    assert report["forbidden_lineage_keys"] is False
    assert report["causality_holdout_episodes_per_task"] >= 2
    assert report["causality_audit_windows_per_task"] >= 8
    assert (
        report["causality_audit_windows_per_task"]
        % report["causality_audit_batch_size"]
        == 0
    )
    assert report["replay_windows_per_episode"] >= 1
    assert report["estimated_run_storage_gib"] < 30.0
    assert report["holdout_teacher_visual_features"] is False
