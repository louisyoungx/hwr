from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hwr.apps import evaluate_tool_kinematics as app


def _summary(value: float = 0.005) -> dict[str, object]:
    return {
        "count": 846,
        "finite": True,
        "euclidean_error_m": {
            "mean": value,
            "median": value,
            "p95": value,
            "max": value,
        },
        "absolute_axis_error_m": {
            axis: {"p95": value, "max": value} for axis in ("x", "y", "z")
        },
    }


def _task_reports() -> list[dict[str, object]]:
    return [
        {
            "task_id": task_id,
            "by_arm": {arm: _summary() for arm in app.ARM_ORDER},
            "terminals": [
                {
                    "terminal_id": f"{task_id}|state|{arm}",
                    "task_id": task_id,
                    "arm": arm,
                    "euclidean_error_m": 0.005,
                    "absolute_error_m": [0.003, 0.004, 0.0],
                }
                for arm in app.ARM_ORDER
            ],
        }
        for task_id in app.TASK_IDS
    ]


def _evaluation(*, deterministic: bool = True) -> dict[str, object]:
    task_arm_replay = {
        "expected_task_arm_count": 6,
        "task_arm_count": 6,
        "all_bit_identical": deterministic,
        "task_arms": [
            {
                "task_id": task_id,
                "arm": arm,
                "bit_identical": deterministic,
                "first_payload": {"sha256": "3" * 64, "bytes": 3},
                "second_payload": {"sha256": "3" * 64, "bytes": 3},
            }
            for task_id in app.TASK_IDS
            for arm in app.ARM_ORDER
        ],
    }
    checks = {
        "three_frozen_tasks_present": True,
        "robot_joint_site_mapping_complete_and_consistent": True,
        "every_planned_state_arm_has_unique_finite_terminal": True,
        "frame_invariance_within_tolerance": True,
        "latency_free_same_state": True,
        "evaluator_private_truth_did_not_enter_action": True,
        "deterministic_measurement_payload_bit_identical": deterministic,
        "deterministic_report_payload_bit_identical": deterministic,
        "deterministic_task_arm_payloads_bit_identical": deterministic,
        "all_measurement_contract_gates": deterministic,
    }
    return {
        "state_grid": {
            "state_count": 153,
            "identity": {"sha256": "1" * 64, "bytes": 1},
        },
        "task_reports": _task_reports(),
        "aggregate": {
            "all_task_arm_states": _summary(),
            "task_arm_count": 6,
            "weakest_task_arm": {
                "task_id": app.TASK_IDS[0],
                "arm": "left",
                **_summary(),
            },
        },
        "frame_invariance": {"passed": True},
        "mappings": [],
        "model_identities": [],
        "action_isolation_audit": {"passed": True, "violations": []},
        "deterministic_replay": {
            "passed": deterministic,
            "measurement_payload_bit_identical": deterministic,
            "report_payload_bit_identical": deterministic,
            "first_measurement_payload": {"sha256": "2" * 64, "bytes": 2},
            "second_measurement_payload": {"sha256": "2" * 64, "bytes": 2},
            "task_arm_replay": task_arm_replay,
        },
        "checks": checks,
    }


def _identities() -> dict[str, object]:
    root = Path(app.__file__).resolve().parents[3]
    return {
        "binding": {
            "path": str(app.BINDING_PATH),
            "sha256": app.BINDING_SHA256,
            "bytes": app.BINDING_BYTES,
        },
        "task_config": {
            "path": str(app.TASK_PATH),
            "sha256": app.TASK_SHA256,
            "bytes": app.TASK_BYTES,
        },
        "robot_model_source": {
            "path": str(app.ROBOT_PATH),
            "sha256": app.ROBOT_SHA256,
            "bytes": app.ROBOT_BYTES,
        },
        "sources": {
            name: app._file_identity(root, root / path)
            for name, path in app.SOURCE_PATHS.items()
        },
    }


def test_frozen_source_identities_match_current_repository() -> None:
    root = Path(app.__file__).resolve().parents[3]

    identities = app._source_identities(root)

    assert identities == _identities()
    for name, path in app.SOURCE_PATHS.items():
        content = (root / path).read_bytes()
        assert identities["sources"][name] == {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }


def test_formal_models_publish_complete_recursive_xml_input_identities() -> None:
    root = Path(app.__file__).resolve().parents[3]
    _, bindings = app.load_default_formal_household_catalogs(root)
    shared = {
        f"assets/mujoco/common/robot_{name}.xml"
        for name in (
            "actuators", "assets", "body", "contacts", "defaults", "sensors"
        )
    }

    for task_id in app.TASK_IDS:
        identity = app.recursive_xml_input_identity(
            root, bindings[task_id].model_path
        )
        paths = {value["path"] for value in identity["dependencies"]}
        assert identity["entry_model"] in paths
        assert paths == shared | {identity["entry_model"]}
        assert identity["identity"]["sha256"]


def test_current_p52_sources_pass_executable_action_isolation_audit() -> None:
    root = Path(app.__file__).resolve().parents[3]

    report = app._action_isolation_audit(root)

    assert report["passed"] is True
    assert report["audited_source_count"] == 2
    assert report["violations"] == []


def test_contract_runs_three_tasks_with_matching_mapping_and_state_identity(
    tmp_path, monkeypatch
) -> None:
    tasks = {task_id: SimpleNamespace(task_id=task_id) for task_id in app.TASK_IDS}
    bindings = {
        task_id: SimpleNamespace(task_id=task_id, model_path=tmp_path / f"{index}.xml")
        for index, task_id in enumerate(app.TASK_IDS)
    }
    monkeypatch.setattr(
        app,
        "load_default_formal_household_catalogs",
        lambda root: (tasks, bindings),
    )
    monkeypatch.setattr(
        app,
        "_file_identity",
        lambda root, path: {
            "path": path.name,
            "sha256": hashlib.sha256(path.name.encode()).hexdigest(),
            "bytes": 1,
        },
    )
    monkeypatch.setattr(
        app,
        "recursive_xml_input_identity",
        lambda root, path: {
            "entry_model": path.name,
            "dependencies": [],
            "identity": {"sha256": "4" * 64, "bytes": 4},
        },
    )
    monkeypatch.setattr(
        app,
        "_action_isolation_audit",
        lambda root: {"passed": True, "violations": []},
    )

    class Evaluator:
        def __init__(self, task_id, binding, model_identity):
            del binding, model_identity
            self.task_id = task_id

        def joint_domain(self):
            ranges = ((-1.0, 1.0),) * app.JOINTS_PER_ARM
            return {
                "qpos0": {arm: (0.0,) * app.JOINTS_PER_ARM for arm in app.ARM_ORDER},
                "joint_ranges": {arm: ranges for arm in app.ARM_ORDER},
            }

        def robot_model_identity(self):
            return {"robot": "shared"}

        def mapping_report(self):
            return {
                "task_id": self.task_id,
                "robot_mapping": {"robot": "shared"},
                "mapping_complete": True,
            }

        def measure(self, states):
            terminals = [
                {
                    "terminal_id": f"{self.task_id}|{state.state_id}|{arm}",
                    "task_id": self.task_id,
                    "arm": arm,
                    "euclidean_error_m": 0.005,
                    "absolute_error_m": (0.003, 0.004, 0.0),
                    "latency_free_same_state": True,
                }
                for state in states
                for arm in app.ARM_ORDER
            ]
            return {
                "task_id": self.task_id,
                "planned_state_count": len(states),
                "planned_terminal_count": len(terminals),
                "terminal_count": len(terminals),
                "unique_finite_terminals": True,
                "by_arm": {
                    arm: app.aggregate_task_reports.__globals__["summarize_terminals"](
                        [value for value in terminals if value["arm"] == arm]
                    )
                    for arm in app.ARM_ORDER
                },
                "terminals": terminals,
            }

        def frame_invariance_fixture(self, state):
            del state
            return {"passed": True, "max_absolute_error_m": 0.0}

    monkeypatch.setattr(app, "_ToolSiteEvaluator", Evaluator)

    evaluation = app._evaluate_contract(tmp_path)

    assert evaluation["state_grid"]["state_count"] == 153
    assert [value["task_id"] for value in evaluation["task_reports"]] == list(
        app.TASK_IDS
    )
    assert evaluation["aggregate"]["task_arm_count"] == 6
    assert evaluation["checks"]["robot_joint_site_mapping_complete_and_consistent"]
    assert evaluation["checks"][
        "every_planned_state_arm_has_unique_finite_terminal"
    ]
    assert evaluation["checks"]["evaluator_private_truth_did_not_enter_action"]


def test_report_applies_frozen_decision_and_all_claim_flags() -> None:
    report = app._build_report(
        "a" * 40,
        ("python", "-m", app.MODULE_NAME),
        _evaluation(),
    )

    assert report["decision"] == "accepted as FK agreement contract evidence"
    assert report["measurement_contract_valid"] is True
    assert report["mujoco_sites_are_evaluator_private_labels"] is True
    assert report["action_isolation_audit"]["passed"] is True
    assert report["observation_latency_queue_used"] is False
    assert report["policy_action_modified"] is False
    assert all(report[name] is value for name, value in app.CLAIM_FLAGS.items())

    invalid = _evaluation(deterministic=False)
    assert app._build_report("a" * 40, (), invalid)["decision"] == "invalid"
    isolation_failure = _evaluation()
    isolation_failure["action_isolation_audit"]["passed"] = False
    isolation_failure["checks"][
        "evaluator_private_truth_did_not_enter_action"
    ] = False
    isolation_failure["checks"]["all_measurement_contract_gates"] = False
    assert (
        app._build_report("a" * 40, (), isolation_failure)["decision"]
        == "invalid"
    )


def test_runner_atomically_writes_hash_bound_deterministic_artifacts(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "p52"
    evaluation = _evaluation()
    evaluation.pop("deterministic_replay")
    evaluation["checks"].pop("deterministic_measurement_payload_bit_identical")
    evaluation["checks"].pop("deterministic_report_payload_bit_identical")
    evaluation["checks"].pop("deterministic_task_arm_payloads_bit_identical")
    evaluation["checks"].pop("all_measurement_contract_gates")
    monkeypatch.setattr(app, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(app, "_source_identities", lambda root: _identities())
    monkeypatch.setattr(app, "_require_clean_source", lambda root, identities: None)
    monkeypatch.setattr(app, "_evaluate_contract", lambda root: evaluation)

    result = app.run(app.build_parser().parse_args(["--output", str(output)]))
    report = json.loads((output / "report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())

    assert result["decision"] == "accepted as FK agreement contract evidence"
    assert report["deterministic_replay"]["passed"] is True
    assert report["deterministic_replay"]["report_payload_bit_identical"] is True
    assert report["deterministic_replay"]["task_arm_replay"]["task_arm_count"] == 6
    assert all(
        value["bit_identical"]
        for value in report["deterministic_replay"]["task_arm_replay"]["task_arms"]
    )
    assert set(path.name for path in output.iterdir()) == {
        "report.json",
        "manifest.json",
    }
    for name, identity in manifest["artifacts"].items():
        payload = (output / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    assert manifest["source_identities"] == _identities()
    assert set(manifest["source_identities"]["sources"]) == set(app.SOURCE_PATHS)
    assert manifest["state_grid_identity"] == evaluation["state_grid"]["identity"]
    assert all(manifest[name] is value for name, value in app.CLAIM_FLAGS.items())
    assert not output.with_name(output.name + ".tmp").exists()


def test_clean_source_gate_rejects_dirty_and_nonancestor_source(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M dirty.py\n", returncode=0),
    )
    with pytest.raises(RuntimeError, match="clean committed source"):
        app._require_clean_source(tmp_path, _identities())

    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(returncode=1),
        )
    )
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="frozen document commit"):
        app._require_clean_source(tmp_path, _identities())


def test_clean_source_gate_explicitly_enumerates_untracked_files(
    monkeypatch, tmp_path
) -> None:
    commands = []
    results = iter(
        (
            SimpleNamespace(stdout="", returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0),
        )
    )

    def run(command, **kwargs):
        del kwargs
        commands.append(command)
        return next(results)

    monkeypatch.setattr(app.subprocess, "run", run)

    app._require_clean_source(tmp_path, _identities())

    assert commands[0] == (
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
