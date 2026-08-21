"""Run the frozen R0001-P52 policy-FK to MuJoCo tool-site measurement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hwr.adapters.mujoco import model as mujoco_model
from hwr.adapters.mujoco.bindings import MujocoTaskBinding
from hwr.adapters.mujoco.model import MujocoModelBundle
from hwr.adapters.mujoco.names import ARM_JOINTS, SECONDARY_ARM_JOINTS
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.eval.tool_kinematics import (
    AGREEMENT_MAX_METERS,
    AGREEMENT_P95_METERS,
    ARM_ORDER,
    CENTRAL_RANGE,
    FRAME_INVARIANCE_TOLERANCE,
    HALTON_BASES,
    HALTON_SAMPLE_COUNT,
    HALTON_SCHEMA,
    JOINTS_PER_ARM,
    MATERIAL_MISMATCH_P95_METERS,
    PROPOSAL_ID,
    QUANTILE_METHOD,
    SINGLE_JOINT_FRACTIONS,
    STATE_SEED,
    TASK_IDS,
    KinematicState,
    aggregate_task_reports,
    audit_action_isolation,
    frame_invariance_report,
    frozen_decision,
    frozen_state_grid,
    measure_task,
    recursive_xml_input_identity,
    state_grid_report,
    task_arm_replay_status,
    world_site_to_base,
)


MODULE_NAME = "hwr.apps.evaluate_tool_kinematics"
REPORT_SCHEMA = "hwr.p52-tool-kinematics-report/v1"
MANIFEST_SCHEMA = "hwr.p52-tool-kinematics-artifacts/v1"
FAILURE_SCHEMA = "hwr.p52-tool-kinematics-failure/v1"
FROZEN_DOCUMENT_COMMIT = "4385ceee2fffcbd23788b498d258747dc273465c"
FROZEN_PARENT_COMMIT = "0fea5f3fce43a9d00ab902138ff1aea63015f1d0"
BINDING_PATH = Path("configs/adapters/mujoco/formal_3d_v1.json")
BINDING_SHA256 = "7984ef2544bb618269681d274257a598b02621371a26de002bfdd8bbf7decab6"
BINDING_BYTES = 3051
TASK_PATH = Path("configs/tasks/formal_3d_v1.json")
TASK_SHA256 = "fa180803a86b42bc633dbf119fa596dd74d3b5c18bf4c2e4f75be97dcccb2a7d"
TASK_BYTES = 5480
ROBOT_PATH = Path("assets/mujoco/common/robot_body.xml")
ROBOT_SHA256 = "a936257039b0037e978897f63e3d012431ff902c311f4428935996d6747da095"
ROBOT_BYTES = 14003
SOURCE_PATHS = {
    "p52_app": Path("src/hwr/apps/evaluate_tool_kinematics.py"),
    "p52_core": Path("src/hwr/eval/tool_kinematics.py"),
    "target_selection_fk_source": Path("src/hwr/eval/target_selection.py"),
    "mujoco_model_loader": Path("src/hwr/adapters/mujoco/model.py"),
    "mujoco_names": Path("src/hwr/adapters/mujoco/names.py"),
    "mujoco_training_catalog": Path("src/hwr/adapters/mujoco/training_catalog.py"),
    "mujoco_binding_loader": Path("src/hwr/adapters/mujoco/bindings.py"),
    "formal_task_catalog": Path("src/hwr/scenarios/formal3d.py"),
}
TOOL_SITE_NAMES = {"left": "left_grasp_center", "right": "right_grasp_center"}
JOINT_NAMES = {
    "left": SECONDARY_ARM_JOINTS,
    "right": ARM_JOINTS,
}
FRAME_FIXTURES = (
    ("identity", (0.0, 0.0, 0.22), 0.0),
    ("translation", (1.25, -0.75, 0.47), 0.0),
    ("yaw_positive", (-0.40, 1.10, 0.22), 1.20),
    ("translation_yaw_negative", (0.90, 0.35, 0.31), -2.10),
)
CLAIM_FLAGS = {
    "measurement_only": True,
    "training_executed": False,
    "policy_inference_executed": False,
    "closed_loop_capability_episode_executed": False,
    "capability_claim_allowed": False,
    "task_success_claim_allowed": False,
    "generalization_claim_allowed": False,
    "hardware_safety_claim_allowed": False,
    "action_causality_claim_allowed": False,
    "policy_action_modified": False,
    "evaluator_private_site_used_for_action": False,
    "evaluator_private_qpos_used_for_action": False,
    "object_truth_used": False,
    "observation_latency_queue_used": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, arguments.output)
    staging = output.with_name(output.name + ".tmp")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    command = [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        "--output",
        str(arguments.output),
    ]
    identities = _source_identities(root)
    try:
        _require_clean_source(root, identities)
        first = _evaluate_contract(root)
        second = _evaluate_contract(root)
        first_payload = _canonical_bytes(first)
        second_payload = _canonical_bytes(second)
        measurement_equal = first_payload == second_payload
        task_arm_replay = task_arm_replay_status(
            first["task_reports"], second["task_reports"]
        )
        replay = {
            "measurement_payload_bit_identical": measurement_equal,
            "first_measurement_payload": _identity(first_payload),
            "second_measurement_payload": _identity(second_payload),
            "task_arm_replay": task_arm_replay,
        }
        first_evaluation = _with_determinism(
            first, replay, report_equal=True
        )
        second_evaluation = _with_determinism(
            second, replay, report_equal=True
        )
        first_report = _build_report(source_commit, command, first_evaluation)
        second_report = _build_report(source_commit, command, second_evaluation)
        report_equal = (
            _canonical_bytes(first_report) == _canonical_bytes(second_report)
        )
        evaluation = _with_determinism(
            first, replay, report_equal=report_equal
        )
        replay_evaluation = _with_determinism(
            second, replay, report_equal=report_equal
        )
        report = _build_report(source_commit, command, evaluation)
        replay_report = _build_report(
            source_commit, command, replay_evaluation
        )
        if (
            _canonical_bytes(report) == _canonical_bytes(replay_report)
        ) != report_equal:
            raise RuntimeError("P52 report determinism accounting is inconsistent")
        artifacts = {"report.json": _json_bytes(report)}
        manifest = _manifest(
            source_commit,
            command,
            identities,
            evaluation,
            artifacts,
            status="complete",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "proposal_id": PROPOSAL_ID,
            "source_commit": source_commit,
            "error_type": type(error).__name__,
            "error": str(error),
            **CLAIM_FLAGS,
        }
        artifacts = {"failure.json": _json_bytes(failure)}
        manifest = _manifest(
            source_commit,
            command,
            identities,
            None,
            artifacts,
            status="failed",
        )
        artifacts["manifest.json"] = _json_bytes(manifest)
        _create_output(output, artifacts)
        raise
    return {
        "output": str(output),
        "decision": report["decision"],
        "aggregate_p95_m": report["aggregate"]["all_task_arm_states"][
            "euclidean_error_m"
        ]["p95"],
        "aggregate_max_m": report["aggregate"]["all_task_arm_states"][
            "euclidean_error_m"
        ]["max"],
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "manifest_sha256": hashlib.sha256(
            artifacts["manifest.json"]
        ).hexdigest(),
    }


def _with_determinism(
    evaluation: Mapping[str, object],
    replay: Mapping[str, object],
    *,
    report_equal: bool,
) -> dict[str, object]:
    measurement_equal = bool(replay["measurement_payload_bit_identical"])
    task_arm_replay = replay["task_arm_replay"]
    checks = {
        **evaluation["checks"],
        "deterministic_measurement_payload_bit_identical": measurement_equal,
        "deterministic_report_payload_bit_identical": report_equal,
        "deterministic_task_arm_payloads_bit_identical": bool(
            task_arm_replay["all_bit_identical"]
        ),
    }
    checks["all_measurement_contract_gates"] = all(checks.values())
    replay_by_task_arm = {
        (value["task_id"], value["arm"]): value
        for value in task_arm_replay["task_arms"]
    }
    task_reports = [
        {
            **report,
            "by_arm": {
                arm: {
                    **report["by_arm"][arm],
                    "deterministic_replay": replay_by_task_arm.get(
                        (report["task_id"], arm)
                    ),
                }
                for arm in ARM_ORDER
            },
        }
        for report in evaluation["task_reports"]
    ]
    return {
        **evaluation,
        "task_reports": task_reports,
        "deterministic_replay": {
            **replay,
            "report_payload_bit_identical": report_equal,
            "passed": (
                measurement_equal
                and report_equal
                and task_arm_replay["all_bit_identical"]
            ),
        },
        "checks": checks,
    }


def _evaluate_contract(root: Path) -> dict[str, object]:
    isolation_audit = _action_isolation_audit(root)
    tasks, bindings = load_default_formal_household_catalogs(root)
    if tuple(tasks) != TASK_IDS or tuple(bindings) != TASK_IDS:
        raise RuntimeError("P52 task/binding catalog differs from frozen order")
    task_reports, mappings, model_identities = [], [], []
    reference_domain = None
    reference_model_identity = None
    reference_states = None
    frame_fixture = None
    for task_id in TASK_IDS:
        binding = bindings[task_id]
        model_identity = recursive_xml_input_identity(root, binding.model_path)
        model_identities.append({"task_id": task_id, **model_identity})
        evaluator = _ToolSiteEvaluator(task_id, binding, model_identity)
        domain = evaluator.joint_domain()
        states = frozen_state_grid(domain["qpos0"], domain["joint_ranges"])
        grid = state_grid_report(states)
        task_report = evaluator.measure(states)
        task_reports.append(task_report)
        mappings.append(evaluator.mapping_report())
        if reference_domain is None:
            reference_domain = _canonical_bytes(domain)
            reference_model_identity = evaluator.robot_model_identity()
            reference_states = grid
            frame_fixture = evaluator.frame_invariance_fixture(states[0])
        elif (
            _canonical_bytes(domain) != reference_domain
            or evaluator.robot_model_identity() != reference_model_identity
            or grid["identity"] != reference_states["identity"]
        ):
            raise RuntimeError("P52 robot joint/site mapping differs across tasks")
    if reference_states is None or frame_fixture is None:
        raise RuntimeError("P52 produced no task measurements")
    aggregate = aggregate_task_reports(task_reports)
    expected_states = int(reference_states["state_count"])
    expected_terminals = expected_states * len(ARM_ORDER)
    mapping_consistent = len(
        {_canonical_bytes(mapping["robot_mapping"]) for mapping in mappings}
    ) == 1
    checks = {
        "three_frozen_tasks_present": [value["task_id"] for value in task_reports]
        == list(TASK_IDS),
        "robot_joint_site_mapping_complete_and_consistent": mapping_consistent
        and all(value["mapping_complete"] for value in mappings),
        "every_planned_state_arm_has_unique_finite_terminal": all(
            value["planned_state_count"] == expected_states
            and value["planned_terminal_count"] == expected_terminals
            and value["terminal_count"] == expected_terminals
            and value["unique_finite_terminals"]
            for value in task_reports
        ),
        "frame_invariance_within_tolerance": bool(frame_fixture["passed"]),
        "latency_free_same_state": all(
            terminal["latency_free_same_state"]
            for value in task_reports
            for terminal in value["terminals"]
        ),
        "evaluator_private_truth_did_not_enter_action": (
            isolation_audit["passed"]
        ),
    }
    return {
        "state_grid": reference_states,
        "task_reports": task_reports,
        "aggregate": aggregate,
        "frame_invariance": frame_fixture,
        "mappings": mappings,
        "model_identities": model_identities,
        "action_isolation_audit": isolation_audit,
        "checks": checks,
    }


class _ToolSiteEvaluator:
    """Evaluator-private direct MuJoCo kinematics without rendering or actions."""

    def __init__(
        self,
        task_id: str,
        binding: MujocoTaskBinding,
        model_identity: Mapping[str, object],
    ) -> None:
        self.task_id = task_id
        self.binding = binding
        self.bundle = MujocoModelBundle.load(binding.model_path, object_joint_name=None)
        self.model = self.bundle.model
        self.data = _new_data(self.model)
        self.model_identity = dict(model_identity)
        self.joint_ids = {
            "left": self.bundle.ids.secondary_arm_joints,
            "right": self.bundle.ids.arm_joints,
        }
        self.site_ids = {
            arm: _site_id(self.model, name) for arm, name in TOOL_SITE_NAMES.items()
        }

    def joint_domain(self) -> dict[str, object]:
        return {
            "qpos0": {
                arm: [
                    float(self.model.qpos0[self.model.jnt_qposadr[joint]])
                    for joint in self.joint_ids[arm]
                ]
                for arm in ARM_ORDER
            },
            "joint_ranges": {
                arm: [
                    [float(value) for value in self.model.jnt_range[joint]]
                    for joint in self.joint_ids[arm]
                ]
                for arm in ARM_ORDER
            },
        }

    def mapping_report(self) -> dict[str, object]:
        mapping = self.robot_model_identity()
        return {
            "task_id": self.task_id,
            "task_binding_identity": {
                "task_id": self.binding.task_id,
                "model": self.model_identity,
            },
            "scene_model": {
                "name": self.binding.model_path.name,
                "nq": int(self.model.nq),
                "nv": int(self.model.nv),
            },
            "robot_mapping": mapping,
            "mapping_complete": (
                self.binding.task_id == self.task_id
                and all(len(self.joint_ids[arm]) == JOINTS_PER_ARM for arm in ARM_ORDER)
                and all(self.site_ids[arm] >= 0 for arm in ARM_ORDER)
                and all(
                    int(self.model.jnt_limited[joint]) == 1
                    and float(self.model.jnt_range[joint][0])
                    < float(self.model.jnt_range[joint][1])
                    for arm in ARM_ORDER
                    for joint in self.joint_ids[arm]
                )
            ),
        }

    def robot_model_identity(self) -> dict[str, object]:
        domain = self.joint_domain()
        return {
            "base_body_name": self.model.body(self.bundle.ids.base_body).name,
            "base_joint_name": self.model.joint(self.bundle.ids.base_joint).name,
            "joint_names": {arm: list(JOINT_NAMES[arm]) for arm in ARM_ORDER},
            "joint_qpos_addresses": {
                arm: [
                    int(self.model.jnt_qposadr[joint])
                    for joint in self.joint_ids[arm]
                ]
                for arm in ARM_ORDER
            },
            "joint_ranges": domain["joint_ranges"],
            "qpos0": domain["qpos0"],
            "tool_site_names": dict(TOOL_SITE_NAMES),
            "tool_site_ids": {
                arm: int(self.site_ids[arm]) for arm in ARM_ORDER
            },
        }

    def measure(self, states: Sequence[KinematicState]) -> dict[str, object]:
        return {
            **measure_task(self.task_id, states, self._sites_for_state),
            "robot_model": {
                **self.robot_model_identity(),
                "scene_model_name": self.binding.model_path.name,
                "scene_model_nq": int(self.model.nq),
                "scene_model_nv": int(self.model.nv),
            },
            "task_binding_identity": {
                "task_id": self.binding.task_id,
                "model": self.model_identity,
            },
        }

    def frame_invariance_fixture(
        self, state: KinematicState
    ) -> dict[str, object]:
        measurements = []
        for fixture_id, position, yaw in FRAME_FIXTURES:
            sites = self._sites_for_state(
                state,
                base_position=position,
                base_yaw=yaw,
            )
            measurements.append((fixture_id, sites))
        return {
            **frame_invariance_report(measurements),
            "state_id": state.state_id,
            "base_pose_fixtures": [
                {
                    "fixture_id": fixture_id,
                    "world_position_m": list(position),
                    "yaw_rad": yaw,
                }
                for fixture_id, position, yaw in FRAME_FIXTURES
            ],
        }

    def _sites_for_state(
        self,
        state: KinematicState,
        *,
        base_position: Sequence[float] = (0.0, 0.0, 0.22),
        base_yaw: float = 0.0,
    ) -> dict[str, tuple[float, float, float]]:
        mujoco_model.mujoco.mj_resetData(self.model, self.data)
        base_address = int(
            self.model.jnt_qposadr[self.bundle.ids.base_joint]
        )
        half_yaw = float(base_yaw) / 2.0
        self.data.qpos[base_address : base_address + 7] = (
            *base_position,
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        )
        for arm in ARM_ORDER:
            for value, joint in zip(
                state.arm_joint_position(arm),
                self.joint_ids[arm],
                strict=True,
            ):
                self.data.qpos[self.model.jnt_qposadr[joint]] = value
        mujoco_model.mujoco.mj_forward(self.model, self.data)
        base_rotation = self.data.xmat[
            self.bundle.ids.base_body
        ].reshape(3, 3)
        base_origin = self.data.xpos[self.bundle.ids.base_body]
        return {
            arm: world_site_to_base(
                self.data.site_xpos[self.site_ids[arm]],
                base_origin,
                base_rotation,
            )
            for arm in ARM_ORDER
        }


def _build_report(
    source_commit: str,
    command: Sequence[str],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    checks = evaluation["checks"]
    aggregate = evaluation["aggregate"]
    decision = frozen_decision(
        checks,
        aggregate["all_task_arm_states"],
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "source_commit": source_commit,
        "command": list(command),
        "decision": decision,
        "hypothesis_thresholds_m": {
            "agreement_aggregate_p95_maximum": AGREEMENT_P95_METERS,
            "agreement_aggregate_max_maximum": AGREEMENT_MAX_METERS,
            "material_mismatch_aggregate_p95_strictly_above": (
                MATERIAL_MISMATCH_P95_METERS
            ),
        },
        "measurement_contract_valid": all(checks.values()),
        "latency_free_same_physical_state": True,
        "mujoco_sites_are_evaluator_private_labels": True,
        "policy_fk_function": "hwr.eval.target_selection._tool_position",
        "state_grid": evaluation["state_grid"],
        "task_reports": evaluation["task_reports"],
        "aggregate": aggregate,
        "frame_invariance": evaluation["frame_invariance"],
        "mappings": evaluation["mappings"],
        "action_isolation_audit": evaluation["action_isolation_audit"],
        "deterministic_replay": evaluation["deterministic_replay"],
        "checks": checks,
        **CLAIM_FLAGS,
    }


def _manifest(
    source_commit: str,
    command: Sequence[str],
    identities: Mapping[str, object],
    evaluation: Mapping[str, object] | None,
    artifacts: Mapping[str, bytes],
    *,
    status: str,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "status": status,
        "source_commit": source_commit,
        "frozen_parent_commit": FROZEN_PARENT_COMMIT,
        "frozen_document_commit": FROZEN_DOCUMENT_COMMIT,
        "frozen_document_commit_is_ancestor": status == "complete",
        "historical_research_loop_documents_unchanged": status == "complete",
        "command": list(command),
        "source_identities": identities,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "mujoco": importlib.metadata.version("mujoco"),
        },
        "constants": {
            "task_ids": list(TASK_IDS),
            "arm_order": list(ARM_ORDER),
            "joint_names": {
                arm: list(JOINT_NAMES[arm]) for arm in ARM_ORDER
            },
            "tool_site_names": dict(TOOL_SITE_NAMES),
            "state_seed": STATE_SEED,
            "single_joint_fractions": list(SINGLE_JOINT_FRACTIONS),
            "halton_sample_count": HALTON_SAMPLE_COUNT,
            "halton_bases": list(HALTON_BASES),
            "halton_schema": HALTON_SCHEMA,
            "central_range_fraction": list(CENTRAL_RANGE),
            "quantile_method": QUANTILE_METHOD,
            "frame_invariance_tolerance_m": FRAME_INVARIANCE_TOLERANCE,
        },
        "state_grid_identity": (
            None if evaluation is None else evaluation["state_grid"]["identity"]
        ),
        "model_identities": (
            None if evaluation is None else evaluation["model_identities"]
        ),
        "action_isolation_audit": (
            None if evaluation is None else evaluation["action_isolation_audit"]
        ),
        **CLAIM_FLAGS,
        "artifacts": {
            name: _identity(content) for name, content in artifacts.items()
        },
    }


def _source_identities(root: Path) -> dict[str, object]:
    return {
        "binding": _file_identity(root, root / BINDING_PATH),
        "task_config": _file_identity(root, root / TASK_PATH),
        "robot_model_source": _file_identity(root, root / ROBOT_PATH),
        "sources": {
            name: _file_identity(root, root / path)
            for name, path in SOURCE_PATHS.items()
        },
    }


def _action_isolation_audit(root: Path) -> dict[str, object]:
    return audit_action_isolation(
        {
            path.as_posix(): (root / path).read_text(encoding="utf-8")
            for path in (
                SOURCE_PATHS["p52_app"],
                SOURCE_PATHS["p52_core"],
            )
        }
    )


def _require_clean_source(
    root: Path,
    identities: Mapping[str, object],
) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("P52 runner requires clean committed source")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", FROZEN_DOCUMENT_COMMIT, "HEAD"),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("P52 frozen document commit is not an ancestor")
    history = tuple(
        f"docs/research-loop/{index:04d}" for index in range(1, 9)
    )
    unchanged = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            FROZEN_PARENT_COMMIT,
            "HEAD",
            "--",
            *history,
        ),
        cwd=root,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("P52 historical research-loop documents drifted")
    frozen = (
        ("binding", BINDING_SHA256, BINDING_BYTES),
        ("task_config", TASK_SHA256, TASK_BYTES),
        ("robot_model_source", ROBOT_SHA256, ROBOT_BYTES),
    )
    for name, sha256, size in frozen:
        identity = identities[name]
        if identity["sha256"] != sha256 or identity["bytes"] != size:
            raise RuntimeError(f"P52 {name} identity differs from frozen contract")


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("P52 runner requires a full Git source commit")
    return commit


def _file_identity(root: Path, path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        **_identity(content),
    }


def _site_id(model, name: str) -> int:
    site = int(
        mujoco_model.mujoco.mj_name2id(
            model,
            mujoco_model.mujoco.mjtObj.mjOBJ_SITE,
            name,
        )
    )
    if site < 0:
        raise ValueError(f"P52 model is missing evaluator-private site {name}")
    return site


def _new_data(model):
    return mujoco_model.mujoco.MjData(model)


def _create_output(output: Path, artifacts: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    staging.mkdir()
    try:
        for name, content in artifacts.items():
            _atomic_write(staging / name, content)
        os.replace(staging, output)
    except BaseException:
        for path in staging.glob("*"):
            path.unlink()
        if staging.exists():
            staging.rmdir()
        raise


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _identity(content: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["decision"].startswith("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
