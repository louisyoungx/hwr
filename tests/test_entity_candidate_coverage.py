from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from hwr.adapters.mujoco.entity_candidate_mapping import (
    EntityCandidateMappingError,
    build_entity_role_table,
    preflight_entity_role_tables,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.apps import evaluate_entity_candidate_coverage as app


@dataclass(frozen=True)
class _Object:
    body: str


@dataclass(frozen=True)
class _Articulation:
    articulation_id: str
    handle_geom: str


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 .1"/>
            <body name="robot"><geom name="robot_geom" size=".1"/></body>
            <body name="item"><geom name="item_visual" size=".1"/></body>
            <body name="drawer">
              <geom name="handle" size=".1"/>
              <geom name="drawer_container" size=".1"/>
            </body>
            <body name="container"><geom name="container" size=".1"/></body>
          </worldbody>
        </mujoco>
        """
    )


def _binding(*, conflict: bool = False):
    target = "drawer_container" if conflict else "container"
    return SimpleNamespace(
        task_id="fixture/v1",
        objects={"item": _Object("item")},
        articulation=_Articulation("drawer", "handle"),
        allowed_robot_contact_roles={
            "target_container": frozenset({target}),
            "floor_support": frozenset({"floor"}),
        },
    )


def test_mapping_uses_exact_body_and_is_deterministic() -> None:
    model = _model()
    robot = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robot"
    )

    first = build_entity_role_table(
        model, _binding(), robot_root_body=robot
    )
    second = build_entity_role_table(
        model, _binding(), robot_root_body=robot
    )

    item = next(
        value for value in first["geoms"]
        if value["geom_name"] == "item_visual"
    )
    assert item["label"] == "object:item"
    assert item["role"] == "manipulated_object"
    assert first == second
    assert len(first["sha256"]) == 64


def test_mapping_keeps_multi_role_body_conflict_fail_closed() -> None:
    model = _model()
    robot = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robot"
    )

    with pytest.raises(
        EntityCandidateMappingError, match="multiple task roles"
    ):
        build_entity_role_table(
            model, _binding(conflict=True), robot_root_body=robot
        )


def test_three_formal_scenes_stop_at_real_kitchen_conflict() -> None:
    root = Path(__file__).resolve().parents[1]
    _, bindings = load_default_formal_household_catalogs(root)

    with pytest.raises(
        EntityCandidateMappingError, match="preflight is infeasible"
    ) as error:
        preflight_entity_role_tables(bindings, app.TASK_IDS)

    assert error.value.details["failed_task_id"] == (
        "store_kitchen_items_3d/v1"
    )
    assert [
        value["task_id"] for value in error.value.details["completed_tasks"]
    ] == list(app.TASK_IDS[:2])
    assert error.value.details["episode_count"] == 0
    assert error.value.details["physical_acquisition_count"] == 0


def _result(stdout="", returncode=0):
    return SimpleNamespace(
        stdout=stdout, returncode=returncode
    )


def _successful_git_results(root: Path, source_commit: str):
    document = (root / app.FROZEN_DOCUMENT_PATH).read_bytes()
    trees = list(app.HISTORICAL_TREES.values())
    return iter(
        (
            _result(""),
            _result(returncode=0),
            _result(document),
            _result(app.FROZEN_DOCUMENT_BLOB + "\n"),
            _result(returncode=0),
            *(_result(value + "\n") for value in trees),
        )
    )


def test_source_gate_accepts_frozen_clean_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    results = _successful_git_results(root, "a" * 40)

    report = app.require_preflight_source(
        root, "a" * 40, runner=lambda *args, **kwargs: next(results)
    )

    assert report["passed"] is True
    assert report["historical_research_loop_trees"] == app.HISTORICAL_TREES


@pytest.mark.parametrize(
    ("results", "message"),
    (
        (lambda root: iter((_result("?? untracked.py\n"),)), "clean committed"),
        (lambda root: iter((_result(" M tracked.py\n"),)), "clean committed"),
        (
            lambda root: iter((_result(""), _result(returncode=1))),
            "not an ancestor",
        ),
        (
            lambda root: iter(
                (
                    _result(""),
                    _result(returncode=0),
                    _result(b"changed"),
                    _result(app.FROZEN_DOCUMENT_BLOB + "\n"),
                )
            ),
            "document content drifted",
        ),
        (
            lambda root: iter(
                (
                    _result(""),
                    _result(returncode=0),
                    _result((root / app.FROZEN_DOCUMENT_PATH).read_bytes()),
                    _result("f" * 40 + "\n"),
                )
            ),
            "document content drifted",
        ),
        (
            lambda root: iter(
                (
                    _result(""),
                    _result(returncode=0),
                    _result((root / app.FROZEN_DOCUMENT_PATH).read_bytes()),
                    _result(app.FROZEN_DOCUMENT_BLOB + "\n"),
                    _result(returncode=1),
                )
            ),
            "target/config/XML drifted",
        ),
    ),
)
def test_source_gate_rejects_dirty_ancestry_and_frozen_drift(
    results, message: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    values = results(root)

    with pytest.raises(RuntimeError, match=message):
        app.require_preflight_source(
            root, "a" * 40, runner=lambda *args, **kwargs: next(values)
        )


def test_source_gate_rejects_historical_tree_drift() -> None:
    root = Path(__file__).resolve().parents[1]
    values = list(_successful_git_results(root, "a" * 40))
    values[-1] = _result("f" * 40 + "\n")
    results = iter(values)

    with pytest.raises(RuntimeError, match="historical research trees"):
        app.require_preflight_source(
            root, "a" * 40, runner=lambda *args, **kwargs: next(results)
        )


def test_five_frozen_inputs_are_bound_and_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {}
    replacements = {}
    for name, (_, _) in app.FROZEN_INPUTS.items():
        content = json.dumps({"name": name}).encode()
        target = tmp_path / f"{name}.json"
        target.write_bytes(content)
        payloads[name] = content
        replacements[name] = (target.relative_to(tmp_path), hashlib.sha256(
            content
        ).hexdigest())
    monkeypatch.setattr(app, "FROZEN_INPUTS", replacements)

    records = app.read_frozen_inputs(tmp_path)

    assert set(records) == {
        "plan", "capsules", "e1_report", "e1_manifest", "e2_report"
    }
    (tmp_path / replacements["e2_report"][0]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="e2_report"):
        app.read_frozen_inputs(tmp_path)


def test_real_five_input_hashes_and_lineage_are_valid() -> None:
    root = Path(__file__).resolve().parents[1]

    records = app.read_frozen_inputs(root)
    app.validate_frozen_input_lineage(records)

    assert set(records) == {
        "plan", "capsules", "e1_report", "e1_manifest", "e2_report"
    }


def test_output_overwrite_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(app, "FORMAL_PLAN", Path("plan.json"))
    monkeypatch.setattr(app, "FORMAL_CAPSULES", Path("capsules.json"))

    with pytest.raises(FileExistsError, match="already exists"):
        app._validate_arguments(
            tmp_path,
            SimpleNamespace(
                plan=Path("plan.json"),
                historical_capsules=Path("capsules.json"),
            ),
            output,
        )


def test_runner_writes_zero_episode_atomic_failure_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan.json"
    capsules = tmp_path / "capsules.json"
    output = tmp_path / "output"
    plan.write_text("{}")
    capsules.write_text("{}")
    monkeypatch.setattr(app, "FORMAL_PLAN", plan)
    monkeypatch.setattr(app, "FORMAL_CAPSULES", capsules)
    monkeypatch.setattr(app, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(app, "candidate_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        app, "require_preflight_source",
        lambda root, commit: {"passed": True},
    )
    inputs = {
        name: {
            "path": name,
            "sha256": "a" * 64,
            "bytes": 2,
            "content": b"{}",
        }
        for name in app.FROZEN_INPUTS
    }
    monkeypatch.setattr(app, "read_frozen_inputs", lambda root: inputs)
    monkeypatch.setattr(app, "validate_frozen_input_lineage", lambda values: None)
    monkeypatch.setattr(app, "source_identities", lambda root: {})
    monkeypatch.setattr(
        app, "load_default_formal_household_catalogs",
        lambda root: ({}, {}),
    )
    monkeypatch.setattr(
        app,
        "preflight_entity_role_tables",
        lambda bindings, tasks: (_ for _ in ()).throw(
            EntityCandidateMappingError(
                "frozen entity mapping preflight is infeasible",
                details={
                    "failed_task_id": app.TASK_IDS[2],
                    "completed_task_count": 2,
                    "completed_tasks": [],
                    "episode_count": 0,
                    "physical_acquisition_count": 0,
                },
            )
        ),
    )
    monkeypatch.setattr(app, "mujoco_runtime_version", lambda: "test")
    monkeypatch.setattr(
        app.importlib.metadata, "version", lambda name: "test"
    )
    monkeypatch.setattr(
        app, "candidate_commit_is_ancestor", lambda *args: True
    )

    result = app.run(
        SimpleNamespace(
            plan=plan,
            historical_capsules=capsules,
            output=output,
        )
    )

    failure = json.loads((output / "failure.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert result["decision"] == "inconclusive_design_infeasible"
    assert failure["episode_count"] == 0
    assert failure["physical_acquisition_count"] == 0
    assert failure["measurement_evidence_accepted"] is False
    assert manifest["status"] == "failed"
    assert manifest["preflight_only"] is True
    assert set(manifest["fixed_inputs"]) == set(app.FROZEN_INPUTS)
    assert not output.with_name(output.name + ".tmp").exists()

