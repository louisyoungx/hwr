from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from hwr.adapters.mujoco.entity_candidate_mapping import (
    ALIAS_PROPOSAL_ID,
    ALIAS_SCHEMA,
    EntityAlias,
    EntityCandidateMappingError,
    TaskAliasContract,
    TaskVisibleGeom,
    build_exact_geom_role_table,
    classify_segmentation_entity,
    load_entity_alias_contracts,
    preflight_exact_geom_role_tables,
)
from hwr.adapters.mujoco.training_catalog import (
    load_default_formal_household_catalogs,
)
from hwr.apps import audit_entity_candidate_mapping as app


@dataclass(frozen=True)
class _Object:
    body: str
    collision_geom: str


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
            <geom name="wall" type="box" size=".1 .1 .1"/>
            <body name="robot">
              <geom name="robot_geom" size=".1"/>
              <geom size=".1"/>
            </body>
            <body name="item">
              <geom name="item_visual" size=".1"/>
              <geom name="item_collision" size=".1"/>
            </body>
            <body name="drawer">
              <geom name="handle_visual" size=".1"/>
              <geom name="handle" size=".1"/>
              <geom name="drawer_container" size=".1"/>
              <site name="target_site" size=".01"/>
            </body>
            <body name="other">
              <geom name="other_visual" size=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _binding(
    *,
    object_geom: str = "item_collision",
    target_geom: str = "drawer_container",
):
    return SimpleNamespace(
        task_id="fixture/v1",
        objects={"item": _Object("item", object_geom)},
        articulation=_Articulation("drawer", "handle"),
        allowed_robot_contact_roles={
            "target_container": frozenset({target_geom}),
            "floor_support": frozenset({"floor"}),
            "manipulated_object": frozenset({object_geom}),
            "articulation": frozenset({"handle"}),
        },
    )


def _contract(
    *,
    source: str = "item_visual",
    target: str = "item_collision",
    role: str = "manipulated_object",
    instance: str | None = "item",
    inventory: tuple[TaskVisibleGeom, ...] | None = None,
) -> TaskAliasContract:
    return TaskAliasContract(
        "fixture/v1",
        (EntityAlias(source, target, role, instance),),
        (
            TaskVisibleGeom(source, role, instance),
            TaskVisibleGeom(
                "handle_visual", "articulation", "drawer"
            ),
            TaskVisibleGeom(
                "drawer_container", "target_container", None
            ),
        )
        if inventory is None
        else inventory,
    )


def _robot_body(model: mujoco.MjModel) -> int:
    return int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    )


def _geom(table: dict[str, object], name: str) -> dict[str, object]:
    return next(
        value for value in table["geoms"] if value["geom_name"] == name
    )


def _real_contract():
    root = Path(__file__).resolve().parents[1]
    _, bindings = load_default_formal_household_catalogs(root)
    aliases = load_entity_alias_contracts(root / app.ALIASES_PATH)
    return root, bindings, aliases


def test_exact_geom_mapping_allows_distinct_roles_on_one_body() -> None:
    model = _model()
    aliases = TaskAliasContract(
        "fixture/v1",
        (
            EntityAlias(
                "item_visual",
                "item_collision",
                "manipulated_object",
                "item",
            ),
            EntityAlias(
                "handle_visual", "handle", "articulation", "drawer"
            ),
        ),
        (
            TaskVisibleGeom(
                "item_visual", "manipulated_object", "item"
            ),
            TaskVisibleGeom(
                "handle_visual", "articulation", "drawer"
            ),
            TaskVisibleGeom(
                "drawer_container", "target_container", None
            ),
        ),
    )

    table = build_exact_geom_role_table(
        model,
        _binding(),
        aliases,
        robot_root_body=_robot_body(model),
    )

    assert _geom(table, "handle_visual")["role"] == "articulation"
    assert _geom(table, "drawer_container")["role"] == "target_container"
    drawer = next(
        value for value in table["bodies"]
        if value["body_name"] == "drawer"
    )
    assert drawer["geom_roles"] == ["articulation", "target_container"]


def test_exact_claim_does_not_spread_to_unaliased_same_body_geom() -> None:
    model = _model()
    contract = TaskAliasContract(
        "fixture/v1",
        (
            EntityAlias(
                "handle_visual", "handle", "articulation", "drawer"
            ),
        ),
        (
            TaskVisibleGeom(
                "handle_visual", "articulation", "drawer"
            ),
            TaskVisibleGeom(
                "drawer_container", "target_container", None
            ),
        ),
    )

    table = build_exact_geom_role_table(
        model,
        _binding(),
        contract,
        robot_root_body=_robot_body(model),
    )

    assert _geom(table, "item_collision")["role"] == "manipulated_object"
    assert _geom(table, "item_visual")["role"] == "other_furniture"
    assert _geom(table, "item_visual")["claim_kind"] == "fallback"


def test_same_exact_geom_conflict_is_invalid() -> None:
    model = _model()

    with pytest.raises(
        EntityCandidateMappingError, match="conflicting task roles"
    ) as error:
        build_exact_geom_role_table(
            model,
            _binding(target_geom="item_collision"),
            TaskAliasContract("fixture/v1", (), ()),
            robot_root_body=_robot_body(model),
        )

    assert error.value.details["failure_kind"] == "exact_geom_role_conflict"
    assert error.value.details["geom_name"] == "item_collision"


@pytest.mark.parametrize(
    ("contract", "message"),
    (
        (
            _contract(source="other_visual"),
            "crosses body boundary",
        ),
        (
            _contract(source="handle"),
            "independent exact task claim",
        ),
        (
            _contract(target="other_visual"),
            "not an exact claimed geom",
        ),
        (
            _contract(role="target_container", instance=None),
            "role or instance differs",
        ),
        (
            TaskAliasContract(
                "fixture/v1",
                (
                    EntityAlias(
                        "item_visual",
                        "item_collision",
                        "manipulated_object",
                        "item",
                    ),
                    EntityAlias(
                        "handle_visual",
                        "item_visual",
                        "manipulated_object",
                        "item",
                    ),
                ),
                (),
            ),
            "chain or cycle",
        ),
    ),
)
def test_alias_validation_fails_closed(
    contract: TaskAliasContract,
    message: str,
) -> None:
    model = _model()

    with pytest.raises(EntityCandidateMappingError, match=message):
        build_exact_geom_role_table(
            model,
            _binding(),
            contract,
            robot_root_body=_robot_body(model),
        )


def test_inventory_role_mismatch_is_rejected() -> None:
    model = _model()
    contract = _contract(
        inventory=(
            TaskVisibleGeom(
                "item_visual", "target_container", None
            ),
        )
    )

    with pytest.raises(
        EntityCandidateMappingError,
        match="inventory does not resolve exactly",
    ) as error:
        build_exact_geom_role_table(
            model,
            _binding(),
            contract,
            robot_root_body=_robot_body(model),
        )

    assert error.value.details["failure_kind"] == (
        "task_visible_inventory_mismatch"
    )


def test_frozen_alias_file_is_exact_and_complete() -> None:
    _, _, contracts = _real_contract()

    assert list(contracts) == list(app.TASK_IDS)
    assert sum(len(value.aliases) for value in contracts.values()) == 8
    assert [
        value.source_visual_geom
        for task in contracts.values()
        for value in task.aliases
    ] == [
        "toy_duck_visual",
        "toy_football_visual",
        "storage_basket_visual",
        "dining_cup_visual",
        "dining_plate_visual",
        "cleaner_yellow_visual",
        "cleaner_pink_visual",
        "drawer_handle_visual",
    ]


def test_alias_loader_rejects_duplicate_tasks_and_extra_fields(
    tmp_path: Path,
) -> None:
    task = {
        "task_id": "fixture/v1",
        "aliases": [],
        "task_visible_inventory": [],
    }
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": ALIAS_SCHEMA,
                "proposal_id": ALIAS_PROPOSAL_ID,
                "tasks": [task, task],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EntityCandidateMappingError, match="duplicate alias task"):
        load_entity_alias_contracts(path)

    path.write_text(
        json.dumps(
            {
                "schema_version": ALIAS_SCHEMA,
                "proposal_id": ALIAS_PROPOSAL_ID,
                "tasks": [{**task, "unexpected": True}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EntityCandidateMappingError, match="task fields differ"):
        load_entity_alias_contracts(path)


def test_real_three_scene_inventory_guards_and_determinism() -> None:
    _, bindings, aliases = _real_contract()

    first = preflight_exact_geom_role_tables(
        bindings, aliases, app.TASK_IDS
    )
    second = preflight_exact_geom_role_tables(
        bindings, aliases, app.TASK_IDS
    )
    evaluation = app.evaluate_mapping_contract(
        bindings,
        aliases,
        {"passed": True, "violations": []},
    )

    assert first == second
    assert [first[task]["sha256"] for task in app.TASK_IDS] == [
        second[task]["sha256"] for task in app.TASK_IDS
    ]
    assert [first[task]["geom_count"] for task in app.TASK_IDS] == [
        70,
        72,
        77,
    ]
    assert [first[task]["body_count"] for task in app.TASK_IDS] == [
        30,
        32,
        33,
    ]
    assert app.negative_guard_audit(first)["mismatch_count"] == 0
    assert evaluation["report"]["decision"] == (
        "accepted as exact-geom evaluator mapping contract"
    )
    guards = evaluation["report"]["negative_guards"]
    assert guards["illegal_object_type_case_count"] == 3
    assert guards["illegal_object_types_fail_closed"] is True
    assert evaluation["report"]["checks"][
        "illegal_segmentation_object_types_fail_closed"
    ] is True
    assert evaluation["report"]["checks"]["all_mapping_contract_gates"] is True
    kitchen = first[app.TASK_IDS[2]]
    drawer = next(
        value for value in kitchen["bodies"]
        if value["body_name"] == "kitchen_drawer"
    )
    assert drawer["geom_roles"] == ["articulation", "target_container"]
    assert _geom(kitchen, "drawer_handle_visual")[
        "canonical_exact_claimed_geom"
    ] == "drawer_handle"
    assert _geom(kitchen, "drawer_frame_left")["role"] == "other_furniture"


def test_sites_background_and_illegal_object_types_are_unknown() -> None:
    model = _model()
    aliases = TaskAliasContract(
        "fixture/v1",
        (
            EntityAlias(
                "item_visual",
                "item_collision",
                "manipulated_object",
                "item",
            ),
            EntityAlias(
                "handle_visual", "handle", "articulation", "drawer"
            ),
        ),
        (
            TaskVisibleGeom(
                "item_visual", "manipulated_object", "item"
            ),
            TaskVisibleGeom(
                "handle_visual", "articulation", "drawer"
            ),
            TaskVisibleGeom(
                "drawer_container", "target_container", None
            ),
        ),
    )
    table = build_exact_geom_role_table(
        model,
        _binding(),
        aliases,
        robot_root_body=_robot_body(model),
    )
    site_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
    )

    assert classify_segmentation_entity(table, -1, -1)["role"] == "background"
    assert classify_segmentation_entity(
        table, site_id, int(mujoco.mjtObj.mjOBJ_SITE)
    )["role"] == "unknown_site"
    assert classify_segmentation_entity(
        table, 0, int(mujoco.mjtObj.mjOBJ_CAMERA)
    )["role"] == "unknown"


def test_illegal_type_guard_rejects_and_isolation_failure_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bindings, aliases = _real_contract()
    monkeypatch.setattr(
        app,
        "classify_segmentation_entity",
        lambda table, object_id, object_type: {"role": "robot"},
    )
    rejected = app.evaluate_mapping_contract(
        bindings, aliases, {"passed": True}
    )
    assert rejected["report"]["decision"] == "rejected_design_not_expressive"

    monkeypatch.undo()
    invalid = app.evaluate_mapping_contract(
        bindings, aliases, {"passed": False, "violations": [{}]}
    )
    assert invalid["report"]["decision"] == "invalid"


def test_current_candidate_policy_action_sources_pass_ast_isolation() -> None:
    root = Path(__file__).resolve().parents[1]

    report = app.audit_alias_isolation(root)

    assert report["passed"] is True
    assert report["audited_source_count"] >= 30
    assert report["violations"] == []


@pytest.mark.parametrize(
    "source",
    (
        "import hwr.adapters.mujoco.entity_candidate_mapping\n",
        "from hwr.adapters.mujoco import entity_candidate_mapping\n",
        (
            "from hwr.adapters.mujoco.entity_candidate_mapping "
            "import load_entity_alias_contracts\n"
        ),
        "__import__('hwr.apps.audit_entity_candidate_mapping')\n",
        "path = 'configs/eval/entity_candidate_aliases_v1.json'\n",
    ),
)
def test_ast_isolation_rejects_alias_imports_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    path = tmp_path / "policy.py"
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(app, "_isolation_paths", lambda root: (path,))

    report = app.audit_alias_isolation(tmp_path)

    assert report["passed"] is False
    assert len(report["violations"]) >= 1


def test_frozen_document_history_and_input_provenance_are_complete() -> None:
    root, bindings, _ = _real_contract()

    identities = app._source_identities(
        root, bindings, root / app.ALIASES_PATH
    )

    frozen = identities["frozen_document"]
    assert frozen["commit_is_ancestor"] is True
    assert frozen["content_matches"] is True
    assert frozen["blob_matches"] is True
    context = identities["frozen_context"]
    assert context["commit_is_ancestor"] is True
    assert context["content_matches"] is True
    assert context["blob_matches"] is True
    trees = identities["historical_research_loop_trees"]
    assert trees["actual_matches_context"] is True
    assert trees["actual"] == trees["declared_by_frozen_context"]
    assert list(trees["actual"]) == [
        f"docs/research-loop/{index:04d}" for index in range(1, 12)
    ]
    assert set(identities["recursive_xml"]) == set(app.TASK_IDS)
    assert {
        "aliases",
        "binding",
        "task_config",
        "mapping",
        "audit_app",
        "frozen_context",
        "frozen_document",
    } <= set(identities["files"])
    for record in identities["files"].values():
        assert record["bytes"] > 0
        assert len(record["sha256"]) == 64


def test_context_tree_inventory_requires_exact_order_and_valid_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / app.FROZEN_CONTEXT_PATH).read_text(encoding="utf-8")
    parsed = app._context_tree_inventory(content)

    assert list(parsed) == [
        f"docs/research-loop/{index:04d}" for index in range(1, 12)
    ]
    with pytest.raises(RuntimeError, match="incomplete"):
        app._context_tree_inventory(
            content.replace(
                "| `docs/research-loop/0011/` "
                "| `85bb445726ecb8e35ff4d8e90606874e2ee36fe4` |\n",
                "",
            )
        )
    with pytest.raises(RuntimeError, match="invalid"):
        app._context_tree_inventory(
            content.replace(
                "416912b7dc1c19611bcfc4375028180014a1989b",
                "not-a-tree",
            )
        )


@pytest.mark.parametrize(
    "field", ("commit_is_ancestor", "content_matches", "blob_matches")
)
def test_source_gate_rejects_each_frozen_context_identity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    valid = {
        "commit_is_ancestor": True,
        "content_matches": True,
        "blob_matches": True,
    }
    identities = {
        "frozen_document": valid,
        "frozen_context": {**valid, field: False},
        "historical_research_loop_trees": {"actual_matches_context": True},
    }
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", returncode=0),
    )

    with pytest.raises(RuntimeError, match="frozen context drifted"):
        app._require_clean_source(tmp_path, identities)


def test_source_gate_rejects_tree_inventory_not_matching_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = {
        "commit_is_ancestor": True,
        "content_matches": True,
        "blob_matches": True,
    }
    identities = {
        "frozen_document": valid,
        "frozen_context": valid,
        "historical_research_loop_trees": {"actual_matches_context": False},
    }
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", returncode=0),
    )

    with pytest.raises(RuntimeError, match="historical research trees"):
        app._require_clean_source(tmp_path, identities)


def test_output_and_staging_overwrite_are_rejected(tmp_path: Path) -> None:
    root = tmp_path
    aliases = root / app.ALIASES_PATH
    aliases.parent.mkdir(parents=True)
    aliases.write_text("{}", encoding="utf-8")
    output = root / app.FORMAL_OUTPUT
    output.parent.mkdir(parents=True)
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        app._validate_arguments(root, aliases, output)

    output.rmdir()
    output.with_name(output.name + ".tmp").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        app._validate_arguments(root, aliases, output)


def test_atomic_output_writes_all_artifacts_and_cleans_failed_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "success"
    artifacts = {
        "tables.json": b"tables\n",
        "report.json": b"report\n",
        "manifest.json": b"manifest\n",
    }

    app._create_output(output, artifacts)

    assert {
        path.name: path.read_bytes() for path in output.iterdir()
    } == artifacts
    assert not output.with_name(output.name + ".tmp").exists()

    failed = tmp_path / "failed"
    real_replace = app.os.replace

    def fail_final_replace(source, destination):
        if Path(source) == failed.with_name(failed.name + ".tmp"):
            raise OSError("synthetic final rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(app.os, "replace", fail_final_replace)
    with pytest.raises(OSError, match="synthetic final rename failure"):
        app._create_output(failed, artifacts)
    assert not failed.exists()
    assert not failed.with_name(failed.name + ".tmp").exists()
