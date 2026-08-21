"""Pure contracts for the frozen R0001-P52 tool-kinematics measurement."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from xml.etree import ElementTree

import numpy as np

from hwr.eval.target_selection import _tool_position


PROPOSAL_ID = "R0001-P52"
TASK_IDS = (
    "tidy_living_room_3d/v1",
    "clear_dining_table_3d/v1",
    "store_kitchen_items_3d/v1",
)
ARM_ORDER = ("left", "right")
JOINTS_PER_ARM = 6
STATE_SEED = 20_265_201
HALTON_SAMPLE_COUNT = 128
SINGLE_JOINT_FRACTIONS = (0.20, 0.80)
CENTRAL_RANGE = (0.10, 0.90)
HALTON_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
FRAME_INVARIANCE_TOLERANCE = 1.0e-12
AGREEMENT_P95_METERS = 0.01
AGREEMENT_MAX_METERS = 0.02
MATERIAL_MISMATCH_P95_METERS = 0.03
STATE_GRID_SCHEMA = "hwr.p52-tool-kinematics-state-grid/v1"
TERMINAL_SCHEMA = "hwr.p52-tool-kinematics-terminal/v1"
HALTON_SCHEMA = "digit-permuted-halton-fixed-zero/v1"
QUANTILE_METHOD = "linear"
POLICY_LATERAL_OFFSET = {"left": 0.31, "right": -0.31}
ISOLATION_AUDIT_SCHEMA = "hwr.p52-action-isolation-audit/v1"
MODEL_INPUT_SCHEMA = "hwr.p52-mujoco-xml-inputs/v1"
FORBIDDEN_ACTION_SYMBOLS = frozenset({
    "hwr.core.embodied.DualArmAction", "hwr.core.embodied.DualArmActionFrame",
    "hwr.eval.target_selection.primitive_action",
    "hwr.eval.target_selection.select_candidate_index",
    "hwr.eval.target_selection.select_control_index",
    "hwr.eval.target_selection.candidate_scores",
})
FORBIDDEN_CALL_NAMES = frozenset(
    {
        "DualArmAction", "DualArmActionFrame", "primitive_action",
        "select_candidate_index", "select_control_index", "candidate_scores",
        "apply", "step", "mj_step", "mj_step1", "mj_step2",
        "_write_controls", "_advance_tool_targets",
    }
)


class ToolKinematicsContractError(ValueError):
    """Raised when the frozen state or measurement contract is incomplete."""


@dataclass(frozen=True)
class KinematicState:
    state_id: str
    state_kind: str
    joint_position: tuple[float, ...]

    def arm_joint_position(self, arm: str) -> tuple[float, ...]:
        try:
            offset = ARM_ORDER.index(arm) * JOINTS_PER_ARM
        except ValueError as error:
            raise ToolKinematicsContractError(f"unknown arm: {arm}") from error
        return self.joint_position[offset : offset + JOINTS_PER_ARM]

    def record(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "state_kind": self.state_kind,
            "joint_position": {
                arm: list(self.arm_joint_position(arm)) for arm in ARM_ORDER
            },
        }


def frozen_state_grid(
    qpos0: Mapping[str, Sequence[float]],
    joint_ranges: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[KinematicState, ...]:
    homes, ranges = _validated_joint_domain(qpos0, joint_ranges)
    flattened_home = tuple(value for arm in ARM_ORDER for value in homes[arm])
    states = [KinematicState("qpos0", "qpos0", flattened_home)]
    for arm_index, arm in enumerate(ARM_ORDER):
        for joint_index in range(JOINTS_PER_ARM):
            low, high = ranges[arm][joint_index]
            for fraction in SINGLE_JOINT_FRACTIONS:
                values = list(flattened_home)
                values[arm_index * JOINTS_PER_ARM + joint_index] = (
                    low + fraction * (high - low)
                )
                states.append(
                    KinematicState(
                        f"single-{arm}-joint-{joint_index + 1}-p{round(fraction * 100)}",
                        "single_joint",
                        tuple(values),
                    )
                )
    unit_vectors = scrambled_halton(
        HALTON_SAMPLE_COUNT,
        len(HALTON_BASES),
        seed=STATE_SEED,
    )
    central_low, central_high = CENTRAL_RANGE
    flattened_ranges = tuple(
        value for arm in ARM_ORDER for value in ranges[arm]
    )
    for sample_index, unit in enumerate(unit_vectors):
        values = tuple(
            low + (central_low + (central_high - central_low) * coordinate)
            * (high - low)
            for coordinate, (low, high) in zip(
                unit, flattened_ranges, strict=True
            )
        )
        states.append(
            KinematicState(
                f"halton-{sample_index:03d}",
                "scrambled_halton_central",
                values,
            )
        )
    expected = 1 + len(ARM_ORDER) * JOINTS_PER_ARM * 2 + HALTON_SAMPLE_COUNT
    if len(states) != expected or len({state.state_id for state in states}) != expected:
        raise ToolKinematicsContractError("frozen state grid is incomplete")
    return tuple(states)


def state_grid_report(states: Sequence[KinematicState]) -> dict[str, object]:
    records = [state.record() for state in states]
    payload = {
        "schema_version": STATE_GRID_SCHEMA,
        "seed": STATE_SEED,
        "arm_order": list(ARM_ORDER),
        "joints_per_arm": JOINTS_PER_ARM,
        "qpos0_state_count": 1,
        "single_joint_state_count": len(ARM_ORDER) * JOINTS_PER_ARM * 2,
        "scrambled_halton_state_count": HALTON_SAMPLE_COUNT,
        "halton": {
            "schema_version": HALTON_SCHEMA,
            "bases": list(HALTON_BASES),
            "start_index": 1,
            "digit_zero_fixed": True,
            "central_range_fraction": list(CENTRAL_RANGE),
        },
        "state_count": len(records),
        "states": records,
    }
    identity = _identity(_canonical_bytes(payload))
    return {**payload, "identity": identity}


def scrambled_halton(
    count: int,
    dimension: int,
    *,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    if count <= 0 or not 1 <= dimension <= len(HALTON_BASES):
        raise ToolKinematicsContractError("Halton shape differs from frozen domain")
    if not 0 <= seed < 2**64:
        raise ToolKinematicsContractError("Halton seed must be uint64")
    bases = HALTON_BASES[:dimension]
    permutations = tuple(
        _digit_permutation(base, ordinal, seed)
        for ordinal, base in enumerate(bases)
    )
    return tuple(
        tuple(
            _radical_inverse(index, base, permutation)
            for base, permutation in zip(bases, permutations, strict=True)
        )
        for index in range(1, count + 1)
    )


def policy_tool_position(
    joint_position: Sequence[float],
    arm: str,
) -> tuple[float, float, float]:
    values = _finite_vector(joint_position, JOINTS_PER_ARM, "joint position")
    try:
        lateral = POLICY_LATERAL_OFFSET[arm]
    except KeyError as error:
        raise ToolKinematicsContractError(f"unknown arm: {arm}") from error
    position = _tool_position(values, lateral)
    return tuple(float(value) for value in position)


def world_site_to_base(
    site_world: Sequence[float],
    base_world_position: Sequence[float],
    world_from_base_rotation: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    site = np.asarray(_finite_vector(site_world, 3, "site position"), np.float64)
    origin = np.asarray(
        _finite_vector(base_world_position, 3, "base position"), np.float64
    )
    rotation = np.asarray(world_from_base_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ToolKinematicsContractError("base rotation must be finite 3x3")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12, rtol=0.0):
        raise ToolKinematicsContractError("base rotation must be orthonormal")
    result = rotation.T @ (site - origin)
    return tuple(float(value) for value in result)


def measure_task(
    task_id: str,
    states: Sequence[KinematicState],
    read_sites_in_base: Callable[
        [KinematicState], Mapping[str, Sequence[float]]
    ],
) -> dict[str, object]:
    terminals: list[dict[str, object]] = []
    for state in states:
        sites = read_sites_in_base(state)
        if set(sites) != set(ARM_ORDER):
            raise ToolKinematicsContractError("tool-site arm mapping is incomplete")
        for arm in ARM_ORDER:
            joints = state.arm_joint_position(arm)
            policy = np.asarray(policy_tool_position(joints, arm), np.float64)
            site = np.asarray(
                _finite_vector(sites[arm], 3, "base-frame site position"),
                np.float64,
            )
            difference = policy - site
            absolute = np.abs(difference)
            if not np.isfinite(difference).all():
                raise ToolKinematicsContractError(
                    "policy/site difference contains nonfinite values"
                )
            terminals.append(
                {
                    "schema_version": TERMINAL_SCHEMA,
                    "terminal_id": f"{task_id}|{state.state_id}|{arm}",
                    "task_id": task_id,
                    "state_id": state.state_id,
                    "state_kind": state.state_kind,
                    "arm": arm,
                    "joint_position": list(joints),
                    "policy_tool_position_base_m": policy.tolist(),
                    "mujoco_tool_site_position_base_m": site.tolist(),
                    "error_vector_m": difference.tolist(),
                    "absolute_error_m": absolute.tolist(),
                    "euclidean_error_m": float(np.linalg.norm(difference)),
                    "finite": True,
                    "latency_free_same_state": True,
                }
            )
    expected = len(states) * len(ARM_ORDER)
    terminal_ids = [str(value["terminal_id"]) for value in terminals]
    complete = (
        len(terminals) == expected
        and len(set(terminal_ids)) == expected
        and all(bool(value["finite"]) for value in terminals)
    )
    return {
        "task_id": task_id,
        "planned_state_count": len(states),
        "planned_terminal_count": expected,
        "terminal_count": len(terminals),
        "unique_finite_terminals": complete,
        "by_arm": {
            arm: summarize_terminals(
                [value for value in terminals if value["arm"] == arm]
            )
            for arm in ARM_ORDER
        },
        "terminals": terminals,
    }


def summarize_terminals(
    terminals: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not terminals:
        raise ToolKinematicsContractError("cannot summarize empty terminals")
    errors = np.asarray(
        [value["euclidean_error_m"] for value in terminals], dtype=np.float64
    )
    axes = np.asarray(
        [value["absolute_error_m"] for value in terminals], dtype=np.float64
    )
    if axes.shape != (len(terminals), 3) or not (
        np.isfinite(errors).all() and np.isfinite(axes).all()
    ):
        raise ToolKinematicsContractError("terminal errors must be finite xyz values")
    return {
        "count": len(terminals),
        "finite": True,
        "euclidean_error_m": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p95": _quantile(errors, 0.95),
            "max": float(np.max(errors)),
        },
        "absolute_axis_error_m": {
            axis: {
                "p95": _quantile(axes[:, index], 0.95),
                "max": float(np.max(axes[:, index])),
            }
            for index, axis in enumerate(("x", "y", "z"))
        },
    }


def aggregate_task_reports(
    task_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    terminals = [
        terminal
        for report in task_reports
        for terminal in report["terminals"]
    ]
    task_arms = [
        {
            "task_id": report["task_id"],
            "arm": arm,
            **report["by_arm"][arm],
        }
        for report in task_reports
        for arm in ARM_ORDER
    ]
    if not task_arms:
        raise ToolKinematicsContractError("task-arm reports are empty")
    weakest = max(
        task_arms,
        key=lambda value: (
            value["euclidean_error_m"]["p95"],
            value["euclidean_error_m"]["max"],
            value["task_id"],
            value["arm"],
        ),
    )
    return {
        "all_task_arm_states": summarize_terminals(terminals),
        "task_arm_count": len(task_arms),
        "weakest_task_arm": weakest,
    }


def frame_invariance_report(
    fixture_coordinates: Sequence[
        tuple[str, Mapping[str, Sequence[float]]]
    ],
) -> dict[str, object]:
    if len(fixture_coordinates) < 2:
        raise ToolKinematicsContractError("frame fixture requires transformed poses")
    reference = {
        arm: np.asarray(
            _finite_vector(fixture_coordinates[0][1][arm], 3, "fixture site"),
            dtype=np.float64,
        )
        for arm in ARM_ORDER
    }
    fixtures = []
    maximum = 0.0
    for fixture_id, coordinates in fixture_coordinates:
        arms = {}
        for arm in ARM_ORDER:
            value = np.asarray(
                _finite_vector(coordinates[arm], 3, "fixture site"),
                dtype=np.float64,
            )
            absolute = np.abs(value - reference[arm])
            error = float(np.max(absolute))
            maximum = max(maximum, error)
            arms[arm] = {
                "base_frame_site_position_m": value.tolist(),
                "absolute_error_from_reference_m": absolute.tolist(),
                "max_absolute_error_m": error,
            }
        fixtures.append({"fixture_id": fixture_id, "arms": arms})
    return {
        "fixture_count": len(fixtures),
        "tolerance_m": FRAME_INVARIANCE_TOLERANCE,
        "max_absolute_error_m": maximum,
        "passed": maximum <= FRAME_INVARIANCE_TOLERANCE,
        "fixtures": fixtures,
    }


def frozen_decision(
    checks: Mapping[str, bool],
    aggregate_summary: Mapping[str, object],
) -> str:
    if not checks or not all(checks.values()):
        return "invalid"
    errors = aggregate_summary["euclidean_error_m"]
    p95, maximum = float(errors["p95"]), float(errors["max"])
    if p95 <= AGREEMENT_P95_METERS and maximum <= AGREEMENT_MAX_METERS:
        return "accepted as FK agreement contract evidence"
    if p95 > MATERIAL_MISMATCH_P95_METERS:
        return "accepted as material FK mismatch evidence"
    return "inconclusive"


def audit_action_isolation(
    sources: Mapping[str, str],
) -> dict[str, object]:
    violations: list[dict[str, object]] = []
    audited_sources = []
    for path, source in sorted(sources.items()):
        content = source.encode("utf-8")
        audited_sources.append({"path": path, **_identity(content)})
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as error:
            violations.append(
                {
                    "path": path,
                    "line": error.lineno or 0,
                    "column": error.offset or 0,
                    "kind": "syntax_error",
                    "symbol": error.msg,
                }
            )
            continue
        aliases = _symbol_aliases(tree)
        for node in ast.walk(tree):
            for kind, symbol in _forbidden_node_uses(node, aliases):
                violations.append(
                    {
                        "path": path,
                        "line": int(getattr(node, "lineno", 0)),
                        "column": int(getattr(node, "col_offset", 0)),
                        "kind": kind,
                        "symbol": symbol,
                    }
                )
    violations.sort(
        key=lambda value: (
            value["path"],
            value["line"],
            value["column"],
            value["kind"],
            value["symbol"],
        )
    )
    return {
        "schema_version": ISOLATION_AUDIT_SCHEMA,
        "passed": not violations,
        "audited_source_count": len(audited_sources),
        "audited_sources": audited_sources,
        "forbidden_action_symbols": sorted(FORBIDDEN_ACTION_SYMBOLS),
        "forbidden_call_names": sorted(FORBIDDEN_CALL_NAMES),
        "violations": violations,
    }


def recursive_xml_input_identity(root: Path, entry: Path) -> dict[str, object]:
    repository = root.resolve()
    entry = entry.resolve()
    pending, records = [entry], {}
    while pending:
        path = pending.pop()
        relative = _relative_input_path(repository, path)
        if relative in records:
            continue
        content = path.read_bytes()
        document = ElementTree.fromstring(content)
        includes = []
        for element in document.iter():
            if element.tag.rsplit("}", 1)[-1] != "include":
                continue
            include = element.attrib.get("file")
            if not include:
                raise ToolKinematicsContractError("MuJoCo include is missing file")
            dependency = (path.parent / include).resolve()
            includes.append(_relative_input_path(repository, dependency))
            pending.append(dependency)
        records[relative] = {
            "path": relative,
            **_identity(content),
            "includes": sorted(includes),
        }
    dependencies = [records[path] for path in sorted(records)]
    payload = {
        "schema_version": MODEL_INPUT_SCHEMA,
        "entry_model": _relative_input_path(repository, entry),
        "dependencies": dependencies,
    }
    return {**payload, "identity": _identity(_canonical_bytes(payload))}


def task_arm_replay_status(
    first_reports: Sequence[Mapping[str, object]],
    second_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    first_by_task = _unique_task_reports(first_reports)
    second_by_task = _unique_task_reports(second_reports)
    records = []
    extras = sorted((set(first_by_task) | set(second_by_task)) - set(TASK_IDS))
    task_ids = (*TASK_IDS, *extras)
    for task_id in task_ids:
        for arm in ARM_ORDER:
            first_payload = _task_arm_payload(first_by_task.get(task_id), arm)
            second_payload = _task_arm_payload(second_by_task.get(task_id), arm)
            first_content = None if first_payload is None else _canonical_bytes(first_payload)
            second_content = None if second_payload is None else _canonical_bytes(second_payload)
            records.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "bit_identical": (
                        first_content is not None
                        and second_content is not None
                        and first_content == second_content
                    ),
                    "first_payload": None if first_content is None else _identity(first_content),
                    "second_payload": None if second_content is None else _identity(second_content),
                }
            )
    expected = len(TASK_IDS) * len(ARM_ORDER)
    return {
        "expected_task_arm_count": expected,
        "task_arm_count": len(records),
        "all_bit_identical": (
            len(records) == expected
            and all(value["bit_identical"] for value in records)
        ),
        "task_arms": records,
    }


def _validated_joint_domain(
    qpos0: Mapping[str, Sequence[float]],
    joint_ranges: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[
    dict[str, tuple[float, ...]],
    dict[str, tuple[tuple[float, float], ...]],
]:
    if set(qpos0) != set(ARM_ORDER) or set(joint_ranges) != set(ARM_ORDER):
        raise ToolKinematicsContractError("joint domain must contain both arms")
    homes, ranges = {}, {}
    for arm in ARM_ORDER:
        homes[arm] = _finite_vector(qpos0[arm], JOINTS_PER_ARM, "qpos0")
        values = tuple(
            _finite_vector(value, 2, "joint range")
            for value in joint_ranges[arm]
        )
        if len(values) != JOINTS_PER_ARM or any(low >= high for low, high in values):
            raise ToolKinematicsContractError("joint ranges differ from six valid ranges")
        if any(
            not low <= home <= high
            for home, (low, high) in zip(homes[arm], values, strict=True)
        ):
            raise ToolKinematicsContractError("qpos0 is outside a joint range")
        ranges[arm] = values
    return homes, ranges


def _digit_permutation(base: int, dimension: int, seed: int) -> tuple[int, ...]:
    domain = f"{HALTON_SCHEMA}|{seed}|{dimension}|{base}|".encode("ascii")
    shuffled = sorted(
        range(1, base),
        key=lambda digit: hashlib.sha256(domain + str(digit).encode()).digest(),
    )
    return (0, *shuffled)


def _radical_inverse(
    index: int,
    base: int,
    permutation: Sequence[int],
) -> float:
    result, factor = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += permutation[digit] * factor
        factor /= base
    return result


def _symbol_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".", 1)[0]
                aliases[local] = item.name if item.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = (
                        f"{node.module}.{item.name}"
                    )
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            value = _resolved_symbol(node.value, aliases)
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != value:
                    aliases[target.id] = value
                    changed = True
        if not changed:
            break
    return aliases


def _forbidden_node_uses(
    node: ast.AST,
    aliases: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    uses = []
    if isinstance(node, ast.Import):
        for item in node.names:
            reason = _forbidden_import(item.name)
            if reason:
                uses.append(("forbidden_import", reason))
    elif isinstance(node, ast.ImportFrom) and node.module:
        for item in node.names:
            reason = _forbidden_import(f"{node.module}.{item.name}")
            if reason:
                uses.append(("forbidden_import", reason))
    elif isinstance(node, ast.Call):
        symbol = _resolved_symbol(node.func, aliases)
        reason = _forbidden_call(symbol)
        if reason:
            uses.append(("forbidden_call", reason))
        dynamic = _forbidden_dynamic_lookup(node, aliases)
        if dynamic:
            uses.append(("forbidden_dynamic_lookup", dynamic))
    elif isinstance(node, (ast.Name, ast.Attribute)):
        symbol = _resolved_symbol(node, aliases)
        reason = _forbidden_reference(symbol)
        if reason:
            uses.append(("forbidden_reference", reason))
    return tuple(uses)


def _resolved_symbol(
    node: ast.AST,
    aliases: Mapping[str, str],
) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolved_symbol(node.value, aliases)
        return node.attr if owner is None else f"{owner}.{node.attr}"
    return None


def _forbidden_import(symbol: str) -> str | None:
    if symbol in FORBIDDEN_ACTION_SYMBOLS:
        return symbol
    tail = symbol.rsplit(".", 1)[-1]
    if (
        (
            tail == "*"
            and symbol.rsplit(".", 1)[0]
            in {"hwr.core.embodied", "hwr.eval.target_selection"}
        )
        or tail in FORBIDDEN_CALL_NAMES
        or tail.endswith("Action")
        or tail.endswith("ActionFrame")
        or tail.endswith("Backend")
        or tail == "backend"
        or tail.endswith("_backend")
    ):
        return symbol
    module = symbol.rsplit(".", 1)[0] if "." in symbol else symbol
    module_tail = module.rsplit(".", 1)[-1]
    if (
        module.startswith("hwr.adapters.mujoco.")
        and (module_tail == "backend" or module_tail.endswith("_backend"))
    ):
        return symbol
    return None


def _forbidden_call(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    if symbol in FORBIDDEN_ACTION_SYMBOLS:
        return symbol
    tail = symbol.rsplit(".", 1)[-1]
    return (
        symbol
        if (
            tail in FORBIDDEN_CALL_NAMES
            or tail.endswith("Action")
            or tail.endswith("ActionFrame")
        )
        else None
    )


def _forbidden_reference(symbol: str | None) -> str | None:
    if symbol is None or "." not in symbol:
        return None
    if symbol in FORBIDDEN_ACTION_SYMBOLS:
        return symbol
    return _forbidden_call(symbol)


def _forbidden_dynamic_lookup(
    node: ast.Call,
    aliases: Mapping[str, str],
) -> str | None:
    function = _resolved_symbol(node.func, aliases)
    if function not in {"getattr", "builtins.getattr"} and not str(function).endswith(".__getattribute__"):
        return None
    index = 0 if function.endswith("__getattribute__") else 1
    if len(node.args) <= index or not isinstance(node.args[index], ast.Constant):
        return None
    name = node.args[index].value
    return str(name) if name in FORBIDDEN_CALL_NAMES else None


def _relative_input_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ToolKinematicsContractError(
            "MuJoCo XML dependency escapes repository root"
        ) from error
    if not path.is_file():
        raise ToolKinematicsContractError(
            f"MuJoCo XML dependency is unavailable: {relative.as_posix()}"
        )
    return relative.as_posix()


def _unique_task_reports(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result = {}
    for report in reports:
        task_id = str(report["task_id"])
        if task_id in result:
            raise ToolKinematicsContractError("duplicate task replay report")
        result[task_id] = report
    return result


def _task_arm_payload(
    report: Mapping[str, object] | None,
    arm: str,
) -> dict[str, object] | None:
    if report is None or arm not in report["by_arm"]:
        return None
    return {
        "task_id": report["task_id"],
        "arm": arm,
        "summary": report["by_arm"][arm],
        "terminals": [
            value for value in report["terminals"] if value["arm"] == arm
        ],
    }


def _finite_vector(
    values: Sequence[float],
    length: int,
    name: str,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ToolKinematicsContractError(
            f"{name} must contain {length} finite values"
        )
    return result


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method=QUANTILE_METHOD))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _identity(content: bytes) -> dict[str, object]:
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
