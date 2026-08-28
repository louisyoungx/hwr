"""Pure schema, cohort, and reachability logic for the R0001-P87 oracle."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any
REGISTRY_SCHEMA = "hwr.r0017-experiment-contract-registry/v1"
P50_SCHEMA = "hwr.p50-acquisition-capsule-index/v1"
P79_SCHEMA = "hwr.p79-candidate-bank/v1"
SUPPORTED_STRATA = dict(
    task=("task_id",), observation_latency=("observation_latency_steps",),
    action_latency=("action_latency_steps",),
    latency_pair=("observation_latency_steps", "action_latency_steps"),
    cell=("task_id", "observation_latency_steps", "action_latency_steps"))
DENOMINATOR_RULES = dict(
    all_episodes={"candidate_count_minimum": 0},
    choice_opportunity={"candidate_count_minimum": 2},
    empty={"candidate_count_maximum": 0},
    nonempty={"candidate_count_minimum": 1})
EXPOSURE_POLICIES = {
    "include_exposed", "exclude_matching_outcome_fields", "historical_design_audit"}
EXPOSURE_FIELDS = {
    "safe_b2_entry", "wall_seconds", "lineage_match", "safety_intervention",
    "runtime_terminal", "invalid_force", "conservation_difference"}
class ContractOracleError(ValueError):
    """Raised with a stable category when an experiment contract is invalid."""
    def __init__(self, category: str, detail: object | None = None) -> None:
        self.category = category
        self.detail = detail
        super().__init__(category if detail is None else f"{category}: {detail}")
def validate_registry(registry: Mapping[str, object]) -> None:
    """Validate registry syntax without consulting any scientific outcome."""
    required = {
        "schema_version", "proposal_id", "sample_unit", "sources",
        "expected_cohort", "denominators", "strata",
        "result_exposure_ledger", "contracts"}
    if set(registry) != required:
        raise ContractOracleError("registry_keys")
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ContractOracleError("registry_schema")
    if registry.get("proposal_id") != "R0001-P87":
        raise ContractOracleError("registry_proposal")
    if registry.get("sample_unit") != "Episode":
        raise ContractOracleError("sample_unit")
    _validate_sources(_mapping(registry.get("sources"), "sources"))
    _validate_expected(_mapping(registry.get("expected_cohort"), "expected_cohort"))
    _validate_denominators(_mapping(registry.get("denominators"), "denominators"))
    strata = _mapping(registry.get("strata"), "strata")
    if strata != {name: list(fields) for name, fields in SUPPORTED_STRATA.items()}:
        raise ContractOracleError("strata_schema")
    ledger = _list(registry.get("result_exposure_ledger"), "exposure_ledger")
    _validate_ledger(ledger)
    contracts = _list(registry.get("contracts"), "contracts")
    if not contracts:
        raise ContractOracleError("contract_missing")
    seen: set[str] = set()
    for value in contracts:
        contract = _mapping(value, "contract_type")
        _validate_contract(contract, set(registry["denominators"]))
        identity = str(contract["contract_id"])
        if identity in seen:
            raise ContractOracleError("contract_duplicate", identity)
        seen.add(identity)
def _validate_sources(sources: Mapping[str, object]) -> None:
    expected_files = {
        "p50": {"capsules.json", "plan.json", "report.json", "manifest.json"},
        "p79": {"bank.json", "manifest.json"},
        "p83": {"report.json", "manifest.json"},
    }
    if set(sources) != set(expected_files):
        raise ContractOracleError("source_scope")
    for source, names in expected_files.items():
        value = _mapping(sources[source], "source_identity")
        if set(value) != {"path", "files"} or not _safe_relative(value["path"]):
            raise ContractOracleError("source_scope", source)
        files = _mapping(value["files"], "source_files")
        if set(files) != names:
            raise ContractOracleError("source_scope", source)
        if any(not _sha256(digest) for digest in files.values()):
            raise ContractOracleError("input_hash", source)
def _validate_expected(expected: Mapping[str, object]) -> None:
    scalar = {
        "episode_count", "nonempty_count", "empty_count",
        "choice_opportunity_count", "cell_count"}
    if set(expected) != scalar | {
        "task_nonempty_counts", "latency_pair_nonempty_counts"}:
        raise ContractOracleError("expected_cohort_keys")
    for name in scalar:
        _nonnegative_int(expected[name], "expected_cohort")
    for name in ("task_nonempty_counts", "latency_pair_nonempty_counts"):
        values = _mapping(expected[name], "expected_cohort")
        if not values or any(not isinstance(key, str) for key in values):
            raise ContractOracleError("expected_cohort")
        for count in values.values():
            _nonnegative_int(count, "expected_cohort")
def _validate_denominators(denominators: Mapping[str, object]) -> None:
    if set(denominators) != set(DENOMINATOR_RULES):
        raise ContractOracleError("denominator_partition")
    for name, semantics in DENOMINATOR_RULES.items():
        definition = _mapping(denominators[name], "denominator_definition")
        if set(definition) != set(semantics) | {"expected_count"}:
            raise ContractOracleError("denominator_semantics", name)
        if any(definition.get(key) != value for key, value in semantics.items()):
            raise ContractOracleError("denominator_semantics", name)
        _nonnegative_int(definition["expected_count"], "denominator_count")
def _validate_ledger(ledger: list[object]) -> None:
    seen: set[str] = set()
    for raw in ledger:
        row = _mapping(raw, "exposure_entry")
        if "episode_id" not in row:
            raise ContractOracleError("exposure_episode")
        if "fields" not in row:
            raise ContractOracleError("exposure_fields")
        if "source" not in row or set(row) != {"episode_id", "fields", "source"}:
            raise ContractOracleError("exposure_entry")
        identity = row.get("episode_id")
        if not _sha256(identity):
            raise ContractOracleError("exposure_episode")
        if identity in seen:
            raise ContractOracleError("exposure_duplicate", identity)
        seen.add(str(identity))
        fields = _list(row.get("fields"), "exposure_fields")
        if (
            not fields
            or len(fields) != len(set(fields))
            or any(field not in EXPOSURE_FIELDS for field in fields)
        ):
            raise ContractOracleError("exposure_fields", identity)
        if not isinstance(row.get("source"), str) or not row["source"]:
            raise ContractOracleError("exposure_source")
def _validate_contract(contract: Mapping[str, object],
                       denominator_names: set[str]) -> None:
    required = {
        "contract_id", "claim_scope", "confirmatory", "denominator",
        "exposure_policy", "outcome_field", "stratum_minimums",
        "target_eligibility", "target_minimum"}
    if not required <= set(contract) or set(contract) - required != {
        "forbidden_claim_scope"
    } and set(contract) != required:
        raise ContractOracleError("contract_keys")
    if not isinstance(contract["contract_id"], str) or not contract["contract_id"]:
        raise ContractOracleError("contract_id")
    if type(contract["confirmatory"]) is not bool:
        raise ContractOracleError("confirmatory")
    if contract["denominator"] not in denominator_names:
        raise ContractOracleError("denominator_name")
    if contract["target_eligibility"] not in denominator_names:
        raise ContractOracleError("target_eligibility")
    if contract["exposure_policy"] not in EXPOSURE_POLICIES:
        raise ContractOracleError("exposure_policy")
    if not isinstance(contract["outcome_field"], str) or not contract["outcome_field"]:
        raise ContractOracleError("outcome_field")
    _nonnegative_int(contract["target_minimum"], "target_minimum")
    claims = _string_set(contract["claim_scope"], "claim_scope")
    forbidden = _string_set(contract.get("forbidden_claim_scope", []), "claim_scope")
    allowed = {"overall", *SUPPORTED_STRATA}
    if "overall" not in claims or not claims <= allowed or not forbidden <= allowed:
        raise ContractOracleError("claim_scope")
    if claims & forbidden:
        raise ContractOracleError("claim_scope_overlap")
    minimums = _mapping(contract["stratum_minimums"], "stratum_minimums")
    if not set(minimums) <= set(SUPPORTED_STRATA):
        raise ContractOracleError("stratum_name")
    for value in minimums.values():
        if isinstance(value, Mapping):
            if not value:
                raise ContractOracleError("stratum_minimum")
            for minimum in value.values():
                _nonnegative_int(minimum, "stratum_minimum")
        else:
            _nonnegative_int(value, "stratum_minimum")
def build_cohort(capsules: Mapping[str, object], bank: Mapping[str, object],
                 registry: Mapping[str, object]) -> dict[str, object]:
    """Independently join minimal P50/P79 Episode structure."""
    validate_registry(registry)
    if capsules.get("schema_version") != P50_SCHEMA:
        raise ContractOracleError("p50_schema")
    if bank.get("schema_version") != P79_SCHEMA:
        raise ContractOracleError("p79_schema")
    p50_rows = _list(capsules.get("episodes"), "p50_episodes")
    p79_rows = _list(bank.get("episodes"), "p79_episodes")
    if capsules.get("capsule_count") != len(p50_rows):
        raise ContractOracleError("p50_episode_count")
    if bank.get("episode_count") != len(p79_rows):
        raise ContractOracleError("p79_episode_count")
    if bank.get("source_acquisition") != registry["sources"]["p50"]["path"]:
        raise ContractOracleError("p79_source_identity")
    p50 = _episode_map(p50_rows, "p50")
    p79 = _episode_map(p79_rows, "p79")
    if set(p50) != set(p79):
        raise ContractOracleError("episode_identity_mismatch")
    records: list[dict[str, object]] = []
    for identity in sorted(p50):
        left, right = p50[identity], p79[identity]
        if left.get("task_id") != right.get("task_id"):
            raise ContractOracleError("task_identity", identity)
        if left.get("cell_id") != right.get("cell_id"):
            raise ContractOracleError("cell_identity", identity)
        observation, action = _latency(left, identity)
        parsed = _cell_latency(right.get("cell_id"))
        if parsed != (observation, action):
            raise ContractOracleError("latency_identity", identity)
        right_count = _candidate_count(right, "candidate_count")
        task = left.get("task_id")
        if not isinstance(task, str) or not task:
            raise ContractOracleError("task_key", identity)
        record = {
            "episode_id": identity,
            "task_id": task,
            "cell_id": str(left["cell_id"]),
            "observation_latency_steps": observation,
            "action_latency_steps": action,
            "candidate_count": right_count,
        }
        record["denominators"] = sorted(
            name
            for name, definition in registry["denominators"].items()
            if _matches_definition(right_count, definition)
        )
        records.append(record)
    result = {
        "schema_version": "hwr.r0017-experiment-contract-cohort/v1",
        "sample_unit": registry["sample_unit"],
        "episodes": records,
    }
    result["summary"] = _cohort_summary(records)
    _validate_cohort(result, registry)
    return result
def _episode_map(rows: list[object],
                 source: str) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw in rows:
        row = _mapping(raw, f"{source}_episode")
        identity = row.get("planned_episode_id")
        if not isinstance(identity, str) or not identity:
            raise ContractOracleError("episode_missing", source)
        if identity in result:
            raise ContractOracleError("episode_duplicate", source)
        result[identity] = row
    return result
def _latency(row: Mapping[str, object], identity: str) -> tuple[int, int]:
    planned = _mapping(row.get("planned_latency"), "latency_missing")
    try:
        observation = planned["observation_steps"]
        action = planned["action_steps"]
    except KeyError as error:
        raise ContractOracleError("latency_missing", identity) from error
    if (
        type(observation) is not int
        or type(action) is not int
        or observation <= 0
        or action <= 0
    ):
        raise ContractOracleError("latency_key", identity)
    return observation, action
def _cell_latency(value: object) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ContractOracleError("cell_key")
    parts = value.rsplit("-obs-", 1)
    if len(parts) != 2 or "-action-" not in parts[1]:
        raise ContractOracleError("cell_key", value)
    observation, action = parts[1].split("-action-", 1)
    try:
        return int(observation), int(action)
    except ValueError as error:
        raise ContractOracleError("cell_key", value) from error
def _candidate_count(row: Mapping[str, object], category: str) -> int:
    candidate_set = _mapping(row.get("candidate_set"), category)
    count = candidate_set.get("candidate_count")
    if type(count) is not int or count < 0:
        raise ContractOracleError(category)
    return count
def _cohort_summary(records: list[Mapping[str, object]]) -> dict[str, object]:
    nonempty = [row for row in records if row["candidate_count"] >= 1]
    return {
        "episode_count": len(records),
        "nonempty_count": len(nonempty),
        "empty_count": sum(row["candidate_count"] == 0 for row in records),
        "singleton_count": sum(row["candidate_count"] == 1 for row in records),
        "choice_opportunity_count": sum(
            row["candidate_count"] >= 2 for row in records
        ),
        "task_nonempty_counts": dict(
            sorted(Counter(str(row["task_id"]) for row in nonempty).items())
        ),
        "latency_pair_nonempty_counts": dict(
            sorted(Counter(_stratum_key(row, "latency_pair") for row in nonempty).items())
        ),
        "cell_counts": dict(
            sorted(Counter(str(row["cell_id"]) for row in records).items())
        ),
    }
def _validate_cohort(cohort: Mapping[str, object],
                     registry: Mapping[str, object]) -> None:
    records = cohort["episodes"]
    summary = cohort["summary"]
    expected = registry["expected_cohort"]
    for name in (
        "episode_count",
        "nonempty_count",
        "empty_count",
        "choice_opportunity_count",
    ):
        if summary[name] != expected[name]:
            category = (
                "candidate_count"
                if name in {
                    "nonempty_count",
                    "empty_count",
                    "choice_opportunity_count",
                }
                else "expected_cohort"
            )
            raise ContractOracleError(category, name)
    if summary["task_nonempty_counts"] != expected["task_nonempty_counts"]:
        raise ContractOracleError("task_key")
    if (
        summary["latency_pair_nonempty_counts"]
        != expected["latency_pair_nonempty_counts"]
    ):
        raise ContractOracleError("latency_key")
    cell_counts = summary["cell_counts"]
    if len(cell_counts) != expected["cell_count"] or len(set(cell_counts.values())) != 1:
        raise ContractOracleError("cell_key")
    membership = {
        name: sum(name in row["denominators"] for row in records)
        for name in registry["denominators"]
    }
    for name, definition in registry["denominators"].items():
        if membership[name] != definition["expected_count"]:
            raise ContractOracleError("denominator_count", name)
    all_ids = {row["episode_id"] for row in records}
    empty = {row["episode_id"] for row in records if "empty" in row["denominators"]}
    nonempty = {
        row["episode_id"] for row in records if "nonempty" in row["denominators"]
    }
    if empty & nonempty or empty | nonempty != all_ids:
        raise ContractOracleError("denominator_partition")
    ledger_ids = {row["episode_id"] for row in registry["result_exposure_ledger"]}
    if not ledger_ids <= all_ids:
        raise ContractOracleError("exposure_episode")
def analyze_registry(registry: Mapping[str, object],
                     cohort: Mapping[str, object]) -> dict[str, object]:
    """Analyze every contract without encoding a formal expected verdict."""
    validate_registry(registry)
    _validate_cohort(cohort, registry)
    analyses = [analyze_contract(value, registry, cohort)
                for value in registry["contracts"]]
    return {
        "schema_version": "hwr.r0017-experiment-contract-analysis/v1",
        "proposal_id": registry["proposal_id"],
        "sample_unit": registry["sample_unit"],
        "contracts": analyses,
        "metrics": {
            "contract_count": len(analyses),
            "reachable_contract_count": sum(row["reachable"] for row in analyses),
            "rejected_contract_count": sum(not row["reachable"] for row in analyses),
            "solver_agreement_count": sum(row["solver_agreement"] for row in analyses),
            "valid_accepted_witness_count": sum(
                row["reachable"] and row["witness_verification"]["passed"]
                for row in analyses),
            "valid_contradiction_count": sum(
                not row["reachable"] and row["contradiction_verification"]["passed"]
                for row in analyses),
            "denominator_conservation_count": sum(
                row["denominator"]["conserved"] for row in analyses),
            "exposure_policy_valid_count": sum(
                row["exposure"]["policy_valid"] for row in analyses),
            "private_outcome_read_count": 0, "sample_unit_violation_count": 0}}
def analyze_contract(contract: Mapping[str, object], registry: Mapping[str, object],
                     cohort: Mapping[str, object]) -> dict[str, object]:
    analytic = solve_analytic(contract, registry, cohort)
    enumeration = solve_enumeration(contract, registry, cohort)
    combine_solver_results(analytic, enumeration)
    witness = enumeration.get("accepted_witness")
    witness_check = (verify_assignment(contract, registry, cohort, witness)
                     if witness is not None else
                     {"passed": False, "errors": ["witness_absent"]})
    contradiction = analytic["contradictions"][0] if analytic["contradictions"] else None
    contradiction_check = (verify_contradiction(
        contract, registry, cohort, contradiction) if contradiction is not None
        else {"passed": False, "errors": ["contradiction_absent"]})
    reachable = bool(analytic["reachable"])
    if reachable and not witness_check["passed"]:
        raise ContractOracleError("witness_verifier")
    if not reachable and not contradiction_check["passed"]:
        raise ContractOracleError("contradiction_verifier")
    return {
        "contract_id": contract["contract_id"],
        "verdict": "eligible" if reachable else "rejected_contract",
        "reachable": reachable,
        "solver_agreement": True,
        "denominator": analytic["denominator"],
        "exposure": analytic["exposure"],
        "target_eligibility_count": analytic["target_eligibility_count"],
        "strata": analytic["strata"],
        "worst_stratum_coverage": (witness_check["worst_stratum_coverage"]
                                   if reachable else
                                   analytic["worst_stratum_coverage"]),
        "solver_a": analytic, "solver_b": enumeration,
        "witness_verification": witness_check,
        "contradiction_verification": contradiction_check}
def solve_analytic(contract: Mapping[str, object], registry: Mapping[str, object],
                   cohort: Mapping[str, object]) -> dict[str, object]:
    context = _contract_context(contract, registry, cohort)
    contradictions = list(context["structural_contradictions"])
    required = int(contract["target_minimum"])
    eligible_count = len(context["eligible_ids"])
    if required > eligible_count:
        contradictions.append({
            "category": "required_gt_eligible", "scope": "overall",
            "required": required, "available": eligible_count})
    strata: dict[str, dict[str, dict[str, int]]] = {}
    coverages: list[dict[str, object]] = []
    for name, floors in context["minimums"].items():
        rows = {}
        for key, minimum in floors.items():
            available = context["eligible_strata"][name].get(key, 0)
            rows[key] = {"minimum": minimum, "eligible": available,
                         "denominator": context["effective_strata"][name].get(key, 0)}
            coverages.append({
                "stratum": name, "key": key, "minimum": minimum,
                "available": available, "slack": available - minimum})
            if minimum > available:
                contradictions.append({
                    "category": "stratum_required_gt_eligible", "scope": name,
                    "key": key, "required": minimum, "available": available})
        strata[name] = rows
    contradictions = sorted(contradictions, key=lambda row: (
        str(row["category"]), str(row.get("scope", "")),
        str(row.get("key", ""))))
    return {
        "solver": "analytic_necessary_boundary",
        "reachable": not contradictions,
        "contradictions": contradictions,
        "denominator": context["denominator"],
        "exposure": context["exposure"],
        "target_eligibility_count": eligible_count,
        "strata": strata,
        "worst_stratum_coverage": (min(
            coverages, key=lambda row: (row["slack"], row["stratum"], row["key"]))
            if coverages else None)}
def solve_enumeration(contract: Mapping[str, object],
                      registry: Mapping[str, object],
                      cohort: Mapping[str, object]) -> dict[str, object]:
    context = _independent_context(contract, registry, cohort)
    eligible = sorted(context["eligible_ids"]); assignment_space = 1 << len(eligible)
    if context["contradictions"]:
        if _assignment_passes(set(), context, int(contract["target_minimum"])):
            raise ContractOracleError("enumeration_prune")
        return {
            "solver": "assignment_enumeration", "reachable": False,
            "accepted_witness": None, "assignment_space": assignment_space,
            "enumerated_assignment_count": 1,
            "pruned_assignment_count": assignment_space - 1, "exhaustive": True,
            "independent_rejections": context["contradictions"]}
    examined = 0; lower = int(contract["target_minimum"])
    for size in range(max(0, lower), len(eligible) + 1):
        for selected in combinations(eligible, size):
            examined += 1
            if _assignment_passes(set(selected), context, lower):
                return {
                    "solver": "assignment_enumeration", "reachable": True,
                    "accepted_witness": {
                        "positive_episode_ids": list(selected),
                        "negative_episode_ids": sorted(
                            set(context["effective_ids"]) - set(selected)),
                    },
                    "assignment_space": assignment_space, "exhaustive": False,
                    "enumerated_assignment_count": examined,
                    "pruned_assignment_count": assignment_space - examined,
                    "independent_rejections": []}
    return {
        "solver": "assignment_enumeration", "reachable": False,
        "accepted_witness": None, "assignment_space": assignment_space,
        "enumerated_assignment_count": examined,
        "pruned_assignment_count": assignment_space - examined,
        "exhaustive": True, "independent_rejections": []}
def verify_assignment(contract: Mapping[str, object], registry: Mapping[str, object],
                      cohort: Mapping[str, object],
                      witness: object) -> dict[str, object]:
    """Independently recompute every gate for a proposed accepted assignment."""
    context = _contract_context(contract, registry, cohort)
    errors = [str(value["category"])
              for value in context["structural_contradictions"]]
    if not isinstance(witness, Mapping):
        return {"passed": False, "errors": [*errors, "witness_type"]}
    selected_raw = witness.get("positive_episode_ids")
    if not isinstance(selected_raw, list) or any(
        not isinstance(value, str) for value in selected_raw):
        return {"passed": False, "errors": [*errors, "witness_type"]}
    selected = set(selected_raw)
    if len(selected) != len(selected_raw):
        errors.append("witness_duplicate")
    if not selected <= context["eligible_ids"]:
        errors.append("witness_ineligible")
    if len(selected) < int(contract["target_minimum"]):
        errors.append("witness_total")
    negative_raw = witness.get("negative_episode_ids")
    if negative_raw is not None and (
        not isinstance(negative_raw, list)
        or any(not isinstance(value, str) for value in negative_raw)
        or len(negative_raw) != len(set(negative_raw))
        or set(negative_raw) != context["effective_ids"] - selected
    ):
        errors.append("witness_negative_partition")
    by_id = {row["episode_id"]: row for row in cohort["episodes"]}
    coverages: list[dict[str, object]] = []
    for name, floors in context["minimums"].items():
        counts = Counter(_stratum_key(by_id[identity], name) for identity in selected)
        for key, minimum in floors.items():
            count = counts[key]
            coverages.append({
                "stratum": name, "key": key, "minimum": minimum,
                "positive_count": count, "slack": count - minimum})
            if count < minimum:
                errors.append(f"witness_stratum:{name}")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "positive_count": len(selected),
        "worst_stratum_coverage": (min(
            coverages, key=lambda row: (
                row["slack"], row["stratum"], row["key"]))
            if coverages else None)}
def verify_contradiction(
    contract: Mapping[str, object], registry: Mapping[str, object],
    cohort: Mapping[str, object], contradiction: object) -> dict[str, object]:
    """Recompute the declared contradiction without either solver context."""
    if not isinstance(contradiction, Mapping):
        return {"passed": False, "errors": ["contradiction_type"]}
    rows = cohort["episodes"]; base, eligible, exposed = _independent_sets(
        contract, registry, rows)
    category, expected = contradiction.get("category"), None
    if category == "historical_confirmatory" and contract["confirmatory"] and (
            contract["exposure_policy"] == "historical_design_audit"):
        expected = {"category": category}
    elif category == "confirmatory_include_exposed" and contract[
            "confirmatory"] and contract["exposure_policy"] == "include_exposed" and (
            exposed & base):
        expected = {"category": category, "episode_count": len(exposed & base)}
    elif category == "claim_without_minimum":
        scope = contradiction.get("scope")
        if scope in contract["claim_scope"] and scope not in contract["stratum_minimums"]:
            expected = {"category": category, "scope": scope}
    elif category == "required_gt_eligible":
        expected = {"category": category, "scope": "overall",
                    "required": int(contract["target_minimum"]),
                    "available": len(eligible)}
    elif category in {
            "stratum_key_unknown", "stratum_key_missing",
            "stratum_required_gt_eligible"}:
        expected = _verify_stratum_contradiction(
            category, contradiction, contract, rows, base, eligible)
    passed = dict(contradiction) == expected
    return {"passed": passed, "errors": [] if passed else ["contradiction_false"]}
def _verify_stratum_contradiction(
    category: str, claimed: Mapping[str, object], contract: Mapping[str, object],
    rows: Sequence[Mapping[str, object]], base: set[str],
    eligible: set[str]) -> dict[str, object] | None:
    scope = claimed.get("scope")
    if scope not in contract["stratum_minimums"]: return None
    base_keys = {_stratum_key(row, str(scope)) for row in rows
                 if row["episode_id"] in base}
    raw = contract["stratum_minimums"][scope]
    if category == "stratum_key_unknown" and isinstance(raw, Mapping):
        return {"category": category, "scope": scope,
                "keys": sorted(set(raw) - base_keys)}
    if category == "stratum_key_missing" and isinstance(raw, Mapping):
        return {"category": category, "scope": scope,
                "keys": sorted(base_keys - set(raw))}
    key = claimed.get("key")
    minimum = raw.get(key) if isinstance(raw, Mapping) else raw
    available = sum(row["episode_id"] in eligible and _stratum_key(
        row, str(scope)) == key for row in rows)
    return {"category": category, "scope": scope, "key": key,
            "required": int(minimum), "available": available}
def combine_solver_results(analytic: Mapping[str, object],
                           enumeration: Mapping[str, object]) -> None:
    if bool(analytic.get("reachable")) != bool(enumeration.get("reachable")):
        raise ContractOracleError("invalid_solver_disagreement")
def validate_denominator_accounting(
    contract: Mapping[str, object], registry: Mapping[str, object],
    cohort: Mapping[str, object], denominator: Mapping[str, object]) -> None:
    expected = _contract_context(contract, registry, cohort)["denominator"]
    if dict(denominator) != expected:
        raise ContractOracleError("denominator_conservation")
def _independent_context(
    contract: Mapping[str, object], registry: Mapping[str, object],
    cohort: Mapping[str, object]) -> dict[str, Any]:
    """Build Solver B constraints without Solver A's shared context."""
    _validate_contract(contract, set(registry["denominators"]))
    rows = cohort["episodes"]
    base, eligible, exposed = _independent_sets(contract, registry, rows)
    policy = contract["exposure_policy"]; contradictions: list[dict[str, object]] = []
    excluded = (exposed & base if policy == "exclude_matching_outcome_fields" else set())
    effective = base - excluded; by_id = {row["episode_id"]: row for row in rows}
    if policy == "historical_design_audit" and contract["confirmatory"]:
        contradictions.append({"category": "historical_confirmatory"})
    if policy == "include_exposed" and contract["confirmatory"] and exposed & base:
        contradictions.append({"category": "confirmatory_include_exposed",
                               "episode_count": len(exposed & base)})
    minimums: dict[str, dict[str, int]] = {}
    base_rows = [by_id[key] for key in base]
    for claim in contract["claim_scope"]:
        if claim != "overall" and claim not in contract["stratum_minimums"]:
            contradictions.append({"category": "claim_without_minimum", "scope": claim})
    for name, raw in contract["stratum_minimums"].items():
        keys = sorted({_stratum_key(row, name) for row in base_rows})
        if isinstance(raw, Mapping):
            unknown = sorted(set(raw) - set(keys))
            missing = sorted(set(keys) - set(raw))
            if unknown: contradictions.append(
                {"category": "stratum_key_unknown", "scope": name, "keys": unknown})
            if missing: contradictions.append(
                {"category": "stratum_key_missing", "scope": name, "keys": missing})
            minimums[name] = {
                str(key): int(raw[key]) for key in raw if key in keys}
        else:
            minimums[name] = {key: int(raw) for key in keys}
    required = int(contract["target_minimum"])
    if required > len(eligible): contradictions.append(
        {"category": "required_gt_eligible", "scope": "overall",
         "required": required, "available": len(eligible)})
    for name, floors in minimums.items():
        counts = Counter(_stratum_key(by_id[identity], name) for identity in eligible)
        for key, minimum in floors.items():
            if minimum > counts[key]: contradictions.append(
                {"category": "stratum_required_gt_eligible", "scope": name,
                 "key": key, "required": minimum, "available": counts[key]})
    contradictions.sort(key=lambda row: (
        str(row["category"]), str(row.get("scope", "")), str(row.get("key", ""))))
    return {"effective_ids": effective, "eligible_ids": eligible,
            "minimums": minimums, "by_id": by_id, "contradictions": contradictions}
def _independent_sets(
    contract: Mapping[str, object], registry: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> tuple[set[str], set[str], set[str]]:
    definitions = registry["denominators"]
    base = {row["episode_id"] for row in rows if _matches_definition(
        int(row["candidate_count"]), definitions[contract["denominator"]])}
    exposed = {row["episode_id"] for row in registry["result_exposure_ledger"]
               if contract["outcome_field"] in row["fields"]}
    excluded = (exposed & base if contract["exposure_policy"]
                == "exclude_matching_outcome_fields" else set())
    eligible = {row["episode_id"] for row in rows if row["episode_id"] in base - excluded
                and _matches_definition(int(row["candidate_count"]),
                    definitions[contract["target_eligibility"]])}
    return base, eligible, exposed
def _assignment_passes(selected: set[str], context: Mapping[str, Any],
                       target: int) -> bool:
    if context["contradictions"] or len(selected) < target or not (
            selected <= context["eligible_ids"]):
        return False
    for name, floors in context["minimums"].items():
        counts = Counter(
            _stratum_key(context["by_id"][identity], name) for identity in selected)
        if any(counts[key] < minimum for key, minimum in floors.items()): return False
    return True
def _contract_context(contract: Mapping[str, object],
                      registry: Mapping[str, object],
                      cohort: Mapping[str, object]) -> dict[str, Any]:
    _validate_contract(contract, set(registry["denominators"]))
    rows = cohort["episodes"]
    base = {
        row["episode_id"]
        for row in rows
        if contract["denominator"] in row["denominators"]
    }
    matching_exposed = {
        row["episode_id"]
        for row in registry["result_exposure_ledger"]
        if contract["outcome_field"] in row["fields"]
    }
    policy = str(contract["exposure_policy"])
    structural: list[dict[str, object]] = []
    if policy == "historical_design_audit" and contract["confirmatory"]:
        structural.append({"category": "historical_confirmatory"})
    if policy == "include_exposed" and contract["confirmatory"] and matching_exposed:
        structural.append(
            {
                "category": "confirmatory_include_exposed",
                "episode_count": len(matching_exposed & base),
            }
        )
    excluded = (
        matching_exposed & base
        if policy == "exclude_matching_outcome_fields"
        else set()
    )
    effective = base - excluded
    eligible = {
        row["episode_id"]
        for row in rows
        if row["episode_id"] in effective
        and contract["target_eligibility"] in row["denominators"]
    }
    universe_rows = [row for row in rows if row["episode_id"] in base]
    effective_rows = [row for row in rows if row["episode_id"] in effective]
    eligible_rows = [row for row in rows if row["episode_id"] in eligible]
    minimums: dict[str, dict[str, int]] = {}
    for claim in contract["claim_scope"]:
        if claim != "overall" and claim not in contract["stratum_minimums"]:
            structural.append(
                {"category": "claim_without_minimum", "scope": claim}
            )
    for name, raw in contract["stratum_minimums"].items():
        keys = sorted({_stratum_key(row, name) for row in universe_rows})
        if isinstance(raw, Mapping):
            unknown = sorted(set(raw) - set(keys))
            missing = sorted(set(keys) - set(raw))
            if unknown:
                structural.append(
                    {"category": "stratum_key_unknown", "scope": name, "keys": unknown}
                )
            if missing:
                structural.append(
                    {"category": "stratum_key_missing", "scope": name, "keys": missing}
                )
            minimums[name] = {
                str(key): int(value) for key, value in raw.items() if key in keys
            }
        else:
            minimums[name] = {key: int(raw) for key in keys}
    eligible_strata = {
        name: Counter(_stratum_key(row, name) for row in eligible_rows)
        for name in minimums
    }
    denominator = {
        "name": contract["denominator"],
        "base_count": len(base),
        "excluded_count": len(excluded),
        "effective_count": len(effective),
        "conserved": len(base) == len(excluded) + len(effective)
        and not excluded & effective,
        "excluded_episode_ids": sorted(excluded),
    }
    exposure = {
        "policy": policy,
        "policy_valid": policy in EXPOSURE_POLICIES,
        "matching_episode_count": len(matching_exposed & base),
        "confirmatory_compatible": not any(
            row["category"] in {"historical_confirmatory", "confirmatory_include_exposed"}
            for row in structural
        ),
    }
    return {
        "base_ids": base,
        "effective_ids": effective,
        "eligible_ids": eligible,
        "minimums": minimums,
        "eligible_strata": eligible_strata,
        "effective_strata": {
            name: Counter(_stratum_key(row, name) for row in effective_rows)
            for name in minimums
        },
        "structural_contradictions": structural,
        "denominator": denominator,
        "exposure": exposure,
    }
def _stratum_key(row: Mapping[str, object], name: str) -> str:
    if name == "task": return str(row["task_id"])
    observation = int(row["observation_latency_steps"])
    action = int(row["action_latency_steps"])
    if name == "observation_latency": return f"o{observation}"
    if name == "action_latency": return f"a{action}"
    if name == "latency_pair": return f"o{observation}-a{action}"
    if name == "cell": return str(row["cell_id"])
    raise ContractOracleError("stratum_name", name)
def _matches_definition(candidate_count: int,
                        definition: Mapping[str, object]) -> bool:
    minimum = definition.get("candidate_count_minimum")
    maximum = definition.get("candidate_count_maximum")
    return (
        (minimum is None or candidate_count >= int(minimum))
        and (maximum is None or candidate_count <= int(maximum))
    )
def _mapping(value: object, category: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping): raise ContractOracleError(category)
    return value
def _list(value: object, category: str) -> list[Any]:
    if not isinstance(value, list): raise ContractOracleError(category)
    return value
def _nonnegative_int(value: object, category: str) -> None:
    if type(value) is not int or value < 0: raise ContractOracleError(category)
def _string_set(value: object, category: str) -> set[str]:
    values = _list(value, category)
    if len(values) != len(set(values)) or any(not isinstance(item, str) for item in values):
        raise ContractOracleError(category)
    return set(values)
def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
def _safe_relative(value: object) -> bool:
    return (isinstance(value, str) and bool(value)
            and not value.startswith("/") and ".." not in value.split("/"))
