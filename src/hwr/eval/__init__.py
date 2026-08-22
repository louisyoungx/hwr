"""Closed-loop policy evaluation."""

from typing import Callable, Mapping, Sequence

from hwr.eval.bimanual import (
    BimanualAcceptanceCriteria,
    BimanualEpisodeEvaluation,
    BimanualEvaluationReport,
    assess_bimanual_acceptance,
    combine_bimanual_reports,
    evaluate_bimanual_policy,
)
from hwr.eval.candidate_funnel import CandidateFunnelContractError
from hwr.eval.evaluator import EvaluationReport, evaluate_policy
from hwr.eval.formal_visual import FormalEvaluationReport, evaluate_formal_visual_policy
from hwr.eval.seed_contract import (
    SEED_SCHEMA,
    PlannedEpisodeSeed,
    derive_domain_seed,
    plan_episode_seeds,
    planned_episode_id,
    random_seed_salt,
    read_seed_salt,
    require_seed_reveal,
    seed_commitment,
    seed_lineage_manifest,
    validate_episode_seed_plan,
    verify_seed_reveal,
)
from hwr.eval.stability import (
    MultiObjectStabilityCriterion,
    PlacementSample,
    StabilityConfig,
    StablePlacementCriterion,
    TargetVolume,
)

LatencySampler = Callable[[str, int], tuple[int, int]]


def formal_latency_sampler(tasks, bindings) -> LatencySampler:
    from hwr.adapters.mujoco.target_selection_diagnostic import (
        TargetSelectionDiagnostic,
    )

    diagnostics = {
        task_id: TargetSelectionDiagnostic(task, bindings[task_id])
        for task_id, task in tasks.items()
    }
    return lambda task_id, seed: diagnostics[task_id].sample_latencies(seed)


def validate_plan_contract(
    plan: Mapping[str, object],
    *,
    salt: str | None,
    expected_schema: str,
    expected_proposal_id: str,
    expected_plan_id: str,
    expected_commitment: str,
    expected_cells: Sequence[tuple[str, int, int]],
    acquisition_steps: int,
    latency_sampler: LatencySampler,
) -> dict[str, object]:
    expected_cells = tuple(expected_cells)
    reveal = plan.get("salt_reveal") if salt is None else salt
    try:
        reveal_valid = (
            isinstance(reveal, str)
            and plan.get("salt_reveal") == reveal
            and verify_seed_reveal(expected_commitment, reveal)
            and plan.get("commitment_verified") is True
        )
    except ValueError:
        reveal_valid = False
    checks = {
        "schema": plan.get("schema_version") == expected_schema,
        "proposal": plan.get("proposal_id") == expected_proposal_id,
        "mode": plan.get("mode") == "formal",
        "plan_id": plan.get("plan_id") == expected_plan_id,
        "seed_schema": plan.get("seed_schema") == SEED_SCHEMA,
        "commitment": plan.get("salt_commitment") == expected_commitment,
        "reveal": reveal_valid,
        "role_excluded": plan.get("role_enters_seed_derivation") is False,
        "natural_rejection": (
            plan.get("natural_evaluation_latency_rejection") is True
            and plan.get("reset_latency_override_used") is False
            and plan.get("replacement_seed_allowed") is False
        ),
        "limits": (
            plan.get("maximum_candidate_ordinal") == 95
            and plan.get("maximum_latency_sampler_calls") == 1_152
            and plan.get("planned_control_steps_per_episode") == acquisition_steps
            and plan.get("planned_episode_count") == 24
            and plan.get("planned_control_step_count") == 24 * acquisition_steps
            and plan.get("validation_replay_count") == 24
            and plan.get("maximum_physical_acquisition_count") == 48
            and plan.get("maximum_control_step_count") == 48 * acquisition_steps
        ),
    }
    cells = plan.get("cells")
    episodes = plan.get("episodes")
    rejected = plan.get("rejected_seed_audit")
    if not all(isinstance(value, list) for value in (cells, episodes, rejected)):
        raise CandidateFunnelContractError("plan lists are missing")
    expected_cell_rows = [
        {
            "cell_id": f"cell-{ordinal:02d}-obs-{observation}-action-{action}",
            "cell_ordinal": ordinal,
            "task_id": task,
            "observation_latency_steps": observation,
            "action_latency_steps": action,
            "replicate_count": 2,
        }
        for ordinal, (task, observation, action) in enumerate(expected_cells)
    ]
    checks["cells"] = cells == expected_cell_rows
    checks["episodes"] = len(episodes) == 24
    checks["episode_order"] = [
        (value.get("cell_ordinal"), value.get("replicate_ordinal"))
        for value in episodes
    ] == [
        (cell_ordinal, replicate)
        for cell_ordinal in range(len(expected_cells))
        for replicate in range(2)
    ]
    checks["candidate_audit"] = _validate_candidate_audit(
        episodes,
        rejected,
        expected_cell_rows,
        expected_plan_id,
        reveal,
        latency_sampler,
    )
    if not all(checks.values()):
        raise CandidateFunnelContractError(
            "plan contract differs",
            details={"checks": checks},
        )
    return {"checks": checks, "passed": True}


def _validate_candidate_audit(
    episodes,
    rejected,
    cells,
    plan_id: str,
    salt: str,
    latency_sampler: LatencySampler,
) -> bool:
    checked_environment, checked_policy = set(), set()
    known_cells = {cell["cell_id"] for cell in cells}
    if (
        len(episodes) != 24
        or len(episodes) + len(rejected) > 1_152
        or any(
            not isinstance(value, Mapping)
            or value.get("cell_id") not in known_cells
            or not isinstance(value.get("candidate_ordinal"), int)
            or isinstance(value.get("candidate_ordinal"), bool)
            for value in (*episodes, *rejected)
        )
    ):
        return False
    cell_ordinals = {cell["cell_id"]: cell["cell_ordinal"] for cell in cells}
    if [
        (cell_ordinals[value["cell_id"]], value["candidate_ordinal"])
        for value in rejected
    ] != sorted(
        (cell_ordinals[value["cell_id"]], value["candidate_ordinal"])
        for value in rejected
    ):
        return False
    for cell in cells:
        accepted = [
            value for value in episodes if value.get("cell_id") == cell["cell_id"]
        ]
        declined = [
            value for value in rejected if value.get("cell_id") == cell["cell_id"]
        ]
        cell_records = [
            *((value, True) for value in accepted),
            *((value, False) for value in declined),
        ]
        cell_records.sort(key=lambda value: value[0]["candidate_ordinal"])
        if (
            [value.get("candidate_ordinal") for value, _ in cell_records]
            != list(range(len(cell_records)))
            or not 2 <= len(cell_records) <= 96
            or [value.get("candidate_ordinal") for value in accepted]
            != sorted(value.get("candidate_ordinal") for value in accepted)
            or [value.get("candidate_ordinal") for value in declined]
            != sorted(value.get("candidate_ordinal") for value in declined)
            or [value.get("replicate_ordinal") for value in accepted] != [0, 1]
            or accepted[-1].get("candidate_ordinal") != len(cell_records) - 1
            or any(value.get("replacement") is not False for value in accepted)
            or any(
                value.get("accepted") is not False
                or value.get("rejection_reason")
                != "natural_latency_cell_mismatch"
                or "replicate_ordinal" in value
                for value in declined
            )
        ):
            return False
        matched_count = 0
        for record, is_accepted in cell_records:
            ordinal = record.get("candidate_ordinal")
            try:
                expected_id = planned_episode_id(
                    plan_id, cell["task_id"], cell["cell_id"], ordinal
                )
                environment = derive_domain_seed(salt, "environment", expected_id)
                policy = derive_domain_seed(salt, "policy", expected_id)
                sampled = tuple(latency_sampler(cell["task_id"], environment))
            except (KeyError, TypeError, ValueError):
                return False
            matched = sampled == (
                cell["observation_latency_steps"],
                cell["action_latency_steps"],
            )
            matched_count += matched
            common = (
                record.get("planned_episode_id") == expected_id
                and record.get("task_id") == cell["task_id"]
                and record.get("cell_ordinal") == cell["cell_ordinal"]
                and record.get("environment_seed") == environment
                and record.get("policy_rng_seed") == policy
                and (
                    record.get("sampled_observation_latency_steps"),
                    record.get("sampled_action_latency_steps"),
                )
                == sampled
                and record.get("latency_matched") is matched
                and record.get("accepted") is is_accepted
            )
            if not common or matched != is_accepted:
                return False
            if environment in checked_environment or policy in checked_policy:
                return False
            if environment in checked_policy or policy in checked_environment:
                return False
            checked_environment.add(environment)
            checked_policy.add(policy)
        if matched_count != 2:
            return False
    return True


__all__ = [
    "BimanualAcceptanceCriteria",
    "BimanualEpisodeEvaluation",
    "BimanualEvaluationReport",
    "EvaluationReport",
    "FormalEvaluationReport",
    "MultiObjectStabilityCriterion",
    "PlacementSample",
    "PlannedEpisodeSeed",
    "SEED_SCHEMA",
    "StabilityConfig",
    "StablePlacementCriterion",
    "TargetVolume",
    "assess_bimanual_acceptance",
    "combine_bimanual_reports",
    "derive_domain_seed",
    "evaluate_bimanual_policy",
    "evaluate_policy",
    "evaluate_formal_visual_policy",
    "formal_latency_sampler",
    "plan_episode_seeds",
    "planned_episode_id",
    "random_seed_salt",
    "read_seed_salt",
    "require_seed_reveal",
    "seed_commitment",
    "seed_lineage_manifest",
    "validate_episode_seed_plan",
    "validate_plan_contract",
    "verify_seed_reveal",
]
