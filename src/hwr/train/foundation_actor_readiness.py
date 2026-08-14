"""Evidence-based Actor training and collection admission gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


ACTOR_READINESS_SCHEMA = "hwr.foundation-actor-readiness/v3"


EXPLORATION_CHECKS = (
    "minimum_replay_episodes",
    "one_step_physical_action_utilization",
    "active_action_dimensions",
    "action_effective_rank",
    "data_action_probe_ratio",
    "data_action_probe_bootstrap_lower_bound",
    "data_action_probe_all_tasks",
)

TASK_INTERACTION_CHECKS = (
    "external_contact_coverage",
    "controlled_physical_motion_coverage",
    "severe_collision_positive_coverage",
    "severe_collision_negative_coverage",
)

CALIBRATION_CHECKS = EXPLORATION_CHECKS[1:]


@dataclass(frozen=True)
class FoundationActorReadinessCriteria:
    minimum_replay_episodes: int
    consecutive_passes: int = 2
    minimum_active_dimension_fraction: float = 0.75
    minimum_effective_rank: float = 6.0
    minimum_data_action_ratio: float = 1.05
    minimum_data_action_ratio_p05: float = 1.01
    minimum_contact_episodes_per_task: int = 1
    minimum_controlled_motion_episodes_per_task: int = 1
    minimum_collision_positive_episodes_per_task: int = 1
    minimum_collision_negative_episodes_per_task: int = 1

    def __post_init__(self) -> None:
        counts = (
            self.minimum_replay_episodes,
            self.consecutive_passes,
            self.minimum_contact_episodes_per_task,
            self.minimum_controlled_motion_episodes_per_task,
        )
        collision_counts = (
            self.minimum_collision_positive_episodes_per_task,
            self.minimum_collision_negative_episodes_per_task,
        )
        if min(counts) <= 0 or min(collision_counts) < 0:
            raise ValueError("Actor readiness counts must be positive")
        if not 0.0 < self.minimum_active_dimension_fraction <= 1.0:
            raise ValueError("Actor readiness active dimension fraction is invalid")
        if self.minimum_effective_rank <= 0.0:
            raise ValueError("Actor readiness effective rank is invalid")
        if min(
            self.minimum_data_action_ratio,
            self.minimum_data_action_ratio_p05,
        ) <= 1.0:
            raise ValueError("Actor readiness data action ratios must exceed one")


class FoundationActorReadinessTracker:
    """Require repeated task-blind physical evidence; revoke on any later failure."""

    def __init__(self, criteria: FoundationActorReadinessCriteria) -> None:
        self.criteria = criteria
        self.consecutive_passes = 0
        self.exploration_unlocked = False
        self.task_actor_consecutive_passes = 0
        self.task_actor_unlocked = False
        self.exploration_actor_update_count = 0
        self.task_actor_update_count = 0
        self.last_assessment: dict[str, object] | None = None

    def assess(
        self,
        diagnostic: Mapping[str, object],
        data_action_probe: Mapping[str, object],
        action_coverage: Mapping[str, object],
        interaction_coverage: Mapping[str, object],
        *,
        replay_episodes: int,
    ) -> dict[str, object]:
        checks = {
            "minimum_replay_episodes": (
                replay_episodes >= self.criteria.minimum_replay_episodes
            ),
            "one_step_physical_action_utilization": _physical_causality_passed(
                diagnostic
            ),
            "active_action_dimensions": float(
                action_coverage["active_dimension_fraction"]
            )
            >= self.criteria.minimum_active_dimension_fraction,
            "action_effective_rank": float(action_coverage["effective_rank"])
            >= self.criteria.minimum_effective_rank,
            "data_action_probe_ratio": float(
                data_action_probe["state_only_to_state_action_ratio"]
            )
            >= self.criteria.minimum_data_action_ratio,
            "data_action_probe_bootstrap_lower_bound": float(
                data_action_probe["bootstrap"]["ratio_p05"]
            )
            >= self.criteria.minimum_data_action_ratio_p05,
            "data_action_probe_all_tasks": _probe_partitions_pass(
                data_action_probe, self.criteria
            ),
            "external_contact_coverage": _coverage_partitions_pass(
                interaction_coverage,
                "unilateral_contact_episode_count",
                self.criteria.minimum_contact_episodes_per_task,
            ),
            "controlled_physical_motion_coverage": _coverage_partitions_pass(
                interaction_coverage,
                "controlled_motion_episode_count",
                self.criteria.minimum_controlled_motion_episodes_per_task,
            ),
            "severe_collision_positive_coverage": _coverage_partitions_pass(
                interaction_coverage,
                "severe_collision_positive_episode_count",
                self.criteria.minimum_collision_positive_episodes_per_task,
            ),
            "severe_collision_negative_coverage": _coverage_partitions_pass(
                interaction_coverage,
                "severe_collision_negative_episode_count",
                self.criteria.minimum_collision_negative_episodes_per_task,
            ),
        }
        exploration_passed = _selected_checks_pass(checks, EXPLORATION_CHECKS)
        self.consecutive_passes = (
            self.consecutive_passes + 1 if exploration_passed else 0
        )
        self.exploration_unlocked = (
            self.consecutive_passes >= self.criteria.consecutive_passes
        )
        deployment_passed = _deployment_causality_passed(diagnostic)
        task_interaction_passed = _selected_checks_pass(
            checks, TASK_INTERACTION_CHECKS
        )
        task_passed = (
            self.exploration_unlocked
            and task_interaction_passed
            and deployment_passed
        )
        self.task_actor_consecutive_passes = (
            self.task_actor_consecutive_passes + 1 if task_passed else 0
        )
        self.task_actor_unlocked = (
            self.task_actor_consecutive_passes >= self.criteria.consecutive_passes
        )
        assessment = {
            "schema_version": ACTOR_READINESS_SCHEMA,
            "passed_this_cycle": task_passed,
            "exploration_passed_this_cycle": exploration_passed,
            "task_passed_this_cycle": task_passed,
            "task_interaction_passed_this_cycle": task_interaction_passed,
            "unlocked": self.task_actor_unlocked,
            "exploration_unlocked": self.exploration_unlocked,
            "task_actor_unlocked": self.task_actor_unlocked,
            "consecutive_passes": self.consecutive_passes,
            "task_actor_consecutive_passes": self.task_actor_consecutive_passes,
            "criteria": asdict(self.criteria),
            "checks": checks,
            "replay_episodes": replay_episodes,
            "action_coverage": dict(action_coverage),
            "data_action_probe": dict(data_action_probe),
            "interaction_coverage": dict(interaction_coverage),
            "exploration_actor_update_count": self.exploration_actor_update_count,
            "task_actor_update_count": self.task_actor_update_count,
            "task_semantic_fields": [],
        }
        self.last_assessment = assessment
        return assessment

    def record_task_actor_updates(self, count: int) -> None:
        if count < 0:
            raise ValueError("Actor readiness update count cannot be negative")
        self.task_actor_update_count += count
        self._sync_update_counts()

    def record_exploration_actor_updates(self, count: int) -> None:
        if count < 0:
            raise ValueError("exploration Actor update count cannot be negative")
        self.exploration_actor_update_count += count
        self._sync_update_counts()

    @property
    def exploration_ready_for_collection(self) -> bool:
        return self.exploration_unlocked and self.exploration_actor_update_count > 0

    @property
    def task_actor_ready_for_collection(self) -> bool:
        return self.task_actor_unlocked and self.task_actor_update_count > 0

    def _sync_update_counts(self) -> None:
        if self.last_assessment is not None:
            self.last_assessment["exploration_actor_update_count"] = (
                self.exploration_actor_update_count
            )
            self.last_assessment["task_actor_update_count"] = (
                self.task_actor_update_count
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": ACTOR_READINESS_SCHEMA,
            "criteria": asdict(self.criteria),
            "consecutive_passes": self.consecutive_passes,
            "exploration_unlocked": self.exploration_unlocked,
            "task_actor_consecutive_passes": self.task_actor_consecutive_passes,
            "task_actor_unlocked": self.task_actor_unlocked,
            "exploration_actor_update_count": self.exploration_actor_update_count,
            "task_actor_update_count": self.task_actor_update_count,
            "last_assessment": self.last_assessment,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        expected = {
            "schema_version": ACTOR_READINESS_SCHEMA,
            "criteria": asdict(self.criteria),
        }
        if any(value.get(name) != item for name, item in expected.items()):
            raise ValueError("Actor readiness checkpoint differs")
        self.consecutive_passes = int(value["consecutive_passes"])
        self.exploration_unlocked = bool(value["exploration_unlocked"])
        self.task_actor_consecutive_passes = int(
            value["task_actor_consecutive_passes"]
        )
        self.task_actor_unlocked = bool(value["task_actor_unlocked"])
        self.exploration_actor_update_count = int(
            value["exploration_actor_update_count"]
        )
        self.task_actor_update_count = int(value["task_actor_update_count"])
        last = value.get("last_assessment")
        self.last_assessment = dict(last) if isinstance(last, Mapping) else None
        if min(
            self.consecutive_passes,
            self.task_actor_consecutive_passes,
            self.exploration_actor_update_count,
            self.task_actor_update_count,
        ) < 0:
            raise ValueError("Actor readiness checkpoint counters are invalid")
        if self.exploration_unlocked != (
            self.consecutive_passes >= self.criteria.consecutive_passes
        ):
            raise ValueError("exploration readiness checkpoint state differs")
        if self.task_actor_unlocked != (
            self.task_actor_consecutive_passes >= self.criteria.consecutive_passes
        ):
            raise ValueError("task Actor readiness checkpoint state differs")


def _physical_causality_passed(diagnostic: Mapping[str, object]) -> bool:
    one_step = diagnostic.get("one_step_action_utilization")
    partitions = diagnostic.get("partitions")
    if not isinstance(one_step, Mapping) or not isinstance(partitions, Mapping):
        return False
    aggregate = one_step.get("assessment")
    statistics = one_step.get("shuffle_statistics")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("passed") is not True
        or not isinstance(statistics, Mapping)
        or statistics.get("robust_passed") is not True
    ):
        return False
    required = {"visual_latent", "proprioception"}
    if set(aggregate.get("components", {})) != required:
        return False
    for value in partitions.values():
        if not isinstance(value, Mapping):
            return False
        physical = value.get("one_step_action_utilization")
        if not isinstance(physical, Mapping):
            return False
        assessment = physical.get("assessment")
        statistics = physical.get("shuffle_statistics")
        if (
            not isinstance(assessment, Mapping)
            or assessment.get("passed") is not True
            or set(assessment.get("components", {})) != required
            or not isinstance(statistics, Mapping)
            or statistics.get("robust_passed") is not True
        ):
            return False
    return True


def _probe_partitions_pass(
    probe: Mapping[str, object], criteria: FoundationActorReadinessCriteria
) -> bool:
    partitions = probe.get("partitions")
    if not isinstance(partitions, Mapping) or not partitions:
        return False
    return all(
        isinstance(value, Mapping)
        and float(value.get("state_only_to_state_action_ratio", 0.0))
        >= criteria.minimum_data_action_ratio
        and isinstance(value.get("bootstrap"), Mapping)
        and float(value["bootstrap"].get("ratio_p05", 0.0))
        >= criteria.minimum_data_action_ratio_p05
        for value in partitions.values()
    )


def _coverage_partitions_pass(
    coverage: Mapping[str, object], field: str, minimum: int
) -> bool:
    partitions = coverage.get("partitions")
    return bool(partitions) and isinstance(partitions, Mapping) and all(
        isinstance(value, Mapping) and int(value.get(field, 0)) >= minimum
        for value in partitions.values()
    )


def _selected_checks_pass(
    checks: Mapping[str, bool], names: tuple[str, ...]
) -> bool:
    return all(checks.get(name) is True for name in names)


def failed_exploration_calibration_checks(
    assessment: Mapping[str, object] | None,
) -> tuple[str, ...]:
    if not isinstance(assessment, Mapping):
        raise RuntimeError("foundation calibration has no readiness assessment")
    checks = assessment.get("checks")
    if not isinstance(checks, Mapping):
        raise RuntimeError("foundation calibration checks are missing")
    return tuple(name for name in CALIBRATION_CHECKS if checks.get(name) is not True)


def _deployment_causality_passed(diagnostic: Mapping[str, object]) -> bool:
    assessment = diagnostic.get("assessment")
    statistics = diagnostic.get("shuffle_statistics")
    partitions = diagnostic.get("partitions")
    if (
        not isinstance(assessment, Mapping)
        or assessment.get("passed") is not True
        or not isinstance(statistics, Mapping)
        or statistics.get("robust_passed") is not True
        or not isinstance(partitions, Mapping)
    ):
        return False
    return all(
        isinstance(value, Mapping)
        and value.get("assessment", {}).get("passed") is True
        and value.get("shuffle_statistics", {}).get("robust_passed") is True
        for value in partitions.values()
    )
