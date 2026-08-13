"""Evidence-based Actor training and collection admission gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


ACTOR_READINESS_SCHEMA = "hwr.foundation-actor-readiness/v1"


@dataclass(frozen=True)
class FoundationActorReadinessCriteria:
    minimum_replay_episodes: int
    consecutive_passes: int = 2
    minimum_active_dimension_fraction: float = 0.75
    minimum_effective_rank: float = 6.0
    minimum_data_action_ratio: float = 1.05
    minimum_data_action_ratio_p05: float = 1.01

    def __post_init__(self) -> None:
        if min(self.minimum_replay_episodes, self.consecutive_passes) <= 0:
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
        self.task_actor_update_count = 0
        self.last_assessment: dict[str, object] | None = None

    def assess(
        self,
        diagnostic: Mapping[str, object],
        data_action_probe: Mapping[str, object],
        action_coverage: Mapping[str, object],
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
        }
        current_passed = all(checks.values())
        self.consecutive_passes = self.consecutive_passes + 1 if current_passed else 0
        self.exploration_unlocked = (
            self.consecutive_passes >= self.criteria.consecutive_passes
        )
        deployment_passed = _deployment_causality_passed(diagnostic)
        task_passed = self.exploration_unlocked and deployment_passed
        self.task_actor_consecutive_passes = (
            self.task_actor_consecutive_passes + 1 if task_passed else 0
        )
        self.task_actor_unlocked = (
            self.task_actor_consecutive_passes >= self.criteria.consecutive_passes
        )
        assessment = {
            "schema_version": ACTOR_READINESS_SCHEMA,
            "passed_this_cycle": current_passed,
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
            "task_actor_update_count": self.task_actor_update_count,
            "task_semantic_fields": [],
        }
        self.last_assessment = assessment
        return assessment

    def record_task_actor_updates(self, count: int) -> None:
        if count < 0:
            raise ValueError("Actor readiness update count cannot be negative")
        self.task_actor_update_count += count

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": ACTOR_READINESS_SCHEMA,
            "criteria": asdict(self.criteria),
            "consecutive_passes": self.consecutive_passes,
            "exploration_unlocked": self.exploration_unlocked,
            "task_actor_consecutive_passes": self.task_actor_consecutive_passes,
            "task_actor_unlocked": self.task_actor_unlocked,
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
        self.task_actor_update_count = int(value["task_actor_update_count"])
        last = value.get("last_assessment")
        self.last_assessment = dict(last) if isinstance(last, Mapping) else None
        if min(
            self.consecutive_passes,
            self.task_actor_consecutive_passes,
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
