from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import random

import pytest

from hwr.core.embodied import (
    ActionChunk,
    DualArmAction,
    DualArmActionFrame,
    DualArmObservation,
    DualArmProprioception,
    NaturalLanguageInstruction,
)
from hwr.core.runtime import PolicySpec, RuntimeStepOutcome
from hwr.core.types import EpisodeResult
from hwr.eval import (
    BimanualAcceptanceCriteria,
    BimanualEpisodeEvaluation,
    BimanualEvaluationReport,
    assess_bimanual_acceptance,
    combine_bimanual_reports,
    evaluate_bimanual_policy,
    plan_episode_seeds,
)
from hwr.eval.bimanual import AblationMode, _wilson_interval


TASK_ID = "bimanual-eval/v1"


def _observation(step: int) -> DualArmObservation:
    return DualArmObservation(
        timestamp_ns=step * 50_000_000,
        sequence_id=step,
        task_id=TASK_ID,
        instruction=NaturalLanguageInstruction("Complete the test with both arms"),
        proprioception=DualArmProprioception(
            (0.0,) * 6,
            (0.0,) * 6,
            (0.0,) * 6,
            (0.0,) * 6,
            0.25,
            0.75,
            (0.0, 0.0, 0.0),
            (0.0, 0.0),
        ),
        cameras=(),
    )


class _Environment:
    def __init__(self) -> None:
        self.step = 0
        self.applied: list[DualArmAction] = []
        self.received_seeds: list[int] = []
        self.observations: list[DualArmObservation] = []
        self._result = None

    def reset(self, *, seed: int, task_id: str) -> DualArmObservation:
        assert task_id == TASK_ID
        self.received_seeds.append(seed)
        self.step = 0
        self.applied.clear()
        self._result = None
        observation = _observation(0)
        self.observations.append(observation)
        return observation

    def observe(self) -> DualArmObservation:
        return _observation(self.step)

    def apply(self, frame: DualArmActionFrame) -> RuntimeStepOutcome:
        self.applied.append(frame.action)
        self.step += 1
        terminated = self.step == 2
        if terminated:
            self._result = EpisodeResult(True, "stable", self.step, {})
        return RuntimeStepOutcome(
            self.observe(),
            terminated=terminated,
            info={
                "applied_action": frame,
                "safety_intervened": False,
            },
        )

    def result(self):
        return self._result

    def task_audit(self):
        return {
            "stable_steps": 40,
            "maximum_concurrent_steps": 20,
            "left_contact_steps": 20,
            "right_contact_steps": 20,
            "simultaneous_contact_steps": 20,
            "severe_collision_count": 0,
            "maximum_forbidden_force": 12.0,
        }

    def close(self) -> None:
        pass


@dataclass
class _Policy:
    feedback: list[DualArmAction]
    received_seeds: list[int] | None = None
    action_trace: list[float] | None = None
    generator: random.Random | None = None

    def spec(self) -> PolicySpec:
        return PolicySpec("reloaded-actor", 1, 1, 20.0, 12)

    def reset(self, *, task_id: str, seed: int) -> None:
        del task_id
        if self.received_seeds is not None:
            self.received_seeds.append(seed)
        self.generator = random.Random(seed)
        self.feedback.clear()

    def infer(self, observations) -> ActionChunk:
        del observations
        value = self.generator.random() if self.generator is not None else 0.1
        if self.action_trace is not None:
            self.action_trace.append(value)
        action = DualArmAction(
            value,
            0.2,
            (0.3,) * 6,
            (0.4,) * 6,
            1.0,
            1.0,
        )
        return ActionChunk((action,), 1)

    def record_applied_action(self, action: DualArmAction) -> None:
        self.feedback.append(action)

    def close(self) -> None:
        pass


def _episode(
    seed: int,
    ablation: AblationMode,
    *,
    success: bool,
    steps: int = 100,
    safety_interventions: int = 0,
) -> BimanualEpisodeEvaluation:
    return BimanualEpisodeEvaluation(
        task_id=TASK_ID,
        environment_seed=seed,
        policy_rng_seed=seed + 10_000,
        planned_episode_id=hashlib.sha256(
            f"{seed}:{ablation}".encode()
        ).hexdigest(),
        seed_commitment="a" * 64,
        ablation=ablation,
        success=success,
        reason="stable" if success else "timeout",
        steps=steps,
        stable_steps=40 if success else 0,
        maximum_concurrent_steps=20 if success else 0,
        left_contact_steps=20 if success else 0,
        right_contact_steps=20 if success else 0,
        simultaneous_contact_steps=20 if success else 0,
        severe_collisions=0,
        maximum_forbidden_force=12.0,
        safety_interventions=safety_interventions,
        action_sources=("learned:reloaded-actor",),
        audit={},
    )


def _acceptance_report(
    normal_interventions: tuple[int, ...],
    *,
    normal_success: bool = True,
    normal_steps: int = 100,
) -> BimanualEvaluationReport:
    episodes = [
        _episode(
            seed,
            "none",
            success=normal_success,
            steps=normal_steps,
            safety_interventions=interventions,
        )
        for seed, interventions in enumerate(normal_interventions)
    ]
    for ablation in ("lock_left", "lock_right"):
        episodes.extend(
            _episode(seed, ablation, success=False)
            for seed in range(len(normal_interventions))
        )
    return BimanualEvaluationReport(
        "reloaded-actor", "acceptance-fixture", tuple(episodes)
    )


def _assess(
    report: BimanualEvaluationReport,
) -> dict[str, object]:
    return assess_bimanual_acceptance(report, {TASK_ID: 20.0})


def test_bimanual_evaluation_uses_runtime_feedback_and_locks_left_arm() -> None:
    environments: list[_Environment] = []

    def factory() -> _Environment:
        environment = _Environment()
        environments.append(environment)
        return environment

    policy = _Policy([])
    episode_seeds = plan_episode_seeds(
        "evaluator-fixture",
        TASK_ID,
        "lock_left",
        1,
        "test-seed-isolation",
        environment_seeds=(9001,),
    )
    report = evaluate_bimanual_policy(
        TASK_ID,
        2,
        factory,
        policy,
        episode_seeds,
        ablation="lock_left",
    )

    assert report.episodes[0].success
    assert report.episodes[0].action_sources == ("learned:reloaded-actor",)
    assert environments[0].applied[0].left_arm == (0.0,) * 6
    assert environments[0].applied[0].left_gripper == 0.25
    assert environments[0].applied[0].right_arm == (0.4,) * 6
    assert policy.feedback == environments[0].applied


def test_acceptance_requires_normal_success_and_both_single_arm_ablations() -> None:
    policy = _Policy([])
    reports = [
        evaluate_bimanual_policy(
            TASK_ID,
            2,
            _Environment,
            policy,
            plan_episode_seeds(
                "acceptance-evaluator-fixture",
                TASK_ID,
                mode,
                1,
                "test-seed-isolation",
                environment_seeds=(9001,),
            ),
            ablation=mode,
        )
        for mode in ("none", "lock_left", "lock_right")
    ]
    report = combine_bimanual_reports(reports)
    criteria = BimanualAcceptanceCriteria(
        minimum_unseen_episodes=1,
        minimum_success_rate=1.0,
        maximum_ablation_success_rate=1.0,
        minimum_stable_seconds=2.0,
    )

    assessment = assess_bimanual_acceptance(
        report, {TASK_ID: 20.0}, criteria
    )

    assert not assessment["passed"]
    assert assessment["tasks"][0]["normal_success_rate"] == 1.0
    assert assessment["tasks"][0]["ablation_success_rates"] == {
        "lock_left": 1.0,
        "lock_right": 1.0,
    }


def test_environment_and_policy_receive_only_their_planned_seed_domains() -> None:
    environments: list[_Environment] = []

    def factory() -> _Environment:
        environment = _Environment()
        environments.append(environment)
        return environment

    policy_seeds: list[int] = []
    policy = _Policy([], policy_seeds)
    episode_seeds = plan_episode_seeds(
        "domain-fixture",
        TASK_ID,
        "none",
        2,
        "domain-separation",
        environment_seeds=(9001, 9002),
    )

    report = evaluate_bimanual_policy(
        TASK_ID,
        2,
        factory,
        policy,
        episode_seeds,
    )

    assert [environment.received_seeds for environment in environments] == [
        [9001],
        [9002],
    ]
    assert policy_seeds == [
        episode.policy_rng_seed for episode in episode_seeds
    ]
    assert not set(policy_seeds) & {9001, 9002}
    assert [
        (
            episode.environment_seed,
            episode.policy_rng_seed,
            episode.planned_episode_id,
            episode.seed_commitment,
        )
        for episode in report.episodes
    ] == [
        (
            episode.environment_seed,
            episode.policy_rng_seed,
            episode.planned_episode_id,
            episode.seed_commitment,
        )
        for episode in episode_seeds
    ]


def test_raw_environment_seed_sequence_has_no_policy_fallback() -> None:
    policy_seeds: list[int] = []

    def unexpected_environment() -> _Environment:
        pytest.fail("raw seed validation happened after environment creation")

    with pytest.raises(TypeError, match="planned seed records"):
        evaluate_bimanual_policy(
            TASK_ID,
            2,
            unexpected_environment,
            _Policy([], policy_seeds),
            [9001],  # type: ignore[list-item]
        )

    assert policy_seeds == []


def test_compatibility_environment_observation_and_policy_trace_replay() -> None:
    legacy_environment = _Environment()
    legacy_observation = legacy_environment.reset(seed=9001, task_id=TASK_ID)
    legacy_audit = legacy_environment.task_audit()
    first_environments: list[_Environment] = []
    second_environments: list[_Environment] = []
    first_trace: list[float] = []
    second_trace: list[float] = []
    episode_seeds = plan_episode_seeds(
        "compatibility-fixture",
        TASK_ID,
        "none",
        1,
        "compatibility-replay",
        environment_seeds=(9001,),
    )

    first = evaluate_bimanual_policy(
        TASK_ID,
        2,
        lambda: _capture_environment(first_environments),
        _Policy([], action_trace=first_trace),
        episode_seeds,
    )
    second = evaluate_bimanual_policy(
        TASK_ID,
        2,
        lambda: _capture_environment(second_environments),
        _Policy([], action_trace=second_trace),
        episode_seeds,
    )

    assert first_environments[0].received_seeds == [9001]
    assert second_environments[0].received_seeds == [9001]
    assert first_environments[0].observations[0] == legacy_observation
    assert first.episodes[0].audit == legacy_audit
    assert first_environments[0].observations == second_environments[0].observations
    assert first_trace == second_trace
    assert first.to_dict() == second.to_dict()


def _capture_environment(environments: list[_Environment]) -> _Environment:
    environment = _Environment()
    environments.append(environment)
    return environment


def test_wilson_interval_does_not_treat_fourteen_of_twenty_as_seventy_percent() -> None:
    lower, upper = _wilson_interval(14, 20, 0.95)

    assert lower < 0.50
    assert upper > 0.70


def test_acceptance_rejects_high_safety_intervention_burden_alone() -> None:
    assessment = _assess(_acceptance_report((2,) * 40))
    task = assessment["tasks"][0]
    burden = task["safety_intervention_burden"]

    assert not assessment["passed"]
    assert task["normal_success_rate"] == 1.0
    assert task["normal_success_interval"]["lower"] >= 0.70
    assert task["ablation_success_rates"] == {
        "lock_left": 0.0,
        "lock_right": 0.0,
    }
    assert burden["empirical_p95"] == 0.02
    assert burden["checks"] == {
        "empirical_p95": False,
        "bootstrap_upper": True,
        "maximum": True,
    }


def test_acceptance_allows_zero_safety_interventions_and_reports_v3_schema() -> None:
    assessment = _assess(_acceptance_report((0,) * 40))
    burden = assessment["tasks"][0]["safety_intervention_burden"]

    assert assessment["passed"]
    assert assessment["schema_version"] == "hwr.bimanual-acceptance/v3"
    assert assessment["criteria"]["maximum_safety_intervention_rate_p95"] == 0.01
    assert (
        assessment["criteria"][
            "maximum_safety_intervention_rate_bootstrap_upper"
        ]
        == 0.02
    )
    assert assessment["criteria"]["maximum_safety_intervention_rate"] == 0.05
    assert burden["empirical_p95"] == 0.0
    assert burden["maximum"] == 0.0
    assert burden["bootstrap"]["samples"] == 2_000
    assert burden["bootstrap"]["seed"] == 20_260_913
    assert burden["bootstrap"]["p95_upper"] == 0.0
    assert len(burden["bootstrap"]["p95_distribution"]) == 2_000


def test_acceptance_rejects_one_extreme_episode_by_maximum_rate() -> None:
    report = _acceptance_report((0,) * 99 + (6,))
    assessment = _assess(report)
    burden = assessment["tasks"][0]["safety_intervention_burden"]

    assert not assessment["passed"]
    assert burden["empirical_p95"] == 0.0
    assert burden["bootstrap"]["p95_upper"] == 0.0
    assert burden["maximum"] == 0.06
    assert burden["checks"] == {
        "empirical_p95": True,
        "bootstrap_upper": True,
        "maximum": False,
    }


def test_safety_intervention_bootstrap_is_reproducible() -> None:
    report = _acceptance_report((0,) * 35 + (1,) * 5)

    first = _assess(report)["tasks"][0]["safety_intervention_burden"]["bootstrap"]
    second = _assess(report)["tasks"][0]["safety_intervention_burden"]["bootstrap"]

    assert first == second
    assert first["seed"] == 20_260_913
    assert len(first["p95_distribution"]) == 2_000
    assert len(set(first["p95_distribution"])) > 1


def test_safety_intervention_bootstrap_seed_uses_sorted_task_index() -> None:
    first = _acceptance_report((0,) * 40)
    second_task = "another-task/v1"
    second = tuple(
        replace(
            episode,
            task_id=second_task,
            planned_episode_id=hashlib.sha256(
                f"{second_task}:{episode.planned_episode_id}".encode()
            ).hexdigest(),
        )
        for episode in first.episodes
    )
    assessment = assess_bimanual_acceptance(
        BimanualEvaluationReport(
            "reloaded-actor",
            "multi-task-fixture",
            first.episodes + second,
        ),
        {TASK_ID: 20.0, second_task: 20.0},
    )

    assert [
        task["safety_intervention_burden"]["bootstrap"]["seed"]
        for task in assessment["tasks"]
    ] == [20_260_913, 20_365_642]


def test_low_intervention_non_moving_policy_still_fails_success_gate() -> None:
    assessment = _assess(
        _acceptance_report((0,) * 40, normal_success=False, normal_steps=0)
    )
    task = assessment["tasks"][0]
    burden = task["safety_intervention_burden"]

    assert not assessment["passed"]
    assert task["normal_success_rate"] == 0.0
    assert burden["passed"]
    assert burden["empirical_p95"] == 0.0
    assert burden["maximum"] == 0.0
