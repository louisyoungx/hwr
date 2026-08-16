from __future__ import annotations

from dataclasses import dataclass, replace

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
)
from hwr.eval.bimanual import AblationMode, _wilson_interval


TASK_ID = "bimanual-eval/v1"


def _observation(step: int) -> DualArmObservation:
    return DualArmObservation(
        timestamp_ns=step * 50_000_000,
        sequence_id=step,
        task_id=TASK_ID,
        instruction=NaturalLanguageInstruction("双臂完成测试"),
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
        self._result = None

    def reset(self, *, seed: int, task_id: str) -> DualArmObservation:
        del seed
        assert task_id == TASK_ID
        self.step = 0
        self.applied.clear()
        self._result = None
        return _observation(0)

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

    def spec(self) -> PolicySpec:
        return PolicySpec("reloaded-actor", 1, 1, 20.0, 12)

    def reset(self, *, task_id: str, seed: int) -> None:
        del task_id, seed
        self.feedback.clear()

    def infer(self, observations) -> ActionChunk:
        del observations
        action = DualArmAction(
            0.1,
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
        seed=seed,
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
    return BimanualEvaluationReport("reloaded-actor", tuple(episodes))


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
    report = evaluate_bimanual_policy(
        TASK_ID,
        2,
        factory,
        policy,
        [9001],
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
            [9001],
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
        replace(episode, task_id=second_task) for episode in first.episodes
    )
    assessment = assess_bimanual_acceptance(
        BimanualEvaluationReport(
            "reloaded-actor",
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
