from __future__ import annotations

from dataclasses import dataclass

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
    assess_bimanual_acceptance,
    combine_bimanual_reports,
    evaluate_bimanual_policy,
)
from hwr.eval.bimanual import _wilson_interval


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
