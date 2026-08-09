from __future__ import annotations

from hwr.core.runtime import PolicySpec, StepOutcome
from hwr.core.types import ActionFrame, EpisodeResult, ObservationFrame
from hwr.eval import evaluate_formal_visual_policy


TASK_ID = "formal-eval/v1"


class _Environment:
    def __init__(self) -> None:
        self.step = 0
        self._result = None

    def reset(self, *, seed: int, task_id: str) -> ObservationFrame:
        del seed
        self.step = 0
        return ObservationFrame(0, 0, task_id, "instruction_following")

    def observe(self) -> ObservationFrame:
        return ObservationFrame(self.step, self.step, TASK_ID, "instruction_following")

    def apply(self, action: ActionFrame) -> StepOutcome:
        self.step += 1
        terminated = self.step == 2
        if terminated:
            self._result = EpisodeResult(
                True,
                "stable",
                self.step,
                {"stable_steps": 40, "severe_collisions": 0},
            )
        return StepOutcome(self.observe(), terminated=terminated, info={"applied_action": action})

    def result(self):
        return self._result

    def audit_snapshot(self):
        return {"stable_steps": 40, "severe_collision_count": 0}

    def close(self) -> None:
        pass


class _LearnedPolicy:
    def spec(self) -> PolicySpec:
        return PolicySpec("learned-checkpoint:v1", 1, 1, 20.0, 6)

    def reset(self, *, task_id: str, seed: int) -> None:
        del task_id, seed

    def infer(self, observations):
        observation = observations[-1]
        return (
            ActionFrame(
                observation.timestamp_ns,
                observation.timestamp_ns,
                observation.timestamp_ns + 1,
                "learned:learned-checkpoint:v1",
                arm_command=(0.0,) * 6,
                policy_version="learned-checkpoint:v1",
            ),
        )

    def close(self) -> None:
        pass


def test_formal_evaluator_records_only_learned_actions() -> None:
    report = evaluate_formal_visual_policy(
        TASK_ID,
        3,
        _Environment,
        _LearnedPolicy(),
        [100, 101],
    )

    assert report.success_rate == 1.0
    assert report.to_dict()["severe_collision_count"] == 0
    assert report.episodes[0].action_sources == ("learned:learned-checkpoint:v1",)
