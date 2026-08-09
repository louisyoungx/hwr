"""Capture learned-policy closed-loop rollouts without changing simulation state."""

from __future__ import annotations

from dataclasses import dataclass

from hwr.core.runtime import Policy
from hwr.core.types import EpisodeEvent, EpisodeResult
from hwr.sim import Household2DEnv, HouseholdTaskSpec, RobotSpec, SimulationSnapshot


@dataclass(frozen=True)
class RolloutFrame:
    snapshot: SimulationSnapshot
    events: tuple[EpisodeEvent, ...] = ()


@dataclass(frozen=True)
class RolloutTrace:
    task_id: str
    policy_id: str
    seed: int
    control_hz: float
    frames: tuple[RolloutFrame, ...]
    result: EpisodeResult

    @property
    def duration_seconds(self) -> float:
        return (len(self.frames) - 1) / self.control_hz


def capture_rollout(
    task_spec: HouseholdTaskSpec,
    robot_spec: RobotSpec,
    policy: Policy,
    *,
    seed: int,
) -> RolloutTrace:
    """Run one real policy/environment loop and retain immutable render frames."""
    environment = Household2DEnv(robot_spec, task_spec)
    frames: list[RolloutFrame] = []
    try:
        observation = environment.reset(seed=seed, task_id=task_spec.task_id)
        policy.reset(task_id=task_spec.task_id, seed=seed)
        frames.append(RolloutFrame(environment.snapshot()))
        for _ in range(task_spec.max_steps):
            action_chunk = policy.infer((observation,))
            if not action_chunk:
                raise RuntimeError("policy returned an empty action chunk")
            outcome = environment.apply(action_chunk[0])
            observation = outcome.observation
            frames.append(RolloutFrame(environment.snapshot(), outcome.events))
            if outcome.terminated or outcome.truncated:
                break
        result = environment.result()
        if result is None:
            raise RuntimeError("rollout ended without an episode result")
        return RolloutTrace(
            task_id=task_spec.task_id,
            policy_id=policy.spec().policy_id,
            seed=seed,
            control_hz=robot_spec.control_hz,
            frames=tuple(frames),
            result=result,
        )
    finally:
        environment.close()
