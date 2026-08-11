"""Outcome-only validation for autonomously discovered reset frontiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from hwr.core.embodied import DualArmAction, DualArmActionFrame
from hwr.core.runtime import RuntimeStepOutcome
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.core.embodied import DualArmObservation
from hwr.tasks import PrivilegedTaskState
from hwr.train.frontier_curriculum import FrontierEntry, OutcomeFrontierCurriculum


class FrontierValidationConfig(Protocol):
    actuator_initial_dwell_probability: float
    actuator_dwell_closed_probability: float
    paired_gripper_exploration_probability: float
    actuator_dwell_steps: int
    frontier_minimum_contact_stability_steps: int
    frontier_reset_validation_steps: int


class FrontierProbeBackend(Protocol):
    def reset(
        self,
        *,
        seed: int,
        task_id: str,
        initial_state: PhysicalStateSnapshot | None = None,
    ) -> DualArmObservation: ...

    def apply(self, frame: DualArmActionFrame) -> RuntimeStepOutcome: ...

    def privileged_training_state(self) -> PrivilegedTaskState: ...


@dataclass(frozen=True)
class FrontierResetProbe:
    contact_steps: int = 0
    validated: bool = False
    reproduced: bool = False


@dataclass(frozen=True)
class PreparedFrontierReset:
    observation: DualArmObservation
    reset_seed: int
    probe: FrontierResetProbe
    applied: bool


def prepare_frontier_reset(
    environment: FrontierProbeBackend,
    frontier: OutcomeFrontierCurriculum,
    entry: FrontierEntry | None,
    *,
    task_id: str,
    episode_seed: int,
    source_seed: int,
    config: FrontierValidationConfig,
) -> PreparedFrontierReset:
    probe = probe_frontier_reset(
        environment, frontier, entry, seed=source_seed, config=config
    )
    active = entry if not probe.validated or probe.reproduced else None
    reset_seed = source_seed if active is not None else episode_seed
    observation = environment.reset(
        seed=reset_seed,
        task_id=task_id,
        initial_state=(active.snapshot if active else None),
    )
    return PreparedFrontierReset(
        observation,
        reset_seed,
        probe,
        applied=active is not None,
    )


def probe_frontier_reset(
    environment: FrontierProbeBackend,
    frontier: OutcomeFrontierCurriculum,
    entry: FrontierEntry | None,
    *,
    seed: int,
    config: FrontierValidationConfig,
) -> FrontierResetProbe:
    if (
        entry is None
        or entry.signature not in (1, 2, 3)
        or not entry.snapshot.runtime_state
    ):
        return FrontierResetProbe()
    observation = environment.reset(
        seed=seed,
        task_id=entry.snapshot.task_id,
        initial_state=entry.snapshot,
    )
    streak = 0
    longest = 0
    for _ in range(config.frontier_reset_validation_steps):
        outcome = environment.apply(_closed_dwell_frame(observation.timestamp_ns))
        observation = outcome.observation
        physical = frontier.outcome_from_metrics(
            environment.privileged_training_state().metrics
        )
        signature = int(physical.left_contact) | (int(physical.right_contact) << 1)
        if (
            bool(outcome.info["physics_advanced"])
            and not physical.severe_collision
            and signature == entry.signature
        ):
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
        if outcome.terminated or outcome.truncated:
            break
    reproduced = frontier.report_reset_outcome(entry, longest)
    return FrontierResetProbe(
        longest,
        validated=reproduced is not None,
        reproduced=bool(reproduced),
    )


def validate_frontier_reset(
    frontier: OutcomeFrontierCurriculum,
    entry: FrontierEntry | None,
    metrics: Mapping[str, object],
    config: FrontierValidationConfig,
) -> tuple[int, bool, bool]:
    if entry is None:
        return 0, False, False
    fields = {
        1: "left_contact_steps",
        2: "right_contact_steps",
        3: "simultaneous_contact_steps",
    }
    name = fields.get(entry.signature)
    if name is None:
        return 0, False, False
    contact_steps = int(metrics[name])
    if not legacy_frontier_reset_validation_enabled(config):
        return contact_steps, False, False
    reproduced = frontier.report_reset_outcome(entry, contact_steps)
    if reproduced is None:
        return contact_steps, False, False
    return contact_steps, True, reproduced


def legacy_frontier_reset_validation_enabled(
    config: FrontierValidationConfig,
) -> bool:
    return (
        config.actuator_initial_dwell_probability == 1.0
        and config.actuator_dwell_closed_probability == 1.0
        and config.paired_gripper_exploration_probability == 1.0
        and config.actuator_dwell_steps
        >= config.frontier_minimum_contact_stability_steps
    )


def _closed_dwell_frame(timestamp_ns: int) -> DualArmActionFrame:
    period_ns = 50_000_000
    return DualArmActionFrame(
        timestamp_ns,
        timestamp_ns,
        timestamp_ns + 2 * period_ns,
        "autonomous_frontier_validation",
        DualArmAction(0.0, 0.0, (0.0,) * 6, (0.0,) * 6, 1.0, 1.0),
    )
