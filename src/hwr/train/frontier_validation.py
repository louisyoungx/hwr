"""Outcome-only validation for autonomously discovered reset frontiers."""

from __future__ import annotations

from typing import Mapping, Protocol

from hwr.train.frontier_curriculum import FrontierEntry, OutcomeFrontierCurriculum


class FrontierValidationConfig(Protocol):
    actuator_initial_dwell_probability: float
    actuator_dwell_closed_probability: float
    paired_gripper_exploration_probability: float
    actuator_dwell_steps: int
    frontier_minimum_contact_stability_steps: int


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
    if not frontier_reset_validation_enabled(config):
        return contact_steps, False, False
    reproduced = frontier.report_reset_outcome(entry, contact_steps)
    if reproduced is None:
        return contact_steps, False, False
    return contact_steps, True, reproduced


def frontier_reset_validation_enabled(config: FrontierValidationConfig) -> bool:
    return (
        config.actuator_initial_dwell_probability == 1.0
        and config.actuator_dwell_closed_probability == 1.0
        and config.paired_gripper_exploration_probability == 1.0
        and config.actuator_dwell_steps
        >= config.frontier_minimum_contact_stability_steps
    )
