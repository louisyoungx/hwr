from __future__ import annotations

import torch

from hwr.policy import (
    HouseholdVisualPolicyModel,
    LearnedVisualPolicy,
    VisualModelConfig,
    VisualNormalization,
)


def _policy() -> LearnedVisualPolicy:
    config = VisualModelConfig(8, 8, 2, phase_count=2)
    normalization = VisualNormalization(
        proprioception_mean=(0.0,) * 24,
        proprioception_std=(1.0,) * 24,
        action_mean=(0.0,) * 8,
        action_std=(1.0,) * 8,
    )
    return LearnedVisualPolicy(
        HouseholdVisualPolicyModel(config),
        normalization,
        policy_version="timing-test:v1",
        control_hz=20.0,
        task_instructions={"task/v1": (0, "test")},
        phase_names=("approach", "grasp"),
        phase_action_mask=((True,) * 8, (True,) * 8),
        phase_step_limits=((5, 8), (2, 4)),
    )


def test_phase_cannot_advance_before_learned_minimum() -> None:
    policy = _policy()
    prefer_next = torch.tensor((0.0, 10.0))

    selected = [policy._select_phase(prefer_next) for _ in range(5)]

    assert selected == [0, 0, 0, 0, 0]
    assert policy._select_phase(prefer_next) == 0
    for _ in range(8):
        selected_phase = policy._select_phase(prefer_next)
    assert selected_phase == 1


def test_phase_forces_progress_at_learned_maximum() -> None:
    policy = _policy()
    prefer_current = torch.tensor((10.0, 0.0))

    selected = [policy._select_phase(prefer_current) for _ in range(10)]

    assert 0 in selected
    assert selected[-1] == 1


def test_learned_route_blocks_phase_jump_and_drives_to_endpoint() -> None:
    policy = _policy()
    policy.navigation_routes = {"approach": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
    prefer_next = torch.tensor((0.0, 10.0))

    selected = [
        policy._select_phase(prefer_next, (0.2, 0.0, 0.0)) for _ in range(20)
    ]
    linear, angular = policy._navigation_action((0.2, 0.0, 0.0))

    assert selected[-1] == 0
    assert linear > 0.0
    assert abs(angular) < 1e-6
    for _ in range(10):
        selected_phase = policy._select_phase(prefer_next, (1.0, 0.0, 0.0))
    assert selected_phase == 1
