from __future__ import annotations

from types import SimpleNamespace

from hwr.core import PhysicalStateSnapshot
from hwr.train import FrontierCurriculumConfig, FrontierOutcome, OutcomeFrontierCurriculum
from hwr.train.frontier_validation import probe_frontier_reset


class _ProbeEnvironment:
    def __init__(self) -> None:
        self.reset_count = 0
        self.apply_count = 0
        self.frames = []
        self.observation = SimpleNamespace(timestamp_ns=0)

    def reset(self, *, seed, task_id, initial_state=None):
        self.reset_count += 1
        self.observation = SimpleNamespace(timestamp_ns=0)
        return self.observation

    def apply(self, frame):
        self.apply_count += 1
        self.frames.append(frame)
        self.observation = SimpleNamespace(timestamp_ns=self.apply_count * 50_000_000)
        return SimpleNamespace(
            observation=self.observation,
            info={"physics_advanced": True},
            terminated=False,
            truncated=False,
        )

    def privileged_training_state(self):
        return SimpleNamespace(
            metrics={
                "left_reach_distance": 0.04,
                "right_reach_distance": 0.05,
                "left_contact": 1.0,
                "right_contact": 1.0,
                "severe_collisions": 0.0,
                "support_contact": 1.0,
                "payload_linear_speed": 0.0,
                "payload_angular_speed": 0.0,
            }
        )


def test_complete_contact_frontier_is_probed_outside_actor_experience() -> None:
    frontier = OutcomeFrontierCurriculum(
        ("tray",),
        FrontierCurriculumConfig(
            reset_probability=1.0,
            minimum_contact_stability_steps=2,
        ),
    )
    snapshot = PhysicalStateSnapshot(
        "tray", "test/v1", (0.0,), runtime_state=(1.0,)
    )
    assert frontier.consider(
        "tray",
        snapshot,
        FrontierOutcome(0.04, 0.05, True, True),
        source_episode=3,
        source_step=4,
        contact_stability_steps=2,
    )
    entry = frontier.entries["tray"][0]
    environment = _ProbeEnvironment()
    config = SimpleNamespace(
        frontier_minimum_contact_stability_steps=2,
        frontier_reset_validation_steps=2,
    )

    result = probe_frontier_reset(
        environment, frontier, entry, seed=11, config=config
    )

    assert result.contact_steps == 2
    assert result.validated is True
    assert result.reproduced is True
    assert environment.reset_count == 1
    assert environment.apply_count == 2
    assert all(frame.source == "autonomous_frontier_validation" for frame in environment.frames)
    assert all(frame.action.vector() == (0.0,) * 14 + (1.0, 1.0) for frame in environment.frames)
