from __future__ import annotations

import pytest

from hwr.core.embodied import (
    DUAL_ARM_ACTION_DIM,
    DUAL_ARM_TOOL_TWIST_FIELDS,
    DUAL_ARM_TOOL_TWIST_REFLECTION_SIGNS,
    ActionChunk,
    DualArmAction,
    DualArmActionFrame,
    DualArmObservation,
    DualArmProprioception,
    FrozenLanguageEmbedding,
    NaturalLanguageInstruction,
)
from hwr.core.types import CameraFrame


def _action(value: float = 0.1) -> DualArmAction:
    return DualArmAction(value, -value, (value,) * 6, (-value,) * 6, 0.0, 1.0)


def test_natural_instruction_is_normalized_without_symbolic_parsing() -> None:
    instruction = NaturalLanguageInstruction("  Put the cup\nbeside the sink  ")

    assert instruction.text == "Put the cup beside the sink"
    assert not hasattr(instruction, "object_token")
    assert not hasattr(instruction, "skill_plan")


def test_frozen_language_embedding_requires_weight_identity() -> None:
    embedding = FrozenLanguageEmbedding("local-text/v1", "a" * 64, (0.1, -0.2))

    assert embedding.weights_sha256 == "a" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        FrozenLanguageEmbedding("local-text/v1", "unknown", (0.1,))


def test_dual_arm_action_chunk_round_trips_vectors() -> None:
    action = _action()
    chunk = ActionChunk((action, _action(0.2)), valid_steps=1)

    assert len(action.vector()) == DUAL_ARM_ACTION_DIM
    assert DualArmAction.from_vector(action.vector()) == action
    assert chunk.vectors()[0] == action.vector()
    assert DUAL_ARM_TOOL_TWIST_FIELDS == ("vx", "vy", "vz", "wx", "wy", "wz")
    assert DUAL_ARM_TOOL_TWIST_REFLECTION_SIGNS == (1, -1, 1, -1, 1, -1)


def test_dual_arm_action_rejects_incomplete_arm() -> None:
    with pytest.raises(ValueError, match="six commands"):
        DualArmAction(0.0, 0.0, (0.0,) * 5, (0.0,) * 6, 0.0, 0.0)


def test_dual_arm_runtime_contract_keeps_raw_inputs_and_side_ownership() -> None:
    observation = DualArmObservation(
        timestamp_ns=10,
        sequence_id=2,
        task_id="bimanual_test",
        instruction=NaturalLanguageInstruction("Pick up the tray with both hands"),
        proprioception=DualArmProprioception(
            left_joint_position=(0.1,) * 6,
            left_joint_velocity=(0.2,) * 6,
            right_joint_position=(-0.1,) * 6,
            right_joint_velocity=(-0.2,) * 6,
            left_gripper_position=0.25,
            right_gripper_position=0.75,
            base_pose=(1.0, 2.0, 0.3),
            base_twist=(0.0, 0.1),
        ),
        cameras=(CameraFrame("head_rgb", 10, 2, 1, 1, payload=b"rgb"),),
    )

    assert observation.instruction.text == "Pick up the tray with both hands"
    assert observation.proprioception.vector()[:6] == (0.1,) * 6
    assert observation.camera("head_rgb").payload == b"rgb"
    assert not hasattr(observation, "privileged_state")


def test_dual_arm_action_frame_validates_time_and_source() -> None:
    frame = DualArmActionFrame(10, 10, 20, "actor", _action())

    assert frame.action.vector() == _action().vector()
    with pytest.raises(ValueError, match="inverted"):
        DualArmActionFrame(10, 20, 19, "actor", _action())
    with pytest.raises(ValueError, match="source"):
        DualArmActionFrame(10, 10, 20, "", _action())
